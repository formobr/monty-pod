"""A queued event is never reported undelivered while it merely waits its turn.

`_sender_loop` takes whatever is in the outbox as one bounded window (`ACK_WINDOW`). When several
callers append under load, a frame that lands in a LATER window still gets delivered, but a caller
whose `_await_delivery` wall is measured only from its own append — never adjusted for how long an
EARLIER window took to settle — can see that wall expire first and be told False, even though its
frame goes on to deliver cleanly moments later. The caller's verdict must depend only on whether its
own frame is ever delivered, never on how many frames were ahead of it.

This is a seam test against a fake CP peer: real ack wire frames are fed back through `conn.recv()`,
so they travel the stream's own `_read_loop` / `_validate_ack` / `_record_clock_sample_locked` path —
nothing about the stream's internal ACK bookkeeping (`_acks`, `_ack_waiters`) or its `_append`/
`_sender_loop` scheduling is poked or monkeypatched directly; only the transport seam (`_ws_client`,
same as tests/test_cp_retry.py) and the derived-wall module constants are. The other seam is the
injected clock: the caller-wall arithmetic is driven by it, never by real elapsed time. Every blocking
wait below either ends on a real event (a write landing, an append happening, a thread finishing) with
no timeout at all, or — where a genuine regression could otherwise hang forever — ends on a generous
real-time BACKSTOP that never decides the normal pass path (it is never close to being reached when
the code under test is correct) and, on firing, fails the test outright rather than nudging the clock:
a clock nudge driven by elapsed real time would itself depend on machine speed, exactly the class of
flake this slice closes.
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

import pytest

from podagent import event_stream
from podagent.event_stream import EventStream
from test_cp_retry import _ack, _job  # reuse the real ack wire shape, not an invented one (TESTING.md §3b.3)

_CLOSE = object()
_BACKSTOP_S = 30.0  # a generous safety net against a genuine hang, never a value anything in this
                     # file's own correctness arithmetic depends on. Ordinary queue/thread handoff never
                     # takes anywhere near this long, so it is never reached on the normal pass path; it
                     # exists solely to turn a regression that leaves something stuck forever into a
                     # clear, bounded failure instead of hanging the whole suite. It never touches the
                     # injected clock — only `step` (test 1) and `_delivery_wall_s()` (test 2) do that,
                     # on a fixed, real-time-independent schedule driven by the test's own actions.


def _bounded_get(q: "queue.Queue[dict[str, Any]]", what: str) -> dict[str, Any]:
    """The next real item, or a clear failure — never a hang — if none arrives within `_BACKSTOP_S`."""
    try:
        return q.get(timeout=_BACKSTOP_S)
    except queue.Empty:
        pytest.fail(f"{what} never arrived — the sender appears stuck")


def _bounded_join(threads: list[threading.Thread]) -> None:
    """Every thread joined, or a clear failure — never a hang — if one is still alive after
    `_BACKSTOP_S`."""
    for t in threads:
        t.join(timeout=_BACKSTOP_S)
    assert all(not t.is_alive() for t in threads), "not every caller settled"


def _wait_for_all_appended(
        stream: EventStream, n: int, threads: list[threading.Thread]) -> None:
    """Blocks until every caller's frame is durably in the outbox, or until every one of `threads` has
    already returned — whichever happens first; never forever. This parks on the stream's own delivery
    condition variable, notified on every append AND (see each test's `_call`) by a finishing caller
    thread's own `finally` — so a regression that makes `send_event` raise BEFORE appending is noticed
    the instant that thread exits, rather than hanging here. The explicit check afterwards turns that
    exact scenario into an immediate, clear failure instead of a silently short count."""
    with stream._work:
        stream._work.wait_for(
            lambda: stream.pending_count() >= n or all(not t.is_alive() for t in threads))
    assert stream.pending_count() >= n, (
        f"only {stream.pending_count()}/{n} caller frame(s) were durably appended, but every caller "
        "thread already finished — a caller must have raised before appending")


class _FakeClock:
    """A monotonic clock the test drives by hand; never tied to real elapsed time."""

    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.Lock()
        self._conditions: list[threading.Condition] = []
        self._read_cv = threading.Condition(self._lock)
        self._readers_since_advance: set[str] = set()

    def __call__(self) -> float:
        with self._lock:
            self._readers_since_advance.add(threading.current_thread().name)
            self._read_cv.notify_all()
            return self._now

    def wait_read_by(self, thread_name: str) -> None:
        """Returns once `thread_name` has read the clock at the time set by the latest `advance()` — i.e.
        that thread has demonstrably looked at the new time. The bound is only a hang net."""
        with self._read_cv:
            if not self._read_cv.wait_for(
                    lambda: thread_name in self._readers_since_advance, timeout=_BACKSTOP_S):
                pytest.fail(f"{thread_name} never read the advanced clock")

    def attach(self, condition: threading.Condition) -> None:
        """Wires this clock to a stream condition: `advance()` then notifies it, so a caller parked in
        `_await_delivery`'s `_work.wait(...)` (or the window watchdog parked on `_ack_cv`) wakes THE
        INSTANT this clock moves past its deadline, rather than depending on a real-time poll tick
        landing inside some narrow window — the exact race a scheduling-starved batch run could miss."""
        with self._lock:
            self._conditions.append(condition)

    def advance(self, delta: float) -> None:
        with self._lock:
            self._now += delta
            self._readers_since_advance.clear()
            conditions = list(self._conditions)
        for condition in conditions:
            with condition:
                condition.notify_all()


class _FakeConn:
    """A fake CP peer's end of the socket. `send` captures what the pod wrote; `recv` blocks for a
    real ack wire frame the test hands back — the normal `_read_loop`/`_validate_ack` path processes
    it exactly as it would a real server's reply, so a real ack-path regression still shows up here."""

    def __init__(self, written: "queue.Queue[dict[str, Any]]", incoming: "queue.Queue[Any]") -> None:
        self._written = written
        self._incoming = incoming
        self.closed = threading.Event()
        self._recv_cv = threading.Condition()
        self._recv_calls = 0

    def send(self, data: str) -> None:
        self._written.put(json.loads(data))

    def recv(self) -> str:
        with self._recv_cv:
            self._recv_calls += 1
            self._recv_cv.notify_all()
        item = self._incoming.get()  # blocks for the next real ack; no wall-clock bound
        if item is _CLOSE:
            raise OSError("fake connection closed")
        return item

    def wait_recv_calls(self, n: int) -> None:
        """Returns once the reader has come back for its `n`th frame, i.e. fully handled the first n-1.
        The bound is only a hang net for a reader stuck behind an earlier frame; a reader that gets there
        never depends on it."""
        with self._recv_cv:
            if not self._recv_cv.wait_for(lambda: self._recv_calls >= n, timeout=_BACKSTOP_S):
                pytest.fail(f"the reader never came back for frame {n} — it is stuck handling an earlier one")

    def close(self) -> None:
        self.closed.set()
        self._incoming.put(_CLOSE)


def test_a_frame_queued_behind_an_earlier_window_is_reported_delivered(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event_stream, "ACK_WINDOW", 2)
    # OPEN_WALL_S/FRAME_WALL_S are left at their real defaults deliberately: `_exchange_window`'s write/
    # ACK wall is read off the injected clock (see `_exchange_window`), so no real elapsed time — however
    # slow this machine is — can ever make it expire on its own; only this test's own `clock.advance()`
    # decides that, below.
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 0)

    written: "queue.Queue[dict[str, Any]]" = queue.Queue()
    incoming: "queue.Queue[Any]" = queue.Queue()
    monkeypatch.setattr(event_stream, "_ws_client", type(
        "_Client", (), {"connect": staticmethod(lambda *_a, **_k: _FakeConn(written, incoming))}))

    clock = _FakeClock()
    n = 6  # ACK_WINDOW=2 caps every dispatch at 2, so 6 frames always take at least 3 windows
    stream = EventStream(
        "http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json", clock=clock)
    clock.attach(stream._work)
    try:
        results: list[bool | BaseException] = [False] * n

        def _call(i: int) -> None:
            try:
                results[i] = stream.send_event(
                    {"stage": "test", "status": "step", "step": f"e{i}"}, wait=True)
            except BaseException as e:  # noqa: BLE001 - captured for the assertion below
                results[i] = e
            finally:
                with stream._work:
                    stream._work.notify_all()

        threads = [threading.Thread(target=_call, args=(i,), daemon=True) for i in range(n)]
        for t in threads:
            t.start()
        # Force the split for real: with ACK_WINDOW=2, 6 frames durably in the outbox before the sender
        # ever dispatches take at least 3 windows, no matter how the sender happens to interleave with
        # these 6 appends otherwise.
        _wait_for_all_appended(stream, n, threads)

        # Advance a FIXED amount before every real ack, rather than assuming a fixed window layout
        # (which window gets which frame is a real thread-scheduling race). `step` must stay BELOW
        # `window_cap`: a window can hold up to ACK_WINDOW frames sharing that window's own, single
        # write/ACK wall (`FRAME_WALL_S`, computed fresh at that window's own dispatch) — advancing by
        # `step` once per ack inside the same window must never, by itself, exceed that wall, or the
        # window's own exchange times out and the sender falls into its background-retry wait, which
        # nothing but a write this test is blocked waiting for can ever clear (a deadlock, not a race).
        # `step` must stay ABOVE `total_floor`: a frame kept waiting across several windows must still
        # accumulate enough elapsed (simulated) time that its ORIGINAL, unextended wall (measured only
        # from its own append) would already be gone by the time its turn finally comes — exactly the
        # queueing-dependent false negative this slice closes. Each dispatched frame's own wall always
        # restarts fresh at its own dispatch (see `_sender_loop`), so two acks within the SAME window
        # never add up past one wall on the EXTENDED side — only the unextended, pre-fix side needs
        # `total_floor` to be exceeded. This is the ONLY place the clock moves in this test: a stuck
        # sender below fails via `_bounded_get`'s real-time backstop, never via a clock advance.
        window_cap = event_stream.FRAME_WALL_S / event_stream.ACK_WINDOW
        total_floor = event_stream._delivery_wall_s() / (n - 1)
        assert total_floor < window_cap, (
            "no step value can satisfy both bounds with these constants — ACK_WINDOW, FRAME_WALL_S "
            "or n must change together")
        step = (total_floor + window_cap) / 2
        for _ in range(n):
            frame = _bounded_get(written, "a queued frame's write")
            clock.advance(step)
            incoming.put(_ack(frame))

        _bounded_join(threads)
    finally:
        stream.close()

    assert results == [True] * n, results


def test_a_caller_behind_a_never_acked_head_is_told_false_within_one_wall(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A head frame the CP never definitively ACKs gets RE-dispatched every retry cycle. A re-dispatch of
    an already-attempted frame is not a new window queued ahead of anyone, so it must not keep granting
    the caller behind it fresh patience — that caller's bound must stay at roughly one-or-two windows'
    wall, not grow with the head's retry count (which can run for its own, much longer, dead-letter
    budget): a caller behind a head the CP never ACKs still gets an answer within a bounded wait.

    Only public surface is touched: `send_event`, `pending_count()` (via `_wait_for_all_appended`) and
    the fake connection's own `written`/`incoming` queues — never the stream's internal ACK or
    retry-attempt bookkeeping.

    `_sender_loop` grants a deadline extension only the FIRST time a given frame identity is ever
    sliced into a window (tracked by `_window_dispatched`, which a key enters once and never leaves
    while merely retried) — never again on any later re-dispatch of that SAME identity, with or
    without a reply. So this only needs to prove the head's very FIRST dispatch — answered once with
    an ordinary unmatched-409 transport-retry reply, never a real ack-wall timeout — then a single
    clock advance well past the queued caller's now-fixed (at-most-once-extended) deadline: whatever
    the head goes on to do afterward (more retries, eventually its own separately bounded dead-letter
    cycle) cannot add anything further to that deadline, so this test does not need to keep watching
    or answering it. The clock only ever moves by that one fixed, derived amount — never a real-time
    poll deciding whether or how much to advance — so the normal pass path never waits on elapsed real
    time at all; the only real-time bound left is `_BACKSTOP_S`, a pure safety net for a genuine hang.
    """
    # ACK_WINDOW starts at 0 — nothing can be dispatched — until BOTH frames are durably appended
    # below. Without this gate there is a genuine race: the sender wakes the instant "head" is
    # appended and may slice and dispatch its very first (and ONLY extension-granting) window before
    # "behind" even exists, so the extension loop in `_sender_loop` finds no deadline yet to extend —
    # "behind" would then carry only its bare, unextended wall, exactly the bug this slice closes, but
    # here as a test artifact rather than a real one. Restored to 1 right after both appends land,
    # which is what actually keeps the stuck head and the queued caller apart from then on.
    monkeypatch.setattr(event_stream, "ACK_WINDOW", 0)
    # OPEN_WALL_S/FRAME_WALL_S are left at their real defaults, exactly like the first test — the write/
    # ACK wall is read off the injected clock, so no real elapsed time decides this test's outcome.
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 0)

    written: "queue.Queue[dict[str, Any]]" = queue.Queue()
    incoming: "queue.Queue[Any]" = queue.Queue()
    monkeypatch.setattr(event_stream, "_ws_client", type(
        "_Client", (), {"connect": staticmethod(lambda *_a, **_k: _FakeConn(written, incoming))}))
    # Every frame actually put on the wire — the proof that "behind" settles on its OWN wall rather
    # than via some other path (for example the head eventually dead-lettering, which would reopen
    # admission and let "behind" itself reach the wire).
    sent_steps: list[str | None] = []

    clock = _FakeClock()
    stream = EventStream(
        "http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json", clock=clock)
    clock.attach(stream._work)
    try:
        stream.send_event({"stage": "test", "status": "step", "step": "head"}, wait=False)

        result: dict[str, bool | BaseException] = {}
        done = threading.Event()

        def _call() -> None:
            try:
                result["behind"] = stream.send_event(
                    {"stage": "test", "status": "step", "step": "behind"}, wait=True)
            except BaseException as e:  # noqa: BLE001 - captured for the assertion below
                result["behind"] = e
            finally:
                done.set()
                with stream._work:
                    stream._work.notify_all()

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        # The behind frame's deadline is set at ITS OWN append; it must be durably appended before any
        # clock advance below, or its append could race past that advance and see a deadline measured
        # from clock time that has already moved on. With ACK_WINDOW still 0, the sender cannot yet
        # dispatch anything, so this ordering is guaranteed rather than merely likely.
        _wait_for_all_appended(stream, 2, [t])

        # Only now let the sender actually dispatch anything: "head" is durably at the outbox HEAD and
        # "behind" right behind it, so head's first dispatch — the only one that ever grants an
        # extension — is guaranteed to find "behind"'s deadline already in place to extend.
        monkeypatch.setattr(event_stream, "ACK_WINDOW", 1)
        with stream._work:
            stream._work.notify_all()

        def _reply_409(frame: dict[str, Any]) -> None:
            sent_steps.append(frame.get("event", {}).get("step"))
            incoming.put(_ack(frame, status=409, error="simulated transient control-plane backpressure"))

        # The head's one and only dispatch that can ever grant an extension (see docstring): answer it
        # with an ordinary transport-retryable 409, never a real ack-wall timeout.
        _reply_409(_bounded_get(written, "the head's first dispatch"))

        # "behind"'s deadline is now fixed at its original wall plus this ONE extension, for good — no
        # later re-dispatch of the head can add to it. A single jump well past that fixed point is
        # therefore enough, regardless of whatever the head itself goes on to do afterward.
        clock.advance(3 * event_stream._delivery_wall_s())
    finally:
        stream.close()

    assert done.wait(timeout=_BACKSTOP_S), (
        "'behind' never settled — a genuine regression, not a hang: the backstop above never touches "
        "the clock, so this only fires if the retry/settlement path itself is broken")
    assert result["behind"] is False
    # "behind" never reached the wire at all: were the regression this test guards against still present
    # (every re-dispatch of the stuck head compounding the queued caller's patience, or the head instead
    # being driven all the way to dead-letter, which reopens admission), "behind" would eventually have
    # been dispatched and this would fail.
    assert "behind" not in sent_steps


def _open_stream(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, attach_work: bool = True,
) -> "tuple[EventStream, _FakeClock, list[_FakeConn], queue.Queue[dict[str, Any]], queue.Queue[Any]]":
    """A stream on a fake peer whose dispatches are gated (ACK_WINDOW 0) until the test opens them."""
    monkeypatch.setattr(event_stream, "ACK_WINDOW", 0)
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 0)
    written: "queue.Queue[dict[str, Any]]" = queue.Queue()
    incoming: "queue.Queue[Any]" = queue.Queue()
    conns: list[_FakeConn] = []

    def _connect(*_a: Any, **_k: Any) -> _FakeConn:
        conns.append(_FakeConn(written, incoming))
        return conns[-1]

    monkeypatch.setattr(event_stream, "_ws_client", type("_Client", (), {"connect": staticmethod(_connect)}))
    clock = _FakeClock()
    stream = EventStream(
        "http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json", clock=clock)
    if attach_work:
        clock.attach(stream._work)
    clock.attach(stream._ack_cv)
    return stream, clock, conns, written, incoming


def _release_window(stream: EventStream, monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    monkeypatch.setattr(event_stream, "ACK_WINDOW", size)
    with stream._work:
        stream._work.notify_all()


def test_acks_are_registered_while_the_outbox_fsync_is_stuck(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prod 2026-10-24 ack_wall loop: a job frame arriving mid-window makes the pod write its inbox and
    job_ack durably. With the disk busy, that write must not hold back the registration of the acks that
    arrive behind it — the socket stays open and the window completes the moment the disk answers."""
    # `advance()` must not take `_work`: the stuck durable write holds it, and the window watchdog under
    # test parks on `_ack_cv` alone, so that is the only condition this clock needs to wake.
    stream, clock, conns, written, incoming = _open_stream(tmp_path, monkeypatch, attach_work=False)
    armed, release, stuck = threading.Event(), threading.Event(), threading.Event()
    real_fsync = event_stream.os.fsync

    def slow_fsync(fd: int) -> None:
        if armed.is_set():
            stuck.set()  # a durable write is now parked on the disk
            release.wait()  # the disk answers only when the test says so, never after some elapsed time
        real_fsync(fd)

    monkeypatch.setattr(event_stream.os, "fsync", slow_fsync)
    try:
        for step in ("a", "b"):
            stream.send_event({"stage": "test", "status": "step", "step": step}, wait=False)
        _release_window(stream, monkeypatch, 2)
        frames = [_bounded_get(written, "a window frame's write") for _ in range(2)]

        armed.set()  # from here on every durable write blocks
        incoming.put(json.dumps(_job("c-slow-disk")))
        for frame in frames:
            incoming.put(_ack(frame))
        # The reader must get all the way through the job frame AND both acks (4th recv = back for more)
        # although the job's durable write is stuck.
        conns[0].wait_recv_calls(4)
        # Any parked fsync holds `_work`, and the job's own persist is itself armed, so the job's receipt
        # cannot be appended before `release.set()`: no receipt window can be open at the jump below.
        # `stuck` just proves the disk is wedged before the clock moves.
        assert stuck.wait(_BACKSTOP_S), "the job's durable write never started — the intake is stuck"
        clock.advance(10 * event_stream.FRAME_WALL_S)
        # The watchdog decides on its own thread. The window has been judged once it has released its ack
        # waiters (all after its stall check, and before any failure it would report for missing acks), so
        # wait for exactly that, on the lock the ack path uses (`_state` is held by the stuck write).
        with stream._ack_cv:
            assert stream._ack_cv.wait_for(lambda: not stream._ack_waiters, timeout=_BACKSTOP_S), (
                "the window was never judged — the watchdog appears stuck")
        # A close needs `_state`, which the stuck write holds, so the verdict is read once it is free: the
        # disk answers, and the next frame must go out on the SAME socket (a window that stalled would
        # have closed it and replayed a/b on a second one instead).
        release.set()
        # The receipt is appended only once the stuck write is through. Wait on that state of the stream
        # (not on time) before appending "after", so the receipt provably leaves first; whether the two
        # share a window is the sender's own slicing, so every frame read is acked rather than assumed to
        # ride in a window with the next.
        with stream._work:
            assert stream._work.wait_for(
                lambda: any(f["type"] == "job_ack" for f in stream._outbox), timeout=_BACKSTOP_S), (
                "the job's receipt was never appended once the disk answered")
        stream.send_event({"stage": "test", "status": "step", "step": "after"}, wait=False)
        receipt = _bounded_get(written, "the job's receipt")
        incoming.put(_ack(receipt))
        nxt = _bounded_get(written, "the frame after the window")
        incoming.put(_ack(nxt))
        assert receipt["type"] == "job_ack" and nxt["event"]["step"] == "after"
        assert len(conns) == 1 and not conns[0].closed.is_set(), (
            "the socket was closed although every ack had arrived")
    finally:
        release.set()
        stream.close()


def test_a_window_whose_acks_keep_arriving_is_not_closed_by_the_stall_bound(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The close is a stall bound: the window's total age may pass FRAME_WALL_S as long as an ack keeps
    arriving inside it."""
    stream, clock, conns, written, incoming = _open_stream(tmp_path, monkeypatch)
    results: list[bool | BaseException] = [False, False]

    def _call(i: int) -> None:
        try:
            results[i] = stream.send_event({"stage": "test", "status": "step", "step": f"e{i}"}, wait=True)
        except BaseException as e:  # noqa: BLE001 - captured for the assertion below
            results[i] = e
        finally:
            with stream._work:
                stream._work.notify_all()

    threads = [threading.Thread(target=_call, args=(i,), daemon=True) for i in range(2)]
    try:
        for t in threads:
            t.start()
        _wait_for_all_appended(stream, 2, threads)
        _release_window(stream, monkeypatch, 2)
        frames = [_bounded_get(written, "a window frame's write") for _ in range(2)]
        part = 0.6 * event_stream.FRAME_WALL_S
        clock.advance(part)
        incoming.put(_ack(frames[0]))
        conns[0].wait_recv_calls(2)
        clock.advance(part)  # 1.2 walls since the window opened, 0.6 since its last sign of life
        # The window watchdog runs on the sender thread; ack 2 must not be registered before it has
        # judged the stall bound at the new time, or a total-wall regression would go unseen.
        clock.wait_read_by("pod-stream-sender")
        assert not conns[0].closed.is_set()
        incoming.put(_ack(frames[1]))
        _bounded_join(threads)
        # Checked before `stream.close()`, which closes the connection itself.
        assert not conns[0].closed.is_set()
    finally:
        stream.close()
    assert results == [True, True], results


def test_a_window_whose_acks_stop_arriving_is_closed_on_the_stall_bound(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream, clock, conns, written, incoming = _open_stream(tmp_path, monkeypatch)
    try:
        stream.send_event({"stage": "test", "status": "step", "step": "silent"}, wait=False)
        _release_window(stream, monkeypatch, 1)
        _bounded_get(written, "the frame's write")  # never acked
        clock.advance(event_stream.FRAME_WALL_S)
        assert conns[0].closed.wait(_BACKSTOP_S), "a silent peer never made the window give up"
    finally:
        stream.close()
