"""The pod has one control-plane lane: a strict, durable, ordered WebSocket.

These are transport tests, not mocks of the happy path.  Each failure shape was capable of losing a
terminal when events and results used different in-memory/HTTP mechanisms.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
from websockets.sync.server import serve

from podagent import cp, event_stream
from podagent.event_stream import DeliveryPending, EventStream, FrameRejected, TransportUnhealthy


def _event(**extra: Any) -> dict[str, Any]:
    return {"stage": "test", "status": "step", **extra}


def _result(corr_id: str = "c", **extra: Any) -> dict[str, Any]:
    return {
        "job_id": "j",
        "session_id": "s",
        "corr_id": corr_id,
        "stage": "ops",
        "status": "ok",
        **extra,
    }


def _job(corr_id: str = "c", *, target_worker_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "infer",
        "session_id": "s",
        "corr_id": corr_id,
        "request": {
            "infer_version": 5,
            "job_id": "j",
            "kind": "face_probe",
            "model": "m",
            "put_url": "https://storage.example/out?sig=x",
            "face_probe": {
                "video_url": "https://storage.example/in?sig=x",
                "shots": [[0.0, 1.0]],
                "stride": 1,
                "frame_diff": False,
            },
        },
    }
    if target_worker_id:
        body["target_worker_id"] = target_worker_id
    return {
        "type": "job",
        "delivery_id": corr_id,
        "job": body,
    }


def _ack(frame: dict[str, Any], status: int = 202, **extra: Any) -> str:
    return json.dumps({
        "type": "ack",
        "stream_id": frame["stream_id"],
        "seq": frame["seq"],
        "status": status,
        **extra,
    })


@contextmanager
def _server(handler: Callable[[Any], None]) -> Iterator[str]:
    server = serve(handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.socket.getsockname()[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def _wait_for(predicate: Callable[[], bool], wall: float = 3.0) -> None:
    deadline = time.monotonic() + wall
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _fast_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event_stream, "FRAME_WALL_S", 0.15)
    monkeypatch.setattr(event_stream, "OPEN_WALL_S", 0.15)
    monkeypatch.setattr(event_stream, "REOPEN_BACKOFF_S", 0.01)
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 1)
    monkeypatch.setattr(event_stream, "ACK_WINDOW", 4)


def test_result_is_a_typed_frame_and_requires_correlation(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        with pytest.raises(TransportUnhealthy, match="invalid result payload"):
            stream.send_result({"job_id": "j", "status": "ok"})
        assert stream.send_result(_result())
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert seen[0] == {
        "type": "result",
        "stream_id": seen[0]["stream_id"],
        "seq": 1,
        "result": _result(),
    }
    assert seen[1]["type"] == "event"
    acked = seen[1]["event"]
    assert acked["phase"] == "result_acked" and acked["corr_id"] == "c"
    assert acked["op"] == "ops" and set(acked["timings"]) == {"delivery_s"}


def test_single_sender_preserves_append_order(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        for n in range(6):
            assert stream.send_event(_event(step=f"n{n}"), wait=False)
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert [f["event"]["step"] for f in seen] == [f"n{n}" for n in range(6)]
    assert [f["seq"] for f in seen] == list(range(1, 7))
    assert len({f["stream_id"] for f in seen}) == 1


def test_delayed_first_ack_still_receives_a_full_bounded_window_without_log_loss(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    first_window_received = threading.Event()
    sender_gate = threading.Event()
    real_sender = EventStream._sender_loop

    def gated_sender(stream: EventStream) -> None:
        assert sender_gate.wait(1)
        real_sender(stream)

    monkeypatch.setattr(EventStream, "_sender_loop", gated_sender)

    def handler(ws: Any) -> None:
        first = [json.loads(ws.recv()) for _ in range(event_stream.ACK_WINDOW)]
        seen.extend(first)
        first_window_received.set()
        for frame in first:
            ws.send(_ack(frame))
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        count = event_stream.ACK_WINDOW * 2 + 1
        for n in range(count):
            assert stream.send_event(_event(phase="log", step=f"n{n}"), wait=False)
        assert len(json.loads((tmp_path / "outbox.json").read_text())["frames"]) == count
        sender_gate.set()
        assert first_window_received.wait(1), "the first ACK was withheld until a whole window arrived"
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert [frame["event"]["step"] for frame in seen] == [f"n{n}" for n in range(count)]
    assert [frame["seq"] for frame in seen] == list(range(1, count + 1))


def test_out_of_order_acks_are_not_settled_ahead_of_the_durable_head(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    later_acks_sent = threading.Event()
    release_head = threading.Event()
    sender_gate = threading.Event()
    real_sender = EventStream._sender_loop
    monkeypatch.setattr(
        EventStream, "_sender_loop",
        lambda stream: sender_gate.wait(1) and real_sender(stream),
    )

    def handler(ws: Any) -> None:
        frames = [json.loads(ws.recv()) for _ in range(3)]
        ws.send(_ack(frames[2]))
        ws.send(_ack(frames[1]))
        later_acks_sent.set()
        assert release_head.wait(1)
        ws.send(_ack(frames[0]))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        for n in range(3):
            stream.send_event(_event(step=f"n{n}"), wait=False)
        sender_gate.set()
        assert later_acks_sent.wait(1)
        time.sleep(0.03)
        assert stream.pending_count() == 3, "later ACKs must not retire around an unacknowledged head"
        release_head.set()
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()


def test_later_window_4xx_latches_admission_before_its_predecessors_settle(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verdict_sent = threading.Event()
    release_predecessors = threading.Event()
    sender_gate = threading.Event()
    real_sender = EventStream._sender_loop
    monkeypatch.setattr(
        EventStream, "_sender_loop",
        lambda stream: sender_gate.wait(1) and real_sender(stream),
    )

    def handler(ws: Any) -> None:
        frames = [json.loads(ws.recv()) for _ in range(3)]
        ws.send(_ack(frames[2], status=422, error="late frame rejected"))
        verdict_sent.set()
        assert release_predecessors.wait(1)
        ws.send(_ack(frames[0]))
        ws.send(_ack(frames[1]))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        for n in range(3):
            stream.send_event(_event(step=f"n{n}"), wait=False)
        sender_gate.set()
        assert verdict_sent.wait(1)
        _wait_for(lambda: isinstance(stream._admission_error, FrameRejected))
        assert stream.pending_count() == 3, "the durable head is intentionally still unacknowledged"
        with pytest.raises(FrameRejected, match="422"):
            stream.claim(0.1)
        release_predecessors.set()
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()


def test_mid_window_disconnect_replays_only_unacked_frames_with_the_same_ids(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    connections: list[list[tuple[str, int]]] = []
    handler_lock = threading.Lock()
    sender_gate = threading.Event()
    real_sender = EventStream._sender_loop
    monkeypatch.setattr(
        EventStream, "_sender_loop",
        lambda stream: sender_gate.wait(1) and real_sender(stream),
    )

    def handler(ws: Any) -> None:
        with handler_lock:
            connection_no = len(connections)
            connections.append([])
        if connection_no == 0:
            frames = [json.loads(ws.recv()) for _ in range(event_stream.ACK_WINDOW)]
            connections[0].extend((frame["stream_id"], frame["seq"]) for frame in frames)
            for frame in frames[:2]:
                ws.send(_ack(frame))
            return
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            connections[connection_no].append((frame["stream_id"], frame["seq"]))
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        for n in range(event_stream.ACK_WINDOW):
            stream.send_event(_event(step=f"n{n}"), wait=False)
        sender_gate.set()
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert len(connections) == 2
    assert [seq for _, seq in connections[0]] == [1, 2, 3, 4]
    assert connections[1] == connections[0][2:], "ACKed prefix must not cross the reconnect again"


def test_reconnect_replays_worker_capacity_bootstrap(tmp_path: Path) -> None:
    """A fresh API socket must receive capacity again before admitting more claims."""
    connections: list[list[dict[str, Any]]] = []
    lock = threading.Lock()

    def handler(ws: Any) -> None:
        with lock:
            no = len(connections)
            connections.append([])
        try:
            while True:
                frame = json.loads(ws.recv())
                connections[no].append(frame)
                if no == 0:
                    # Force a reconnect with both the original frame and the
                    # bootstrap event still durable.
                    return
                ws.send(_ack(frame))
        except Exception:
            return

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        stream.set_bootstrap_event(_event(phase="capacity", capacity={"max_inflight": 5, "max_parallel": 5}))
        stream.send_event(_event(step="work"), wait=False)
        _wait_for(lambda: len(connections) >= 2 and stream.pending_count() == 0)
        stream.close()

    replayed = connections[1]
    assert any(frame["type"] == "event" and frame["event"].get("phase") == "capacity"
               for frame in replayed)


def test_bootstrap_persist_failure_closes_admission_loudly(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A reconnect that cannot durably queue capacity may not quietly look like an idle socket."""
    class _Connection:
        pass

    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    stream.set_bootstrap_event(_event(phase="capacity", capacity={"max_inflight": 1, "max_parallel": 1}))
    monkeypatch.setattr(event_stream, "_ws_client", type(
        "_Client", (), {"connect": staticmethod(lambda *_a, **_k: _Connection())}))
    monkeypatch.setattr(stream, "_append", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))
    failed: list[str] = []

    def fail_connection(_conn: Any, why: str) -> None:
        failed.append(why)
        with stream._state:
            stream._conn = None

    monkeypatch.setattr(stream, "_fail_connection", fail_connection)
    assert stream._ensure_open() is False
    assert failed == ["bootstrap append: OSError: disk full"]
    assert "bootstrap append failed" in capsys.readouterr().err
    stream.close()


def test_stale_reader_cannot_wake_the_reconnected_socket_window(tmp_path: Path) -> None:
    class _Conn:
        def close(self) -> None:
            pass

    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    old_conn, new_conn = _Conn(), _Conn()
    waiter = threading.Event()
    key = (stream._stream_id, 1)
    with stream._state:
        stream._conn = new_conn
        stream._ack_waiters[key] = waiter
    stream._fail_connection(old_conn, "late old reader")
    assert stream._conn is new_conn
    assert not waiter.is_set(), "the old socket must not manufacture an ACK wakeup on the new window"
    with stream._state:
        stream._ack_waiters.clear()
        stream._conn = None
    stream.close()


def test_hanging_send_is_force_closed_inside_frame_wall_and_keeps_durable_admission_latch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release_send = threading.Event()
    send_entered = threading.Event()
    socket_closed = threading.Event()

    class _Socket:
        def shutdown(self, how: int) -> None:
            assert how == event_stream.socket.SHUT_RDWR
            release_send.set()

        def close(self) -> None:
            socket_closed.set()

    class _HungConn:
        socket = _Socket()

        def send(self, _body: str) -> None:
            send_entered.set()
            assert release_send.wait(2), "the transport watchdog never interrupted send()"
            raise OSError("socket force-closed")

    monkeypatch.setattr(event_stream, "FRAME_WALL_S", 0.05)
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 0)
    path = tmp_path / "outbox.json"
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    conn = _HungConn()
    with stream._state:
        stream._conn = conn
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)

    started = time.monotonic()
    with pytest.raises(DeliveryPending, match="remains unacknowledged"):
        stream.send_event(_event(phase="must-survive"), wait=True)
    assert time.monotonic() - started < 0.5
    assert send_entered.is_set() and release_send.is_set() and socket_closed.is_set()
    assert isinstance(stream._admission_error, DeliveryPending)
    with pytest.raises(DeliveryPending):
        stream.claim(0.01)
    disk = json.loads(path.read_text())
    assert len(disk["frames"]) == 1
    assert disk["frames"][0]["event"]["phase"] == "must-survive"
    stream.close()


def test_result_and_its_ack_event_cannot_overtake_prior_or_already_appended_frames(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    sender_gate = threading.Event()
    real_sender = EventStream._sender_loop
    monkeypatch.setattr(
        EventStream, "_sender_loop",
        lambda stream: sender_gate.wait(1) and real_sender(stream),
    )

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        stream.send_event(_event(step="before-1"), wait=False)
        stream.send_event(_event(step="before-2"), wait=False)
        stream.send_result(_result(), wait=False)
        stream.send_event(_event(step="after"), wait=False)
        sender_gate.set()
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert [frame["type"] for frame in seen] == ["event", "event", "result", "event", "event"]
    assert [frame["seq"] for frame in seen] == [1, 2, 3, 4, 5]
    assert seen[3]["event"]["step"] == "after"
    assert seen[4]["event"]["phase"] == "result_acked"


def test_ack_lost_after_send_replays_the_same_identity(tmp_path: Path) -> None:
    seen: list[tuple[str, int]] = []
    first = True

    def handler(ws: Any) -> None:
        nonlocal first
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            if frame["type"] == "result":
                seen.append((frame["stream_id"], frame["seq"]))
                if first:
                    first = False
                    return  # The peer may have committed the frame; the ACK alone was lost.
                ws.send(_ack(frame, duplicate=True))
            else:
                ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        assert stream.send_result(_result())
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    assert len(seen) == 2
    assert seen[0] == seen[1], "ambiguity must replay the original dedupe identity"


def test_process_restart_replays_old_identity_and_new_frames_use_a_new_stream(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    rejected_seen: list[dict[str, Any]] = []
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 0)

    def unavailable(ws: Any) -> None:
        frame = json.loads(ws.recv())
        rejected_seen.append(frame)
        ws.send(_ack(frame, status=503, error="try later"))

    with _server(unavailable) as base:
        first = EventStream(base, "token", outbox_path=path)
        with pytest.raises(DeliveryPending):
            first.send_result(_result(), wait=True)
        first.close()

    old_identity = (rejected_seen[0]["stream_id"], rejected_seen[0]["seq"])
    accepted: list[dict[str, Any]] = []

    def accepting(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            accepted.append(frame)
            ws.send(_ack(frame))

    with _server(accepting) as base:
        second = EventStream(base, "token", outbox_path=path)
        _wait_for(lambda: second.pending_count() == 0)
        assert second.send_event(_event(phase="received"), wait=True)
        second.close()

    assert (accepted[0]["stream_id"], accepted[0]["seq"]) == old_identity
    assert accepted[1]["type"] == "event" and accepted[1]["event"]["phase"] == "result_acked"
    assert accepted[1]["stream_id"] != old_identity[0] and accepted[1]["seq"] == 1
    assert accepted[2]["stream_id"] == accepted[1]["stream_id"] and accepted[2]["seq"] == 2


@pytest.mark.parametrize("reply", [
    {"type": "mystery"},
    {"type": "ack", "stream_id": "wrong", "seq": 1, "status": 202, "extra": True},
])
def test_unknown_or_malformed_ack_never_retires_a_frame(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: dict[str, Any]) -> None:
    monkeypatch.setattr(event_stream, "MAX_REOPENS", 0)

    def handler(ws: Any) -> None:
        ws.recv()
        ws.send(json.dumps(reply))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        with pytest.raises(DeliveryPending):
            stream.send_event(_event(phase="started"), wait=True)
        assert stream.pending_count() == 1
        stream.close()


def test_4xx_is_durably_dead_lettered_and_fails_the_caller(tmp_path: Path) -> None:
    path = tmp_path / "outbox.json"

    def handler(ws: Any) -> None:
        frame = json.loads(ws.recv())
        ws.send(_ack(frame, status=422, error="bad result"))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        with pytest.raises(FrameRejected, match="422"):
            stream.send_result(_result())
        assert stream.pending_count() == 0
        stream._accept_job(_job())
        assert stream._inbox == {}
        with pytest.raises(FrameRejected):
            stream.claim(0.1)
        stream.close()

    state = json.loads(path.read_text())
    assert state["frames"] == []
    assert len(state["rejected"]) == 1
    assert state["rejected"][0]["ack"]["error"] == "bad result"


def test_async_4xx_dead_letters_without_accumulating_an_outcome(tmp_path: Path) -> None:
    path = tmp_path / "outbox.json"

    def handler(ws: Any) -> None:
        frame = json.loads(ws.recv())
        ws.send(_ack(frame, status=409, error="duplicate content key"))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        assert stream.send_event(_event(phase="started"))
        _wait_for(lambda: bool(json.loads(path.read_text())["rejected"]))
        assert stream.pending_count() == 0
        assert stream._delivery_outcomes == {}
        with pytest.raises(FrameRejected):
            stream.claim(0.1)
        stream.close()


def test_persist_failure_while_retiring_keeps_frame_and_unblocks_caller(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"

    def handler(ws: Any) -> None:
        frame = json.loads(ws.recv())
        ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        real_persist = stream._persist_locked
        calls = 0

        def fail_retire() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk became read-only")
            real_persist()

        monkeypatch.setattr(stream, "_persist_locked", fail_retire)
        with pytest.raises(TransportUnhealthy, match="read-only"):
            stream.send_event(_event(phase="work_finished"), wait=True)
        assert stream.pending_count() == 1
        assert len(json.loads(path.read_text())["frames"]) == 1
        stream.close()


def test_unwritable_outbox_refuses_construction_before_claim(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        EventStream,
        "_persist_locked",
        lambda _self: (_ for _ in ()).throw(OSError("read-only outbox")),
    )
    with pytest.raises(OSError, match="read-only outbox"):
        EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")


def test_retry_exhaustion_retains_the_frame(tmp_path: Path) -> None:
    path = tmp_path / "outbox.json"
    hits = 0

    def handler(ws: Any) -> None:
        nonlocal hits
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            hits += 1
            ws.send(_ack(frame, status=503, error="overloaded"))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        with pytest.raises(DeliveryPending):
            stream.send_result(_result())
        assert stream.pending_count() == 1
        with pytest.raises(DeliveryPending):
            stream.claim(0.1)
        stream.close()

    state = json.loads(path.read_text())
    assert hits == 2
    assert len(state["frames"]) == 1
    assert state["rejected"] == []


def test_jobs_are_typed_and_unwrapped(tmp_path: Path) -> None:
    def handler(ws: Any) -> None:
        ws.send(json.dumps(_job()))
        time.sleep(0.1)

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        assert stream.claim(1) == _job()["job"]
        stream.close()


def test_targeted_warm_job_is_accepted_and_keeps_worker_fence(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The target is a strict transport field: admission retains it while
    dispatch continues to use product session/correlation identity."""
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    stream._accept_job(_job("warm-corr", target_worker_id="fleet-worker-7"))
    admitted = stream.claim(0.2)
    assert admitted["corr_id"] == "warm-corr"
    assert admitted["session_id"] == "s"
    assert admitted["target_worker_id"] == "fleet-worker-7"
    stream.close()


def test_python_invalid_but_addressed_job_gets_correlated_terminal_before_latch(
        tmp_path: Path) -> None:
    """A Go-addressable claim must release credit even when Python rejects its payload."""
    path = tmp_path / "outbox.json"
    malformed = _job("bad-python-job")
    # Keep the transport address intact while violating the strict PodJob
    # schema.  This is the production ingress seam, before _dispatch_loop.
    del malformed["job"]["request"]["model"]
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    stream._accept_job(malformed)
    frames = json.loads(path.read_text())["frames"]
    results = [frame["result"] for frame in frames if frame["type"] == "result"]
    assert len(results) == 1
    assert results[0]["job_id"] == "j"
    assert results[0]["session_id"] == "s"
    assert results[0]["corr_id"] == "bad-python-job"
    assert results[0]["stage"] == "dispatch"
    assert results[0]["status"] == "error"
    assert results[0]["error"].startswith("invalid server job frame:")
    assert any(frame["type"] == "event" and frame["event"]["corr_id"] == "bad-python-job"
               for frame in frames)
    stream.close()


def test_python_invalid_job_is_terminalized_and_next_delivery_is_admitted(tmp_path: Path) -> None:
    """The reader stays up through the terminal ACK, then accepts the next claim credit."""
    malformed = _job("bad-python-job")
    del malformed["job"]["request"]["model"]
    seen: list[dict[str, Any]] = []

    def handler(ws: Any) -> None:
        ws.send(json.dumps(malformed))
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))
            if frame["type"] == "result":
                ws.send(json.dumps(_job("next-job")))
                return

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        assert stream.claim(2.0)["corr_id"] == "next-job"
        stream.close()

    assert any(frame["type"] == "result" and frame["result"]["corr_id"] == "bad-python-job"
               for frame in seen)


def test_inbox_replays_after_receive_before_run_crash(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    first = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    first._accept_job(_job("work/infer/result.json"))
    first.close()

    second = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    monkeypatch.setattr(second, "_ensure_open", lambda: True)
    assert second.claim(0.2)["corr_id"] == "work/infer/result.json"
    second.close()


def test_reconnect_duplicate_job_is_not_queued_or_run_twice(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    stream._accept_job(_job())
    assert stream.claim(0.2)["corr_id"] == "c"
    stream._accept_job(_job())
    assert stream.claim(0.05) is None
    stream.close()


def test_conflicting_duplicate_delivery_latches_before_queued_work_can_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    stream._accept_job(_job())
    conflicting = _job()
    conflicting["job"]["request"]["model"] = "different"
    with pytest.raises(TransportUnhealthy, match="different content"):
        stream._accept_job(conflicting)
    with pytest.raises(TransportUnhealthy, match="different content"):
        stream.claim(0.1)
    assert "c" in stream._inbox
    stream.close()


def test_valid_infer_golden_crosses_real_event_stream_with_one_result_identity(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from podagent import main as agent_main
    from podagent.cp import ControlPlane

    seen: list[dict[str, Any]] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame)
            ws.send(_ack(frame))

    req = json.loads(
        (Path(__file__).parents[1] / "contracts/examples/infer_request.face_probe.json").read_text())
    corr = "work/session/result.json"
    yunet = tmp_path / "yunet.onnx"

    class _Probe:
        def run(self, _params: Any, _put_url: str) -> float:
            return 0.125

    probe_module = types.ModuleType("podagent.infer_probe")
    probe_module.ProbeService = _Probe
    monkeypatch.setitem(sys.modules, "podagent.infer_probe", probe_module)

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        plane = object.__new__(ControlPlane)
        plane._stream = stream
        assert agent_main._run_infer(
            req, plane, {}, {(yunet, req["model"]): _Probe()}, {}, yunet, False,
            corr_id=corr, session_id="session-1")
        _wait_for(lambda: stream.pending_count() == 0)
        stream.close()

    result = next(frame["result"] for frame in seen if frame["type"] == "result")
    assert result["corr_id"] == corr == result["result_key"]
    assert result["session_id"] == "session-1" and result["kind"] == "face_probe"


def test_result_append_and_inbox_retirement_are_one_durable_transition(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    corr = "work/infer/result.json"
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    stream._accept_job(_job(corr))
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    assert stream.claim(0.2)["corr_id"] == corr
    assert stream.send_result(_result(corr, result_key=corr), wait=False)
    state = json.loads(path.read_text())
    assert state["inbox"] == {}
    assert state["frames"][0]["result"]["corr_id"] == corr
    stream.close()


def test_result_append_failure_rolls_back_seq_waiter_and_inbox(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    corr = "work/infer/result.json"
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=path)
    stream._accept_job(_job(corr))
    monkeypatch.setattr(stream, "_ensure_open", lambda: True)
    assert stream.claim(0.2)["corr_id"] == corr
    monkeypatch.setattr(
        stream, "_persist_locked",
        lambda: (_ for _ in ()).throw(OSError("fsync failed for https://u:p@store/x?token=secret")),
    )
    with pytest.raises(TransportUnhealthy, match=r"durable append failed.*\[redacted-url\]"):
        stream.send_result(_result(corr, result_key=corr))
    assert stream._next_seq == 1
    assert stream._outbox == [] and stream._delivery_waiters == {}
    assert corr in stream._inbox
    with pytest.raises(TransportUnhealthy):
        stream.claim(0.1)
    stream.close()


def test_replayed_job_before_pending_result_ack_does_not_reconnect_livelock_or_rerun(
        tmp_path: Path) -> None:
    corr = "work/infer/result.json"
    seen: list[str] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            seen.append(frame["type"])
            if frame["type"] == "result":
                ws.send(json.dumps(_job(corr)))
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=tmp_path / "outbox.json")
        stream._accept_job(_job(corr))
        assert stream.claim(0.2)["corr_id"] == corr
        assert stream.send_result(_result(corr, result_key=corr))
        _wait_for(lambda: stream.pending_count() == 0)
        assert stream.claim(0.1) is None
        stream.close()
    assert seen == ["result", "event"]


def test_result_acked_persist_failure_replays_result_after_restart(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "outbox.json"
    accepted: list[str] = []

    def handler(ws: Any) -> None:
        while True:
            try:
                frame = json.loads(ws.recv())
            except Exception:
                return
            accepted.append(frame["type"])
            ws.send(_ack(frame))

    with _server(handler) as base:
        stream = EventStream(base, "token", outbox_path=path)
        real_persist = stream._persist_locked
        calls = 0

        def fail_result_retire() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("result_acked fsync failed")
            real_persist()

        monkeypatch.setattr(stream, "_persist_locked", fail_result_retire)
        with pytest.raises(TransportUnhealthy, match="result_acked fsync failed"):
            stream.send_result(_result())
        stream.close()

    disk = json.loads(path.read_text())
    assert [f["type"] for f in disk["frames"]] == ["result"]

    with _server(handler) as base:
        replay = EventStream(base, "token", outbox_path=path)
        _wait_for(lambda: replay.pending_count() == 0)
        replay.close()
    assert accepted == ["result", "result", "event"]


def test_stuck_sender_hits_caller_wall_without_leaking_waiter_or_outcome(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    release = threading.Event()
    monkeypatch.setattr(event_stream, "_delivery_wall_s", lambda: 0.05)

    def stuck(_frames: list[dict[str, Any]]) -> dict[tuple[str, int], tuple[str, None, float]]:
        release.wait(1)
        return {}

    monkeypatch.setattr(stream, "_deliver_window", stuck)
    assert stream.send_result(_result()) is False
    assert stream._delivery_waiters == {} and stream._delivery_outcomes == {}
    assert isinstance(stream._admission_error, DeliveryPending)
    with pytest.raises(DeliveryPending, match="caller delivery wall"):
        stream.claim(0.01)
    release.set()
    assert stream._delivery_outcomes == {}
    assert stream.pending_count() == 1
    stream.close()


def test_control_plane_delegates_one_result_without_a_second_lane() -> None:
    calls: list[tuple[dict[str, Any], bool]] = []

    class _Stream:
        def send_result(self, payload: dict[str, Any], *, wait: bool = True) -> bool:
            calls.append((payload, wait))
            return True

    plane = object.__new__(cp.ControlPlane)
    plane._stream = _Stream()
    plane.send_result({
        "job_id": "j", "stage": "ops", "status": "ok", "session_id": "s", "corr_id": "c"})

    assert calls == [({
        "job_id": "j", "stage": "ops", "status": "ok", "session_id": "s", "corr_id": "c"}, True)]
