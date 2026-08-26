"""A step's own PUT no longer sits on the DAG's critical path for a hand-off that never reads it
(runner.PUT_DRAIN_WHY). Every test is NEGATIVE (docs/TESTING.md): watched fail with the drain reverted.
Hermetic: no control plane, no network, no pack fetch, no real sleeps — events and an injected clock stand in."""
from __future__ import annotations

import concurrent.futures as cf
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from podagent import cp  # noqa: E402
from podagent.ops import runner  # noqa: E402

_OP = "media.scale"          # one required input port ('src'), one required output port ('dst')


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.results: list[dict] = []

    def send_event(self, payload: dict, *, wait: bool = False) -> bool:
        self.events.append(payload)
        return True

    def send_result(self, payload: dict, *, wait: bool = True) -> bool:
        self.results.append(payload)
        return True

    def timeline_context(self, corr_id: str) -> dict:
        return {"complete": True, "incomplete_reasons": [], "pod_clock_id": "c1",
                "attempt_id": "a1", "delivery": None, "clock_sync": []}

    @property
    def terminal(self) -> dict:
        return self.events[-1]


class _B:
    def __init__(self, port, *, url=None, urls=None, from_step=None, from_port=None, path=None, retain=None):
        self.port, self.url, self.urls = port, url, urls
        self.from_step, self.from_port, self.path, self.retain = from_step, from_port, path, retain


class _Step:
    def __init__(self, sid, op=_OP, *, needs=(), inputs=(), outputs=(), optional=False):
        self.id, self.op, self.params = sid, op, {}
        self.needs, self.optional = list(needs), optional
        self.inputs, self.outputs = list(inputs), list(outputs)


class _Chain:
    def __init__(self, steps, job_id: str = "j-1") -> None:
        self.steps, self.job_id, self.pack = steps, job_id, None


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """The real `run_chain` DAG/drain, with the pack, the registry and the transport stubbed."""
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.registry, "assert_pod_safe", lambda *a, **k: None)
    monkeypatch.setattr(runner, "preflight_chain", lambda chain: None)
    monkeypatch.setattr(runner.pack, "activate_or_mismatch", lambda ref: tmp_path)
    monkeypatch.setattr(runner, "log", lambda *a, **k: None)
    monkeypatch.setattr(runner.inputcache, "enabled", lambda: False)
    runner._reset_step_slots()
    yield
    runner._reset_step_slots()


def _src(tmp_path: Path) -> Path:
    p = tmp_path / "in.bin"
    p.write_bytes(b"x" * 16)
    return p


def _write_handler(*, on_step: dict[str, threading.Event] | None = None):
    def _fn(*, params, inputs, outputs):
        sid = outputs["dst"].parent.name
        if on_step and sid in on_step:
            on_step[sid].set()
        outputs["dst"].write_bytes(b"y" * 8)
    return _fn


# ── (a) DAG scheduling no longer waits on a step's own puts ─────────────────────────────────────────

def test_b_computes_while_as_put_is_still_pending(monkeypatch, wired, tmp_path):
    put_started, put_release, b_started = threading.Event(), threading.Event(), threading.Event()

    def _upload(src_path, url, ct=None):
        put_started.set()
        assert put_release.wait(timeout=5), "A's put was never released — the test itself is broken"

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _write_handler(on_step={"B": b_started}))
    monkeypatch.setattr(runner, "upload", _upload)

    a = _Step("A", inputs=[_B("src", path=str(_src(tmp_path)))], outputs=[_B("dst", url="https://x/a.out")])
    b = _Step("B", needs=["A"], inputs=[_B("src", from_step="A")])
    cp_ = _CP()

    thread = threading.Thread(target=runner.run_chain, args=(_Chain([a, b]), cp_))
    thread.start()
    try:
        assert put_started.wait(timeout=5), "A's put never started"
        assert b_started.wait(timeout=5), "B's compute never ran while A's put was still blocked"
    finally:
        put_release.set()
        thread.join(timeout=5)
    assert not thread.is_alive(), "the chain never finished once the put was released"
    assert cp_.results and cp_.results[-1]["status"] == "ok"


def test_sibling_url_readers_finds_a_same_chain_address_collision():
    a = _Step("A", outputs=[_B("dst", url="https://x/shared.out")])
    b = _Step("B", needs=["A"], inputs=[_B("src", url="https://x/shared.out")])
    assert runner._sibling_url_readers(_Chain([a, b])) == {"A"}


def test_sibling_url_readers_is_empty_for_an_ordinary_from_step_handoff():
    a = _Step("A", outputs=[_B("dst", url="https://x/a.out")])
    b = _Step("B", needs=["A"], inputs=[_B("src", from_step="A")])
    assert runner._sibling_url_readers(_Chain([a, b])) == set()


# ── (b) a failed put still fails the chain, loudly, before any terminal ─────────────────────────────

def test_a_failed_put_fails_the_chain_and_no_terminal_ever_reports_done(monkeypatch, wired, tmp_path):
    def _upload(src_path, url, ct=None):
        raise RuntimeError("503 from the store")

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _write_handler())
    monkeypatch.setattr(runner, "upload", _upload)

    step = _Step("s1", inputs=[_B("src", path=str(_src(tmp_path)))],
                 outputs=[_B("dst", url="https://x/s1.out")])
    cp_ = _CP()

    with pytest.raises(RuntimeError, match="503 from the store"):
        runner.run_chain(_Chain([step]), cp_)

    assert cp_.results == [], "the terminal must never report done after a failed put"
    assert not any(e.get("phase") == "work_finished" for e in cp_.events)


def test_an_optional_steps_failed_put_does_not_fail_the_chain(monkeypatch, wired, tmp_path):
    def _upload(src_path, url, ct=None):
        raise RuntimeError("403 from one candidate host")

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _write_handler())
    monkeypatch.setattr(runner, "upload", _upload)

    step = _Step("s1", inputs=[_B("src", path=str(_src(tmp_path)))],
                 outputs=[_B("dst", url="https://x/s1.out")], optional=True)
    cp_ = _CP()

    runner.run_chain(_Chain([step]), cp_)      # must NOT raise — a fan-out arm, not the chain

    assert cp_.results and cp_.results[-1]["status"] == "ok"
    terminal = cp_.terminal
    assert terminal["steps"] == [], "a step whose put failed must not ALSO claim steps= (F2: no contradiction)"
    assert terminal["skipped"] == ["s1"], "the failed arm must still be booked, same as a compute failure"
    assert [row["id"] for row in terminal["timings"]["steps"]] == ["s1"], \
        "the row must still land — a duration for a chain that completed around it is not 'worse than none'"
    assert any(r.startswith("put_failed") for r in terminal["timeline"]["pod"]["incomplete_reasons"])


# ── (c) per-step put timings still land on the right step row ───────────────────────────────────────

def test_put_timings_land_on_the_right_step_row(monkeypatch, wired, tmp_path):
    def _upload(src_path, url, ct=None):
        if url.endswith("s1.out"):
            runner.retry.add(0.4)
        elif url.endswith("s2.out"):
            runner.retry.add(0.9)

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _write_handler())
    monkeypatch.setattr(runner, "upload", _upload)

    src = _src(tmp_path)
    s1 = _Step("s1", inputs=[_B("src", path=str(src))], outputs=[_B("dst", url="https://x/s1.out")])
    s2 = _Step("s2", inputs=[_B("src", path=str(src))], outputs=[_B("dst", url="https://x/s2.out")])
    cp_ = _CP()

    runner.run_chain(_Chain([s1, s2]), cp_)

    rows = {row["id"]: row for row in cp_.terminal["timings"]["steps"]}
    assert rows["s1"]["legs"]["put_retry"] == pytest.approx(0.4, abs=1e-3)
    assert rows["s2"]["legs"]["put_retry"] == pytest.approx(0.9, abs=1e-3)


# ── (d) the workspace is not removed while a drained put still reads it ─────────────────────────────

def test_workspace_survives_until_the_last_put_returns(monkeypatch, wired, tmp_path):
    captured: dict[str, str] = {}
    real_mkdtemp = runner.tempfile.mkdtemp

    def _mkdtemp(*a, **k):
        made = real_mkdtemp(*a, **k)
        captured["tmp"] = made
        return made

    put_started, put_release = threading.Event(), threading.Event()

    def _upload(src_path, url, ct=None):
        put_started.set()
        assert put_release.wait(timeout=5), "the put was never released — the test itself is broken"

    monkeypatch.setattr(runner.tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _write_handler())
    monkeypatch.setattr(runner, "upload", _upload)

    step = _Step("s1", inputs=[_B("src", path=str(_src(tmp_path)))],
                 outputs=[_B("dst", url="https://x/s1.out")])
    cp_ = _CP()

    thread = threading.Thread(target=runner.run_chain, args=(_Chain([step]), cp_))
    thread.start()
    try:
        assert put_started.wait(timeout=5), "the put never started"
        assert Path(captured["tmp"]).exists(), "the workspace vanished before its own put returned"
    finally:
        put_release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not Path(captured["tmp"]).exists(), "the workspace must be removed once the drain settles"


# ── (e) the drain respects the chain's own deadline, proven without sleeping for it ────────────────

def _two_call_clock(first: float, later: float):
    calls = {"n": 0}

    def _clock() -> float:
        calls["n"] += 1
        return first if calls["n"] == 1 else later
    return _clock


def test_the_clock_is_untouched_until_wait_or_settle_actually_blocks():
    """The regression this pins: a deadline anchored at chain CONSTRUCTION lets the DAG's own compute time
    eat the put-drain's budget, so a long chain fails a put that would land in milliseconds. `enqueue()`
    must not open a window either — only `wait()`/`settle()`, each its OWN, at their OWN entry."""
    calls: list[str] = []

    def _clock() -> float:
        calls.append("tick")
        return 0.0

    pool = cf.ThreadPoolExecutor(max_workers=2)
    try:
        drain = runner._PutDrain(pool, 1.0, clock=_clock)
        assert calls == [], "construction must not open a deadline window"
        drain.enqueue("s1", lambda: None)
        assert calls == [], "enqueue must not open a deadline window either"
        drain.wait()
        assert calls, "wait() must open its own window once it actually starts blocking"
        after_wait = len(calls)
        drain.settle()
        assert len(calls) > after_wait, "settle() must open its OWN fresh window, not reuse wait()'s"
    finally:
        pool.shutdown(wait=True)


def test_drain_wait_raises_once_its_deadline_passes_with_work_outstanding():
    pool = cf.ThreadPoolExecutor(max_workers=2)
    hang = threading.Event()
    try:
        drain = runner._PutDrain(pool, 1.0, clock=_two_call_clock(0.0, 1000.0))
        drain.enqueue("stuck", hang.wait)
        with pytest.raises(runner.ChainError, match="put drain"):
            drain.wait()
    finally:
        hang.set()
        pool.shutdown(wait=True)


def test_settle_skips_a_put_wait_already_gave_up_on():
    """F1: a hung REQUIRED put must not burn a SECOND full window in `finally` after `wait()` already
    spent one and raised — `settle()` must not even touch the clock for a future `wait()` already booked
    stuck, or a rented box pays 2x the deadline for one put the CP was already told about."""
    calls = {"n": 0}

    def _clock() -> float:
        calls["n"] += 1
        return 0.0 if calls["n"] == 1 else 1000.0

    pool = cf.ThreadPoolExecutor(max_workers=2)
    hang = threading.Event()
    try:
        drain = runner._PutDrain(pool, 1.0, clock=_clock)
        drain.enqueue("stuck", hang.wait)
        with pytest.raises(runner.ChainError, match="put drain"):
            drain.wait()
        after_wait = calls["n"]
        drain.settle()
        assert calls["n"] == after_wait, "settle() must not re-poll a future wait() already gave up on"
    finally:
        hang.set()
        pool.shutdown(wait=True)


def test_drain_settle_is_best_effort_and_never_raises_past_its_deadline():
    pool = cf.ThreadPoolExecutor(max_workers=2)
    hang = threading.Event()
    try:
        drain = runner._PutDrain(pool, 1.0, clock=_two_call_clock(0.0, 1000.0))
        drain.enqueue("stuck", hang.wait)      # a REQUIRED step's stuck put — settle() still must not raise
        drain.settle()      # must return, not raise, even though the future is still pending
    finally:
        hang.set()
        pool.shutdown(wait=True)


def test_wait_returns_an_optional_steps_timeout_instead_of_raising():
    pool = cf.ThreadPoolExecutor(max_workers=2)
    hang = threading.Event()
    try:
        drain = runner._PutDrain(pool, 1.0, clock=_two_call_clock(0.0, 1000.0))
        drain.enqueue("stuck", hang.wait, optional=True)
        failures = drain.wait()
        assert [sid for sid, _exc in failures] == ["stuck"]
    finally:
        hang.set()
        pool.shutdown(wait=True)


# ── (f) the pool-sizing fallback matches transport_cap()'s derivation, not a bare constant ──────────

def test_store_pool_fallback_matches_transport_cap_derivation(monkeypatch):
    monkeypatch.delenv("OPS_MAX_TRANSFERS", raising=False)
    monkeypatch.delenv("OPS_MAX_PARALLEL", raising=False)
    assert cp._store_pool() == runner.transport_cap()[0]


def test_store_pool_fallback_tracks_a_narrower_box_too(monkeypatch):
    monkeypatch.delenv("OPS_MAX_TRANSFERS", raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert cp._store_pool() == runner.transport_cap()[0]


def test_store_pool_respects_the_env_when_set(monkeypatch):
    monkeypatch.setenv("OPS_MAX_TRANSFERS", "37")
    assert cp._store_pool() == 37
