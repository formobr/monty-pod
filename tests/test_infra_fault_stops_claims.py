"""TRK-123 / MISC-200: an infrastructure-class run_error stops this pod's claiming and says so on the wire
(runner.INFRA_FAULT_WHY); no pod-side latch keeps a pod «ready» but never taking work."""
from __future__ import annotations

import threading
import time

import pytest

from podagent import event_stream
from podagent import main as agent_main
from podagent.event_stream import DELIVERY_PENDING_MAX_ATTEMPTS, DeliveryPending, EventStream, TransportUnhealthy
from podagent.ops import gpu_admission, registry, runner


def _step(tmp_path):
    op = next(o for o in registry.all_ops().values() if any(not p.optional for p in o.outputs))
    required = next(p.id for p in op.outputs if not p.optional)
    src = tmp_path / "in.bin"
    src.write_bytes(b"x")
    return type("S", (), {
        "id": "s", "op": op.op, "params": {}, "needs": [],
        "inputs": [type("B", (), {"port": p.id, "url": None, "from_step": None, "path": str(src)})()
                   for p in op.inputs],
        "outputs": [type("B", (), {"port": required, "url": None, "urls": None})()]})()


def _run_failing(tmp_path, monkeypatch, exc: BaseException, name: str):
    """Run one step whose handler raises `exc`, with the sink main's ops lane installs. Returns (events,
    coordinator)."""
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)

    def _boom(**_kw):
        raise exc

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _boom)
    coordinator = agent_main.RestartCoordinator()
    events: list[dict] = []
    token = runner.infra_fault_sink.set(coordinator.report_infra_fault)
    try:
        with pytest.raises(type(exc)):
            runner._run_step(_step(tmp_path), runner.Workspace(tmp_path / name), {},
                             emit=lambda **kw: events.append(kw))
    finally:
        runner.infra_fault_sink.reset(token)
    return events, coordinator


class _PollCounter:
    def __init__(self) -> None:
        self.calls = 0

    def poll_job(self):
        self.calls += 1
        return None


class _StreamCP:
    def __init__(self) -> None:
        self.order: list[str] = []

    def close_stream(self) -> None:
        self.order.append("close_stream")


def test_an_infra_class_run_error_stops_claiming_and_reports_it(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_main.time, "sleep", lambda _s: None)
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")

    # An infrastructure class the pool condemns on (registry/pod_defect_classes.yaml `classes:`).
    events, coordinator = _run_failing(
        tmp_path, monkeypatch, gpu_admission.GpuAdmissionTimeout("op 'x' timed out waiting for GPU admission"),
        "infra")
    phases = [e["phase"] for e in events]
    assert phases[-2:] == ["run_error", "infra_fault"], phases
    fault = events[-1]
    assert fault["timings"]["error_class"] == "gpu_admission_timeout"
    assert fault["outcome"] == "error" and fault["error_type"] == "GpuAdmissionTimeout"
    assert coordinator.infra_fault() and coordinator.infra_class == "gpu_admission_timeout"
    assert not coordinator.restart_requested(), "an infra fault is not a pack restart"

    cp = _PollCounter()
    agent_main._dispatch_loop(cp, None, None, None, lambda _j: None, once=True, coordinator=coordinator)
    assert cp.calls == 0, "a fenced pod must not claim even once more"

    # ...and it stops honestly after its in-flight work drained: stream closed, liveness mark cleared.
    (tmp_path / "podagent.alive").write_text("x")
    stream_cp = _StreamCP()
    with pytest.raises(SystemExit) as excinfo:
        agent_main._drain_and_stop_on_infra_fault(stream_cp, coordinator, drain_s=0.01,
                                                  sleep=lambda _s: None, kill_orphans=lambda: None)
    assert excinfo.value.code == 5
    assert stream_cp.order == ["close_stream"]
    assert not (tmp_path / "podagent.alive").exists()

    # NEGATIVE: a plain op error — and the texts the registry rules NOT infrastructure (a CUDA OOM is
    # job-caused, MISC-209) — change nothing: no infra_fault, and the claim loop keeps polling.
    for i, exc in enumerate((RuntimeError("boom: handler bug"),
                             RuntimeError("CUDA error: out of memory"))):
        events, coordinator = _run_failing(tmp_path, monkeypatch, exc, f"plain{i}")
        assert "infra_fault" not in [e["phase"] for e in events]
        assert events[-1]["phase"] == "run_error"
        assert "error_class" not in events[-1].get("timings", {})
        assert not coordinator.infra_fault()
        cp = _PollCounter()
        agent_main._dispatch_loop(cp, None, None, None, lambda _j: None, once=True, coordinator=coordinator)
        assert cp.calls == 1


def test_ops_lane_installs_the_infra_fault_sink(monkeypatch):
    """_run_ops is what hands the coordinator to the runner; without it the classifier reaches no one."""
    seen: list[object] = []

    def _fake_run_chain(chain, cp, **_kw):
        sink = runner.infra_fault_sink.get()
        seen.append(sink)
        sink("vulkan_false")
        return {}

    class _CP:
        interrupted = 0

        def interrupt_claim(self):
            self.interrupted += 1

    monkeypatch.setattr(runner, "run_chain", _fake_run_chain)
    coordinator = agent_main.RestartCoordinator()
    cp = _CP()
    agent_main._run_ops(object(), cp, corr_id="c", session_id="s", coordinator=coordinator)
    assert seen and seen[0] is not None
    assert coordinator.infra_class == "vulkan_false"
    assert cp.interrupted == 1, "the claim the loop is blocked in must be woken, not waited out"
    assert runner.infra_fault_sink.get() is None, "the sink must not leak past the chain it was set for"


def test_storage_error_clears_and_delivery_pending_is_bounded(tmp_path, monkeypatch):
    # ── _storage_error clears once a later durable write succeeds ──
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    try:
        real_replace = event_stream.os.replace
        broken = {"on": True}

        def _replace(src, dst):
            if broken["on"]:
                raise OSError("disk full")
            return real_replace(src, dst)

        monkeypatch.setattr(event_stream.os, "replace", _replace)
        ev = {"stage": "ops", "status": "step", "phase": "probe"}
        with pytest.raises(TransportUnhealthy, match="durable append failed"):
            stream.send_event(ev)
        assert stream._storage_error is not None and stream._admission_error is not None
        with pytest.raises(TransportUnhealthy, match="durable append failed"):
            stream.send_event(ev)             # still broken: the latch still refuses

        broken["on"] = False
        assert stream.send_event(ev) is True  # the volume recovered: the append goes through
        assert stream._storage_error is None
        assert stream._admission_error is None, "admission the storage latch closed must reopen with it"
    finally:
        monkeypatch.undo()
        stream.close()

    # ── consecutive DeliveryPending waits end at the stream's own cap, in its named refusal ──
    monkeypatch.setattr(agent_main, "_TRANSPORT_UNHEALTHY_BACKOFF_S", 0.0)
    monkeypatch.setattr(agent_main.time, "sleep", lambda _s: None)
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    logs: list[str] = []
    monkeypatch.setattr(agent_main, "_log", logs.append)

    class _AlwaysPending:
        calls = 0

        def poll_job(self):
            self.calls += 1
            raise DeliveryPending("startup replay must clear before admitting work")

    cp = _AlwaysPending()
    with pytest.raises(SystemExit) as excinfo:
        agent_main._dispatch_loop(cp, None, None, None, lambda _j: None)
    assert excinfo.value.code == 4
    assert cp.calls == DELIVERY_PENDING_MAX_ATTEMPTS + 1
    assert f"delivery-pending cap ({DELIVERY_PENDING_MAX_ATTEMPTS}) exceeded" in logs[-1]

    # A healthy poll resets the count: pending bursts separated by a real poll never add up to the cap.
    class _Bursty:
        calls = 0

        def poll_job(self):
            self.calls += 1
            if self.calls % DELIVERY_PENDING_MAX_ATTEMPTS == 0:
                return None
            if self.calls > 3 * DELIVERY_PENDING_MAX_ATTEMPTS:
                raise SystemExit(0)           # test-only way out; production never returns
            raise DeliveryPending("still settling")

    with pytest.raises(SystemExit) as excinfo:
        agent_main._dispatch_loop(_Bursty(), None, None, None, lambda _j: None)
    assert excinfo.value.code == 0


def test_the_real_gpu_admission_timeout_reaches_the_fence(tmp_path, monkeypatch):
    """GpuAdmissionTimeout is raised by the admission wait, before run_started — the heavy_slot_wait_error
    path, never run_error. The fence must see it there, through the real gpu_admission code."""
    op = next(o for o in registry.all_ops().values()
              if o.op in gpu_admission.HEAVY_GPU_OPS and o.budget != "transport"
              and any(not p.optional for p in o.outputs))
    step = _step(tmp_path)
    step.op = op.op
    step.inputs = [type("B", (), {"port": p.id, "url": None, "from_step": None,
                                  "path": str(tmp_path / "in.bin")})() for p in op.inputs]
    step.outputs = [type("B", (), {"port": next(p.id for p in op.outputs if not p.optional),
                                   "url": None, "urls": None})()]
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: (lambda **kw: None))
    real = gpu_admission.admission
    monkeypatch.setattr(runner.gpu_admission, "admission", lambda name, **kw: real(name, **{**kw, "deadline_s": 0.01}))
    gpu_admission._reset_for_tests()
    with gpu_admission._cond:
        gpu_admission._live[object()] = gpu_admission.budget_mib()   # another heavy op holds the whole budget
    coordinator = agent_main.RestartCoordinator()
    events: list[dict] = []
    token = runner.infra_fault_sink.set(coordinator.report_infra_fault)
    try:
        with pytest.raises(gpu_admission.GpuAdmissionTimeout):
            runner._run_step(step, runner.Workspace(tmp_path / "heavy"), {}, emit=lambda **kw: events.append(kw))
    finally:
        runner.infra_fault_sink.reset(token)
        gpu_admission._reset_for_tests()
    phases = [e["phase"] for e in events]
    assert "run_started" not in phases and phases[-2:] == ["heavy_slot_wait_error", "infra_fault"], phases
    assert events[-1]["timings"]["error_class"] == "gpu_admission_timeout"
    assert coordinator.infra_fault()


def test_a_fault_during_a_blocked_claim_wakes_it_and_the_claimed_job_is_released_not_run(tmp_path, monkeypatch):
    # The claim the loop sits in returns at once when the fault reporter fires — no job, no latch on admission.
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    try:
        monkeypatch.setattr(stream, "_ensure_open", lambda: True)
        coordinator = agent_main.RestartCoordinator()
        cp = type("CP", (), {"interrupt_claim": lambda self: stream.interrupt_claim()})()
        out: list[object] = []
        t = threading.Thread(target=lambda: out.append(stream.claim(30.0)))
        t0 = time.monotonic()
        t.start()
        time.sleep(0.05)
        agent_main._infra_fault_reporter(cp, coordinator)("gpu_admission_timeout")
        t.join(timeout=5)
        assert out == [None] and time.monotonic() - t0 < 5, "a woken claim returns None, not after its wall"
        assert stream._admission_error is None, "waking a claim must not latch the stream's admission"
    finally:
        stream.close()

    # A job the in-flight claim already took when the fault landed is released with a terminal, never run.
    class _FaultsMidClaim:
        def __init__(self) -> None:
            self.events: list[dict] = []
            self.results: list[dict] = []

        def poll_job(self):
            coordinator.report_infra_fault("gpu_admission_timeout")   # lands while the claim is in flight
            return {"type": "ops", "corr_id": "c-1", "session_id": "s-1", "chain": {"job_id": "j-1"}}

        def send_event(self, payload, wait=False):
            self.events.append(payload)

        def send_result(self, payload, wait=True):
            self.results.append(payload)

    class _NoPool:
        def submit(self, *_a, **_k):
            raise AssertionError("a job claimed after an infra fault must never reach a pool")

    coordinator = agent_main.RestartCoordinator()
    cp = _FaultsMidClaim()
    agent_main._dispatch_loop(cp, _NoPool(), _NoPool(), _NoPool(), lambda _j: None, coordinator=coordinator)
    assert [r["corr_id"] for r in cp.results] == ["c-1"] and cp.results[0]["status"] == "error"
    assert "InfraFaultRefused" in cp.results[0]["error"]
    assert not [e for e in cp.events if e.get("phase") == "received"], "released before any lifecycle"


def test_storage_recovery_restores_the_verdict_it_displaced(tmp_path, monkeypatch):
    stream = EventStream("http://127.0.0.1:1", "token", outbox_path=tmp_path / "outbox.json")
    try:
        verdict = TransportUnhealthy("worker identity refused")
        with stream._work:
            stream._latch_locked(verdict)
        real_replace = event_stream.os.replace
        broken = {"on": True}
        monkeypatch.setattr(event_stream.os, "replace",
                            lambda a, b: (_ for _ in ()).throw(OSError("disk full")) if broken["on"]
                            else real_replace(a, b))
        with pytest.raises(TransportUnhealthy, match="durable append failed"):
            stream.send_event({"stage": "ops", "status": "step", "phase": "probe"})
        broken["on"] = False
        stream.send_event({"stage": "ops", "status": "step", "phase": "probe"})
        assert stream._storage_error is None
        assert stream._admission_error is verdict, "storage recovery must not reopen what a verdict closed"
    finally:
        monkeypatch.undo()
        stream.close()
