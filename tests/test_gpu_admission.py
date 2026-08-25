"""GPU admission: exclusive, fair, bounded park around the heavy-op HANDLER call, never the chain.
Every test is NEGATIVE (docs/TESTING.md): each was watched fail with its mechanism reverted.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from podagent.ops import gpu_admission, registry, runner


@pytest.fixture(autouse=True)
def _reset():
    gpu_admission._reset_for_tests()
    yield
    gpu_admission._reset_for_tests()


def _spin_until(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.005)


# ── a. FIFO order ────────────────────────────────────────────────────────────────────────────────

def test_admissions_are_granted_in_arrival_order():
    """NEGATIVE: waking whichever thread the scheduler favours would starve a heavy op parked behind a
    slow one — arrival order is the fairness contract."""
    order: list[str] = []
    lock = threading.Lock()
    release_a = threading.Event()

    def _run(name: str, release: threading.Event | None) -> None:
        with gpu_admission.admission("cut.apply", deadline_s=5.0):
            with lock:
                order.append(name)
            if release is not None:
                assert release.wait(timeout=5), f"{name} was never released"

    ta = threading.Thread(target=_run, args=("a", release_a))
    ta.start()
    _spin_until(lambda: order == ["a"])

    tb = threading.Thread(target=_run, args=("b", None))
    tb.start()
    _spin_until(lambda: len(gpu_admission._queue) == 1)

    tc = threading.Thread(target=_run, args=("c", None))
    tc.start()
    _spin_until(lambda: len(gpu_admission._queue) == 2)

    release_a.set()
    for t in (ta, tb, tc):
        t.join(timeout=5)
        assert not t.is_alive()

    assert order == ["a", "b", "c"]


# ── b. re-entrancy ───────────────────────────────────────────────────────────────────────────────

def test_reentrant_admission_refuses_immediately_instead_of_blocking():
    """NEGATIVE: a naive lock would deadlock a thread that invokes a second heavy op from inside the
    first's handler. The short join timeout is the proof it never even parks."""
    caught: list[BaseException] = []

    def _run() -> None:
        with gpu_admission.admission("cut.apply", deadline_s=5.0):
            try:
                with gpu_admission.admission("cut.apply", deadline_s=5.0):
                    pass  # unreachable: the inner admission must refuse before yielding
            except gpu_admission.GpuAdmissionRefused as exc:
                caught.append(exc)

    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=1.0)
    assert not t.is_alive(), "re-entrant admission blocked instead of refusing"
    assert len(caught) == 1 and "cut.apply" in str(caught[0])


# ── c. deadline ──────────────────────────────────────────────────────────────────────────────────

def test_deadline_times_out_and_the_queue_recovers_for_a_fresh_waiter():
    release_holder = threading.Event()

    def _hold() -> None:
        with gpu_admission.admission("cut.apply", deadline_s=5.0):
            assert release_holder.wait(timeout=5), "holder was never released"

    holder = threading.Thread(target=_hold)
    holder.start()
    _spin_until(lambda: gpu_admission._busy)

    with pytest.raises(gpu_admission.GpuAdmissionTimeout, match="cut.apply"):
        with gpu_admission.admission("cut.apply", deadline_s=0.05):
            pytest.fail("a timed-out waiter must never be admitted")

    assert list(gpu_admission._queue) == [], "the timed-out waiter's own token must be gone"

    release_holder.set()
    holder.join(timeout=5)

    admitted = threading.Event()
    with gpu_admission.admission("cut.apply", deadline_s=1.0):
        admitted.set()
    assert admitted.is_set(), "a fresh waiter after the holder released must be admitted, not poisoned"


# ── d. bounded FIFO ──────────────────────────────────────────────────────────────────────────────

def test_bounded_fifo_refuses_once_the_parked_depth_is_reached(monkeypatch):
    monkeypatch.setattr(gpu_admission, "MAX_PARKED", 1)
    release_holder = threading.Event()

    def _hold() -> None:
        with gpu_admission.admission("cut.apply", deadline_s=5.0):
            assert release_holder.wait(timeout=5), "holder was never released"

    holder = threading.Thread(target=_hold)
    holder.start()
    _spin_until(lambda: gpu_admission._busy)

    release_parked = threading.Event()

    def _park() -> None:
        with gpu_admission.admission("cut.apply", deadline_s=5.0):
            assert release_parked.wait(timeout=5), "parked waiter was never released"

    parked = threading.Thread(target=_park)
    parked.start()
    _spin_until(lambda: len(gpu_admission._queue) == 1)

    with pytest.raises(gpu_admission.GpuAdmissionRefused, match="parked"):
        with gpu_admission.admission("cut.apply", deadline_s=1.0):
            pytest.fail("a ninth (here: second, MAX_PARKED=1) waiter must be refused, not parked")

    release_holder.set()
    release_parked.set()
    holder.join(timeout=5)
    parked.join(timeout=5)


# ── e. a PARKED heavy holds no step slot; an ADMITTED one takes it in fixed order; events are wire-legal ──

def test_heavy_op_takes_step_slot_only_after_admission_and_emits_wire_legal_events(monkeypatch, tmp_path: Path):
    """The reserve this module buys: a heavy consults handler_slots() only INSIDE admission (fixed lock
    order), so a park holds nothing and the step budget stays open to light ops the whole wait."""
    from podagent.stream_models import event_payload

    order: list[str] = []
    real_admission = gpu_admission.admission

    def _recording_admission(op_name, deadline_s=gpu_admission.HEAVY_WAIT_DEADLINE_S):
        order.append("admission")
        return real_admission(op_name, deadline_s)

    def _recording_handler_slots(op):
        order.append(f"step_slot:{op.op}")
        return runner.step_slots()

    monkeypatch.setattr(runner.gpu_admission, "admission", _recording_admission)
    monkeypatch.setattr(runner, "handler_slots", _recording_handler_slots)
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.registry, "assert_pod_safe", lambda *a, **k: None)

    def _handler(*, params, inputs, outputs):
        assert gpu_admission._busy, "the heavy handler must run INSIDE admission"
        outputs["dst"].write_bytes(b"cut")
        outputs["durs"].write_text("[]")

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler)

    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")

    step = type("S", (), {
        "id": "s1", "op": "cut.apply", "params": {}, "needs": [], "optional": False,
        "inputs": [type("B", (), {"port": "src", "url": None, "from_step": None, "path": str(src)})()],
        "outputs": [type("B", (), {"port": "dst", "url": None, "urls": None})(),
                    type("B", (), {"port": "durs", "url": None, "urls": None})()],
    })()

    events: list[dict] = []
    runner._run_step(step, runner.Workspace(tmp_path / "ws"), {}, emit=lambda **payload: events.append(payload))

    assert order == ["admission", "step_slot:cut.apply"], order

    heavy = [e for e in events if e.get("phase") in ("heavy_slot_wait_started", "heavy_slot_wait_ended")]
    assert [e["phase"] for e in heavy] == ["heavy_slot_wait_started", "heavy_slot_wait_ended"]
    for e in heavy:
        # worker rides INSIDE timings: the wire event vocabulary is closed (StreamEventFields extra=forbid)
        assert e["op"] == "cut.apply" and e["step"] == "s1" and e["timings"].get("worker"), e
        # the REAL seam: every heavy event must validate against the generated wire model, or the event
        # stream latches dead on the first heavy op of the first montage
        event_payload({"stage": "ops", **e})
    assert "heavy_slot_wait_s" in heavy[1]["timings"], heavy[1]
    assert "bind_s" in heavy[0]["timings"], heavy[0]


# ── f. cancellation during wait ──────────────────────────────────────────────────────────────────

def test_cancellation_during_wait_leaves_queue_and_busy_consistent():
    """A waiter's deadline expiring while the HOLDER's own body is raising must not corrupt shared state:
    both release paths share one finally, and neither may leave the other stuck."""
    holder_ready = threading.Event()
    holder_may_fail = threading.Event()
    holder_exc: list[BaseException] = []

    def _hold_then_fail() -> None:
        try:
            with gpu_admission.admission("cut.apply", deadline_s=5.0):
                holder_ready.set()
                assert holder_may_fail.wait(timeout=5), "holder was never told to fail"
                raise RuntimeError("handler blew up")
        except RuntimeError as exc:
            holder_exc.append(exc)

    holder = threading.Thread(target=_hold_then_fail)
    holder.start()
    assert holder_ready.wait(timeout=2), "holder never reported admitted"

    waiter_exc: list[BaseException] = []

    def _wait_and_timeout() -> None:
        try:
            with gpu_admission.admission("cut.apply", deadline_s=0.05):
                pytest.fail("a timed-out waiter must never be admitted")
        except gpu_admission.GpuAdmissionTimeout as exc:
            waiter_exc.append(exc)

    waiter = threading.Thread(target=_wait_and_timeout)
    waiter.start()
    waiter.join(timeout=2)
    assert waiter_exc, "the parked waiter must have timed out"

    holder_may_fail.set()
    holder.join(timeout=2)
    assert holder_exc, "the holder's own exception must have propagated through admission"

    assert list(gpu_admission._queue) == [], "a token from either unwind path must not linger"
    assert gpu_admission._busy is False, "the holder's failure must still have cleared busy"
