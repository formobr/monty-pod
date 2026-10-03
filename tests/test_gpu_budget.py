"""GPU admission as a VRAM ledger: reservations against ONE measured free-VRAM budget (GPU_ADMISSION_WHY).
Every test is NEGATIVE (docs/TESTING.md): each fails with its mechanism reverted.
"""
from __future__ import annotations

import random
import threading
import time

import pytest

from podagent.ops import gpu_admission as ga

SESSION = ga.NVENC_SESSION_MIB


@pytest.fixture(autouse=True)
def _reset():
    ga._reset_for_tests()
    yield
    ga._reset_for_tests()


def _spin_until(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.005)


def _measured(monkeypatch, free_mib: float | None) -> list[int]:
    """Unmeasured ledger whose card reports `free_mib`; returns the list of probe calls."""
    calls: list[int] = []

    def _probe():
        calls.append(1)
        return free_mib

    monkeypatch.setattr(ga, "_free_vram_mb", _probe)
    ga._reset_for_tests(budget_mib=None)
    return calls


def test_budget_is_one_read_of_free_vram_minus_the_pod_reserve(monkeypatch):
    """NEGATIVE: a re-read per admission would see our own running heavies as 'taken' and shrink the budget
    under them; retyping the reserve would let it drift from every other VRAM budget on the pod."""
    from podagent.infer_cliprank import _VRAM_RESERVE_MB
    calls = _measured(monkeypatch, 8000.0)
    assert ga.budget_mib() == 8000.0 - _VRAM_RESERVE_MB
    ga.release(ga.reserve("cut.apply", deadline_s=1.0))
    ga.release(ga.reserve("cut.apply", deadline_s=1.0))
    assert len(calls) == 1, "the budget must be read once, at the first admission"


def test_reservations_never_exceed_the_budget(monkeypatch):
    """NEGATIVE: many threads reserving mixed needs at once — the sum of live reservations, sampled at every
    admission, must never exceed the budget, and every waiter must eventually get through."""
    _measured(monkeypatch, 4000.0 + 512.0)  # budget 4000 MiB
    budget = ga.budget_mib()
    assert budget == 4000.0
    lock = threading.Lock()
    peaks: list[float] = []
    errors: list[BaseException] = []
    rng = random.Random(7)
    needs = [rng.choice([SESSION, 1500.0, 2500.0, 4000.0]) for _ in range(ga.MAX_PARKED)]

    def _run(need: float) -> None:
        try:
            token = ga.reserve("cut.apply", need, deadline_s=10.0)
            try:
                with ga._cond:
                    total = sum(ga._live.values())
                with lock:
                    peaks.append(total)
                time.sleep(0.01)
            finally:
                ga.release(token)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_run, args=(n,)) for n in needs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()
    assert not errors, errors
    assert len(peaks) == len(needs)
    assert max(peaks) <= budget, (max(peaks), budget)
    assert ga.reserved_mib() == 0.0 and list(ga._queue) == []


def test_two_sessions_fit_side_by_side_when_the_card_has_room(monkeypatch):
    """NEGATIVE: the old one-at-a-time gate would park the second heavy op although the card holds both."""
    _measured(monkeypatch, 2 * SESSION + 512.0)
    a = ga.reserve("cut.apply", deadline_s=1.0)
    b = ga.reserve("media.normalize", deadline_s=0.2)
    assert ga.reserved_mib() == 2 * SESSION
    ga.release(a)
    ga.release(b)


def test_release_returns_memory_and_wakes_a_waiter(monkeypatch):
    """NEGATIVE: a release that forgets to return the memory or to notify leaves the waiter parked until its
    deadline although the card is free."""
    _measured(monkeypatch, 2000.0 + 512.0)
    holder = ga.reserve("cut.apply", 1500.0, deadline_s=1.0)
    admitted = threading.Event()
    got: list[object] = []

    def _wait() -> None:
        got.append(ga.reserve("camera.apply", 1500.0, deadline_s=5.0))
        admitted.set()

    t = threading.Thread(target=_wait)
    t.start()
    _spin_until(lambda: len(ga._queue) == 1)
    assert not admitted.is_set()
    woke = time.monotonic()
    ga.release(holder)
    assert admitted.wait(timeout=1.0), "release must wake the parked waiter"
    assert time.monotonic() - woke < 1.0
    t.join(timeout=2)
    assert ga.reserved_mib() == 1500.0
    ga.release(got[0])
    assert ga.reserved_mib() == 0.0


def test_an_over_budget_waiter_times_out_at_its_deadline(monkeypatch):
    """NEGATIVE: a waiter whose need does not fit beside the live reservations must fail LOUD at its deadline
    — not be admitted over budget, not park forever — and leave no token behind."""
    _measured(monkeypatch, 2000.0 + 512.0)
    holder = ga.reserve("cut.apply", 1500.0, deadline_s=1.0)
    started = time.monotonic()
    with pytest.raises(ga.GpuAdmissionTimeout, match="media.normalize"):
        ga.reserve("media.normalize", 1000.0, deadline_s=0.1)
    waited = time.monotonic() - started
    assert 0.1 <= waited < 1.0, waited
    assert list(ga._queue) == [] and ga.reserved_mib() == 1500.0
    ga.release(holder)


def test_a_need_the_whole_budget_cannot_hold_is_refused_now(monkeypatch):
    _measured(monkeypatch, 2000.0 + 512.0)
    with pytest.raises(ga.GpuAdmissionRefused, match="budget"):
        ga.reserve("cut.apply", 2001.0, deadline_s=5.0)


@pytest.mark.parametrize("free", [None, 100.0])
def test_an_unreadable_or_starved_card_admits_exactly_one_heavy_op_at_a_time(monkeypatch, free):
    """NEGATIVE: an unreadable card guessed wide would let two heavies size from the same number; one read
    below a session would refuse every heavy. Both keep today's exclusivity: budget = exactly one session."""
    _measured(monkeypatch, free)
    assert ga.budget_mib() == SESSION
    release_holder = threading.Event()

    def _hold() -> None:
        with ga.admission("cut.apply", deadline_s=5.0):
            assert release_holder.wait(timeout=5)

    holder = threading.Thread(target=_hold)
    holder.start()
    _spin_until(lambda: bool(ga._live))
    with pytest.raises(ga.GpuAdmissionTimeout):
        with ga.admission("media.normalize", deadline_s=0.05):
            pytest.fail("a second heavy op must not run beside the first on an unreadable card")
    release_holder.set()
    holder.join(timeout=5)
    with ga.admission("media.normalize", deadline_s=1.0):
        assert ga.reserved_mib() == SESSION


def test_admission_stays_exclusive_on_a_wide_card_while_handlers_size_themselves(monkeypatch):
    """NEGATIVE: a flat one-session admission would admit two self-sizing heavies on an 8 GB card, each
    sizing its NVENC fan-out from the same free-VRAM number — the joint OOM. admission() books the whole
    budget until handlers size from their reservation."""
    _measured(monkeypatch, 8000.0)
    release_holder = threading.Event()

    def _hold() -> None:
        with ga.admission("cut.apply", deadline_s=5.0):
            assert release_holder.wait(timeout=5)

    holder = threading.Thread(target=_hold)
    holder.start()
    _spin_until(lambda: bool(ga._live))
    assert ga.reserved_mib() == ga.budget_mib()
    with pytest.raises(ga.GpuAdmissionTimeout):
        with ga.admission("cut.apply", deadline_s=0.05):
            pytest.fail("two self-sizing heavies must not run side by side")
    release_holder.set()
    holder.join(timeout=5)
    assert ga.reserved_mib() == 0.0


def test_the_budget_probe_runs_outside_the_ledger_lock(monkeypatch):
    """NEGATIVE: measuring under _cond would stall every release() and waiter behind a slow nvidia-smi —
    here, with another thread holding _cond, the first measurement must still reach the probe."""
    _measured(monkeypatch, 4000.0)
    probed = threading.Event()
    monkeypatch.setattr(ga, "_free_vram_mb", lambda: (probed.set(), 4000.0)[1])
    lock_held = threading.Event()
    let_go = threading.Event()

    def _hold_lock() -> None:
        with ga._cond:
            lock_held.set()
            let_go.wait(timeout=2)

    holder = threading.Thread(target=_hold_lock)
    holder.start()
    assert lock_held.wait(timeout=1)
    reader = threading.Thread(target=ga.budget_mib)
    reader.start()
    try:
        assert probed.wait(timeout=1), "the budget probe ran under _cond"
    finally:
        let_go.set()
        holder.join(timeout=2)
        reader.join(timeout=2)
    assert ga.budget_mib() == 4000.0 - 512.0
