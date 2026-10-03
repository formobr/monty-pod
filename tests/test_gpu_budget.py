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


def _hold_admission(op: str, need: float, entered: threading.Event, leave: threading.Event,
                    order: list[str] | None = None) -> threading.Thread:
    def _run() -> None:
        with ga.admission(op, deadline_s=5.0, need_mib=need):
            if order is not None:
                order.append(op)
            entered.set()
            assert leave.wait(timeout=5)

    t = threading.Thread(target=_run)
    t.start()
    return t


def test_small_heavy_ops_overlap_and_a_large_one_is_never_overtaken(monkeypatch):
    """NEGATIVE: a one-at-a-time gate would park the second small heavy op; a large need that waited for the
    card to EMPTY (not for exactly its MiB) would stay parked after the first release; a gate that let any
    fitting waiter in would let the stream of small ops behind the large cut.apply overtake — and starve — it."""
    # Two one-session heavies on a 4 GiB budget run side by side.
    _measured(monkeypatch, 4096.0 + 512.0)
    leave = threading.Event()
    a_in, b_in = threading.Event(), threading.Event()
    small = [_hold_admission("media.normalize", SESSION, a_in, leave),
             _hold_admission("camera.apply", SESSION, b_in, leave)]
    assert a_in.wait(timeout=1) and b_in.wait(timeout=1), "two small heavies must overlap inside the budget"
    assert ga.reserved_mib() == 2 * SESSION and list(ga._queue) == []
    leave.set()
    for t in small:
        t.join(timeout=2)
    assert ga.reserved_mib() == 0.0

    # An 8 GiB need on a 10 GiB budget waits for exactly enough, and nobody behind it gets past.
    _measured(monkeypatch, 10240.0 + 512.0)
    held = [ga.reserve("media.cut_proxy", 1024.0, deadline_s=1.0) for _ in range(3)]  # 3072 of 10240 MiB
    order: list[str] = []
    large_in, large_leave = threading.Event(), threading.Event()
    large = _hold_admission("cut.apply", 8192.0, large_in, large_leave, order)
    _spin_until(lambda: len(ga._queue) == 1)
    stream_leave = threading.Event()
    stream_in = [threading.Event() for _ in range(3)]
    stream = []
    for i, ev in enumerate(stream_in):
        # each small need fits in the 7168 MiB left right now — only strict FIFO keeps it parked
        stream.append(_hold_admission(f"small-{i}", SESSION, ev, stream_leave, order))
        _spin_until(lambda n=i: len(ga._queue) == n + 2)
    time.sleep(0.05)
    assert order == [] and ga.reserved_mib() == 3072.0, "a small waiter overtook the large head"

    ga.release(held.pop())  # 2048 + 8192 == budget: exactly enough, two reservations still live
    assert large_in.wait(timeout=1), "the large need must be admitted once exactly its MiB are free"
    assert ga.reserved_mib() == 10240.0 and len(ga._queue) == 3
    time.sleep(0.05)
    assert order == ["cut.apply"], "the stream must still wait behind a full budget"

    large_leave.set()
    large.join(timeout=2)
    assert all(ev.wait(timeout=1) for ev in stream_in)
    assert order == ["cut.apply", "small-0", "small-1", "small-2"], order
    assert ga.reserved_mib() == 2048.0 + 3 * SESSION
    stream_leave.set()
    for t in stream:
        t.join(timeout=2)
    for token in held:
        ga.release(token)
    assert ga.reserved_mib() == 0.0 and list(ga._queue) == []


def test_reentry_with_an_explicit_need_still_refuses(monkeypatch):
    """NEGATIVE: with room left in the budget, a re-entering thread would be admitted a second reservation
    instead of refused — a heavy op invoking a heavy op on its own thread must stay a loud refusal."""
    _measured(monkeypatch, 4096.0 + 512.0)
    with ga.admission("cut.apply", deadline_s=1.0, need_mib=SESSION):
        with pytest.raises(ga.GpuAdmissionRefused, match="already holds"):
            with ga.admission("media.normalize", deadline_s=1.0, need_mib=SESSION):
                pytest.fail("re-entry must refuse")
        assert ga.reserved_mib() == SESSION and list(ga._queue) == []
    assert ga.reserved_mib() == 0.0


def test_an_unreadable_card_admits_one_explicit_need_at_a_time(monkeypatch):
    """NEGATIVE: overlap must come only from a MEASURED budget — on an unreadable card two one-session needs
    must not run side by side."""
    _measured(monkeypatch, None)
    leave, entered = threading.Event(), threading.Event()
    holder = _hold_admission("cut.apply", SESSION, entered, leave)
    assert entered.wait(timeout=1)
    with pytest.raises(ga.GpuAdmissionTimeout):
        with ga.admission("media.normalize", deadline_s=0.05, need_mib=SESSION):
            pytest.fail("a second heavy op must not run beside the first on an unreadable card")
    leave.set()
    holder.join(timeout=2)
    assert ga.reserved_mib() == 0.0
