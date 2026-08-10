"""How wide a chain runs on THIS box.

The cap is a scheduling decision made from two readings — cores and available memory — so every case
here injects both rather than introspecting the machine the tests happen to run on. Each asserts a
specific wrong cap is refused, and each was watched fail with the derivation reverted to the old flat
`min(8, cpu_count)`.
"""
from __future__ import annotations

from podagent.ops import registry, runner


def test_big_box_scales_above_the_old_flat_eight():
    """The whole point: a 32-core box with RAM to match must not run 8-wide. 32//2 = 16 cores-bound,
    96 GiB // 1.5 GiB = 64 memory-bound, so cores are the binding side."""
    cap, why = runner._cap_from(cores=32, mem_avail=96 * 1024**3, env_raw=None)
    assert cap == 16
    assert "core-bound=16" in why and "mem-bound=64" in why


def test_memory_holds_a_big_core_box_down():
    """Cores are not the constraint alone: the measured ~1 GB-RSS-per-encode is why. 32 cores would
    schedule 16, but 6 GiB of available RAM only holds 4 — and 16 encodes on 6 GiB is the swap
    thrashing this cap exists to prevent."""
    cap, why = runner._cap_from(cores=32, mem_avail=6 * 1024**3, env_raw=None)
    assert cap == 4
    assert "mem-bound=4" in why


def test_env_override_wins_over_both_bounds():
    """An operator (and the local harness, which pins 4 for laptop comfort) must be able to say the
    number outright — over a derived cap in either direction."""
    assert runner._cap_from(cores=32, mem_avail=96 * 1024**3, env_raw="4")[0] == 4
    cap, why = runner._cap_from(cores=4, mem_avail=2 * 1024**3, env_raw="12")
    assert cap == 12
    assert "env=12" in why


def test_unreadable_memory_degrades_to_the_core_bound():
    """A kernel that will not report memory must cost a conservative cap, not a crashed chain on a
    rented box."""
    cap, why = runner._cap_from(cores=32, mem_avail=None, env_raw=None)
    assert cap == 16
    assert "mem-bound=unknown" in why


def test_probe_returning_nothing_does_not_raise(monkeypatch):
    """Same degradation through the real entry point: /proc absent AND sysconf refusing."""
    monkeypatch.setattr(runner, "_mem_available_bytes", lambda: None)
    monkeypatch.setattr(runner.os, "cpu_count", lambda: 8)
    monkeypatch.delenv(runner.MAX_PARALLEL_ENV, raising=False)
    assert runner.parallel_cap()[0] == 4


def test_floor_holds_at_one():
    """A one-core box, a box with less RAM than one step's budget, a garbage or zero override: a
    ThreadPoolExecutor with max_workers<1 raises, so the cap may never reach 0 or go negative."""
    assert runner._cap_from(cores=1, mem_avail=64 * 1024**3, env_raw=None)[0] == 1
    assert runner._cap_from(cores=32, mem_avail=100 * 1024**2, env_raw=None)[0] == 1
    assert runner._cap_from(cores=None, mem_avail=None, env_raw=None)[0] >= 1
    assert runner._cap_from(cores=8, mem_avail=None, env_raw="0")[0] == 4
    assert runner._cap_from(cores=8, mem_avail=None, env_raw="not-a-number")[0] == 4
    assert runner._cap_from(cores=8, mem_avail=None, env_raw="-3")[0] == 4


# ── the budget is BOX-wide, not chain-wide ───────────────────────────────────────────────────────

def _budget_probe(monkeypatch, cap, frame):
    """Drive `frame` from six threads at once and report the peak it ever ran concurrently."""
    import threading

    monkeypatch.setattr(runner, "parallel_cap", lambda: (cap, "test"))
    runner._reset_step_slots()
    live = peak = 0
    lock = threading.Lock()
    gate = threading.Event()

    def _work():
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        gate.wait(timeout=0.3)
        with lock:
            live -= 1

    threads = [threading.Thread(target=frame, args=(_work,)) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=5)
    runner._reset_step_slots()
    return peak


def test_concurrent_chains_share_one_cpu_budget(monkeypatch):
    """The agent drains several ops envelopes at once. A cap applied per chain multiplies with them —
    4 chains x cap 8 is 32 ffmpegs on a box sized for 8, which is the swap thrashing MEM_PER_STEP_BYTES
    exists to prevent. The budget wraps the HANDLER CALL now (TRANSPORT_BUDGET_WHY), which is the frame
    those two constants are statements about; it is still ONE counter for the whole box."""
    def frame(work):
        with runner.step_slots():
            work()

    assert _budget_probe(monkeypatch, 2, frame) <= 2


def test_a_transfer_does_not_take_a_slot_sized_for_an_encode(monkeypatch):
    """THE ONE THIS SPLIT EXISTS FOR. `put 208.1 s` of `375.7 s` of b-roll step time was objects going to
    R2 — no core burned, and a one-step `media.fetch` starved behind fifteen-step chains for it. Transport
    is bounded (the disk and the socket count are real), but by its OWN counter, wider than the CPU one."""
    def frame(work):
        with runner.transport_slots():
            work()

    peak = _budget_probe(monkeypatch, 2, frame)
    assert peak > 2, f"transport still capped at the CPU budget ({peak})"
    assert peak <= runner.transport_cap()[0]


def test_the_transport_budget_is_derived_from_the_cpu_one_and_overridable(monkeypatch):
    """It has to scale with the box like the CPU cap does — and an operator who has measured their own link
    may say so. An unreadable knob is not a bound: the derivation stands and the number stays sane."""
    monkeypatch.setattr(runner, "parallel_cap", lambda: (4, "test"))
    monkeypatch.delenv(runner.TRANSPORT_MAX_ENV, raising=False)
    assert runner.transport_cap()[0] == 4 * runner.TRANSFERS_PER_STEP
    monkeypatch.setenv(runner.TRANSPORT_MAX_ENV, "3")
    assert runner.transport_cap()[0] == 3
    monkeypatch.setenv(runner.TRANSPORT_MAX_ENV, "not-a-number")
    assert runner.transport_cap()[0] == 4 * runner.TRANSFERS_PER_STEP
    monkeypatch.setenv(runner.TRANSPORT_MAX_ENV, "0")
    assert runner.transport_cap()[0] == 4 * runner.TRANSFERS_PER_STEP


def test_transport_budgeted_handler_can_use_transport_width_but_cpu_handler_cannot(monkeypatch):
    """`media.fetch` waits on origin/remux I/O, so its handler may use the wider transport budget.

    NEGATIVE: route every handler through `step_slots()` and six fetch siblings peak at the CPU cap;
    route every handler through `transport_slots()` and pixel work can oversubscribe the box. The op
    declaration must choose the one permit explicitly.
    """
    monkeypatch.setattr(runner, "parallel_cap", lambda: (2, "test"))
    runner._reset_step_slots()

    # Keep the actual probe explicit so the test models the runner's `with handler_slots(op)`.
    def frame_for(op):
        def frame(work):
            with runner.handler_slots(op):
                work()
        return frame

    fetch_peak = _budget_probe(monkeypatch, 2, frame_for(registry.get("media.fetch")))
    cpu_peak = _budget_probe(monkeypatch, 2, frame_for(registry.get("media.scale")))
    assert fetch_peak > 2
    assert fetch_peak <= runner.transport_cap()[0]
    assert cpu_peak <= 2


def test_chain_executor_exposes_the_wider_budget():
    """A per-chain executor capped at CPU width would hide transport slots behind an artificial queue."""
    assert runner.executor_workers(5, 20) == 20
    assert runner.executor_workers(20, 5) == 20
