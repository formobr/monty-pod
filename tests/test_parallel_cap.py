"""How wide a chain runs on THIS box.

The cap is a scheduling decision made from two readings — cores and available memory — so every case
here injects both rather than introspecting the machine the tests happen to run on. Each asserts a
specific wrong cap is refused, and each was watched fail with the derivation reverted to the old flat
`min(8, cpu_count)`.
"""
from __future__ import annotations

from podagent.ops import runner


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
