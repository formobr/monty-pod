"""Fair, bounded VRAM ledger around the GPU HANDLER call only, never the chain.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

from ..infer_cliprank import _VRAM_RESERVE_MB, _free_vram_mb

GPU_ADMISSION_WHY: str = """
Heavy ops each size their own NVENC session fan-out from free VRAM observed at start (cut_apply.max_
sessions); two running concurrently that size for the SAME free-VRAM number jointly OOM the card. What is
forbidden is therefore not "two heavy ops at once" but "two heavy ops sizing themselves from the same
free-VRAM number": admission is a LEDGER of reservations against ONE budget, and the ledger is the single
source of sizing — the constraint is the physical card, not cores/RAM step_slots() prices.

The budget is read ONCE, at the first admission, by the reader the pod already has (infer_cliprank.
_free_vram_mb, nvidia-smi memory.free, bounded) minus the reserve every other VRAM budget on this pod keeps
(infer_cliprank._VRAM_RESERVE_MB). Once is the point: a re-read while our own heavies run would see their
allocations as "taken" and double-count them. A card that reports nothing gets exactly ONE session's worth —
today's one-at-a-time exclusivity, never a guess. A measured budget below one session is also floored to
one session: the ledger may only ever be WIDER than the exclusivity it replaces, never stricter (the boot
preflight is what refuses a card too full to run at all).

Light ops are not routed through this gate at all. A PARKED heavy holds neither a step slot nor a
transport slot, so the step budget stays fully available to every light op the whole time it waits; an
ADMITTED heavy additionally takes one ordinary step slot around its handler (runner), so its CPU/RAM stays
priced by the same budget as everyone else's. The order is fixed — admission first, then the step slot —
and lights never take admission, so the two locks cannot cycle.

A SELF-SIZING HANDLER RESERVES THE WHOLE BUDGET. Today's heavy handlers (cut_apply.max_sessions and
kin) still size their NVENC fan-out from their OWN free-VRAM read, not from a reservation the ledger hands
them; a flat one-session reservation would admit two of them side by side, each sizing for the whole card —
exactly the joint OOM this module exists to stop. So admission() — the runner's door — books the full budget,
keeping one-at-a-time for them, and reserve(op, need_mib) with the one-session default is the door for a
handler that sizes from its own reservation. Heavies overlap only once their handlers do that (Q-20/Q-22).

Waiters are admitted strictly in arrival order: the head is admitted when its need fits in what the live
reservations leave, and nobody behind it jumps ahead — a large need parked behind small ones is not starved.

The wait is bounded (repo law: a wait with no deadline is a swallowed error; registered box-side as
deadline.yaml `gpu_heavy_admission_park`) and spends part of the op envelope the box already grants a
claimed heavy op — media.normalize/camera.apply carry 140 s table budgets, cut.apply/media.cut_proxy ride
the wider unmeasured window — never a second, uncoordinated clock. A waiter behind a heavy that runs
longer than this deadline fails LOUD by design: on this card that is the box over-driving one pod, and a
silent multi-minute park would just move the same failure past the point where anyone can read it.

The queue is bounded to mirror OPS_MAX_CHAINS=8: one heavy op per chain in flight is a reasonable park; a
ninth waiter means the box over-drove this pod and must hear that now, as a refusal, not a growing queue.
"""

# Ops whose handler saturates the GPU (GPU_ADMISSION_WHY).
HEAVY_GPU_OPS = frozenset({"cut.apply", "media.normalize", "camera.apply", "media.cut_proxy"})

HEAVY_WAIT_DEADLINE_S = 90.0   # part of the box's own op envelope, never a second clock (GPU_ADMISSION_WHY)
MAX_PARKED = 8                 # mirrors OPS_MAX_CHAINS (GPU_ADMISSION_WHY)


# Free VRAM one 1080p NVDEC+NVENC session needs — the default heavy-op reservation. The engine's own
# measurement: video-editor scripts/montyops/cut_apply.py `_VRAM_PER_SESSION_MB` (NVENC_SIZING_WHY there).
NVENC_SESSION_MIB = 960.0


class GpuAdmissionRefused(RuntimeError):
    """Bounded-FIFO overflow, a need the whole budget cannot hold, or a thread re-entering admission it
    already holds."""


class GpuAdmissionTimeout(RuntimeError):
    """A parked waiter's deadline expired before it reached the head with its need free in the budget."""


_cond = threading.Condition()
_queue: deque = deque()
_live: dict[object, float] = {}   # reservation token -> MiB it holds; the ledger
_holders: set[int] = set()        # thread idents inside admission() (re-entrancy guard)
_budget: float | None = None      # MiB; None until the first admission measures it (GPU_ADMISSION_WHY)


def _reset_for_tests(budget_mib: float | None = NVENC_SESSION_MIB) -> None:
    """Clear the ledger. The default pins today's exclusivity (one session) so no test reads a real card;
    pass None to leave the budget unmeasured, so the next admission reads it through _free_vram_mb."""
    global _budget
    with _cond:
        _queue.clear()
        _live.clear()
        _holders.clear()
        _budget = budget_mib
        _cond.notify_all()


def _measure_budget() -> float:
    free = _free_vram_mb()
    if free is None:
        return NVENC_SESSION_MIB  # unreadable card → exactly one session: today's exclusivity, never a guess
    return max(free - _VRAM_RESERVE_MB, NVENC_SESSION_MIB)


def budget_mib() -> float:
    """The ledger's budget: ONE read of free VRAM at the first admission, minus the pod's reserve."""
    global _budget
    if _budget is not None:       # a single global read; once set it only changes through _reset_for_tests
        return _budget
    measured = _measure_budget()  # outside _cond: a slow nvidia-smi must not stall release() or waiters
    with _cond:
        if _budget is None:       # a racing first admission may have measured too; the first write wins
            _budget = measured
        return _budget


def reserved_mib() -> float:
    with _cond:
        return sum(_live.values())


def reserve(op_name: str, need_mib: float = NVENC_SESSION_MIB,
            deadline_s: float = HEAVY_WAIT_DEADLINE_S) -> object:
    """Park in FIFO order until this waiter leads the queue and `need_mib` fits in what the live reservations
    leave of the budget; return the token release() takes. Refuses or times out, never parks unbounded."""
    budget = budget_mib()
    started = time.monotonic()
    with _cond:
        if not 0 < need_mib <= budget:
            raise GpuAdmissionRefused(
                f"op {op_name!r} refused GPU admission: need {need_mib:.0f} MiB is outside the "
                f"{budget:.0f} MiB budget (waited 0.0s)")
        if len(_queue) >= MAX_PARKED:
            raise GpuAdmissionRefused(
                f"op {op_name!r} refused GPU admission: {len(_queue)} heavy op(s) already parked "
                f"(MAX_PARKED={MAX_PARKED}, waited 0.0s)")
        token = object()
        _queue.append(token)
        deadline = started + deadline_s
        try:
            while not (_queue[0] is token and sum(_live.values()) + need_mib <= budget):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GpuAdmissionTimeout(
                        f"op {op_name!r} timed out waiting for GPU admission after {deadline_s:.1f}s "
                        f"(need {need_mib:.0f} MiB, {sum(_live.values()):.0f}/{budget:.0f} MiB reserved, "
                        f"queue depth {len(_queue) - 1}, waited {time.monotonic() - started:.1f}s)")
                _cond.wait(timeout=remaining)
        except BaseException:
            # any unwind before admitting must drop the token and let the next head re-check the budget
            try:
                _queue.remove(token)
            except ValueError:
                pass
            _cond.notify_all()
            raise
        _queue.popleft()
        _live[token] = need_mib
        _cond.notify_all()  # the new head may fit in what is left
        return token


def release(token: object) -> None:
    """Return a reservation's memory to the budget and wake the waiters. Idempotent."""
    with _cond:
        _live.pop(token, None)
        _cond.notify_all()


@contextmanager
def admission(op_name: str, deadline_s: float = HEAVY_WAIT_DEADLINE_S) -> Iterator[None]:
    """reserve() the WHOLE budget for a self-sizing handler, release() it on any exit (GPU_ADMISSION_WHY)."""
    me = threading.get_ident()
    with _cond:
        if me in _holders:
            # a heavy op invoking a heavy op on ITS OWN thread cannot both hold and wait — refuse, not deadlock
            raise GpuAdmissionRefused(
                f"op {op_name!r} refused GPU admission: this thread already holds it "
                f"(queue depth {len(_queue)}, waited 0.0s)")
    token = reserve(op_name, budget_mib(), deadline_s)
    with _cond:
        _holders.add(me)
    try:
        yield
    finally:
        with _cond:
            _holders.discard(me)
        release(token)
