"""Fair, bounded VRAM ledger around the GPU HANDLER call only, never the chain.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Iterator

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
exactly the joint OOM this module exists to stop. So admission() with no need — the runner's door today —
books the full budget, keeping one-at-a-time for them; admission(op, need_mib=...) (and reserve(op, need_mib)
with the one-session default) is the door for a handler that sizes from its own reservation. Such heavies
OVERLAP while their reservations fit the budget: two 960 MiB sessions run side by side on a 4 GiB card, and a
large need waits only until exactly its MiB are free, not until the card is empty (Q-20).

Waiters are admitted strictly in arrival order (no overtaking): admit = head of the queue AND reserved + need
<= budget. The head lets NOBODY past until it fits itself, even a small need that would fit right now — a
stream of small requests must never starve a large cut.apply parked ahead of them.

The wait is bounded (repo law: a wait with no deadline is a swallowed error; registered box-side as
deadline.yaml `gpu_heavy_admission_park`) by a deadline computed ONCE, at entry, from the queue policy:
(requests parked ahead + reservations admitted) × HEAVY_OP_CEILING_S, the per-op ceiling the box already
sanctions for every heavy op, capped by HEAVY_PARK_CEILING_S. Each op ahead of this one may legitimately
run its whole sanctioned ceiling, so a flat deadline (the old 90 s) timed out the THIRD of three
preview-equivalent heavies that were each well inside their own budget. Computed once is the point: a
deadline re-derived while waiting would stretch every time the queue changed, i.e. never be a deadline.
A waiter that still has not fit when it expires fails LOUD, naming its position and its request in MiB:
on this card that is the box over-driving one pod, and a silent multi-minute park would just move the same
failure past the point where anyone can read it.

A park is VISIBLE: a waiter that cannot be admitted on arrival calls the caller's `on_park` hook (the
runner emits a `heavy_park` event: position, budget, request, deadline) BEFORE it waits, outside the lock.

The queue is bounded to mirror OPS_MAX_CHAINS=8: one heavy op per chain in flight is a reasonable park; a
ninth waiter means the box over-drove this pod and must hear that now, as a refusal, not a growing queue.
"""

# Ops whose handler saturates the GPU (GPU_ADMISSION_WHY).
HEAVY_GPU_OPS = frozenset({"cut.apply", "media.normalize", "camera.apply", "media.cut_proxy"})

# The sanctioned per-op ceiling of every heavy op: the box's _OP_BUDGET_S rows for the four heavies sit AT the
# measured cap, video-editor scripts/op_backend.py:437-444 (_MEASURED_CEILING_S 200.0 − overhead = 140.0 s).
# The park deadline is priced in units of it (GPU_ADMISSION_WHY); deadline.yaml cites the cap below.
HEAVY_OP_CEILING_S = 140.0
# Cap on any one park: four whole heavy ceilings, inside the box's widest op window (scripts/op_backend.py
# _OPS_MAX_BUDGET_S = 600.0) — a park past that would outlive every envelope the box can grant the op.
HEAVY_PARK_CEILING_S = 4 * HEAVY_OP_CEILING_S
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


def park_deadline_s(ahead: int, admitted: int) -> float:
    """The park deadline, priced ONCE at entry from the queue policy (GPU_ADMISSION_WHY)."""
    return min(max(ahead + admitted, 1) * HEAVY_OP_CEILING_S, HEAVY_PARK_CEILING_S)


def reserve(op_name: str, need_mib: float = NVENC_SESSION_MIB, deadline_s: float | None = None,
            on_park: Callable[..., None] | None = None) -> object:
    """Park in FIFO order until this waiter leads the queue and `need_mib` fits in what the live reservations
    leave of the budget; return the token release() takes. Refuses or times out, never parks unbounded.
    `deadline_s` None prices the deadline at entry (park_deadline_s); a waiter that must park first calls
    `on_park(position=, ahead=, admitted=, budget_mib=, need_mib=, deadline_s=)` outside the lock."""
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
        ahead, admitted = len(_queue), len(_live)
        position = ahead + 1  # 1-based place in the park queue at entry
        if deadline_s is None:
            deadline_s = park_deadline_s(ahead, admitted)
        token = object()
        _queue.append(token)
        deadline = started + deadline_s
        parks = not _admissible(token, need_mib, budget)
    try:
        if parks and on_park is not None:
            on_park(position=position, ahead=ahead, admitted=admitted, budget_mib=budget,
                    need_mib=need_mib, deadline_s=deadline_s)
        with _cond:
            while not _admissible(token, need_mib, budget):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GpuAdmissionTimeout(
                        f"op {op_name!r} timed out waiting for GPU admission after {deadline_s:.1f}s "
                        f"at park position {position} (request {need_mib:.0f} MiB, "
                        f"{sum(_live.values()):.0f}/{budget:.0f} MiB reserved, {ahead} ahead + "
                        f"{admitted} admitted at entry, waited {time.monotonic() - started:.1f}s)")
                _cond.wait(timeout=remaining)
            _queue.popleft()
            _live[token] = need_mib
            _cond.notify_all()  # the new head may fit in what is left
            return token
    except BaseException:
        # any unwind before admitting must drop the token and let the next head re-check the budget
        with _cond:
            try:
                _queue.remove(token)
            except ValueError:
                pass
            _cond.notify_all()
        raise


def _admissible(token: object, need_mib: float, budget: float) -> bool:
    """Caller holds _cond: head of the queue AND the need fits beside the live reservations."""
    return _queue[0] is token and sum(_live.values()) + need_mib <= budget


def release(token: object) -> None:
    """Return a reservation's memory to the budget and wake the waiters. Idempotent."""
    with _cond:
        _live.pop(token, None)
        _cond.notify_all()


@contextmanager
def admission(op_name: str, deadline_s: float | None = None, need_mib: float | None = None,
              on_park: Callable[..., None] | None = None) -> Iterator[None]:
    """reserve() `need_mib` — the WHOLE budget when None, for a self-sizing handler — and release() it on any
    exit; heavies with explicit needs overlap while they fit (GPU_ADMISSION_WHY)."""
    me = threading.get_ident()
    with _cond:
        if me in _holders:
            # a heavy op invoking a heavy op on ITS OWN thread cannot both hold and wait — refuse, not deadlock
            raise GpuAdmissionRefused(
                f"op {op_name!r} refused GPU admission: this thread already holds it "
                f"(queue depth {len(_queue)}, waited 0.0s)")
    token = reserve(op_name, budget_mib() if need_mib is None else need_mib, deadline_s, on_park)
    with _cond:
        _holders.add(me)
    try:
        yield
    finally:
        with _cond:
            _holders.discard(me)
        release(token)
