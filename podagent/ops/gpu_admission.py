"""Exclusive, fair, bounded admission around the GPU HANDLER call only, never the chain.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

GPU_ADMISSION_WHY: str = """
Heavy ops each size their own NVENC session fan-out from free VRAM observed at start (cut_apply.max_
sessions); two running concurrently size for the SAME free VRAM and jointly OOM the card, so admission is
one-at-a-time, not a wider budget — the constraint is the physical card, not cores/RAM step_slots() prices.

Light ops are not routed through this gate at all. A PARKED heavy holds neither a step slot nor a
transport slot, so the step budget stays fully available to every light op the whole time it waits; an
ADMITTED heavy additionally takes one ordinary step slot around its handler (runner), so its CPU/RAM stays
priced by the same budget as everyone else's. The order is fixed — admission first, then the step slot —
and lights never take admission, so the two locks cannot cycle.

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


class GpuAdmissionRefused(RuntimeError):
    """Bounded-FIFO overflow, or a thread re-entering admission it already holds."""


class GpuAdmissionTimeout(RuntimeError):
    """A parked waiter's deadline expired before it reached the head with the card free."""


_cond = threading.Condition()
_queue: deque = deque()
_busy = False
_local = threading.local()  # per-thread re-entrancy marker: only THIS thread's holding state matters


def _reset_for_tests() -> None:
    global _busy, _local
    with _cond:
        _queue.clear()
        _busy = False
        _cond.notify_all()
    # a FRESH local, not a flag flip: the flip only reaches the calling thread, and a worker thread that
    # held admission across a reset would keep a stale marker and refuse its next legitimate acquire
    _local = threading.local()


@contextmanager
def admission(op_name: str, deadline_s: float = HEAVY_WAIT_DEADLINE_S) -> Iterator[None]:
    """Park in FIFO order until this token leads the queue and the card is free, or refuse/timeout."""
    global _busy
    if getattr(_local, "holding", False):
        # a heavy op invoking a heavy op on ITS OWN thread cannot both hold and wait — refuse, not deadlock
        raise GpuAdmissionRefused(
            f"op {op_name!r} refused GPU admission: this thread already holds it "
            f"(queue depth {len(_queue)}, waited 0.0s)")

    started = time.monotonic()
    with _cond:
        if len(_queue) >= MAX_PARKED:
            raise GpuAdmissionRefused(
                f"op {op_name!r} refused GPU admission: {len(_queue)} heavy op(s) already parked "
                f"(MAX_PARKED={MAX_PARKED}, waited 0.0s)")
        token = object()
        _queue.append(token)

    deadline = started + deadline_s
    admitted = False
    try:
        with _cond:
            while not (_queue and _queue[0] is token and not _busy):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    try:
                        _queue.remove(token)
                    except ValueError:
                        pass                # a racing admit already popped it; nothing left to undo
                    _cond.notify_all()
                    raise GpuAdmissionTimeout(
                        f"op {op_name!r} timed out waiting for GPU admission after {deadline_s:.1f}s "
                        f"(queue depth {len(_queue)}, waited {time.monotonic() - started:.1f}s)")
                _cond.wait(timeout=remaining)
            _queue.popleft()
            _busy = True
            admitted = True
        _local.holding = True
        try:
            yield
        finally:
            with _cond:
                _busy = False
                _cond.notify_all()
            _local.holding = False
    finally:
        # any unwind reaching here without admitting must still drop the token; remove() is idempotent
        # against a token the admit branch above already popped
        if not admitted:
            with _cond:
                try:
                    _queue.remove(token)
                except ValueError:
                    pass
                _cond.notify_all()
