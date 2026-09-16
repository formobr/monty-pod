"""Which infer kinds may hold this pod's card at once (INFER_KIND_RESIDENCY_WHY)."""
from __future__ import annotations

import gc
from typing import Any, Callable, Mapping

from .infer_cliprank import _VRAM_PER_LANE_MB, _VRAM_RESERVE_MB, _VRAM_WEIGHTS_MB, _free_vram_mb

INFER_KIND_RESIDENCY_WHY = """
TWO INFER KINDS, ONE CARD, AND NOTHING DECIDED WHETHER IT HOLDS BOTH.

`align` (wav2vec2 CTC) rides the one-wide heavy lane, `clip_rank` (SigLIP so400m) rides its own rank lane, so
the two kinds overlap BY DESIGN (main._dispatch_loop) — and each kind's service is cached by weights hash and
never released, so even strictly serial envelopes leave the previous kind's checkpoint resident. On a fleet
card both fit with ~10 GB to spare and this module changes nothing. On a small card the second kind dies:
measured here, clip_rank asked for 20 MiB with 167 MiB free while the process still held 2.89 GiB of align.

THE DECISION IS DERIVED, exactly as the clip_rank lane width already is (infer_cliprank.LANE_SIZING_WHY): the
per-kind residency is measured with nvidia-smi and matches the engine's vram_budget.GpuResidents figure for
figure — align 1500 MiB (1.2 GB fp32 checkpoint + a 20 s window's emissions + the CUDA context, seen at 1.44
GiB in the OOM report, = vram_budget.GPU.ctc_align), clip_rank the fp16 towers plus ONE capped forward (the
two numbers the lane width is already sized from), face_probe's yunet ONNX no card tenant at all — and a card
that cannot hold both kinds runs them one at a time with the idle kind's weights dropped. Never the CPU:
ranking or aligning there moves the work across the placement axis the ruling forbids.

NO NEW WAIT. A narrow card routes clip_rank onto the SAME one-wide lane align already queues on, so the
ordering is the pool's and the bound is the envelope budget the box already grants a claimed infer job; an
unreadable card is treated as narrow, because serialising costs wall clock and guessing costs the envelope.
"""

KIND_VRAM_MIB: Mapping[str, float] = {      # measured; see INFER_KIND_RESIDENCY_WHY for each number
    "align": 1500.0,
    "clip_rank": _VRAM_WEIGHTS_MB + _VRAM_PER_LANE_MB,
    "face_probe": 0.0,
}
RESERVE_MIB = _VRAM_RESERVE_MB


def coresident_mib() -> float:
    """Free MiB a card must report for both weight-holding kinds to be resident together (4748 here; the
    engine's vram_budget.concurrent_mib() reads 4752 off its own rounding of the same measurements)."""
    return KIND_VRAM_MIB["align"] + KIND_VRAM_MIB["clip_rank"] + RESERVE_MIB


def kinds_fit_together(free_mib: float | None) -> tuple[bool, str]:
    """May align and clip_rank be resident at once on a card reporting `free_mib`, and why. Pure — the reading
    is injected."""
    need = coresident_mib()
    if free_mib is None:
        return False, ("the card reports no free VRAM — one infer kind resident at a time, still the GPU, "
                       "never the CPU")
    if free_mib >= need:
        return True, (f"{free_mib:.0f} MiB free >= {need:.0f} MiB for align+clip_rank+reserve — both kinds may "
                      f"hold the card, the lanes stay parallel")
    return False, (f"{free_mib:.0f} MiB free < {need:.0f} MiB for align+clip_rank+reserve — the kinds "
                   f"serialise on one lane and the idle kind's weights are dropped")


def card_holds_both_kinds(probe: Callable[[], float | None] | None = None) -> tuple[bool, str]:
    """The same answer against the LIVE card. A probe failure is a reading, not a raise (this is a budget)."""
    try:
        free = (probe or _free_vram_mb)()
    except (OSError, ValueError) as e:
        return False, f"could not ask the card for free VRAM ({type(e).__name__}: {e}) — one kind at a time"
    return kinds_fit_together(free)


def release_other_kinds(kind: str, caches: Mapping[str, dict[Any, Any]]) -> list[str]:
    """Drop every OTHER kind's cached service so only `kind` holds weights; the allocator then recycles those
    blocks for this kind's load. Safe only where no other kind can be mid-forward — the caller serialises the
    kinds onto one lane first."""
    freed = [other for other, cache in caches.items() if other != kind and cache]
    for other in freed:
        caches[other].clear()       # the service object holds the only reference to the model
    if freed:
        gc.collect()                # the service graph has cycles; the tensors outlive the dict without this
    return freed
