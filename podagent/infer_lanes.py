"""Which infer kinds may hold this pod's card at once (INFER_KIND_RESIDENCY_WHY)."""
from __future__ import annotations

import gc
from typing import Any, Callable, Mapping

from .infer_cliprank import _VRAM_PER_LANE_MB, _VRAM_RESERVE_MB, _VRAM_WEIGHTS_MB, _free_vram_mb
from .ops.gpu_admission import NVENC_SESSION_MIB

INFER_KIND_RESIDENCY_WHY = """
TWO INFER KINDS, ONE CARD, AND NOTHING DECIDED WHETHER IT HOLDS BOTH.

`align` (wav2vec2 CTC) rides the one-wide heavy lane, `clip_rank` (SigLIP so400m) rides its own rank lane, so
the two kinds overlap BY DESIGN (main._dispatch_loop) — and each kind's service is cached by weights hash and
never released, so even strictly serial envelopes leave the previous kind's checkpoint resident. On a fleet
card both fit with ~10 GB to spare and this module changes nothing. On a small card the second kind dies:
measured here, clip_rank asked for 20 MiB with 167 MiB free while the process still held 2.89 GiB of align.

THE DECISION IS DERIVED, exactly as the clip_rank lane width already is (infer_cliprank.LANE_SIZING_WHY): the
per-kind residency is measured with nvidia-smi and matches the engine's own VRAM-budget residency figure for
figure — align 1500 MiB (1.2 GB fp32 checkpoint + a 20 s window's emissions + the CUDA context, seen at 1.44
GiB in the OOM report, = the engine's VRAM-budget align figure), clip_rank the fp16 towers plus ONE capped forward (the
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
    engine's own VRAM-budget concurrent-residency figure reads 4752 off its own rounding of the same measurements)."""
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


SERVED_INFER_KINDS_WHY = """
A POD SERVES THE INFER KINDS IT IS GIVEN, AND SIZES ITS CARD FOR THOSE ALONE.

The engine release smoke replays align / clip_rank outputs from a model-output cassette whenever a model's
passport is unchanged, so those kinds never reach the pod — yet the boot floor demanded the heaviest kind's
residency regardless, and a laptop whose desktop holds 2.5 GB of a 6 GB card refused (free=3236 < 3248) with
no model ever going to load. MONTY_INFER_KINDS (comma list; unset = every kind, the fleet default) is the set
this pod serves: the boot floor is the heaviest SERVED kind + reserve, none at all when no served kind holds
VRAM, and a request for any other kind is refused by name — never loaded lazily, never a silent fallback. An
unknown name refuses boot: a typo must not quietly shrink what the pod serves.
"""

SERVED_INFER_KINDS_ENV = "MONTY_INFER_KINDS"


class UnservedInferKind(RuntimeError):
    """An infer request for a kind this pod was not given (SERVED_INFER_KINDS_WHY)."""


def served_kinds(raw: str | None) -> frozenset[str]:
    """Parse MONTY_INFER_KINDS; None = every kind. Raises ValueError naming any unknown kind."""
    if raw is None:
        return frozenset(KIND_VRAM_MIB)
    kinds = [k.strip() for k in raw.split(",") if k.strip()]
    unknown = sorted(set(kinds) - set(KIND_VRAM_MIB))
    if unknown:
        raise ValueError(f"{SERVED_INFER_KINDS_ENV} names unknown infer kind(s) {', '.join(unknown)} "
                         f"(known: {', '.join(sorted(KIND_VRAM_MIB))})")
    return frozenset(kinds)


RENDER_HEADROOM_WHY = """
A CARD THAT PASSED THE FLOOR MUST STILL BE ABLE TO ENCODE.

The floor used to count only the heaviest infer kind + reserve (clip_rank 2736 + 512 = 3248 MiB), yet every full
pod also serves render ops — and camera.apply on a 6 GB card that cleared 3248 died mid-job on its NVENC open:
«CreateInputBuffer failed: out of memory». So a pod that serves render ops adds the render headroom: ONE 1080p
NVDEC+NVENC session, the engine's measured `_VRAM_PER_SESSION_MB` 960 MiB (the engine's
cut.apply op, NVENC_SIZING_WHY there; mirrored as gpu_admission.NVENC_SESSION_MIB, the unit the
pod's own heavy-op ledger reserves). One session is the largest working set any render op is GUARANTEED: the
heavies (cut.apply, camera.apply, media.normalize, media.cut_proxy) size their fan-out from what is free and run
one at a time under the whole-budget admission (gpu_admission.GPU_ADMISSION_WHY), so the floor that lets ONE of
them start is the one that matters — a wider fan-out is their own sizing, never a boot promise. The headroom is
ADDED to the infer kind's residency, because the heavy lane (align) and a render op are dispatched side by side.
Full floor with every kind: 2736 + 960 + 512 = 4208 MiB. A contour-dry pod renders nothing and keeps no floor.
"""

RENDER_HEADROOM_MIB = NVENC_SESSION_MIB


def vram_floor_mib(kinds: frozenset[str] | set[str], *, render: bool = False) -> float | None:
    """Free MiB a card must report to load the heaviest served kind (+ one NVENC session when the pod serves
    render ops, RENDER_HEADROOM_WHY) + reserve; None when nothing served holds VRAM."""
    heaviest = max((KIND_VRAM_MIB[k] for k in kinds), default=0.0)
    need = heaviest + (RENDER_HEADROOM_MIB if render else 0.0)
    return need + RESERVE_MIB if need > 0 else None


def refuse_unserved(kind: str, kinds: frozenset[str] | set[str]) -> None:
    """Raise before any fetch or load when `kind` is not one this pod serves."""
    if kind not in kinds:
        raise UnservedInferKind(
            f"infer kind {kind!r} is not served by this pod: {SERVED_INFER_KINDS_ENV}="
            f"{','.join(sorted(kinds)) or '(empty)'}")
