"""Which infer kinds may hold this pod's card at once (INFER_KIND_RESIDENCY_WHY)."""
from __future__ import annotations

import gc
import sys
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

HEAVY_LANE_MEASURED_WHY = """
THE HEAVY LANE NEXT TO clip_rank, MEASURED ON THE CARD THAT OOM'd (056 v13, 2026-10-22).

The pod's own classes in the pod image (09adca1, torch 2.11+cu128) on the dev RTX 2060, card-wide nvidia-smi
sampled every 20 ms (the number an OOM is about); the bench script and its JSON results are kept engine-side:
  align (wav2vec2 xlsr-56)   weights resident 1244 MiB · one 20 s window forward +518 MiB over that
  clip_rank (SigLIP so400m)  2322 MiB weights + context · 12-tile lanes at once: 1 → +354, 2 → +726, 3 → +864,
                              4 → +1040 MiB (≤ 414 per lane: the per-lane figure holds)
  person_matte (RVM 1080x1920, ds 0.25) first-frame autotune +958..1010 MiB over everything already resident
                              (+1094 over an idle card, the figure below), 16 MiB weights, 228 between videos
  align + SigLIP + 2 lanes running together: 4310 MiB over the idle card.
The old floor said align = 1500 (weights + forward) and sized the clip_rank lanes against the WHOLE free card, as
if align were never there: 4835 free → 4 lanes, + align 1762 → over the card, «Tried to allocate 72 MiB».
Every kind's service stays cached, so on a card where the kinds coexist align's weights are resident while
person_matte (same one-wide lane) autotunes: the heavy lane holds align_resident + max(align forward + RVM
resident, RVM peak). That is subtracted from the clip_rank lane budget, and it is the second term of the
coexist floor. A card under that floor serialises the kinds with the idle weights dropped (never the CPU).
"""
ALIGN_RESIDENT_MIB = 1244.0     # measured (HEAVY_LANE_MEASURED_WHY)
ALIGN_FORWARD_MIB = 518.0       # one 20 s window (the engine aligns in 20 s chunks)
PERSON_MATTE_PEAK_MIB = 1100.0  # +1094 over an idle card, first-frame cudnn autotune
PERSON_MATTE_RESIDENT_MIB = 228.0

KIND_VRAM_MIB: Mapping[str, float] = {      # measured; see INFER_KIND_RESIDENCY_WHY for each number
    "align": ALIGN_RESIDENT_MIB + ALIGN_FORWARD_MIB,
    "clip_rank": _VRAM_WEIGHTS_MB + _VRAM_PER_LANE_MB,
    "face_probe": 0.0,
    # RVM fp16 at 1080x1920: nvidia-smi peak +1094 MiB over the idle card during the first frame's cudnn autotune
    # (torch max reserved 990), +228 MiB resident between videos — sized at the peak, the moment it must fit.
    # It rides the one-wide heavy lane, so it never runs beside align; it CAN run beside a clip_rank lane.
    "person_matte": PERSON_MATTE_PEAK_MIB,
}
RESERVE_MIB = _VRAM_RESERVE_MB


def heavy_lane_mib(served: "frozenset[str] | set[str] | None" = None) -> float:
    """The peak the one-wide heavy lane can hold beside a clip_rank lane (HEAVY_LANE_MEASURED_WHY): align's weights
    stay resident once loaded, and the lane's transient is align's forward (+ RVM's cached weights) or RVM's
    autotune peak, whichever is larger. `served` None = every kind."""
    kinds = set(KIND_VRAM_MIB) if served is None else set(served)
    align = "align" in kinds
    matte = "person_matte" in kinds
    resident = ALIGN_RESIDENT_MIB if align else 0.0
    transient = max((ALIGN_FORWARD_MIB + (PERSON_MATTE_RESIDENT_MIB if matte else 0.0)) if align else 0.0,
                    PERSON_MATTE_PEAK_MIB if matte else 0.0)
    return resident + transient


def coresident_mib(served: "frozenset[str] | set[str] | None" = None) -> float:
    """Free MiB a card must report for clip_rank (weights + one lane) to run beside the heavy lane:
    2736 + heavy_lane_mib + 512 — 5592 with every kind served (2736 + 1244 + max(518 + 228, 1100) + 512)."""
    return KIND_VRAM_MIB["clip_rank"] + heavy_lane_mib(served) + RESERVE_MIB


def kinds_fit_together(free_mib: float | None, served: "frozenset[str] | set[str] | None" = None
                       ) -> tuple[bool, str]:
    """May align and clip_rank be resident at once on a card reporting `free_mib`, and why. Pure — the reading
    is injected."""
    need = coresident_mib(served)
    if free_mib is None:
        return False, ("the card reports no free VRAM — one infer kind resident at a time, still the GPU, "
                       "never the CPU")
    if free_mib >= need:
        return True, (f"{free_mib:.0f} MiB free >= {need:.0f} MiB for clip_rank+heavy lane+reserve — both "
                      f"kinds may hold the card, the lanes stay parallel")
    return False, (f"{free_mib:.0f} MiB free < {need:.0f} MiB for clip_rank+heavy lane+reserve — the kinds "
                   f"serialise on one lane and the idle kind's weights are dropped")


def card_holds_both_kinds(probe: Callable[[], float | None] | None = None,
                          served: "frozenset[str] | set[str] | None" = None) -> tuple[bool, str]:
    """The same answer against the LIVE card. A probe failure is a reading, not a raise (this is a budget)."""
    try:
        free = (probe or _free_vram_mb)()
    except (OSError, ValueError) as e:
        return False, f"could not ask the card for free VRAM ({type(e).__name__}: {e}) — one kind at a time"
    return kinds_fit_together(free, served)


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


CARD_TURNS_WHY = """
ON A CARD WHERE THE INFER KINDS TAKE TURNS, THE ENCODE IS ONE OF THE TURNS.

A standing pod (decision `the-laptop-holds-a-standing-pod-like-the-fleet`) keeps whatever kind it served last
on the card for its whole life. On a card too small for two kinds (INFER_KIND_RESIDENCY_WHY) that resident
kind sat under every encode: 056 v13 r8 (dev RTX 2060, 6144 MiB, ~1.5-2.5 GiB held by the desktop) had SigLIP
resident (2322 MiB) through cut.apply, whose ledger grant was 1388-1960 MiB, so 1 NVENC session (pm2 pod log
«1768 MiB granted by the pod's admission ledger → 1 session(s)»); apply_cut took 99 s. The ledger also read
its budget ONCE for the pod's life, so what the card held at the first heavy op sized every later one.

So, on such a card only, ONE ledger serves both tenants: a weight-holding infer (align, clip_rank,
person_matte) books the whole budget around its load + forward exactly as a self-sizing heavy op does
(gpu_admission.admission), so the two never run at once and keep FIFO order with its bounded, visible park. A
heavy op, once admitted, drops every cached infer service (all idle by construction — no infer can be inside
the ledger beside it), returns the allocator's blocks to the driver (a separate ffmpeg process cannot use
blocks torch merely caches), and re-reads the budget from the card (gpu_admission.remeasure_held). The dropped
kind reloads from the pod's disk cache when next asked (SigLIP measured 6.5 s, r8 run.log). face_probe holds no
card at all and takes no turn. On a fleet card the kinds coexist, this is never armed, and nothing changes.
"""

_TURNS: dict[str, Any] = {"armed": False, "caches": {}}


def arm_card_turns(armed: bool, caches: Mapping[str, dict[Any, Any]]) -> None:
    """Boot: arm CARD_TURNS_WHY when the kinds take turns on this card, and name the caches a heavy op may drop."""
    _TURNS["armed"] = bool(armed)
    _TURNS["caches"] = caches


def card_turns_armed() -> bool:
    return bool(_TURNS["armed"])


def takes_card_turn(kind: str) -> bool:
    """Does an infer of `kind` take the card's turn? Only a weight-holding kind, only on a turn-taking card."""
    return card_turns_armed() and KIND_VRAM_MIB.get(kind, 0.0) > 0.0


def clear_card_for_render() -> list[str]:
    """Called by an ADMITTED heavy op (CARD_TURNS_WHY): drop every cached infer service, hand the blocks back to
    the driver, re-read the ledger budget. Returns the dropped kinds; [] (and nothing touched) when not armed."""
    if not card_turns_armed():
        return []
    from .ops import gpu_admission

    dropped = release_other_kinds("", _TURNS["caches"])
    torch = sys.modules.get("torch")      # never import torch on a pod that has not loaded a model
    if dropped and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gpu_admission.remeasure_held()
    return dropped


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
