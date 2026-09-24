"""podagent/ops/dry.py — contour-dry stand-in handler, reached via `runner.py`'s dry-vs-pack seam when
`ARM_ENV` is armed. Mocks what is EXPENSIVE or EXTERNAL only; cheap read-only measurement ops resolve
straight through to the pack's own handler (see `_CLASSIFICATION`)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from . import pack, registry

ARM_ENV = "MONTY_OPS_CONTOUR_DRY"

# Mirrored byte-for-byte at scripts/plan_match.py::CONTOUR_DRY_CLAIMS — MISC-62 lock 4 refuses a receipt
# whose tuple has moved on only one side.
CONTOUR_DRY_CLAIMS: dict[str, str] = {
    "taps": "plan-derived, not measured", "pixels": "not rendered", "vram": "not exercised",
    "nvenc": "not exercised", "weights": "cache presence only", "graph": "really built",
    "argv": "really built", "store": "real PUT/GET", "measure": "real on the input",
}

# A stand-in for real work never legitimately runs longer than the smallest thing that could stall it.
_LAVFI_BUDGET_S = 30.0

REAL = "real"   # runs the pack's own handler on the real bound input
STUB = "stub"   # runs the arity-correct placeholder; no real work happens

# Explicit name roster: no contract field (`needs`, `budget`) separates cheap-read-only from heavy/network.
_CLASSIFICATION: dict[str, tuple[str, str]] = {
    "measure.audio": (REAL, "ffmpeg astats envelope + levels over the bound source — read-only"),
    "measure.boundary": (REAL, "numeric join heuristics (RMS/click/framediff) — ffmpeg/ffprobe reads only"),
    "measure.envelope": (REAL, "per-frame RMS-in-dB level contour — ffmpeg astats reads only"),
    "measure.framediff": (REAL, "framediff over tiny per-boundary windows — ffmpeg decode, no encode"),
    "measure.master": (REAL, "loudnorm analysis (loudness/true-peak/LRA) + ffprobe colour/bitrate"),
    "measure.silence": (REAL, "silencedetect spans + RMS envelope — ffmpeg reads only"),
    "measure.source": (REAL, "ffprobe-class ingest numbers (dims, rotation, codec, pix_fmt, ...)"),
    "media.range_frames": (REAL, "Range-only frame reader — decode-only sampling, no full encode"),
    # cassette replays this mp3 against the audio-LLM, so the bytes must be real, not synthetic.
    "cut.audio": (REAL, "per-segment trim/fade/atempo AUDIO-ONLY encode — CPU ffmpeg, no video_encode argv"),
    "media.audio": (REAL, "full-file audio demux to mp3 — CPU ffmpeg, no video_encode argv"),
    # frames/sample_rate/channels are the EXACT decode count (no param carries source duration) — a stub
    # can only guess or refuse, so this pays one real ffmpeg pass on the bound input, like media.audio.
    "media.pcm": (REAL, "full-file audio decode to PCM wav — CPU ffmpeg, no video_encode argv"),

    "camera.apply": (STUB, "GPU (libplacebo) crop-trajectory render to pixels"),
    "cut.apply": (STUB, "per-segment trim/atempo render + concat + crossfade encode"),
    "edit.splice": (STUB, "trim+concat re-encode"),
    "edit.weld": (STUB, "film-burn transition composite + encode"),
    "media.cut_proxy": (STUB, "proxy encode"),
    "media.filmstrip": (STUB, "frame sampling + hstack composite image encode"),
    "media.frames": (STUB, "per-fraction frame extraction, image encode"),
    "media.image_scale": (STUB, "image normalize + re-encode"),
    "media.normalize": (STUB, "canonical ingest encode (GPU_ADMISSION.HEAVY_GPU_OPS)"),
    "media.scale": (STUB, "video downscale + re-encode"),
    "media.sheet": (STUB, "contact-sheet composite + image encode"),
    "media.still": (STUB, "vector rasterise + contain composite + image encode"),
    "media.tag": (STUB, "container remux, not a measurement"),
    "mograph.render": (STUB, "headless-Chrome Remotion render (browser-heavy)"),
    "motion.kenburns": (STUB, "Ken-Burns bake, video encode"),
    "opener.build": (STUB, "cold-open assembly + composite + encode (browser-class)"),

    "media.fetch": (STUB, "origin GET — external network"),
    "media.image_filmstrip": (STUB, "public still-image origin GET + filmstrip render"),
    "media.image_tile": (STUB, "public still-image origin(s) GET + tile composite"),
    "media.range_filmstrip": (STUB, "public clip origin Range GETs + filmstrip render — external network"),
}


def _classify(op_name: str) -> tuple[str, str]:
    row = _CLASSIFICATION.get(op_name)
    if row is None:
        raise registry.OpError(
            f"contour-dry: {op_name!r} has no dry-tier classification — add a REAL/STUB row with a reason "
            f"to podagent/ops/dry.py::_CLASSIFICATION before this op can run under {ARM_ENV}")
    return row


def armed() -> bool:
    return os.environ.get(ARM_ENV, "").strip() not in ("", "0")


class DryStubUnsupportedOutput(RuntimeError):
    """A STUB port's declared output path has an extension the placeholder writer has no codec/container
    mapping for — refused by name here, before ffmpeg gets a mismatched target and raises its own opaque
    CalledProcessError two layers down."""

    def __init__(self, op_name: str, ext: str, kind: str) -> None:
        super().__init__(
            f"contour-dry: {op_name!r} declares a {kind!r} output with extension {ext!r}, which has no "
            f"placeholder codec/container mapping in podagent/ops/dry.py — add one or use a supported "
            f"extension")


# Extension -> ffmpeg codec/container args a 1s lavfi source can honestly be muxed into.
_AUDIO_EXT_ARGS: dict[str, list[str]] = {
    ".mp3": ["-c:a", "libmp3lame", "-q:a", "5"],
    ".wav": ["-c:a", "pcm_s16le"],
    ".m4a": ["-c:a", "aac", "-b:a", "8k"],
    ".aac": ["-c:a", "aac", "-b:a", "8k"],
}
_VIDEO_EXT_ARGS: dict[str, list[str]] = {
    ".mp4": ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "8k"],
    ".mov": ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "8k"],
}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _write_audio(dst: Path, *, op_name: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = _AUDIO_EXT_ARGS.get(dst.suffix.lower())
    if args is None:
        raise DryStubUnsupportedOutput(op_name, dst.suffix.lower(), "audio")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=1", "-t", "1", *args, str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


def _has_audio_stream(path: Path) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, timeout=_LAVFI_BUDGET_S)
    return bool(proc.stdout.strip())


# Only a VIDEO-kind bound input is a candidate — the one file whose real audio a placeholder must reflect.
def _content_input(op: registry.Op, inputs: dict[str, Path]) -> Path | None:
    for port in op.inputs:
        if port.kind != "video":
            continue
        raw = inputs.get(port.id)
        if raw is None:
            continue
        path = Path(raw)
        if path.exists():
            return path
    return None


def _write_video(dst: Path, *, op_name: str, audio_src: Path | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() not in _VIDEO_EXT_ARGS:
        raise DryStubUnsupportedOutput(op_name, dst.suffix.lower(), "video")
    if audio_src is None:
        # No video-kind input bound at all — nothing real to reflect, stays fully synthetic.
        args = _VIDEO_EXT_ARGS[dst.suffix.lower()]
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=1",
               "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=1", "-t", "1", *args, str(dst)]
    elif _has_audio_stream(audio_src):
        # `-shortest` against an infinite colour source sizes the output to the real audio's own duration.
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(audio_src),
               "-f", "lavfi", "-i", "color=c=black:s=64x64",
               "-map", "1:v", "-map", "0:a",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-shortest", str(dst)]
    else:
        # Real input, genuinely no audio track — the placeholder gets none, never a fabricated anullsrc.
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=1",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


def _write_image(dst: Path, *, op_name: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() not in _IMAGE_EXTS:
        raise DryStubUnsupportedOutput(op_name, dst.suffix.lower(), "image")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=64x64", "-frames:v", "1", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


def _cell_colour(url: str, index: int) -> str:
    digest = hashlib.sha256(f"{url}#{index}".encode("utf-8")).digest()
    return f"0x{digest[0]:02x}{digest[1]:02x}{digest[2]:02x}"


# The origin url IS the candidate's identity here; nothing else in params distinguishes one clip's strip
# from another's, and two candidates sharing one placeholder would hide a mis-addressed tile downstream.
def _write_filmstrip(dst: Path, *, op_name: str, params: dict[str, Any]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() not in _IMAGE_EXTS:
        raise DryStubUnsupportedOutput(op_name, dst.suffix.lower(), "image")
    positions = params.get("positions")
    if not isinstance(positions, list) or not positions:
        raise registry.OpError(
            f"contour-dry: {op_name!r} stub cannot lay out a strip without a non-empty `positions` list")
    url = str(params.get("url") or "")
    cell_w, cell_h = int(params["width"]), int(params["height"])
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for i in range(len(positions)):
        cmd += ["-f", "lavfi", "-i", f"color=c={_cell_colour(url, i)}:s={cell_w}x{cell_h}"]
    if len(positions) > 1:
        chain = "".join(f"[{i}:v]" for i in range(len(positions)))
        cmd += ["-filter_complex", f"{chain}hstack=inputs={len(positions)}"]
    cmd += ["-frames:v", "1", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


_IMAGE_SYNTH: dict[str, Callable[..., None]] = {"media.range_filmstrip": _write_filmstrip}


# A field neither params nor bound inputs can honestly produce is refused BY NAME, never guessed
# (MISC-62: cut.apply's placeholder durs.rdurs was exactly that guess — scripts/apply_edl.py:306).
class DryStubUnderivedField(RuntimeError):
    def __init__(self, op_name: str, field: str, *, why: str) -> None:
        self.op_name, self.field = op_name, field
        super().__init__(
            f"contour-dry: {op_name!r} stub cannot derive JSON field {field!r} from its params/inputs "
            f"({why}) — never guessed")


def _synth_cut_apply_durs(params: dict[str, Any], _inputs: dict[str, Path]) -> dict[str, Any]:
    # rdurs read at scripts/apply_edl.py:306/260 and scripts/project.py:66 (needs len == len(keep)).
    speed = float(params.get("speed", 1.0)) or 1.0
    return {"rdurs": [(float(e) - float(s)) / speed for s, e in params["keep"]]}


def _synth_media_sheet_meta(params: dict[str, Any], inputs: dict[str, Path]) -> dict[str, Any]:
    # {cells,drawn,width,height} read at scripts/broll_resolve.py:3105-3114; width/height replay the pure
    # canvas_size() at scripts/montyops/media_sheet.py:150-158; drawn mirrors its own input-presence check.
    n = len(params.get("captions") or [])
    cols = max(1, min(int(params["cols"]), n)) if n else 1
    rows = (n + cols - 1) // cols if n else 1
    gap, head = int(params["gap"]), int(params["head"])
    cellw = int(params["cell_w"]) + gap
    cellh = int(params["cell_h"]) + int(params["caption_h"]) + gap
    drawn = sorted(i for i in range(n) if inputs.get(f"tile{i}") is not None)
    return {"cells": n, "drawn": drawn, "width": gap + cols * cellw, "height": head + rows * cellh}


def _synth_media_image_tile_meta(params: dict[str, Any], _inputs: dict[str, Path]) -> dict[str, Any]:
    # Same shape read at scripts/broll_resolve.py:3104-3114; fused sheet call is cols=len(urls) one row
    # (scripts/montyops/media_image_tile.py:70-73). `drawn` reports every requested cell as drawn — no GET
    # runs under a stub, but broll_resolve.py:2381-2386 drops a candidate whose cell is missing from `drawn`
    # ("preview did not render"), so an honest empty list makes every photo-lane candidate under dry fail at
    # fetch_broll, not just skip the fetch, whenever the LLM picked asset auto/photo (TRK-90/MISC-189).
    n = len(params["urls"])
    return {"cells": n, "drawn": list(range(n)), "width": int(params["width"]) * n, "height": int(params["height"])}


def _synth_media_still_meta(_params: dict[str, Any], _inputs: dict[str, Path]) -> dict[str, Any]:
    # dark/bbox/mark_w/mark_h/width/height/finished/plated/rasterizer read at scripts/broll_resolve.py:2231
    # and scripts/fetch_photo.py:1633-1931; all come off probe(src)'s pixel/host reality, none off params.
    raise DryStubUnderivedField("media.still", "dark", why="pixel/alpha/host facts, not a param function")


def _synth_range_filmstrip_receipt(params: dict[str, Any], _inputs: dict[str, Path]) -> dict[str, Any]:
    # Shape read back by scripts/fetch_broll.py:1537; `object_bytes` is the ORIGIN object's size, which only
    # the GET this stub abolishes could know — so this is empty_receipt()'s shape, never a fabricated green.
    return {
        "schema_version": 1, "status": "failed", "reason": "", "origin_class": "transient",
        "object_bytes": None, "origin_bytes": 0, "proven_bytes": 0,
        "byte_cap": max(0, int(params.get("max_origin_bytes") or 0)),
        "range_requests": 0, "whole_attempts": 0, "whole_reads": 0, "ignored_range_responses": 0,
        "cap_exceeded": False, "outputs_expected": 1, "outputs_present": 1,
    }


_JSON_SYNTH: dict[str, Callable[[dict[str, Any], dict[str, Path]], dict[str, Any]]] = {
    "cut.apply": _synth_cut_apply_durs,
    "media.range_filmstrip": _synth_range_filmstrip_receipt,
    "media.sheet": _synth_media_sheet_meta,
    "media.image_tile": _synth_media_image_tile_meta,
    "media.still": _synth_media_still_meta,
}


def _write_json(dst: Path, *, op_name: str, params: dict[str, Any], inputs: dict[str, Path]) -> None:
    synth = _JSON_SYNTH.get(op_name)
    if synth is None:
        raise registry.OpError(
            f"contour-dry: {op_name!r} declares a JSON output with no plan-derivation rule in "
            f"podagent/ops/dry.py::_JSON_SYNTH — a placeholder JSON is refused, add a synth function")
    doc = synth(params, inputs)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")


_WRITER: dict[str, Callable[..., None]] = {
    "audio": _write_audio,
    "image": _write_image,
}


def _fill_one(dst: Path, kind: str, *, op_name: str, params: dict[str, Any], inputs: dict[str, Path],
              audio_src: Path | None = None) -> None:
    if kind == "json":
        _write_json(dst, op_name=op_name, params=params, inputs=inputs)
        return
    if kind == "video":
        _write_video(dst, op_name=op_name, audio_src=audio_src)
        return
    if kind == "image" and (image_synth := _IMAGE_SYNTH.get(op_name)) is not None:
        image_synth(dst, op_name=op_name, params=params)
        return
    fn = _WRITER.get(kind)
    if fn is None:
        raise registry.OpError(f"contour-dry: no synthesis rule for output kind {kind!r}")
    fn(dst, op_name=op_name)


def _handler(op: registry.Op) -> Callable[..., None]:
    def run(*, params: dict[str, Any], inputs: dict[str, Path], outputs: dict[str, Any]) -> None:
        declared = {p.id: p for p in op.outputs}
        audio_src = _content_input(op, inputs)
        for port_id, dst in outputs.items():
            port = declared[port_id]
            targets = dst if isinstance(dst, list) else [dst]
            for one in targets:
                _fill_one(Path(one), port.kind, op_name=op.op, params=params, inputs=inputs,
                          audio_src=audio_src)
    return run


def resolve(op: registry.Op) -> Callable[..., None]:
    mode, _reason = _classify(op.op)
    if mode == REAL:
        # `runner.py` activates the ops pack before it ever picks this seam, dry or not — safe to call.
        return pack.resolve(op.handler)
    return _handler(op)


def _clip_rank_group_verdict(n: int) -> tuple[list[float], list[None]]:
    """Strictly descending scores over ALL candidates: real SigLIP on identical STUB placeholder pixels
    would score every candidate near-equally, clearing no relevance floor and shortlisting nothing."""
    return [round(max(0.05, 0.95 - 0.05 * i), 4) for i in range(n)], [None] * n


def run_clip_rank(params: Any, put_url: str, progress: Callable[[str], None] | None = None) -> SimpleNamespace:
    """The dry-tier stand-in for `infer_cliprank.ClipRankService.run` — same `(infer_s, timings)` shape,
    no weights loaded, no GPU touched, and the PUT contract kept so the resolver reads it identically."""
    from .. import cp
    from ..models import ClipRankGroupResult, ClipRankPayload

    t0 = time.monotonic()
    groups = []
    for g in params.groups:
        scores, embeds = _clip_rank_group_verdict(len(g.image_urls))
        groups.append(ClipRankGroupResult(scores=scores, embeds=embeds))
    payload = ClipRankPayload(model="contour-dry-stub", groups=groups)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip_rank.json"
        out.write_text(payload.model_dump_json())
        cp.upload(out, put_url, "application/json")
    infer_s = time.monotonic() - t0
    if progress is not None:
        progress(f"contour-dry clip_rank stub: {len(groups)} group(s), deterministic descending scores")
    return SimpleNamespace(infer_s=infer_s, timings={"infer_s": round(infer_s, 3), "work_s": round(infer_s, 3)})
