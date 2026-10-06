"""podagent/ops/dry.py — contour-dry stand-in handler, reached via `runner.py`'s dry-vs-pack seam when
`ARM_ENV` is armed. Mocks what is EXPENSIVE or EXTERNAL only; cheap read-only measurement ops resolve
straight through to the pack's own handler (see `_CLASSIFICATION`). media.tag is REAL here too: a `-c copy`
remux whose output IS the master stamp_deliverables re-checks, so a stub would overwrite a bt709-tagged
master with an untagged placeholder and get it refused off-contract."""
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

# Mirrored byte-for-byte by the engine's plan-match CONTOUR_DRY_CLAIMS — MISC-62 lock 4 refuses a receipt
# whose tuple has moved on only one side. A real media.tag (see `_CLASSIFICATION`) leaves every claim true:
# it re-muxes the master's own streams, so no pixel is rendered and no encoder runs.
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
    # a stub replaced the dry master with an untagged 64x64 black file; the real op is a stream copy.
    "media.tag": (REAL, "deliverable metadata remux — `-c copy`, no encode; keeps the master's colour tags"),
    "media.range_frames": (REAL, "Range-only frame reader — decode-only sampling, no full encode"),
    # a stubbed probe reports fake capability facts, defeating the whole op (TRK-108).
    "probe.ffmpeg_caps": (REAL, "ffmpeg encode/decode candidates + nvidia-smi read against the bound fixture"),
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
    ".m4a": ["-c:a", "aac", "-b:a", "96k"],
    ".aac": ["-c:a", "aac", "-b:a", "96k"],
}
_VIDEO_EXT_ARGS: dict[str, list[str]] = {
    ".mp4": ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k"],
    ".mov": ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k"],
}
# Raster stills the placeholder can honestly mux through ffmpeg (`ffmpeg -formats`/`-muxers` on this
# build: png/jpeg/webp all have a raster muxer; svg does not — confirmed empty `ffmpeg -muxers | grep svg`).
_RASTER_IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".webp"}
# Vector — no ffmpeg muxer exists for it, so its placeholder is written as literal markup, not piped
# through ffmpeg (see _write_image below).
_VECTOR_IMAGE_EXTS = {".svg"}
_IMAGE_EXTS = _RASTER_IMAGE_EXTS | _VECTOR_IMAGE_EXTS


DRY_AUDIO_WHY = """
A DRY FAKE IS NEVER SILENT: THE ENGINE'S MASTER CONTRACT REFUSES A FINAL WITHOUT SOUND.

the engine's master check refuses («OFF-CONTRACT … silent audio: loudness measured as -inf»)
any master whose audio stream measures no programme, or under SILENT_LUFS -40, or outside target ± TOL 3 LU.
anullsrc is bit-exact silence, so every dry master failed the REAL gate. The stand-in is a 997 Hz sine on both
channels at 48 kHz (check_master WANT_SAMPLE_RATE) with peak amplitude 10^(L/20): BS.1770 K-weighting is ~0 dB
at 997 Hz, so a stereo sine of peak L dBFS integrates to L LUFS (measured with ffmpeg loudnorm here: -14 → -14.1,
-20 → -20.1; true peak L + 1.2 dBTP after AAC) — the contract's loudness by construction, no measure pass.
"""

DRY_AUDIO_RATE = 48000
DRY_TONE_LUFS = -14.0   # the brand delivery target check_master.TARGET_LUFS reads (audio.master_lufs)


def dry_tone_lavfi(*, lufs: float = DRY_TONE_LUFS, dur: float | None = None) -> str:
    """The lavfi spec of the never-silent stand-in programme at `lufs` integrated (DRY_AUDIO_WHY)."""
    amp = 10 ** (float(lufs) / 20.0)
    wave = f"{amp:.6f}*sin(2*PI*997*t)"
    return f"aevalsrc={wave}|{wave}:s={DRY_AUDIO_RATE}" + (f":d={dur:.3f}" if dur is not None else "")


def _write_audio(dst: Path, *, op_name: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = _AUDIO_EXT_ARGS.get(dst.suffix.lower())
    if args is None:
        raise DryStubUnsupportedOutput(op_name, dst.suffix.lower(), "audio")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", dry_tone_lavfi(dur=1.0), "-t", "1", *args, str(dst)]
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
               "-f", "lavfi", "-i", dry_tone_lavfi(dur=1.0), "-t", "1", *args, str(dst)]
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
    ext = dst.suffix.lower()
    if ext in _VECTOR_IMAGE_EXTS:
        # No ffmpeg muxer exists for svg — write literal minimal markup instead of piping through ffmpeg.
        dst.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
            '<rect width="64" height="64" fill="#000000"/></svg>',
            encoding="utf-8")
        return
    if ext not in _RASTER_IMAGE_EXTS:
        raise DryStubUnsupportedOutput(op_name, ext, "image")
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
    # Raster only: the hstack composite below is an ffmpeg filter_complex, which has no vector (svg) target.
    if dst.suffix.lower() not in _RASTER_IMAGE_EXTS:
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
# (MISC-62: cut.apply's placeholder durs.rdurs was exactly that guess — the engine's EDL step).
class DryStubUnderivedField(RuntimeError):
    def __init__(self, op_name: str, field: str, *, why: str) -> None:
        self.op_name, self.field = op_name, field
        super().__init__(
            f"contour-dry: {op_name!r} stub cannot derive JSON field {field!r} from its params/inputs "
            f"({why}) — never guessed")


def _synth_cut_apply_durs(params: dict[str, Any], _inputs: dict[str, Path]) -> dict[str, Any]:
    # rdurs read by the engine's EDL step and its project loader (needs len == len(keep)).
    speed = float(params.get("speed", 1.0)) or 1.0
    return {"rdurs": [(float(e) - float(s)) / speed for s, e in params["keep"]]}


def _synth_media_sheet_meta(params: dict[str, Any], inputs: dict[str, Path]) -> dict[str, Any]:
    # {cells,drawn,width,height} read by the engine's b-roll resolver; width/height replay the pure
    # canvas_size() of the engine's media.sheet op; drawn mirrors its own input-presence check.
    n = len(params.get("captions") or [])
    cols = max(1, min(int(params["cols"]), n)) if n else 1
    rows = (n + cols - 1) // cols if n else 1
    gap, head = int(params["gap"]), int(params["head"])
    cellw = int(params["cell_w"]) + gap
    cellh = int(params["cell_h"]) + int(params["caption_h"]) + gap
    drawn = sorted(i for i in range(n) if inputs.get(f"tile{i}") is not None)
    return {"cells": n, "drawn": drawn, "width": gap + cols * cellw, "height": head + rows * cellh}


def _synth_media_image_tile_meta(params: dict[str, Any], _inputs: dict[str, Path]) -> dict[str, Any]:
    # Same shape read by the engine's b-roll resolver; fused sheet call is cols=len(urls) one row
    # (the engine's media.image_tile op). `drawn` reports every requested cell as drawn — no GET
    # runs under a stub, but the engine's b-roll resolver drops a candidate whose cell is missing from `drawn`
    # ("preview did not render"), so an honest empty list makes every photo-lane candidate under dry fail at
    # fetch_broll, not just skip the fetch, whenever the LLM picked asset auto/photo (TRK-90/MISC-189).
    n = len(params["urls"])
    return {"cells": n, "drawn": list(range(n)), "width": int(params["width"]) * n, "height": int(params["height"])}


def _synth_range_filmstrip_receipt(params: dict[str, Any], _inputs: dict[str, Path]) -> dict[str, Any]:
    # Shape read back by the engine's b-roll fetcher and validated by the op pack's range transport receipt
    # (+ run_ledger.RANK_TRANSPORT_FIELDS' strict fold). The strip DID land, so the receipt is green: a "failed"
    # receipt with outputs_present=1 made every dry b-roll candidate a transport failure the real gate counts.
    # Nothing was read from any origin (origin/proven bytes 0, no requests — CONTOUR_DRY_CLAIMS); the dry
    # origin object is the plan's own declared ceiling, the only object size the plan names, so the green
    # receipt's strict inequality origin+proven < object holds without inventing a transfer.
    cap = max(1, int(params.get("max_origin_bytes") or 0))
    return {
        "schema_version": 1, "status": "ok", "reason": "", "origin_class": "transient",
        "object_bytes": cap, "origin_bytes": 0, "proven_bytes": 0, "byte_cap": cap,
        "range_requests": 0, "whole_attempts": 0, "whole_reads": 0, "ignored_range_responses": 0,
        "cap_exceeded": False, "outputs_expected": 1, "outputs_present": 1,
    }


_JSON_SYNTH: dict[str, Callable[[dict[str, Any], dict[str, Path]], dict[str, Any]]] = {
    "cut.apply": _synth_cut_apply_durs,
    "media.range_filmstrip": _synth_range_filmstrip_receipt,
    "media.sheet": _synth_media_sheet_meta,
    "media.image_tile": _synth_media_image_tile_meta,
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


# Mirrors runner.py::_ext's own precedence (runner.py:381-395, "Prefer the extension the binding's
# destination already implies … fall back to the port kind's default"): `_bind_outputs` in runner.py
# builds `dst` for BOTH the real and the dry seam, so by the time it reaches here the bound extension has
# already won over `port.kind` once, at path-naming time (media.fetch declares `dst` as kind `video` but
# is also used for stills, so a binding can hand this a `.jpg`/`.png`/`.svg` path). The stub must pick its
# placeholder WRITER off that same already-resolved extension, not re-derive a `.mp4` path off `kind` and
# refuse it — `kind` is only the fallback, for an extension this tier does not itself recognise.
def _resolve_media_class(kind: str, ext: str) -> str:
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXT_ARGS:
        return "video"
    if ext in _AUDIO_EXT_ARGS:
        return "audio"
    return kind


def _fill_one(dst: Path, kind: str, *, op_name: str, params: dict[str, Any], inputs: dict[str, Path],
              audio_src: Path | None = None) -> None:
    if kind == "json":
        _write_json(dst, op_name=op_name, params=params, inputs=inputs)
        return
    resolved = _resolve_media_class(kind, dst.suffix.lower())
    if resolved == "video":
        _write_video(dst, op_name=op_name, audio_src=audio_src)
        return
    if resolved == "image" and (image_synth := _IMAGE_SYNTH.get(op_name)) is not None:
        image_synth(dst, op_name=op_name, params=params)
        return
    fn = _WRITER.get(resolved)
    if fn is None:
        raise registry.OpError(f"contour-dry: no synthesis rule for output kind {kind!r}")
    fn(dst, op_name=op_name)


DRY_STILL_WHY = """
media.still's sidecar is a MEASUREMENT of pixels, so the dry stand-in measures pixels — the placeholder ones.

The engine reads `dark` (its mark-backing treatment, glass vs shadow), `bbox`/`mark_w`/`mark_h`
(the mark rect), `width`/`height` (its photo fetcher), `plated`/`finished`/`rasterizer`
(the engine's photo fetcher) — the old stub refused with DryStubUnderivedField, so the dry photo and
logo lanes died at the first still. Here the bound `src` (the media.fetch placeholder) is read with the REAL
op's own rules, vendored verbatim from the engine's media.still op (is_dark_image: alpha-
weighted mean luminance of a 48x48 RGBA resize < 110; has_alpha: mode carries alpha and min alpha of a 64x64
resize < 250; mark_probe: alpha bbox at >= 24 as canvas fractions; probe: 0x0 for a vector); a vector is
"rasterised" to a solid placeholder at `width` (rasterizer `contour-dry-stub`), and the plate is a solid
`width`-square canvas in the tone the same luminance rule picks. The real op's plate condition decides whether
`dst` is written at all (vector, plate=always, or real transparency), exactly as `run` does.
"""

_PLATE_DARK, _PLATE_LIGHT = "0x111111", "0xf2f2f2"   # media_still._PLATE_DARK / _PLATE_LIGHT
_DRY_RASTERIZER = "contour-dry-stub"


def _still_is_vector(p: Path) -> bool:
    try:
        head = p.read_bytes()[:256].lstrip()
    except OSError:
        return False
    return p.suffix.lower() == ".svg" or head.startswith(b"<?xml") or head.startswith(b"<svg")


def _still_is_dark(path: Path) -> bool:
    from PIL import Image
    px: Any = Image.open(path).convert("RGBA").resize((48, 48)).load()
    tot, n = 0.0, 0
    for y in range(48):
        for x in range(48):
            r, g, b, a = px[x, y]
            if a < 24:
                continue
            tot += 0.299 * r + 0.587 * g + 0.114 * b
            n += 1
    return bool(n) and (tot / n) < 110


def _still_has_alpha(path: Path) -> bool:
    from PIL import Image
    im = Image.open(path)
    if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
        return False
    return bool(im.convert("RGBA").resize((64, 64)).getchannel("A").getextrema()[0] < 250)


def _still_mark(path: Path) -> dict[str, Any]:
    from PIL import Image
    out: dict[str, Any] = {"dark": _still_is_dark(path), "bbox": [0.0, 0.0, 1.0, 1.0], "mark_w": 0, "mark_h": 0}
    im = Image.open(path)
    w, h = im.width, im.height
    out["mark_w"], out["mark_h"] = int(w), int(h)
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        bb = im.convert("RGBA").getchannel("A").point(lambda a: 255 if a >= 24 else 0).getbbox()
        if bb and w and h:
            out["bbox"] = [bb[0] / w, bb[1] / h, bb[2] / w, bb[3] / h]
    return out


def _solid_png(dst: Path, colour: str, w: int, h: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() not in _RASTER_IMAGE_EXTS:
        raise DryStubUnsupportedOutput("media.still", dst.suffix.lower(), "image")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"color=c={colour}:s={w}x{h}", "-frames:v", "1", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


def _run_media_still(*, params: dict[str, Any], inputs: dict[str, Path], outputs: dict[str, Any]) -> None:
    """media.still under the dry tier (DRY_STILL_WHY): same sidecar shape and plate condition as the real
    `montyops.media_still.run`, measured off the bound placeholder's pixels."""
    from PIL import Image
    raw_src = inputs.get("src")
    src = Path(raw_src) if raw_src is not None else None
    if src is None or not src.exists() or src.stat().st_size == 0:
        raise registry.OpError(f"contour-dry: 'media.still' input src {raw_src!r} is empty — nothing to finish")
    width = int(params.get("width") or 1080)
    plate = str(params.get("plate") or "always")
    if plate not in ("always", "if_transparent", "none"):
        raise registry.OpError(f"contour-dry: 'media.still' unknown plate mode {plate!r}")
    vector = _still_is_vector(src)
    if vector:
        w, h = 0, 0
    else:
        with Image.open(src) as im:
            w, h = int(im.width), int(im.height)
    meta: dict[str, Any] = {"width": w, "height": h, "vector": vector,
                            "alpha": bool(not vector and _still_has_alpha(src)),
                            "plated": False, "finished": False, "rasterizer": None, "width_requested": width}
    dst = outputs.get("dst")
    if dst is not None and (vector or plate == "always" or meta["alpha"]):
        dst = Path(dst)
        with tempfile.TemporaryDirectory(prefix="dry-still-") as td:
            raw = Path(td) / "raw.png"
            if vector:
                _solid_png(raw, "0x000000", width, width)   # the placeholder svg's own fill, rasterised
            else:
                Image.open(src).save(raw)
            mark = _still_mark(raw)
            if plate != "none":
                _solid_png(dst, _PLATE_LIGHT if mark["dark"] else _PLATE_DARK, width, width)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.open(raw).save(dst)
        meta.update(plated=plate != "none", finished=True, rasterizer=_DRY_RASTERIZER if vector else None)
        meta.update(mark)
    meta_p = Path(outputs["meta"])
    meta_p.parent.mkdir(parents=True, exist_ok=True)
    meta_p.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")


# Ops whose dry stand-in is a whole-op function, because the real op's sidecar decides which outputs exist.
_OP_SYNTH: dict[str, Callable[..., None]] = {"media.still": _run_media_still}


def _handler(op: registry.Op) -> Callable[..., None]:
    if (whole := _OP_SYNTH.get(op.op)) is not None:
        return whole
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
