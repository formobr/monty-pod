"""podagent/ops/dry.py — contour-dry stand-in handler, reached via `runner.py`'s dry-vs-pack seam when
`ARM_ENV` is armed. Mocks what is EXPENSIVE or EXTERNAL only; cheap read-only measurement ops resolve
straight through to the pack's own handler (see `_CLASSIFICATION`)."""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
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
    "media.range_filmstrip": (REAL, "Range-only filmstrip reader — decode-only sampling, no full encode"),
    # cassette replays this mp3 against the audio-LLM, so the bytes must be real, not synthetic.
    "cut.audio": (REAL, "per-segment trim/fade/atempo AUDIO-ONLY encode — CPU ffmpeg, no video_encode argv"),
    "media.audio": (REAL, "full-file audio demux to mp3 — CPU ffmpeg, no video_encode argv"),

    "camera.apply": (STUB, "GPU (libplacebo) crop-trajectory render to pixels"),
    "cut.apply": (STUB, "per-segment trim/atempo render + concat + crossfade encode"),
    "edit.splice": (STUB, "trim+concat re-encode"),
    "edit.weld": (STUB, "film-burn transition composite + encode"),
    "media.cut_proxy": (STUB, "proxy encode"),
    "media.filmstrip": (STUB, "frame sampling + hstack composite image encode"),
    "media.frames": (STUB, "per-fraction frame extraction, image encode"),
    "media.image_scale": (STUB, "image normalize + re-encode"),
    "media.normalize": (STUB, "canonical ingest encode (GPU_ADMISSION.HEAVY_GPU_OPS)"),
    "media.pcm": (STUB, "full-file audio decode to PCM wav"),
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


def _write_video(dst: Path, *, op_name: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = _VIDEO_EXT_ARGS.get(dst.suffix.lower())
    if args is None:
        raise DryStubUnsupportedOutput(op_name, dst.suffix.lower(), "video")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=1",
           "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=1", "-t", "1", *args, str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


def _write_image(dst: Path, *, op_name: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() not in _IMAGE_EXTS:
        raise DryStubUnsupportedOutput(op_name, dst.suffix.lower(), "image")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=64x64", "-frames:v", "1", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True, timeout=_LAVFI_BUDGET_S)


def _write_json(dst: Path, *, seed: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('{"contour_dry": true, "seed": "%s"}' % hashlib.sha256(seed.encode()).hexdigest()[:12],
                   encoding="utf-8")


_WRITER: dict[str, Callable[..., None]] = {
    "video": _write_video,
    "audio": _write_audio,
    "image": _write_image,
}


def _fill_one(dst: Path, kind: str, *, seed: str, op_name: str) -> None:
    if kind == "json":
        _write_json(dst, seed=seed)
        return
    fn = _WRITER.get(kind)
    if fn is None:
        raise registry.OpError(f"contour-dry: no synthesis rule for output kind {kind!r}")
    fn(dst, op_name=op_name)


def _handler(op: registry.Op) -> Callable[..., None]:
    def run(*, params: dict[str, Any], inputs: dict[str, Path], outputs: dict[str, Any]) -> None:  # noqa: ARG001
        declared = {p.id: p for p in op.outputs}
        for port_id, dst in outputs.items():
            port = declared[port_id]
            targets = dst if isinstance(dst, list) else [dst]
            for i, one in enumerate(targets):
                _fill_one(Path(one), port.kind, seed=f"{op.op}:{port_id}:{i}", op_name=op.op)
    return run


def resolve(op: registry.Op) -> Callable[..., None]:
    mode, _reason = _classify(op.op)
    if mode == REAL:
        # `runner.py` activates the ops pack before it ever picks this seam, dry or not — safe to call.
        return pack.resolve(op.handler)
    return _handler(op)
