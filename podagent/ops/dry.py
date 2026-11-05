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
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, NamedTuple

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


def _video_frame_rate(path: Path) -> str | None:
    """The first video stream's `r_frame_rate` as ffprobe's exact rational ("30000/1001"), or None when the
    file has no video stream or the rate is unreadable / not a positive rational."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
             "-of", "csv=p=0", str(path)],
            check=True, capture_output=True, text=True, timeout=_LAVFI_BUDGET_S)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    text = proc.stdout.strip().splitlines()[0].strip().rstrip(",") if proc.stdout.strip() else ""
    return _positive_rational(text)


def _positive_rational(text: str) -> str | None:
    """ffprobe's "30000/1001" (or a bare "25") as the exact "n/d" string, None when not a positive rational."""
    num, sep, den = text.strip().partition("/")
    try:
        n, d = int(num), int(den) if sep else 1
    except ValueError:
        return None
    return f"{n}/{d}" if n > 0 and d > 0 else None


class DryStubUnreadableShape(RuntimeError):
    """A probed field of a video placeholder (width, height, SAR, rotation, rate) has no readable value —
    the op is refused by name, never given a default (the placeholder would diverge from its live record)."""

    def __init__(self, op_name: str, field: str, *, why: str) -> None:
        self.op_name, self.field = op_name, field
        super().__init__(
            f"contour-dry: {op_name!r} stub cannot resolve video field {field!r} ({why}) per "
            f"podagent/ops/dry.py::_VIDEO_SHAPE — never defaulted")


class _Probed(NamedTuple):
    """What the engine's media_clock reads off a rendered video: stored dims, SAR, rotation, exact rate."""
    width: int
    height: int
    sar: Fraction
    rotation: int
    rate: str

    @property
    def display(self) -> tuple[int, int]:
        """The decoded (displayed) w/h: stored dims swapped for a 90/270 display matrix (media_clock's rule)."""
        return (self.height, self.width) if abs(self.rotation) % 180 == 90 else (self.width, self.height)


def _probe_video(path: Path, *, op_name: str) -> _Probed | None:
    """The first video stream's probed fields, None when the file has NO video stream at all (audio-only);
    a video stream with an unreadable field is a named refusal."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,sample_aspect_ratio,r_frame_rate:stream_tags=rotate:stream_side_data=rotation",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=_LAVFI_BUDGET_S)
        streams = json.loads(proc.stdout or "{}").get("streams") or []
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
        raise DryStubUnreadableShape(op_name, "probe", why=f"ffprobe failed on {path.name}: {exc}") from exc
    if not streams:
        return None
    st = streams[0]
    w, h = st.get("width"), st.get("height")
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        raise DryStubUnreadableShape(op_name, "width/height", why=f"unreadable dims on {path.name}")
    rate = _positive_rational(str(st.get("r_frame_rate") or ""))
    if rate is None:
        raise DryStubUnreadableShape(op_name, "r_frame_rate", why=f"unreadable rate on {path.name}")
    sn, _, sd = str(st.get("sample_aspect_ratio") or "").partition(":")
    sar = Fraction(1)   # an absent / 0:1 SAR decodes as square pixels
    if sn.isdigit() and sd.isdigit() and int(sn) > 0 and int(sd) > 0:
        sar = Fraction(int(sn), int(sd))
    rot_raw: Any = None
    for sd_item in st.get("side_data_list") or []:
        if isinstance(sd_item, dict) and "rotation" in sd_item:
            rot_raw = sd_item["rotation"]
    if rot_raw is None:
        rot_raw = (st.get("tags") or {}).get("rotate", 0)
    try:
        rotation = int(round(float(rot_raw)))
    except (TypeError, ValueError) as exc:
        raise DryStubUnreadableShape(op_name, "rotation", why=f"unreadable rotation on {path.name}") from exc
    return _Probed(w, h, sar, rotation, rate)


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


@dataclass(frozen=True)
class Shape:
    """Where each probed field of a STUB video op's LIVE output comes from (the engine's media_clock Clock
    probes duration, fps, width, height, rotation of every rendered video and plans on them). Tokens:
      source          the bound video input's own probed value (display w/h, SAR, r_frame_rate)
      param:NAME      the op param that sets it (required)
      param?:NAME     the param when present, else the source's value (rate only)
      cap:NAME        height only: min(source height, param NAME), never upscaling; absent param = source
      scale:NAME      height only: param NAME, or min(source, NAME) when params["height_mode"] == "at_most"
      follow_height   width only: source aspect at the resolved height, nearest even (scale=-2:H)
      square          sar only: 1:1 (the live op squares the pixels)
      baked           rotation only: the live op decodes upright (autorotate), so the output carries no
                      display matrix and its decoded w/h are the source's displayed w/h
      fallback        nothing in params/inputs carries it: the synthetic 64 (w/h) / the legacy rate rule
    The duration is `_VIDEO_DUR`'s rule. `source_port` names the bound video input to read when it is not
    the first one."""
    width: str
    height: str
    sar: str
    rotation: str
    rate: str
    source_port: str | None = None


_ALL_SOURCE = Shape("source", "source", "source", "baked", "source")

# One row per STUB op whose output is a video; a row's comment cites the live op contract it mirrors
# (contracts/ops/<op>.json — the handler code itself lives in the external ops pack). An op missing here is
# refused at resolve() (`_check_video_shape`).
_VIDEO_SHAPE: dict[str, Shape] = {
    # cut.apply: per-segment `fps=fps_grid` of the source's own rate; `scale=-2:max_h` caps height, aspect
    # (so SAR) kept, never upscaling; max_h absent = native. rate is the source's: fps_grid is by contract
    # "the source video frame rate the keep boundaries were snapped to", so the two are the same number.
    # SAR stays the source's (ffmpeg's -2 rounding nudges it by <0.5%; cutgrid.cut_scale_vf, which could
    # square it, is in the external ops pack and was not readable here).
    "cut.apply": Shape("follow_height", "cap:max_h", "source", "baked", "source"),
    # camera.apply: crop trajectory rendered to out_w x out_h; "camera.apply never emits an fps filter".
    "camera.apply": Shape("param:out_w", "param:out_h", "source", "baked", "source"),
    # edit.splice: trim+concat re-encode of the baked video — geometry and clock untouched.
    "edit.splice": _ALL_SOURCE,
    # edit.weld: the cold-open is welded onto the FRONT of the finished `main` master; the output is the
    # master's picture.
    "edit.weld": Shape("source", "source", "source", "baked", "source", source_port="main"),
    # media.cut_proxy: "max_h ... aspect kept, never upscaling" (cut_scale_vf, the same string cut.apply caps
    # with); "timing passthrough".
    "media.cut_proxy": Shape("follow_height", "cap:max_h", "source", "baked", "source"),
    # media.fetch: a bare origin GET with no bound input; nothing declares its geometry or rate.
    "media.fetch": Shape("fallback", "fallback", "square", "baked", "fallback"),
    # media.normalize: width/height are the TRUE displayed dims (rotation baked), pixels squared, CFR `fps`.
    "media.normalize": Shape("param:width", "param:height", "square", "baked", "param:fps"),
    # media.scale: height exact (or at_most), width follows the source aspect (-2), `fps` resamples when given.
    "media.scale": Shape("follow_height", "scale:height", "source", "baked", "param?:fps"),
    # mograph.render: renders at `fps`; the canvas size comes from the bundled composition, not a param.
    "mograph.render": Shape("fallback", "fallback", "square", "baked", "param:fps"),
    # motion.kenburns: a 9:16 clip from a still; neither pixel size nor rate is a param.
    "motion.kenburns": Shape("fallback", "fallback", "square", "baked", "fallback"),
    # opener.build: the clip is w x h at `fps` (a rational string).
    "opener.build": Shape("param:w", "param:h", "square", "baked", "param:fps"),
}

_TOKEN_KINDS: dict[str, frozenset[str]] = {
    "width": frozenset({"source", "param", "follow_height", "fallback"}),
    "height": frozenset({"source", "param", "cap", "scale", "fallback"}),
    "sar": frozenset({"source", "square"}),
    "rotation": frozenset({"baked"}),
    "rate": frozenset({"source", "param", "param?", "fallback"}),
}
for _op, _shape in _VIDEO_SHAPE.items():
    for _field, _kinds in _TOKEN_KINDS.items():
        if getattr(_shape, _field).partition(":")[0] not in _kinds:
            raise registry.OpError(f"podagent/ops/dry.py::_VIDEO_SHAPE[{_op!r}].{_field} = "
                                   f"{getattr(_shape, _field)!r} is not one of {sorted(_kinds)}")


class VideoFields(NamedTuple):
    """The resolved probed fields a placeholder is built with (rate None = the legacy rate rule)."""
    width: int
    height: int
    sar: Fraction
    rotation: int
    rate: str | None


def _needs_source(shape: Shape) -> bool:
    return any(getattr(shape, f).partition(":")[0] in ("source", "cap", "scale", "follow_height", "param?")
               for f in ("width", "height", "sar", "rate"))


def _rational_param(op_name: str, name: str, value: Any) -> str:
    try:
        frac = Fraction(value) if isinstance(value, str) else Fraction(str(value)).limit_denominator(1001)
    except (ValueError, ZeroDivisionError) as exc:
        raise DryStubUnreadableShape(op_name, "rate", why=f"param {name!r}={value!r} is not a rate") from exc
    if frac <= 0:
        raise DryStubUnreadableShape(op_name, "rate", why=f"param {name!r}={value!r} is not positive")
    return f"{frac.numerator}/{frac.denominator}"


def resolve_video_fields(op_name: str, shape: Shape, params: dict[str, Any],
                         probed: _Probed | None) -> VideoFields | None:
    """The row's tokens resolved against the params and the probed source; None = a source-derived field with
    no video source bound (the synthetic 64x64 fallback). An unreadable field raises, never defaults."""
    if probed is None and _needs_source(shape):
        return None
    if probed is None and any(
            getattr(shape, f).partition(":")[0] == "param" and params.get(getattr(shape, f).partition(":")[2]) is None
            for f in ("width", "height", "rate")):
        # No video source is bound AND a param-set field was not sent: nothing real to mirror, so the whole
        # placeholder stays synthetic (with a bound source the same absence is a named refusal below).
        return None

    def param(name: str, field: str) -> Any:
        if params.get(name) is None:
            raise DryStubUnreadableShape(op_name, field, why=f"param {name!r} is absent")
        return params[name]

    disp_w, disp_h = probed.display if probed is not None else (0, 0)
    kind, _, arg = shape.height.partition(":")
    if kind == "cap":
        height = min(disp_h, int(params[arg])) if params.get(arg) is not None else disp_h
    elif kind == "scale":
        height = int(param(arg, "height"))
        if params.get("height_mode") == "at_most":
            height = min(height, disp_h)
    elif kind == "param":
        height = int(param(arg, "height"))
    elif kind == "source":
        height = disp_h
    else:
        height = 64
    wkind, _, warg = shape.width.partition(":")
    if wkind == "follow_height":
        # scale=-2:H rounds the width to the nearest even, halves UP (av_rescale), not Python's banker's round.
        width = disp_w if height == disp_h else 2 * ((disp_w * height + disp_h) // (2 * disp_h))
    elif wkind == "param":
        width = int(param(warg, "width"))
    elif wkind == "source":
        width = disp_w
    else:
        width = 64
    # A baked 90/270 rotation is a transpose, which inverts the pixel aspect (out SAR = 1/in SAR).
    sar = Fraction(1)
    if shape.sar == "source" and probed is not None:
        sar = 1 / probed.sar if abs(probed.rotation) % 180 == 90 else probed.sar
    rkind, _, rarg = shape.rate.partition(":")
    rate: str | None
    if rkind == "source":
        rate = probed.rate if probed is not None else None
    elif rkind == "param":
        rate = _rational_param(op_name, rarg, param(rarg, "rate"))
    elif rkind == "param?":
        rate = _rational_param(op_name, rarg, params[rarg]) if params.get(rarg) is not None else (
            probed.rate if probed is not None else None)
    else:
        rate = None
    return VideoFields(width, height, sar, 0, rate)


def _check_video_shape(op: registry.Op) -> None:
    """A STUB op that writes a video must say, in `_VIDEO_SHAPE`, where each probed field comes from."""
    if op.op in _VIDEO_SHAPE or op.op in _OP_SYNTH:
        return
    if any(p.kind == "video" for p in op.outputs):
        raise registry.OpError(
            f"contour-dry: {op.op!r} is a STUB that writes a video but has no shape row — add one to "
            f"podagent/ops/dry.py::_VIDEO_SHAPE (where its width, height, SAR, rotation and frame rate come "
            f"from) before it can run under {ARM_ENV}")


def _write_video(dst: Path, *, op_name: str, audio_src: Path | None = None, dur: float | None = None,
                 params: dict[str, Any] | None = None, shape_src: Path | None = None) -> None:
    # `dur` (a cut's own reported length) pins the placeholder's length; None keeps each branch's own length.
    # `audio_src` supplies the real audio; the probed fields come from `shape_src` (default: `audio_src`).
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() not in _VIDEO_EXT_ARGS:
        raise DryStubUnsupportedOutput(op_name, dst.suffix.lower(), "video")
    secs = f"{dur:.6f}" if dur is not None else "1"
    # EVERY probed field mirrors the live op's output (`_VIDEO_SHAPE`): the engine plans on the rendered
    # video's clock (duration, fps, width, height, rotation), so a literal 30 fps or 64x64 here moves a beat
    # or a crop and a dry replay realizes different placements than the live run. Without a video source
    # the fallback is 64x64, and r=30 when `dur` is pinned — at r=1 a colour source's last frame carries a
    # full 1s, so d=5.656 would round the container's length UP to 6.0 regardless of a trailing -t; 30 keeps
    # that rounding under one 1/30s frame (the tolerance callers of a `dur`-pinned placeholder check
    # against) — and r=1:d=1 when `dur` is None (no cut length to pin).
    shape = _VIDEO_SHAPE.get(op_name, _ALL_SOURCE)   # a missing row was refused at resolve()
    probe_src = shape_src if shape_src is not None else audio_src
    probed = _probe_video(probe_src, op_name=op_name) if probe_src is not None and _needs_source(shape) else None
    fields = resolve_video_fields(op_name, shape, params or {}, probed)
    size = f"{fields.width}x{fields.height}" if fields is not None else "64x64"
    sar = f",setsar={fields.sar.numerator}/{fields.sar.denominator}" if fields is not None else ""
    source_rate = fields.rate if fields is not None else None
    rate = source_rate or (30 if dur is not None else 1)
    if audio_src is None:
        # No video-kind input bound at all — nothing real to reflect, stays synthetic (params-set fields aside).
        args = _VIDEO_EXT_ARGS[dst.suffix.lower()]
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", f"color=c=black:s={size}:r={rate}:d={secs}{sar}",
               "-f", "lavfi", "-i", dry_tone_lavfi(dur=float(secs)), "-t", secs, *args, str(dst)]
    elif _has_audio_stream(audio_src) and dur is not None:
        # The real audio trimmed (or silence-padded) to the cut's length, so audio and video both end there.
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(audio_src),
               "-f", "lavfi", "-i", f"color=c=black:s={size}:r={rate}{sar}",
               "-map", "1:v", "-map", "0:a", "-af", "apad",
               *_VIDEO_EXT_ARGS[dst.suffix.lower()], "-t", secs, str(dst)]
    elif _has_audio_stream(audio_src):
        # `-shortest` against an infinite colour source sizes the output to the real audio's own duration.
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(audio_src),
               "-f", "lavfi", "-i", f"color=c=black:s={size}" + (f":r={source_rate}" if source_rate else "") + sar,
               "-map", "1:v", "-map", "0:a",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-shortest", str(dst)]
    else:
        # Real input, genuinely no audio track — the placeholder gets none, never a fabricated anullsrc.
        # `-t` must also cap THIS branch (the previous fix missed it): without it the lavfi colour
        # source's own `d=` is the only cap, which at r=1 rounds up (see `rate` above).
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", f"color=c=black:s={size}:r={rate}:d={secs}{sar}",
               "-t", secs,
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


# A video placeholder whose op REPORTS its own length lasts exactly that long (MISC-756: the cut.apply
# placeholder ran the source's full length while its rdurs summed to the cut, so a dry replay probed a length
# the live run never recorded). Every other op's video placeholder keeps its branch's own length.
_VIDEO_DUR: dict[str, Callable[[dict[str, Any]], float]] = {
    "cut.apply": lambda params: sum(_synth_cut_apply_durs(params, {})["rdurs"]),
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
        dur_of = _VIDEO_DUR.get(op_name)
        shape_src = audio_src
        port = _VIDEO_SHAPE.get(op_name, _ALL_SOURCE).source_port
        if port is not None and audio_src is not None:
            raw = inputs.get(port)
            if raw is None or not Path(raw).exists():
                raise DryStubUnreadableShape(op_name, "source", why=f"input {port!r} is not bound")
            shape_src = Path(raw)
        _write_video(dst, op_name=op_name, audio_src=audio_src, params=params, shape_src=shape_src,
                     dur=dur_of(params) if dur_of is not None else None)
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
    _check_video_shape(op)
    return _handler(op)


def _clip_rank_group_verdict(n: int) -> tuple[list[float], list[list[float]]]:
    """Strictly descending scores over ALL candidates: real SigLIP on identical STUB placeholder pixels
    would score every candidate near-equally, clearing no relevance floor and shortlisting nothing. Each
    tile gets a distinct unit embedding (one-hot): a scored tile always has an embed, as on a real pod, and
    distinct ones keep the planner's MMR dedup from collapsing the shortlist."""
    return ([round(max(0.05, 0.95 - 0.05 * i), 4) for i in range(n)],
            [[1.0 if k == i else 0.0 for k in range(n)] for i in range(n)])


# An origin group's tiles were "fetched" by nobody here; each still gets a measure, as on a real pod, so the
# engine's readers of `measures` run in the dry tier. A typical provider thumbnail, not a measured one.
_DRY_ORIGIN_MEASURE = {"bytes": 40000, "w": 512, "h": 512}
# The real judge tile's size and quality (infer_cliprank._JUDGE_LONG_SIDE / _JUDGE_QUALITY), drawn solid.
_DRY_JUDGE_SIDE = 512
_DRY_JUDGE_QUALITY = 82
# Every dry stub progress line starts with this: the engine recognises a stub run by it (DRY_STUB_MARKS).
_CLIP_RANK_STUB_MARK = "contour-dry clip_rank stub"


def run_clip_rank(params: Any, put_url: str, progress: Callable[[str], None] | None = None) -> SimpleNamespace:
    """The dry-tier stand-in for `infer_cliprank.ClipRankService.run` — same `(infer_s, timings)` shape,
    no weights loaded, no GPU touched, and the PUT contract kept so the resolver reads it identically. An
    origin group also carries one deterministic measure per tile, and the payload carries its timings."""
    from .. import cp
    from ..models import ClipRankGroupResult, ClipRankPayload, ClipRankTileMeasure

    t0 = time.monotonic()
    groups = []
    for g in params.groups:
        scores, embeds = _clip_rank_group_verdict(len(g.tile_urls))
        measures = ([ClipRankTileMeasure(**_DRY_ORIGIN_MEASURE) for _ in scores]
                    if g.origin_urls is not None else None)
        groups.append(ClipRankGroupResult(status="ok", scores=scores, embeds=embeds, errors=[None] * len(scores),
                                          measures=measures))
    payload = ClipRankPayload(model="contour-dry-stub", groups=groups,
                              timings={"infer_s": round(time.monotonic() - t0, 3)})
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "clip_rank.json"
        out.write_text(payload.model_dump_json(exclude_none=True))
        cp.upload(out, put_url, "application/json")
    infer_s = time.monotonic() - t0
    if progress is not None:
        progress(f"{_CLIP_RANK_STUB_MARK}: {len(groups)} group(s), deterministic descending scores")
    return SimpleNamespace(infer_s=infer_s, timings={"infer_s": round(infer_s, 3), "work_s": round(infer_s, 3)})


def _write_judge_tile(dst: Path, origin_url: str) -> None:
    """A solid 512x512 JPEG whose colour is a hash of the origin url: two candidates never share one tile."""
    from PIL import Image

    digest = hashlib.sha256(origin_url.encode("utf-8")).digest()
    Image.new("RGB", (_DRY_JUDGE_SIDE, _DRY_JUDGE_SIDE), (digest[0], digest[1], digest[2])).save(
        dst, "JPEG", quality=_DRY_JUDGE_QUALITY)


# Mirrors infer_cliprank's own default when `cp._STORE_POOL_WIDTH` is ever renamed out from under this
# private reach-through — the dry tier degrades to serial PUTs rather than raising at import time.
_FALLBACK_POOL_WIDTH = 8


def run_judge_tiles(params: Any, put_url: str, progress: Callable[[str], None] | None = None) -> SimpleNamespace:
    """The dry-tier stand-in for `infer_cliprank.run_judge_tiles`: no origin fetch — each tile is a
    deterministic JPEG — but its real PUT (in parallel, as on the pod) and the same one result JSON after.
    A failed PUT is a NAMED `put_failed` tile result, never a raise — one bad tile must not cost the judge
    the others nor the result JSON, same as the real pod (infer_cliprank._judge_tile)."""
    import concurrent.futures as cf

    from .. import cp
    from ..infer_cliprank import _redacted_tile_url, _tile_error
    from ..models import ClipRankJudgeTileResult, ClipRankPayload

    t0 = time.monotonic()
    tiles = params.judge_tiles or []
    pool_width = getattr(cp, "_STORE_POOL_WIDTH", _FALLBACK_POOL_WIDTH)
    with tempfile.TemporaryDirectory() as td:
        def one(i: int, tile: Any) -> ClipRankJudgeTileResult:
            dst = Path(td) / f"{i}.jpg"
            _write_judge_tile(dst, tile.origin_url)
            try:
                cp.upload(dst, tile.put_url, "image/jpeg")
            except Exception as exc:  # noqa: BLE001 — a refused or exhausted PUT is this tile's error
                err = _tile_error("put_failed", exc, _redacted_tile_url(tile.put_url))
                return ClipRankJudgeTileResult(status="error", error=err)
            return ClipRankJudgeTileResult(status="ok", bytes=dst.stat().st_size, w=_DRY_JUDGE_SIDE,
                                           h=_DRY_JUDGE_SIDE, cache="hit")

        with cf.ThreadPoolExecutor(max_workers=max(1, min(len(tiles), pool_width)),
                                   thread_name_prefix="dry-judge-tile") as ex:
            futs = [ex.submit(one, i, t) for i, t in enumerate(tiles)]
            results = [f.result() for f in futs]
        timings = {"infer_s": round(time.monotonic() - t0, 3), "judge_tiles_n": float(len(tiles)),
                   "judge_cache_miss_n": 0.0}
        payload = ClipRankPayload(model="contour-dry-stub", groups=[], judge_tiles=results, timings=timings)
        out = Path(td) / "clip_rank.json"
        out.write_text(payload.model_dump_json(exclude_none=True))
        cp.upload(out, put_url, "application/json")
    infer_s = time.monotonic() - t0
    if progress is not None:
        progress(f"{_CLIP_RANK_STUB_MARK}: {len(results)} judge tile(s), deterministic solid JPEGs")
    return SimpleNamespace(infer_s=infer_s, timings={**timings, "infer_s": round(infer_s, 3),
                                                     "work_s": round(infer_s, 3)})
