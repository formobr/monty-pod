"""Render half of the pod: a fully-resolved RenderSpec becomes one ffmpeg pass, the result is
PUT back over presigned URLs. No decisions here — every number was fixed by the planner; this
module only translates those numbers into a filtergraph and an argv."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .cp import ControlPlane, download, upload
from .models import MotionKeyframe, RenderSpec

_P_STYLE = re.compile(r"p\d+")  # NVENC preset names (p1..p7); libx264 can't take these


def _num(x: float) -> str:
    """ffmpeg-friendly number: whole values without a trailing '.0', else shortest round-trip."""
    xf = float(x)
    return str(int(xf)) if xf == int(xf) else repr(xf)


# --- keyframe animation -------------------------------------------------------

def _ease(interp: str, p: str) -> str:
    """Eased fraction as an ffmpeg expr, given `p` already clamped to [0, 1]."""
    if interp == "ease_in":
        return f"({p})*({p})"
    if interp == "ease_out":
        return f"1-(1-({p}))*(1-({p}))"
    if interp == "ease_in_out":
        return f"({p})*({p})*(3-2*({p}))"  # smoothstep
    return f"({p})"  # linear


def anim_expr(keyframes: list[MotionKeyframe], component: int, interp: str, scale: str) -> str:
    """One rect component (0=x,1=y,2=w,3=h) as a per-frame pixel expr over segment time `t`.

    setpts reset PTS, so `t` starts at 0 for the segment. `scale` is the full-dimension expr
    ("iw" for x/w, "ih" for y/h): the normalized fraction is multiplied by it to reach pixels.
    Piecewise linear-in-time with the chosen easing between adjacent keyframes; a clamp on the
    first interval covers t before the first keyframe, the last value covers t past the last."""
    vals = [kf.rect[component] for kf in keyframes]
    if len(keyframes) == 1:
        return f"({_num(vals[0])})*{scale}"

    times = [kf.t for kf in keyframes]
    expr = _num(vals[-1])  # else-branch once t is past the final keyframe
    for i in range(len(keyframes) - 2, -1, -1):
        dt = times[i + 1] - times[i]
        p = "1" if dt <= 0 else f"clip((t-{_num(times[i])})/{_num(dt)},0,1)"
        eased = _ease(interp, p)
        lerp = f"({_num(vals[i])}+({_num(vals[i + 1])}-{_num(vals[i])})*({eased}))"
        expr = f"if(lt(t,{_num(times[i + 1])}),{lerp},{expr})"
    return f"({expr})*{scale}"


def _gpu_crop(keyframes: list[MotionKeyframe], interp: str, w: int, h: int) -> str:
    """A moving/zooming crop on the GPU: one hwupload -> libplacebo (per-frame crop exprs) ->
    hwdownload. crop_* are in source pixels; w/h is the output size."""
    cx = anim_expr(keyframes, 0, interp, "iw")
    cy = anim_expr(keyframes, 1, interp, "ih")
    cw = anim_expr(keyframes, 2, interp, "iw")
    ch = anim_expr(keyframes, 3, interp, "ih")
    return (
        "format=yuv420p,hwupload,"
        f"libplacebo=w={w}:h={h}:crop_x='{cx}':crop_y='{cy}':crop_w='{cw}':crop_h='{ch}',"
        "hwdownload,format=yuv420p,setrange=range=tv"
    )


def _cpu_crop(keyframes: list[MotionKeyframe], w: int, h: int) -> str:
    """CPU fallback: no animation in v1 — a static crop at the first keyframe rect, then scale."""
    x0, y0, w0, h0 = keyframes[0].rect
    return (
        f"crop=w=iw*{_num(w0)}:h=ih*{_num(h0)}:x=iw*{_num(x0)}:y=ih*{_num(y0)},"
        f"scale={w}:{h}:flags=lanczos,setsar=1"
    )


def _atempo_chain(speed: float) -> list[str]:
    """atempo tokens whose product equals `speed`. One instance is limited to [0.5, 2.0], so a
    factor outside that range is split into several whose product is the factor (near-always one)."""
    factors: list[float] = []
    r = float(speed)
    while r > 2.0:
        factors.append(2.0)
        r /= 2.0
    while r < 0.5:
        factors.append(0.5)
        r /= 0.5
    factors.append(r)
    return [f"atempo={_num(f)}" for f in factors]


# --- graph & command ----------------------------------------------------------

def input_ids(spec: RenderSpec) -> list[str]:
    """Unique input ids in first-appearance order; an id's position is its ffmpeg input index."""
    seen: list[str] = []
    for inp in spec.inputs:
        if inp.id not in seen:
            seen.append(inp.id)
    return seen


def build_filtergraph(spec: RenderSpec, gpu: bool) -> str:
    """Pure: the -filter_complex string trimming, speed-adjusting, motion-treating and
    concatenating every timeline segment into [vout]/[aout]."""
    idx = {iid: n for n, iid in enumerate(input_ids(spec))}
    w, h = spec.timeline.width, spec.timeline.height
    motion_by_seg = {m.seg: m for m in spec.motion.segments} if spec.motion else {}

    chains: list[str] = []
    pads: list[str] = []
    for k, seg in enumerate(spec.timeline.segments):
        j = idx[seg.src]
        video = (
            f"[{j}:v]trim=start={_num(seg.in_)}:end={_num(seg.out)},"
            f"setpts=(PTS-STARTPTS)/{_num(seg.speed)}"
        )
        m = motion_by_seg.get(k)
        if m is None:
            video += f",scale={w}:{h}:flags=lanczos,setsar=1"
        elif gpu:
            video += "," + _gpu_crop(m.keyframes, m.interp, w, h)
        else:
            video += "," + _cpu_crop(m.keyframes, w, h)
        chains.append(f"{video}[v{k}]")

        audio = f"[{j}:a]atrim=start={_num(seg.in_)}:end={_num(seg.out)},asetpts=PTS-STARTPTS"
        audio += "," + ",".join(_atempo_chain(seg.speed))
        chains.append(f"{audio}[a{k}]")

        pads.append(f"[v{k}][a{k}]")

    n = len(spec.timeline.segments)
    chains.append(f"{''.join(pads)}concat=n={n}:v=1:a=1[vout][aout]")
    return ";".join(chains)


def build_command(
    spec: RenderSpec, input_paths: dict[str, Path], out_path: Path, gpu: bool
) -> list[str]:
    """Pure: the full ffmpeg argv for this spec. gpu decides the codec at runtime — the spec's
    named encoder is only a hint; a CPU fallback overriding it is allowed mechanics."""
    enc = spec.encode
    cmd = ["ffmpeg", "-y", "-hide_banner"]
    if gpu:
        cmd += ["-init_hw_device", "vulkan"]  # libplacebo runs on a Vulkan device; hwupload derives from it
    for iid in input_ids(spec):
        cmd += ["-i", str(input_paths[iid])]
    cmd += ["-filter_complex", build_filtergraph(spec, gpu)]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    if gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", enc.preset, "-tune", "hq", "-cq", str(enc.cq)]
    else:
        preset = "medium" if _P_STYLE.fullmatch(enc.preset) else enc.preset
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(enc.cq)]
    cmd += [
        "-pix_fmt", enc.pix_fmt,
        "-c:a", "aac", "-b:a", enc.audio_bitrate,
        "-movflags", "+faststart", str(out_path),
    ]
    return cmd


# --- I/O orchestration --------------------------------------------------------

_GPU: bool | None = None


def _gpu_available() -> bool:
    """One cached REAL smoke render — an encoder merely being listed proves nothing about the
    Vulkan/libplacebo/NVENC path actually working on this box."""
    global _GPU
    if _GPU is None:
        try:
            probe = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-init_hw_device", "vulkan",
                 "-f", "lavfi", "-i", "testsrc=duration=0.1:size=64x64:rate=10",
                 "-vf", "format=yuv420p,hwupload,libplacebo=w=32:h=32,hwdownload,format=yuv420p",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=30,
            )
            _GPU = probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _GPU = False
    return _GPU


def render_spec(spec: RenderSpec, cp: ControlPlane) -> None:
    """Fetch inputs, run the single encode pass, PUT every non-cache output, report the event."""
    if spec.mode == "final" and spec.overlays is not None:
        raise NotImplementedError("final overlays composite lands with the mograph bundle input")

    gpu = _gpu_available()
    if not gpu and spec.motion is not None and spec.motion.segments:
        print("no NVENC: camera motion degrades to a static crop at the first keyframe",
              file=sys.stderr)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        input_paths = {
            inp.id: download(inp.url, tmp / inp.id.replace("/", "__")) for inp in spec.inputs
        }
        out = tmp / "render.mp4"
        cmd = build_command(spec, input_paths, out, gpu)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            tail = (exc.stderr or b"")[-2000:]
            detail = tail.decode("utf-8", "replace") if isinstance(tail, bytes) else str(tail)
            raise RuntimeError(f"ffmpeg exited {exc.returncode}: {detail}") from exc

        done: list[str] = []
        for o in spec.outputs:
            if o.kind == "cache":
                print(f"cache output {o.id!r} skipped (v1)", file=sys.stderr)
                continue
            upload(out, o.put_url, "video/mp4")
            done.append(o.id)

    cp.post_event({
        "job_id": spec.job_id,
        "stage": "render",
        "status": "done",
        "outputs": done,
    })
