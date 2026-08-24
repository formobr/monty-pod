"""Render half of the pod: a fully-resolved RenderSpec becomes one ffmpeg pass, the result is
PUT back over presigned URLs. No decisions here — every number was fixed by the planner; this
module only translates those numbers into a filtergraph and an argv."""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from . import finalize as _finalize
from .cp import ControlPlane, download, upload
from .sanitize import safe_error
from .models import MotionKeyframe, RenderSpec, SpecBrollClip, SpecTransition

_P_STYLE = re.compile(r"p\d+")  # NVENC preset names (p1..p7); libx264 can't take these
# delivery signal: TAG bt709 (no convert — untagged made platforms guess the colourspace)
_BT709 = ("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709")
_BT709_SET_PARAMS = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"

# Twin of scripts/montyops/camera_apply.py.VULKAN_PROBE and podagent.main.VULKAN_PROBE; pinned in the
# superproject parity test because the baked camera and live preview must ask the same question.
# NVENC min frame dimension on newer GPUs exceeds 32; a tiny probe frame false-fails the whole GPU.
VULKAN_PROBE = (
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-init_hw_device", "vulkan",
    "-f", "lavfi", "-i", "testsrc=duration=0.1:size=256x256:rate=10",
    "-vf", "format=yuv420p,hwupload,libplacebo=w=256:h=256,hwdownload,format=yuv420p",
    "-c:v", "h264_nvenc", "-f", "null", "-",
)


def _venc(enc, gpu: bool) -> list[str]:
    """Video-encoder argv for the delivery contract: cq (14), bt709 tag, nvenc maxrate/bufsize headroom."""
    if gpu:
        return ["-c:v", "h264_nvenc", "-preset", enc.preset, "-tune", "hq", "-cq", str(enc.cq),
                "-maxrate", "24M", "-bufsize", "32M", "-pix_fmt", enc.pix_fmt, *_BT709]
    preset = "medium" if _P_STYLE.fullmatch(enc.preset) else enc.preset
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(enc.cq), "-pix_fmt", enc.pix_fmt, *_BT709]


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
    """Input ids the MAIN filtergraph consumes (timeline srcs, broll clips, sfx), in spec.inputs
    order — an id's position is its ffmpeg -i index. Cover/caption ASSETS (fonts, logo) are downloaded
    but NOT decoded here (the cover/caption passes read them off disk), so a TTF never becomes a bad -i.

    The MUSIC TRACK is not here either: what the graph mixes is the pre-rendered bed (an extra_input),
    so listing the track opened a decoder for a file no filter ever reads."""
    consumed: set[str] = {seg.src for seg in spec.timeline.segments}
    ov = spec.overlays
    if ov is not None:
        if ov.broll_final:
            consumed.update(c.clip for c in ov.broll_final.broll)
        if ov.sfx:
            consumed.update(s.sound for s in ov.sfx)
    seen: list[str] = []
    for inp in spec.inputs:
        if inp.id in consumed and inp.id not in seen:
            seen.append(inp.id)
    return seen


# b-roll cutaway overlay (final): transition exprs are single-quoted so commas stay literal, not separators.

def _broll_slide_xy(clip: SpecBrollClip, start: float, end: float) -> tuple[str, str] | None:
    """(x_expr, y_expr) overlay offsets for a slide_wipe/push at this cutaway's entry/return seam, or
    None when it hard-cuts (seated at 0,0). Dissolve is NOT here — it's an alpha fade on the clip."""
    def _sel(tr: "SpecTransition | None", phase: str) -> "tuple[float, float, str, str] | None":
        if tr is None or tr.kind not in ("slide_wipe", "push") or tr.direction is None:
            return None
        dur = min(tr.dur, end - start)
        t0 = start if phase == "entry" else end - dur
        p = f"clip((t-{t0:.4f})/{dur:.4f},0,1)"
        e = f"(pow({p},3)*({p}*({p}*6-15)+10))"  # smootherstep 0→1
        if phase == "return":
            e = f"(1-{e})"
        d = tr.direction
        if d == "left":
            x, y = f"W-W*{e}", "0"
        elif d == "right":
            x, y = f"-W+W*{e}", "0"
        elif d == "up":
            x, y = "0", f"H-H*{e}"
        else:  # down
            x, y = "0", f"-H+H*{e}"
        lo, hi = (start, start + dur) if phase == "entry" else (end - dur, end)
        return (lo, hi, x, y)

    pieces = [s for s in (_sel(clip.transition_in, "entry"),
                          _sel(clip.transition_out, "return")) if s is not None]
    if not pieces:
        return None

    def _expr(component: int) -> str:
        expr = "0"  # outside every window → seated
        for lo, hi, x, y in pieces:
            expr = f"if(between(t,{lo:.4f},{hi:.4f}),{(x, y)[component]},{expr})"
        return expr

    return _expr(0), _expr(1)


def _broll_dissolve_frag(clip: SpecBrollClip, start: float, end: float) -> str:
    """Alpha-crossfade fragment spliced into a cutaway's source chain for a `dissolve` seam; '' if none.
    alpha=1 ramps opacity (needs yuva420p) so the host shows through — a naплыв, not a luma fade."""
    ti, to = clip.transition_in, clip.transition_out
    di = ti if (ti and ti.kind == "dissolve") else None
    do = to if (to and to.kind == "dissolve") else None
    if not di and not do:
        return ""
    span = max(end - start, 1e-3)
    frag = ",format=yuva420p"
    if di:
        d = min(di.dur, span)
        frag += f",fade=t=in:st={start:.3f}:d={d:.3f}:alpha=1"
    if do:
        d = min(do.dur, span)
        frag += f",fade=t=out:st={end - d:.3f}:d={d:.3f}:alpha=1"
    return frag


# Ken Burns cutaway move: prepare at 2x supersample, then zoompan down to canvas so sub-pixel pans stay
# smooth (anti-jitter). z/x/y interpolate linearly across the clip per the preset (in=zoom-in, pans slide x/y).
_KB_SS = 2
_KB_PAN = {  # preset -> (fx0, fx1, fy0, fy1); z resolved by _kb_zooms. Unknown preset → "in" (centered zoom).
    "in": (0.5, 0.5, 0.5, 0.5), "out": (0.5, 0.5, 0.5, 0.5),
    "left": (0.85, 0.15, 0.5, 0.5), "right": (0.15, 0.85, 0.5, 0.5),
    "up": (0.5, 0.5, 0.85, 0.15), "down": (0.5, 0.5, 0.15, 0.85),
    "in_left": (0.65, 0.35, 0.5, 0.5), "in_right": (0.35, 0.65, 0.5, 0.5),
    "out_left": (0.35, 0.65, 0.5, 0.5), "out_right": (0.65, 0.35, 0.5, 0.5),
}


def _kb_zooms(preset: str, amount: float, pan_zoom: float) -> tuple[float, float]:
    if preset.startswith("in"):
        return 1.0, 1.0 + amount
    if preset.startswith("out"):
        return 1.0 + amount, 1.0
    return 1.0 + pan_zoom, 1.0 + pan_zoom  # pure pans keep a small constant zoom for pan room


def _kenburns(preset: str, amount: float, pan_zoom: float, n: int, w: int, h: int, fps: float) -> str:
    """The scale-2x → zoompan filter fragment for one cutaway's Ken Burns move (cover mode)."""
    p = preset if preset in _KB_PAN else "in"
    nf = max(1, n - 1)
    z0, z1 = _kb_zooms(p, amount, pan_zoom)
    fx0, fx1, fy0, fy1 = _KB_PAN[p]
    r = lambda v: round(v, 5)  # noqa: E731 — clean filtergraph coeffs (no float dust)
    z = f"({r(z0)}+({r(z1 - z0)})*on/{nf})"
    x = f"(iw-iw/zoom)*({r(fx0)}+({r(fx1 - fx0)})*on/{nf})"
    y = f"(ih-ih/zoom)*({r(fy0)}+({r(fy1 - fy0)})*on/{nf})"
    sw, sh = w * _KB_SS, h * _KB_SS
    return (f"scale={sw}:{sh}:force_original_aspect_ratio=increase:flags=lanczos,crop={sw}:{sh},"
            f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={w}x{h}:fps={_num(fps)}")


def _broll_chains(spec: RenderSpec, idx: dict[str, int], base_label: str) -> list[str]:
    """Overlay every resolved cutaway onto [base_label] → [vout]: Ken Burns move (scale-2x→zoompan per the
    clip's preset/amount), trim [in,in+dur], seat at `start`, ride authored slide/push (overlay x/y) or
    dissolve (alpha fade). Audio untouched."""
    assert spec.overlays is not None and spec.overlays.broll_final is not None
    clips = spec.overlays.broll_final.broll
    w, h = spec.timeline.width, spec.timeline.height
    fps = spec.timeline.fps
    chains: list[str] = []
    prev = f"[{base_label}]"
    last = len(clips) - 1
    for i, c in enumerate(clips):
        if c.dur is None:
            raise ValueError(f"final broll clip {c.clip!r} has no resolved dur")
        start, end = c.start, c.start + c.dur
        frag = _broll_dissolve_frag(c, start, end)
        j = idx[c.clip]
        kb = _kenburns(c.preset, c.amount if c.amount is not None else 0.12, 0.08,
                       max(1, round(c.dur * fps)), w, h, fps)
        chains.append(
            f"[{j}:v]trim=start={_num(c.in_ or 0.0)}:duration={_num(c.dur)},setpts=PTS-STARTPTS,"
            f"fps={_num(fps)},{kb},setpts=PTS-STARTPTS+{start:.3f}/TB{frag}[b{i}]"
        )
        xy = _broll_slide_xy(c, start, end)
        over = f"overlay=x='{xy[0]}':y='{xy[1]}':" if xy else "overlay="
        out_label = "vout" if i == last else f"o{i}"
        chains.append(
            f"{prev}[b{i}]{over}enable='between(t,{start:.3f},{end:.3f})':eof_action=pass[{out_label}]"
        )
        prev = f"[o{i}]"
    return chains


def _has_broll(spec: RenderSpec) -> bool:
    return bool(spec.mode == "final" and spec.overlays is not None
               and spec.overlays.broll_final is not None and spec.overlays.broll_final.broll)


# locked audio chain (add_music.sh + memory voice-audio-chain): voice -20 LUFS denoise-only (no comp/deharsh),
# music bed -33 LUFS, gentle sidechain duck. Master -14 loudnorm is a later step (after cover), not here.
_VOICE_LUFS, _TP, _LRA = -20.0, -1.5, 11
_MUSIC_LUFS = -33.0
_DUCK = 3
# One full-length audio decode/encode runs far above realtime; minutes of source fit well under this.
# A pass that does not is wedged, and a wedge must fail loud, not absorb the stage (deadline law).
_AUDIO_PASS_WALL_S = 300


class _AudioMix(NamedTuple):
    voice_idx: int                            # ffmpeg input index of the base (raw voice)
    bed_idx: int | None                       # music bed input index, or None (no music)
    clean: str                                # voice pre-filter (highpass [+ afftdn if dirty])
    vln: str                                  # measured two-pass loudnorm to -20 LUFS
    dur: float
    sfx: tuple[tuple[int, float, float], ...] = ()   # (sound input index, start seconds, linear gain)


def body_duration(spec: RenderSpec) -> float:
    """The body's own length, from the timeline that defines it. Probing the first SOURCE measured the
    whole recording instead — every second the cut dropped was padded onto the mix as silence."""
    return sum((s.out - s.in_) / s.speed for s in spec.timeline.segments)


def _voice_is_dirty(voice: Path) -> bool:
    """Denoise gate: afftdn on a CLEAN voice adds musical-noise + dulls it (memory voice-audio-chain), so
    denoise ONLY a source whose post-highpass noise floor is above -50 dB."""
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(voice), "-map", "0:a?",
                          "-af", "highpass=f=80,astats=metadata=1:reset=0", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=_AUDIO_PASS_WALL_S)
    for line in res.stderr.splitlines():
        if "Noise floor dB" in line:
            try:
                return float(line.split(":")[-1].strip()) > -50.0
            except ValueError:
                return False
    return False


def _measure_loudnorm(voice: Path, pre: str) -> str:
    """Two-pass loudnorm: measure the cleaned voice, return the loudnorm filter carrying measured_* +
    linear=true (exact delivery). On a measure failure return the plain filter (single-pass dynamic)."""
    vln = f"loudnorm=I={_num(_VOICE_LUFS)}:TP={_num(_TP)}:LRA={_LRA}"
    # -map 0:a:0? : a `-f null` pass with no map decodes the VIDEO too, and this measure only needs
    # the first audio stream (with no audio ffmpeg still exits nonzero — rc is unchecked, the parse miss below handles it).
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(voice),
                          "-map", "0:a:0?",
                          "-af", f"{pre},{vln}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=_AUDIO_PASS_WALL_S)
    out = res.stderr
    try:
        d = json.loads(out[out.rindex("{"):out.rindex("}") + 1])
        return (f"{vln}:measured_I={d['input_i']}:measured_TP={d['input_tp']}"
                f":measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}"
                f":offset={d['target_offset']}:linear=true")
    except (ValueError, KeyError):
        return vln


def _prerender_bed(music: Path, mstart: float, dur: float, tmp: Path) -> Path:
    """Normalize the track to the -33 LUFS bed, then loop+seek it to exactly `dur`. Order matters:
    loudnorm on an infinite loop truncates, so normalize the finite track first, then loop the fixed bed."""
    norm = tmp / "music_norm.flac"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(music),
                    "-af", f"loudnorm=I={_num(_MUSIC_LUFS)}:TP=-2:LRA=11", "-ar", "48000", "-ac", "2",
                    str(norm)], check=True, timeout=_AUDIO_PASS_WALL_S)
    bed = tmp / "music_bed.flac"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stream_loop", "-1",
                    "-ss", _num(mstart), "-i", str(norm), "-t", _num(dur), "-ar", "48000", "-ac", "2",
                    str(bed)], check=True, timeout=_AUDIO_PASS_WALL_S)
    return bed


def _audio_mix_chains(a: _AudioMix) -> list[str]:
    """Voice (clean → measured loudnorm → padded) mixed with the ducked -33 bed (when music), then the
    accent SFX summed on top with a peak-safe limiter → [aout]. Bed level is its normalize, not volume."""
    dur = _num(a.dur)
    chains: list[str] = []
    if a.bed_idx is not None:
        chains += [
            f"[{a.voice_idx}:a]{a.clean},{a.vln},apad=whole_dur={dur},asplit=2[vc1][vc2]",
            f"[{a.bed_idx}:a]volume=1.0[bg0]",
            f"[bg0][vc2]sidechaincompress=threshold=0.06:ratio={_DUCK}:attack=20:release=500[bg]",
            "[vc1][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[premix]",
            "[premix]aresample=192000,alimiter=limit=0.63:attack=5:release=50:level=false,"
            "aresample=48000[amaster]",
        ]
    else:
        chains.append(f"[{a.voice_idx}:a]{a.clean},{a.vln},apad=whole_dur={dur}[amaster]")

    if a.sfx:
        labels = []
        for i, (sidx, at, gain) in enumerate(a.sfx):
            chains.append(f"[{sidx}:a]adelay={int(round(at * 1000))}:all=1,volume={_num(gain)}[sx{i}]")
            labels.append(f"[sx{i}]")
        # amix SUMS sfx onto the master → cap the tips with a lookahead limiter (peak-safe last stage).
        chains.append(f"[amaster]{''.join(labels)}amix=inputs={len(a.sfx) + 1}:normalize=0:duration=first[mx]")
        # 0.79 (−2.05 dB) not 0.84: this limiter runs at 48k without the 192k oversample the premix one has, and inter-sample peaks overshoot ~0.5 dB past the sample ceiling — measured −0.93 dBTP against the −1.0 gate
        chains.append("[mx]alimiter=limit=0.79:attack=5:release=50:level=false[aout]")
    else:
        chains.append("[amaster]anull[aout]")
    return chains


def build_filtergraph(spec: RenderSpec, gpu: bool, audio: _AudioMix | None = None,
                      terminal_bt709: bool = True) -> str:
    """Pure: the -filter_complex string trimming, speed-adjusting, motion-treating and concatenating
    every timeline segment into [vout]/[aout], compositing final b-roll, and mixing music when `audio`."""
    idx = {iid: n for n, iid in enumerate(input_ids(spec))}
    w, h = spec.timeline.width, spec.timeline.height
    motion_by_seg = {m.seg: m for m in spec.motion.segments} if spec.motion else {}

    # `audio` overrides the segment audio with the voice+music mix (built by render_spec's pre-passes).
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

        if audio is None:
            aud = f"[{j}:a]atrim=start={_num(seg.in_)}:end={_num(seg.out)},asetpts=PTS-STARTPTS"
            aud += "," + ",".join(_atempo_chain(seg.speed))
            chains.append(f"{aud}[a{k}]")
            pads.append(f"[v{k}][a{k}]")
        else:
            pads.append(f"[v{k}]")

    n = len(spec.timeline.segments)
    # b-roll composites onto the concatenated base: base video → [vbase], last cutaway overlay → [vout].
    base_v = "vbase" if _has_broll(spec) else "vout"
    if audio is None:
        chains.append(f"{''.join(pads)}concat=n={n}:v=1:a=1[{base_v}][aout]")
    else:
        chains.append(f"{''.join(pads)}concat=n={n}:v=1:a=0[{base_v}]")
    if _has_broll(spec):
        chains += _broll_chains(spec, idx, base_v)
    if audio is not None:
        chains += _audio_mix_chains(audio)
    # Frame-level metadata wins over the encoder context, especially on the Vulkan/NVENC path.
    if terminal_bt709:
        chains.append(f"[vout]{_BT709_SET_PARAMS}[vout]")
    return ";".join(chains)


def build_command(
    spec: RenderSpec, input_paths: dict[str, Path], out_path: Path, gpu: bool,
    extra_inputs: tuple[Path, ...] = (), audio: _AudioMix | None = None,
) -> list[str]:
    """Pure: the full ffmpeg argv for this spec. gpu decides the codec at runtime — the spec's
    named encoder is only a hint; a CPU fallback overriding it is allowed mechanics. extra_inputs
    (the pre-rendered music bed) append after the spec inputs, at the indices `audio` references."""
    enc = spec.encode
    grid = _finalize.declared_grid(spec.timeline.fps)
    cmd = ["ffmpeg", "-y", "-hide_banner"]
    if gpu:
        cmd += ["-init_hw_device", "vulkan"]  # libplacebo runs on a Vulkan device; hwupload derives from it
    for iid in input_ids(spec):
        cmd += ["-i", str(input_paths[iid])]
    for extra in extra_inputs:
        cmd += ["-i", str(extra)]
    cmd += ["-filter_complex", build_filtergraph(spec, gpu, audio)]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    cmd += ["-r", grid, "-fps_mode", "cfr"]
    cmd += _venc(enc, gpu)
    cmd += ["-c:a", "aac", "-b:a", enc.audio_bitrate, "-ar", "48000", "-movflags", "+faststart", str(out_path)]
    return cmd


# Body colour when NOTHING branded crossed. Deliberately pure white: a tenant's own off-white here reads as
# correct on that tenant's videos and silently brands everyone else's — the exact bug this fallback replaces.
_NEUTRAL_FG = "#ffffff"


def _caption_colours(caps, motion_plan) -> tuple[str, str]:
    """(fg, accent) for the burn, from the DATA that crossed — never a literal palette.

    accent is a declared spec field the brain always fills; missing it means the brand did not cross at all,
    and burning a guessed lime onto a delivered master is worse than failing here."""
    tokens = (motion_plan.brand.tokens if motion_plan is not None and motion_plan.brand else {}) or {}
    colors = tokens.get("color") or {}
    accent = caps.accent or colors.get("accent")
    if not accent:
        raise RuntimeError("captions carry no accent colour and no brand tokens crossed — refusing to burn "
                           "captions in a guessed palette (see motion_plan.captions.accent / .brand.tokens)")
    fg = colors.get("fg")
    if not fg:
        fg = _NEUTRAL_FG
        print("[render] captions: no brand fg crossed (motion_plan.brand absent) — burning neutral "
              f"{_NEUTRAL_FG}, NOT a brand off-white", file=sys.stderr)
    return fg, accent


# --- I/O orchestration --------------------------------------------------------

def _safe_send_event(cp: ControlPlane, event: dict) -> None:
    """Swallow+log, never raise: past this seam the master is already encoded (~20 GPU-min paid)."""
    try:
        cp.send_event(event)
    except Exception as exc:
        label = event.get("phase") or event.get("op") or "event"
        print(f"[render] {label}: event send failed, swallowed: {safe_error(exc)}", file=sys.stderr)


_GPU: bool | None = None
_GPU_CACHE_ENV = "MONTY_GPU_PROBE_CACHE"


def _gpu_cache_key() -> str:
    """Identity of the probed configuration: boot, driver, visible devices, the ffmpeg build (a new pod
    image on an unrebooted host must re-probe), probe argv; an unreadable part contributes its own
    marker so a box that cannot say still keys stably."""
    parts = []
    for p in ("/proc/sys/kernel/random/boot_id", "/proc/driver/nvidia/version"):
        try:
            parts.append(Path(p).read_text().splitlines()[0].strip())
        except (OSError, IndexError):
            parts.append(f"{p}?")
    parts.append(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    try:
        v = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        parts.append((v.stdout or "").splitlines()[0].strip() if v.stdout else "ffmpeg?")
    except (OSError, subprocess.SubprocessError, IndexError):
        parts.append("ffmpeg?")
    parts.append(" ".join(VULKAN_PROBE))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _gpu_cache_read(path: Path, key: str) -> bool | None:
    """True only on a matching POSITIVE record; anything else is a miss that re-probes. A cached
    negative would silently demote every later render on this box to CPU, so it does not exist."""
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return True if rec.get("key") == key and rec.get("ok") is True else None


def _gpu_cache_write(path: Path, key: str) -> None:
    try:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"key": key, "ok": True}))
        os.replace(tmp, path)  # atomic; concurrent writers race to an identical record
    except OSError as exc:
        # a foreign-uid file at the fixed path makes every write fail: the probe answered this process,
        # but silence here would hide that every future process re-pays it
        print(f"[render] gpu probe cache not written ({path}): {exc}", file=sys.stderr, flush=True)


def _gpu_available() -> bool:
    """One cached REAL smoke render (a listed encoder proves nothing about the Vulkan/libplacebo/NVENC
    path), its POSITIVE verdict also cached on disk across worker processes — each was re-paying the
    ~2s probe. A stale positive on a died GPU is bounded by the encode itself failing loudly."""
    global _GPU
    if os.environ.get("MONTY_GPU_MOTION") == "0":
        return False
    if _GPU is None:
        cache = Path(os.environ.get(_GPU_CACHE_ENV) or "/tmp/monty_gpu_probe.json")
        key = _gpu_cache_key()
        if _gpu_cache_read(cache, key):
            _GPU = True
            return _GPU
        try:
            probe = subprocess.run(VULKAN_PROBE, capture_output=True, timeout=120)
            _GPU = probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _GPU = False
        if _GPU:
            _gpu_cache_write(cache, key)
    return _GPU


def _unlanded_arms(exc: BaseException | None) -> list[str] | None:
    """The `unlanded_arms` mark, sought through the WHOLE cause/context chain: an error raised while
    reporting the marked one (phase's event send) must not launder the mark away."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if names := getattr(exc, "unlanded_arms", None):
            return list(names)
        exc = exc.__cause__ or exc.__context__
    return None


@contextmanager
def _job_tmpdir():
    """The job's tmp dir, deleted on exit — UNLESS the unwinding error chain carries `unlanded_arms`:
    a pool arm that never landed may still own a writing child, and an rmtree under a live writer is a
    worse defect than a leaked dir (the container's /tmp dies with the pod)."""
    td = tempfile.mkdtemp(prefix="monty-render-")
    keep = False
    try:
        yield Path(td)
    except BaseException as exc:
        if names := _unlanded_arms(exc):
            keep = True
            print(f"[render] leaking tmp {td}: arm(s) {', '.join(names)} never landed — deleting "
                  f"under a possibly-live child is the worse defect", file=sys.stderr, flush=True)
        raise
    finally:
        if not keep:
            shutil.rmtree(td, ignore_errors=True)


# One input's transfer cap per pool wave; download() already retries and resumes inside it.
_DL_WORKERS = 4
_DL_ARM_S = 600.0
_DL_TICK_S = 15.0


def _download_inputs(inputs, tmp: Path) -> dict[str, Path]:
    """Fetch every spec input concurrently, bounded and narrated; the mapping is built in spec order.
    A failed transfer raises (a master missing an input is not renderable) — unstarted arms are
    cancelled, RUNNING ones drained before the raise, never a teardown under a still-writing child."""
    if not inputs:
        return {}
    dests = {inp.id: tmp / inp.id.replace("/", "__") for inp in inputs}
    if len(set(dests.values())) != len(dests):
        # unique ids by contract, but the "/"→"__" flattening is not injective — and two parallel
        # writers on one path would race where the serial loop silently last-wrote
        clash = sorted({i for i in dests if list(dests.values()).count(dests[i]) > 1})
        raise RuntimeError(f"download: input ids collide after path flattening: {clash}")
    workers = min(_DL_WORKERS, len(inputs))
    waves = -(-len(inputs) // workers)
    stop = time.monotonic() + _DL_ARM_S * waves
    ex = cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dl")
    futs = {ex.submit(download, inp.url, dests[inp.id]): inp.id for inp in inputs}
    got: dict[str, Path] = {}
    pending = set(futs)
    try:
        while pending:
            left = stop - time.monotonic()
            if left <= 0:
                raise RuntimeError(
                    f"download: out of patience after {_DL_ARM_S * waves:.0f}s with "
                    + ", ".join(sorted(futs[f] for f in pending)) + " still out")
            done, pending = cf.wait(pending, timeout=min(_DL_TICK_S, left),
                                    return_when=cf.FIRST_COMPLETED)
            for f in done:
                got[futs[f]] = f.result(timeout=0)  # a failed transfer re-raises HERE: first fault wins
            if pending:
                print(f"[render] download: {len(got)}/{len(futs)} in hand, {len(pending)} still out, "
                      f"{max(0.0, stop - time.monotonic()):.0f}s of patience left",
                      file=sys.stderr, flush=True)
    except BaseException as exc:
        from . import render_onepass as _onepass
        for f in pending:
            f.cancel()
        if left := _onepass._drain("download", {f for f in pending if not f.cancelled()}, futs, stop):
            exc.unlanded_arms = left  # read by _job_tmpdir: leak the dir, never race a live child
        raise
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return {inp.id: got[inp.id] for inp in inputs}


def render_spec(spec: RenderSpec, cp: ControlPlane, corr_id: str | None = None,
                session_id: str | None = None) -> None:
    """Fetch inputs, run the single encode pass, PUT every non-cache output, report the event.

    corr_id/session_id are echoed from the claimed envelope onto the terminal (pool result demux)."""
    work_started = time.monotonic()
    common = {"job_id": spec.job_id, "session_id": session_id, "corr_id": corr_id, "stage": "render"}

    @contextmanager
    def phase(op: str, *, guarded: bool = False):
        # guarded=True is the post-spend path: nothing is worth protecting before the encode succeeds,
        # so only a phase entirely at-or-after that point may swallow its own event-send failure.
        base = {**common, "status": "step", "op": op}
        send = (lambda ev: _safe_send_event(cp, ev)) if guarded else cp.send_event
        send({k: v for k, v in {**base, "phase": f"{op}_started"}.items() if v is not None})
        started = time.monotonic()
        try:
            yield
        except BaseException as exc:
            send({
                k: v for k, v in {
                    **base,
                    "phase": f"{op}_error",
                    "outcome": "error",
                    "error_type": type(exc).__name__,
                    "error": safe_error(exc),
                    "timings": {"phase_s": round(time.monotonic() - started, 3)},
                }.items() if v is not None
            })
            raise
        else:
            send({
                k: v for k, v in {
                    **base,
                    "phase": f"{op}_finished",
                    "outcome": "ok",
                    "timings": {"phase_s": round(time.monotonic() - started, 3)},
                }.items() if v is not None
            })
    if spec.mode == "final":
        # function-local: render_onepass imports this module, so a top-level import would cycle.
        from . import render_onepass as _onepass
        _onepass.preflight(spec)  # refuses trims/opener/cover before any subprocess (the only door)
    else:
        _finalize.declared_grid(spec.timeline.fps)  # the ONLY refusal a lost render is worse than

    with phase("gpu_probe"):
        cpu_requested = os.environ.get("MONTY_OPS_CPU_ENCODE") == "1"
        motion_disabled = os.environ.get("MONTY_GPU_MOTION") == "0"
        deliberate_cpu = cpu_requested or motion_disabled
        gpu = not deliberate_cpu and _gpu_available()
    if not gpu and spec.motion is not None and spec.motion.segments:
        if deliberate_cpu:
            print("no NVENC: camera motion degrades to a static crop at the first keyframe", file=sys.stderr)
        elif os.environ.get("MONTY_ALLOW_STATIC_CAMERA") != "1":
            raise RuntimeError(
                "camera.apply: GPU (Vulkan+libplacebo+NVENC) unavailable — refusing a static-crop master; "
                "set MONTY_ALLOW_STATIC_CAMERA=1 to accept degradation")
        else:
            print("camera.apply: GPU unavailable — degrading to a static crop at the first keyframe "
                  "(MONTY_ALLOW_STATIC_CAMERA=1)", file=sys.stderr)

    with _job_tmpdir() as tmp:
        with phase("download"):
            input_paths = _download_inputs(spec.inputs, tmp)
        fin = spec.overlays.finalize if (spec.mode == "final" and spec.overlays is not None) else None

        if spec.mode == "final":
            # preflight already ran above; the one-pass graph is the ONLY final encode core.
            from . import render_onepass as _onepass
            prepared = _onepass.prepare(spec, input_paths, tmp, gpu, phase=phase)
            _onepass.run_encode(prepared, phase=phase)
            master = prepared.master_out
            # The pre-accent reference; its PUT below still gates on `fin` — guard_sync reads an
            # absent presync object as "the tail never ran" (contract), regardless of which core built it.
            presync = prepared.presync_out
            if fin is not None:
                with phase("finalize"):
                    master = _finalize.apply_loudnorm(fin, master, tmp / "fin_ln.mp4")
        else:
            # preview carries no overlays (models.py:423-424) — a single composite pass with no
            # audio pre-pass, no mograph/captions/cover, camera `motion` only.
            out = tmp / "render.mp4"
            cmd = build_command(spec, input_paths, out, gpu, (), None)
            with phase("ffmpeg"):
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as exc:
                    tail = (exc.stderr or b"")[-2000:]
                    detail = tail.decode("utf-8", "replace") if isinstance(tail, bytes) else str(tail)
                    raise RuntimeError(
                        f"ffmpeg exited {exc.returncode}: {safe_error(RuntimeError(detail))}") from exc
            master = out
            presync = master

        # The COMPOSITE is complete here; everything below is the delivery tail. `presync` pins this
        # exact frame: it is the video-identical reference the origin's A/V-sync guard measures the
        # finished master against, so the guard can attribute any drift to the tail and nothing else.
        # Header-only, on the paid-for master, before any PUT — a probe failure becomes a defect
        # entry (finalize.grid_verdict), never an exception that would cost the render.
        defect = _finalize.grid_verdict(master, spec.timeline.fps)
        if defect is not None:
            msg = f"declared vs measured mismatch: {defect}"
            print(f"[render] grid_verify: {msg}", file=sys.stderr)
            # outcome MUST stay "ok" with no error/error_type: the cabinet frontend keys "failed" off
            # outcome alone (progress.mjs isErrorActivity), and a delivered master is not a failure.
            _safe_send_event(cp, {k: v for k, v in {
                **common,
                "status": "step",
                "op": "grid_verify",
                "phase": "grid_verify_degraded",
                "outcome": "ok",
                "timings": {"grid_defect": defect},
            }.items() if v is not None})

        done: list[str] = []
        with phase("upload", guarded=True):
            for o in spec.outputs:
                if o.kind == "cache":
                    print(f"cache output {o.id!r} skipped (v1)", file=sys.stderr)
                    continue
                if o.kind == "cover":
                    # preflight refuses this on mode=final only; without the guard a preview spec
                    # declaring one would PUT the mp4 to the cover URL instead of refusing.
                    raise RuntimeError(
                        f"output {o.id!r} kind=cover has no producer "
                        "(the-cover-weld-arm-is-deleted-with-the-multipass-path)")
                if o.kind == "presync":
                    if fin is not None:
                        upload(presync, o.put_url, "video/mp4")
                        done.append(o.id)
                    continue
                upload(master, o.put_url, "video/mp4")
                done.append(o.id)

    terminal = {
        "job_id": spec.job_id,
        "stage": "render",
        "status": "done",
        "outputs": done,
    }
    if corr_id is not None:
        terminal["corr_id"] = corr_id
    if session_id is not None:
        terminal["session_id"] = session_id
    # Observability and wake-up are separate typed frames. An event may be replayed without ever creating a
    # second business result; the correlated result is the only terminal the awaiting brain consumes.
    _safe_send_event(cp, {
        **terminal,
        "phase": "work_finished",
        "outcome": "ok",
        "timings": {"work_s": round(time.monotonic() - work_started, 3)},
    })
    cp.send_result({
        "job_id": spec.job_id,
        "stage": "render",
        "status": "ok",
        "corr_id": corr_id,
        "outputs": done,
        **({"session_id": session_id} if session_id is not None else {}),
        **({"defects": [defect]} if defect is not None else {}),
    })
