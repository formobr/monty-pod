"""Render half of the pod: a fully-resolved RenderSpec becomes one ffmpeg pass, the result is
PUT back over presigned URLs. No decisions here — every number was fixed by the planner; this
module only translates those numbers into a filtergraph and an argv."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from .cp import ControlPlane, download, upload
from .models import MotionKeyframe, RenderSpec, SpecBrollClip, SpecTransition

_P_STYLE = re.compile(r"p\d+")  # NVENC preset names (p1..p7); libx264 can't take these
# delivery signal: TAG bt709 (no convert — untagged made platforms guess the colourspace)
_BT709 = ("-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709")


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
    """Input ids the MAIN filtergraph consumes (timeline srcs, broll clips, music, sfx), in spec.inputs
    order — an id's position is its ffmpeg -i index. Cover/caption ASSETS (fonts, logo) are downloaded
    but NOT decoded here (the cover/caption passes read them off disk), so a TTF never becomes a bad -i."""
    consumed: set[str] = {seg.src for seg in spec.timeline.segments}
    ov = spec.overlays
    if ov is not None:
        if ov.broll_final:
            consumed.update(c.clip for c in ov.broll_final.broll)
        if ov.music:
            consumed.add(ov.music.track)
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


def _broll_chains(spec: RenderSpec, idx: dict[str, int], base_label: str) -> list[str]:
    """Overlay every resolved cutaway onto [base_label] → [vout]: cover-crop to canvas (Ken Burns bake
    NOT reproduced — fidelity gap vs engine, FINAL_COMPOSITE_BUILD.md §C2), trim [in,in+dur], seat at
    `start`, ride authored slide/push (overlay x/y) or dissolve (alpha fade). Audio untouched."""
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
        chains.append(
            f"[{j}:v]trim=start={_num(c.in_ or 0.0)}:duration={_num(c.dur)},setpts=PTS-STARTPTS,"
            f"fps={_num(fps)},scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={w}:{h},setpts=PTS-STARTPTS+{start:.3f}/TB{frag}[b{i}]"
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


class _AudioMix(NamedTuple):
    voice_idx: int                            # ffmpeg input index of the base (raw voice)
    bed_idx: int | None                       # music bed input index, or None (no music)
    clean: str                                # voice pre-filter (highpass [+ afftdn if dirty])
    vln: str                                  # measured two-pass loudnorm to -20 LUFS
    dur: float
    sfx: tuple[tuple[int, float, float], ...] = ()   # (sound input index, start seconds, linear gain)


def _probe_dur(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip() or 0.0)


def _voice_is_dirty(voice: Path) -> bool:
    """Denoise gate: afftdn on a CLEAN voice adds musical-noise + dulls it (memory voice-audio-chain), so
    denoise ONLY a source whose post-highpass noise floor is above -50 dB."""
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(voice), "-map", "0:a?",
                          "-af", "highpass=f=80,astats=metadata=1:reset=0", "-f", "null", "-"],
                         capture_output=True, text=True)
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
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(voice),
                          "-af", f"{pre},{vln}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True)
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
                    str(norm)], check=True)
    bed = tmp / "music_bed.flac"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stream_loop", "-1",
                    "-ss", _num(mstart), "-i", str(norm), "-t", _num(dur), "-ar", "48000", "-ac", "2",
                    str(bed)], check=True)
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
        chains.append("[mx]alimiter=limit=0.84:attack=5:release=50:level=false[aout]")
    else:
        chains.append("[amaster]anull[aout]")
    return chains


def _music_of(spec: RenderSpec) -> "object | None":
    return spec.overlays.music if (spec.mode == "final" and spec.overlays is not None) else None


def build_filtergraph(spec: RenderSpec, gpu: bool, audio: _AudioMix | None = None) -> str:
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
    return ";".join(chains)


def build_command(
    spec: RenderSpec, input_paths: dict[str, Path], out_path: Path, gpu: bool,
    extra_inputs: tuple[Path, ...] = (), audio: _AudioMix | None = None,
) -> list[str]:
    """Pure: the full ffmpeg argv for this spec. gpu decides the codec at runtime — the spec's
    named encoder is only a hint; a CPU fallback overriding it is allowed mechanics. extra_inputs
    (the pre-rendered music bed) append after the spec inputs, at the indices `audio` references."""
    enc = spec.encode
    cmd = ["ffmpeg", "-y", "-hide_banner"]
    if gpu:
        cmd += ["-init_hw_device", "vulkan"]  # libplacebo runs on a Vulkan device; hwupload derives from it
    for iid in input_ids(spec):
        cmd += ["-i", str(input_paths[iid])]
    for extra in extra_inputs:
        cmd += ["-i", str(extra)]
    cmd += ["-filter_complex", build_filtergraph(spec, gpu, audio)]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    cmd += _venc(enc, gpu)
    cmd += ["-c:a", "aac", "-b:a", enc.audio_bitrate, "-movflags", "+faststart", str(out_path)]
    return cmd


def _burn_captions(caps, src: Path, input_paths: dict, out_path: Path, gpu: bool,
                   w: int, h: int, enc) -> None:
    """Burn the subtitle track onto `src` via libass (one re-encode, audio copied through)."""
    from .captions import build_ass
    if not caps.font or caps.font not in input_paths:
        raise RuntimeError("captions present but no resolved font input on the spec")
    font = input_paths[caps.font]
    words = [wd.model_dump() for wd in caps.words]
    ass = out_path.parent / "captions.ass"
    ass.write_text(build_ass(words, font=font, w=w, h=h, accent=caps.accent or "#d6ff3a",
                             center_y=caps.centerY if caps.centerY is not None else 0.76,
                             style=caps.style or "oneword"), encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-i", str(src),
           "-vf", f"subtitles={ass}:fontsdir={font.parent}"]
    cmd += _venc(enc, gpu) + ["-c:a", "copy", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-2000:]
        detail = tail.decode("utf-8", "replace") if isinstance(tail, bytes) else str(tail)
        raise RuntimeError(f"caption burn ffmpeg exited {exc.returncode}: {detail}") from exc


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
    if spec.mode == "final" and spec.overlays is not None and spec.overlays.trims:
        raise NotImplementedError("final overlay(s) not yet composited on the pod: ['trims']")

    gpu = _gpu_available()
    if not gpu and spec.motion is not None and spec.motion.segments:
        print("no NVENC: camera motion degrades to a static crop at the first keyframe",
              file=sys.stderr)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        input_paths = {
            inp.id: download(inp.url, tmp / inp.id.replace("/", "__")) for inp in spec.inputs
        }
        # music: pre-measure the voice loudnorm + pre-render the -33 LUFS bed (own ffmpeg passes), then
        # the main pass mixes voice+bed instead of copying segment audio (locked add_music.sh chain).
        extra_inputs: tuple[Path, ...] = ()
        audio: _AudioMix | None = None
        music = _music_of(spec)
        sfx = spec.overlays.sfx if (spec.mode == "final" and spec.overlays is not None) else None
        if music is not None or sfx:
            ids = input_ids(spec)
            voice = input_paths[spec.timeline.segments[0].src]
            dur = _probe_dur(voice)
            clean = "highpass=f=80" + (",afftdn=nr=8:nf=-30" if _voice_is_dirty(voice) else "")
            vln = _measure_loudnorm(voice, clean)
            bed_idx: int | None = None
            if music is not None:
                extra_inputs = (_prerender_bed(input_paths[music.track], music.start, dur, tmp),)
                bed_idx = len(ids)
            sfx_tuples = tuple((ids.index(s.sound), s.at, s.gain) for s in (sfx or []))
            audio = _AudioMix(voice_idx=ids.index(spec.timeline.segments[0].src), bed_idx=bed_idx,
                              clean=clean, vln=vln, dur=dur, sfx=sfx_tuples)
        out = tmp / "render.mp4"
        cmd = build_command(spec, input_paths, out, gpu, extra_inputs, audio)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            tail = (exc.stderr or b"")[-2000:]
            detail = tail.decode("utf-8", "replace") if isinstance(tail, bytes) else str(tail)
            raise RuntimeError(f"ffmpeg exited {exc.returncode}: {detail}") from exc

        master = out
        _mp = spec.overlays.motion_plan if (spec.mode == "final" and spec.overlays is not None) else None
        # mograph overlays first (under the captions), then captions, then the cover weld.
        if _mp is not None and _mp.sections:
            from .mograph import composite
            master = composite(_mp, master, input_paths, tmp / "render_mograph.mp4", gpu, spec.encode, tmp)

        # captions burn (libass) BEFORE the cover weld — the subtitle track covers the whole body.
        caps = _mp.captions if _mp is not None else None
        if caps is not None and caps.words:
            captioned = tmp / "render_caps.mp4"
            _burn_captions(caps, master, input_paths, captioned, gpu,
                           spec.timeline.width, spec.timeline.height, spec.encode)
            master = captioned

        # cover is the LAST step: extract the base frame, compose the still, weld it onto the master tail.
        cover = spec.overlays.cover if (spec.mode == "final" and spec.overlays is not None) else None
        if cover is not None:
            from .cover import render_cover
            welded = tmp / "render_cover.mp4"
            render_cover(cover.model_dump(by_alias=True), input_paths[spec.timeline.segments[0].src],
                         master, input_paths, welded, gpu, spec.timeline.width, spec.timeline.height)
            master = welded

        done: list[str] = []
        for o in spec.outputs:
            if o.kind == "cache":
                print(f"cache output {o.id!r} skipped (v1)", file=sys.stderr)
                continue
            upload(master, o.put_url, "video/mp4")
            done.append(o.id)

    cp.post_event({
        "job_id": spec.job_id,
        "stage": "render",
        "status": "done",
        "outputs": done,
    })
