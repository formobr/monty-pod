"""Delivery tail — the LAST thing that happens to a master, on whichever box rendered it.

After the composite (segments + b-roll + mograph + captions + cover weld) the master still has to be
accented, branded and levelled before it is a deliverable:

  1. frame accents  — the Director's resolved zoom_punch / camera_shake / glitch / grain / … chained
                      into ONE pass over the body (podagent.accents).
  2. body logo      — the persistent corner logo, over the BODY only; the cover end-card carries its
                      own logo, so the overlay is disabled for the last `cover_hold` seconds.
  3. watermark      — the animated brand sting -> idle loop, plus its chime mixed once into the audio.
  4. loudnorm       — two-pass loudnorm to the brand's delivery target, as the LAST step, so every
                      video ships at one level.

Order is load-bearing: accents first (they re-slice the picture, and must not smear the static
overlays), then the overlays, then the level. This mirrors the order the engine used when this tail
still ran on the origin, which is what keeps the two transports' masters comparable.

Everything brand-specific arrives as data: the logo/sting/idle are `inputs[]` ids resolved by the
planner, every geometry and level is a number on `overlays.finalize`. The pod holds no brand profile
and reads none — `brands/` and `effects/` do not exist here.

Encoder settings are NOT on the contract. They are tuning, and tuning stays in the handler (the same
rule render_profile.py follows on the planner side). The values below are the delivery chain's own,
deliberately not `spec.encode`: the finalize passes have always run at their own quality, and the
watermark pass is the TERMINAL full-frame encode of the master, so it sets the deliverable's
bitrate/colour signal.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import accents as _accents

# Intermediate finalize passes: cq/crf 16, matching the engine's fx/logo chain.
_MID_GPU = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-cq", "16"]
_MID_CPU = ["-c:v", "libx264", "-preset", "medium", "-crf", "16"]
# bt709 SIGNAL (tag, no convert) — an untagged master makes platforms GUESS the colourspace.
_BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
# TERMINAL encode (the watermark pass): cq14 + unclamped maxrate/bufsize so busy frames aren't
# starved, since the platform re-compresses whatever we ship.
_FINAL_GPU = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-cq", "14",
              "-maxrate", "24M", "-bufsize", "32M", "-pix_fmt", "yuv420p", *_BT709]
_FINAL_CPU = ["-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", *_BT709]

_POS = {
    "bottom-center": "(W-w)/2:H-h-{m}",
    "bottom-right":  "W-w-{m}:H-h-{m}",
    "bottom-left":   "{m}:H-h-{m}",
    "top-center":    "(W-w)/2:{m}",
    "top-right":     "W-w-{m}:{m}",
    "top-left":      "{m}:{m}",
    "center":        "(W-w)/2:(H-h)/2",
}
WM_CANVAS_W = 1200
WM_CANVAS_H = 600


def _run(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-2000:]
        detail = tail.decode("utf-8", "replace") if isinstance(tail, bytes) else str(tail)
        raise RuntimeError(f"{what} ffmpeg exited {exc.returncode}: {detail}") from exc


def _probe(path: Path) -> tuple[int, int, float, float]:
    """(width, height, fps, duration) of the master — the accent macros need the real canvas + rate."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True).stdout.split()
    w, h = int(out[0]), int(out[1])
    num, den = out[2].split("/")
    fps = float(num) / float(den or 1)
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return w, h, fps, float(dur)


def _has_audio(path: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


# --- 1. frame accents ---------------------------------------------------------

def apply_accents(fin, src: Path, out: Path, gpu: bool) -> Path:
    """Chain the resolved frame-accents over the FULL master in one pass. No accents -> `src` unchanged."""
    if not fin.accents:
        return src
    w, h, fps, _ = _probe(src)
    fc = _accents.build_chain_filter(fin.accents, fps=fps, w=w, h=h, gpu=gpu)
    if fc is None:
        return src
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           *(["-init_hw_device", "vulkan"] if gpu else []), "-i", str(src),
           "-filter_complex", fc, "-map", "[vout]", "-map", "0:a?",
           *(_MID_GPU if gpu else _MID_CPU),
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out)]
    _run(cmd, "frame accents")
    return out


# --- 2. persistent body logo --------------------------------------------------

def body_logo_filter(corner: str, width: int, opacity: float, margin: int, body_end: float) -> str:
    """Persistent corner logo over the BODY only (t < body_end); the cover end-card carries its own."""
    x = f"W-w-{margin}" if corner in ("tr", "br") else f"{margin}"
    y = f"H-h-{margin}" if corner in ("bl", "br") else f"{margin}"
    return (f"[1:v]format=rgba,colorchannelmixer=aa={opacity},scale={width}:-1:flags=lanczos[lg];"
            f"[0:v][lg]overlay={x}:{y}:enable='lt(t,{body_end:.3f})'[vout]")


def apply_logo(fin, src: Path, out: Path, input_paths: dict, gpu: bool) -> Path:
    """Bake the brand's persistent corner logo onto the master BODY. `logo` absent (partner /
    --no-logo) -> `src` unchanged."""
    logo = fin.logo
    if logo is None:
        return src
    asset = input_paths.get(logo.asset)
    if asset is None:
        raise RuntimeError(f"finalize.logo.asset {logo.asset!r} is not a resolved inputs[] id")
    # body_end is derived from the master the pod itself just welded, not guessed upstream: the cover
    # tail is exactly `cover_hold` s long, and only this box knows the welded master's real duration.
    body_end = max(0.0, _probe(src)[3] - logo.cover_hold)
    fc = body_logo_filter(logo.corner, logo.width, logo.opacity, logo.margin, body_end)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-i", str(asset),
           "-filter_complex", fc, "-map", "[vout]", "-map", "0:a?",
           *(_MID_GPU if gpu else _MID_CPU),
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out)]
    _run(cmd, "body logo")
    return out


# --- 3. animated watermark ----------------------------------------------------

def watermark_filter(*, base_v: str, sting_v: str, idle_v: str, width: int, overlay_xy: str,
                     base_a: str | None, chime_a: str | None, chime_vol: float, delay: float,
                     out_v: str = "wmv", out_a: str = "wma") -> tuple[str, str, str | None]:
    """The watermark filtergraph, parameterised by input LABELS so it composes into any filter_complex.
    The sting uses the original 1200x600 canvas so the spring overshoot is not clipped; the shorter
    idle is padded to that canvas before concat, anchored left so the animation is preserved."""
    normalize = f"pad={WM_CANVAS_W}:{WM_CANVAS_H}:0:(oh-ih)/2:color=black@0"
    f = (f"[{sting_v}]{normalize},setpts=PTS-STARTPTS[i];"
         f"[{idle_v}]{normalize},setpts=PTS-STARTPTS[d];"
         "[i][d]concat=n=2:v=1:a=0[wm0];"
         f"[wm0]scale={width}:-1[wm]")
    if delay > 0:
        f += f";[wm]setpts=PTS+{delay}/TB[wmd]"
        f += f";[{base_v}][wmd]overlay={overlay_xy}:enable='gte(t,{delay})':shortest=1:format=auto[{out_v}]"
    else:
        f += f";[{base_v}][wm]overlay={overlay_xy}:shortest=1:format=auto[{out_v}]"
    ret_a: str | None = None
    if chime_a is not None:
        ops = []
        if abs(chime_vol - 1.0) > 1e-3:
            ops.append(f"volume={chime_vol}")
        if delay > 0:
            ms = int(delay * 1000)
            ops.append(f"adelay={ms}|{ms}")
        ch = f"[{chime_a}]{','.join(ops)}[chm];" if ops else ""
        cha = "[chm]" if ch else f"[{chime_a}]"
        if base_a is not None:
            f += f";{ch}[{base_a}]{cha}amix=inputs=2:duration=first:dropout_transition=0:normalize=0[{out_a}]"
        else:
            f += f";{ch}{cha}apad[{out_a}]"
        ret_a = out_a
    return f, out_v, ret_a


def apply_watermark(fin, src: Path, out: Path, input_paths: dict, gpu: bool) -> Path:
    """Overlay the animated brand watermark (+ chime). `watermark` absent (paid tier / --no-watermark)
    -> `src` unchanged.

    The .webm alpha lives in a separate VP9 stream that ONLY the `libvpx-vp9` decoder extracts —
    ffmpeg's default native `vp9` decoder drops it and the mark renders as a BLACK BOX. Each webm
    input is therefore forced with `-c:v libvpx-vp9`."""
    wm = fin.watermark
    if wm is None:
        return src
    for ref in (wm.sting, wm.idle):
        if ref not in input_paths:
            raise RuntimeError(f"finalize.watermark asset {ref!r} is not a resolved inputs[] id")
    sting, idle = input_paths[wm.sting], input_paths[wm.idle]
    has_audio = _has_audio(src)
    overlay_xy = (f"{wm.x}:{wm.y}" if wm.x is not None and wm.y is not None
                  else _POS[wm.position].format(m=wm.margin))
    fc, out_v, out_a = watermark_filter(
        base_v="0:v", sting_v="1:v", idle_v="2:v", width=wm.width, overlay_xy=overlay_xy,
        base_a=("0:a" if has_audio else None), chime_a=("1:a" if wm.chime else None),
        chime_vol=wm.chime_volume, delay=wm.delay, out_v="v", out_a="a")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
           "-c:v", "libvpx-vp9", "-i", str(sting),
           "-c:v", "libvpx-vp9", "-stream_loop", "-1", "-i", str(idle),
           "-filter_complex", fc, "-map", f"[{out_v}]"]
    cmd += ["-map", f"[{out_a}]"] if out_a else ["-an"]
    if not has_audio:
        cmd += ["-t", f"{_probe(src)[3]:.3f}"]
    cmd += [*(_FINAL_GPU if gpu else _FINAL_CPU),
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
    _run(cmd, "watermark")
    return out


# --- 4. delivery loudness -----------------------------------------------------

# loudnorm's linear-mode TP is a PREDICTION, not a brickwall (it overshoots ~0.2 dB), so the delivery pass
# aims this far UNDER the declared ceiling. Named, not inline: it has no engine mirror to drift against, so
# a test pins it directly (test_finalize_parity) — otherwise editing it re-levels every master, silently.
TP_HEADROOM_DB = 0.7


def master_af(mv: dict, target: float, tp_aim: float, attenuate_only: bool) -> tuple[str | None, str]:
    """Delivery filter for the MEASURED master. Clean -> normalize to target; a source the planner
    flagged as clipping-hot and already at/under target -> ship as-is (boosting a hot mic only
    amplifies the crackle). Returns (af|None, note)."""
    in_i = float(mv["input_i"])
    if attenuate_only and in_i <= target:
        return None, f"hot source, {in_i} LUFS <= target — shipped clean, no boost (crackle guard)"
    af = (f"loudnorm=I={target}:TP={tp_aim}:LRA=11:linear=true:measured_I={mv['input_i']}"
          f":measured_TP={mv['input_tp']}:measured_LRA={mv['input_lra']}:measured_thresh={mv['input_thresh']}")
    verb = "attenuate" if in_i > target else "normalize"
    return af, f"{in_i} -> {target} LUFS ({verb}{', hot-guarded' if attenuate_only else ''})"


def apply_loudnorm(fin, src: Path, out: Path) -> Path:
    """Two-pass loudnorm to the brand's delivery target, audio-gain only (A/V untouched, video copied).
    Non-critical by contract: a measurement that cannot be parsed leaves the master at source level
    rather than failing a finished render."""
    ln = fin.loudnorm
    if ln is None:
        return src
    tp_aim = round(ln.tp - TP_HEADROOM_DB, 2)
    meas = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
         "-af", f"loudnorm=I={ln.i}:TP={tp_aim}:LRA={ln.lra}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas.stderr, re.S)
    if not m:
        print("[finalize] master loudnorm: couldn't measure -> left at source level")
        return src
    af, note = master_af(json.loads(m.group(0)), ln.i, tp_aim, ln.attenuate_only)
    print(f"[finalize] master: {note}")
    if af is None:
        return src
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
                        "-af", af, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out), "-y"],
                       capture_output=True)
    if r.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        print(f"[finalize] master loudnorm exit {r.returncode} -> left at source level")
        return src
    return out


# --- the tail -----------------------------------------------------------------

def finalize(fin, master: Path, input_paths: dict, tmp: Path, gpu: bool) -> Path:
    """Run the delivery tail over `master`, returning the path to the finished deliverable.

    Each step returns its input unchanged when its block is absent, so a spec that carries only some
    of the tail (a partner deliverable with no logo and no watermark) walks the same code path — the
    two transports cannot diverge on which steps they honour, because there is only one of them."""
    out = apply_accents(fin, master, tmp / "fin_accents.mp4", gpu)
    out = apply_logo(fin, out, tmp / "fin_logo.mp4", input_paths, gpu)
    out = apply_watermark(fin, out, tmp / "fin_wm.mp4", input_paths, gpu)
    return apply_loudnorm(fin, out, tmp / "fin_ln.mp4")
