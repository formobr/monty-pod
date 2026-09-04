"""Delivery-tail BUILDERS: pure filtergraph fragments (body logo, watermark) plus the grid/level
steps (declared_grid, grid_verdict, apply_loudnorm) that render_onepass.assemble folds into its ONE
encode. Brand-specific values (logo/sting/idle assets, geometry, levels) all arrive as spec data."""
from __future__ import annotations

import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path

from .sanitize import safe_error

# bt709 SIGNAL (tag, no convert) — an untagged master makes platforms GUESS the colourspace.
_BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
_BT709_SET_PARAMS = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"
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


def _run(cmd: list[str], what: str, *, timeout_s: int | None = None) -> None:
    try:
        if timeout_s is None:
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_s)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-2000:]
        detail = tail.decode("utf-8", "replace") if isinstance(tail, bytes) else str(tail)
        raise RuntimeError(f"{what} ffmpeg exited {exc.returncode}: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{what} ffmpeg timed out after {timeout_s}s") from exc


_PROBE_TIMEOUT_S = 20  # header-only ffprobe; matches the engine's own precedent (scripts/tag_music.py:_duration)


def _probe(path: Path) -> tuple[int, int, float, int, int, float]:
    """(width, height, fps, fps_num, fps_den, duration) of the master."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True, timeout=_PROBE_TIMEOUT_S).stdout.split()
    w, h = int(out[0]), int(out[1])
    num, den = out[2].split("/")
    fps = float(num) / float(den or 1)
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True, timeout=_PROBE_TIMEOUT_S).stdout.strip()
    return w, h, fps, int(num), int(den), float(dur)


def _grid_rate(fps_num: int, fps_den: int) -> str:
    """Exact rational for -r, built from the probe's num/den — routing it through the rounded float
    `fps` first would reintroduce the drift this declaration exists to kill."""
    return str(Fraction(fps_num, fps_den))


_DELIVERY_SAMPLE_RATE = 48000
_AV_DELTA_FRAME_TOLERANCE = 2.0


# Keyed by the CONVENTIONAL literal (29.97), not the rational's own float — 29.9701 is ~2e-6 relative
# off 29.97, far past the tolerance below, so a genuinely different rate keeps its own exact conversion.
_NTSC_DROP_FRAME = ((29.97, Fraction(30000, 1001)), (23.976, Fraction(24000, 1001)),
                    (59.94, Fraction(60000, 1001)))
_NTSC_SNAP_REL_TOL = 1e-9


def declared_grid(fps: float) -> str:
    """The exact -r rational for a DECLARED timeline.fps. Absent/zero/NaN/infinite cannot become a
    grid, so it refuses HERE (before any subprocess) rather than negotiating one downstream."""
    if fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"timeline.fps must be a finite, positive number, got {fps!r}")
    for alias, std in _NTSC_DROP_FRAME:
        if math.isclose(fps, alias, rel_tol=_NTSC_SNAP_REL_TOL):
            return str(std)
    return str(Fraction(fps).limit_denominator(1001))


def _has_audio(path: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def _probe_audio(path: Path) -> tuple[int, float]:
    """(sample_rate, duration) of the first audio stream — header-only, same cost class as _probe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True, timeout=_PROBE_TIMEOUT_S).stdout.split()
    return int(out[0]), float(out[1])


def grid_verdict(master: Path, declared_fps: float) -> dict | None:
    """Header-only: measured master grid vs the DECLARED one. None on a clean match (zero-cost case),
    else a machine-readable defect record — NEVER raises, ANY exception below becomes a defect."""
    try:
        return _grid_verdict(master, declared_fps)
    except Exception as exc:
        return {"probe_failed": safe_error(exc)}


def _grid_verdict(master: Path, declared_fps: float) -> dict | None:
    declared = declared_grid(declared_fps)
    _w, _h, _fps, m_num, m_den, v_dur = _probe(master)
    defect: dict = {}
    measured = _grid_rate(m_num, m_den)
    if measured != declared:
        defect["video_rate"] = {"declared": declared, "measured": measured}
    if _has_audio(master):
        try:
            a_rate, a_dur = _probe_audio(master)
        except Exception as exc:
            defect["audio_probe_failed"] = safe_error(exc)
        else:
            if a_rate != _DELIVERY_SAMPLE_RATE:
                defect["audio_rate"] = {"declared": _DELIVERY_SAMPLE_RATE, "measured": a_rate}
            tolerance_ms = _AV_DELTA_FRAME_TOLERANCE * 1000.0 / declared_fps
            delta_ms = abs(v_dur - a_dur) * 1000.0
            if delta_ms > tolerance_ms:
                defect["av_duration_delta_ms"] = round(delta_ms, 1)
    return defect or None


def _terminal_bt709(fc: str, out_v: str) -> str:
    """Stamp frame metadata on the actual video pad consumed by the encoder."""
    return f"{fc};[{out_v}]{_BT709_SET_PARAMS}[{out_v}]"


# --- 1. persistent body logo --------------------------------------------------

def body_logo_filter(corner: str, width: int, opacity: float, margin: int, body_end: float, *,
                     base_v: str = "0:v", logo_v: str = "1:v", out_v: str = "vout") -> str:
    """Persistent corner logo over the BODY only (t < body_end); the cover end-card carries its own."""
    # The three labels are parameters for the same reason watermark_filter's are: in a merged graph
    # input 1 is another timeline SOURCE, so a hardcoded [1:v] alpha-blends a video clip as the "logo".
    x = f"W-w-{margin}" if corner in ("tr", "br") else f"{margin}"
    y = f"H-h-{margin}" if corner in ("bl", "br") else f"{margin}"
    return (f"[{logo_v}]format=rgba,colorchannelmixer=aa={opacity},scale={width}:-1:flags=lanczos[lg];"
            f"[{base_v}][lg]overlay={x}:{y}:enable='lt(t,{body_end:.3f})'[{out_v}]")


# --- 2. animated watermark ----------------------------------------------------

def watermark_filter(*, base_v: str, sting_v: str, idle_v: str, width: int, overlay_xy: str,
                     base_a: str | None, chime_a: str | None, chime_vol: float, delay: float,
                     grid: str, sample_rate: int,
                     out_v: str = "wmv", out_a: str = "wma") -> tuple[str, str, str | None]:
    """The watermark filtergraph, parameterised by input LABELS so it composes into any filter_complex.
    The sting uses the original 1200x600 canvas so the spring overshoot is not clipped; the shorter
    idle is padded, anchored left. `grid`/`sample_rate` conform the 60fps/44.1kHz brand assets."""
    normalize = f"pad={WM_CANVAS_W}:{WM_CANVAS_H}:0:(oh-ih)/2:color=black@0,fps={grid}"
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
        ops = [f"aresample={sample_rate}"]
        if abs(chime_vol - 1.0) > 1e-3:
            ops.append(f"volume={chime_vol}")
        if delay > 0:
            ms = int(delay * 1000)
            ops.append(f"adelay={ms}|{ms}")
        ch = f"[{chime_a}]{','.join(ops)}[chm];"
        cha = "[chm]"
        if base_a is not None:
            f += f";{ch}[{base_a}]{cha}amix=inputs=2:duration=first:dropout_transition=0:normalize=0[{out_a}]"
        else:
            f += f";{ch}{cha}apad[{out_a}]"
        ret_a = out_a
    return f, out_v, ret_a


# --- 3. delivery loudness -----------------------------------------------------

# loudnorm's TP is a PREDICTION (it drops to dynamic mode silently when the linear gain would breach it),
# so the alimiter below holds the ceiling; this headroom buys only the AAC encode's ~0.9 dBTP of codec peaks.
TP_HEADROOM_DB = 1.2
_LIMITER_ATTACK_MS = 5   # same transparent brickwall the engine's weld uses (scripts/montyops/edit_weld.py)
_LIMITER_RELEASE_MS = 50


def limiter_af(tp_aim: float) -> str:
    """Brickwall at the TP aim. `limit` is linear amplitude, not dB; level=false keeps it a pure ceiling,
    so a master already under the aim comes out gain-identical."""
    return (f"alimiter=limit={round(10 ** (tp_aim / 20.0), 4)}"
            f":attack={_LIMITER_ATTACK_MS}:release={_LIMITER_RELEASE_MS}:level=false")


def master_af(mv: dict, target: float, tp_aim: float, attenuate_only: bool) -> tuple[str | None, str]:
    """Delivery filter for the MEASURED master. Clean -> normalize to target; a source the planner
    flagged as clipping-hot and already at/under target -> ship as-is (boosting a hot mic only
    amplifies the crackle). Returns (af|None, note)."""
    in_i = float(mv["input_i"])
    if attenuate_only and in_i <= target:
        return None, f"hot source, {in_i} LUFS <= target — shipped clean, no boost (crackle guard)"
    in_tp, in_lra = float(mv["input_tp"]), float(mv["input_lra"])
    # a non-finite measured_I is bit-exact silence across the whole master, not a quiet one — ffmpeg's loudnorm refuses it anyway, so name it here instead of the caller's generic "couldn't apply" fallback.
    if not (math.isfinite(in_i) and math.isfinite(in_tp) and math.isfinite(in_lra)):
        raise RuntimeError(
            f"master measure: no signal across the delivered master's measured span "
            f"(input_i={mv['input_i']!r} input_tp={mv['input_tp']!r} input_lra={mv['input_lra']!r}) — "
            f"refusing to feed a non-finite measured_I into the delivery loudnorm filter")
    af = (f"loudnorm=I={target}:TP={tp_aim}:LRA=11:linear=true:measured_I={mv['input_i']}"
          f":measured_TP={mv['input_tp']}:measured_LRA={mv['input_lra']}:measured_thresh={mv['input_thresh']}"
          f",{limiter_af(tp_aim)}")
    verb = "attenuate" if in_i > target else "normalize"
    return af, f"{in_i} -> {target} LUFS ({verb}{', hot-guarded' if attenuate_only else ''})"


def apply_loudnorm(fin, src: Path, out: Path) -> Path:
    """Two-pass loudnorm + brickwall to the brand's delivery target, audio-gain only (video copied).
    A measurement that cannot be parsed leaves the master at source level rather than failing a finished
    render; a master measured OVER the ceiling raises — shipping it silently is what the box refuses."""
    ln = fin.loudnorm
    if ln is None:
        return src
    tp_aim = round(ln.tp - TP_HEADROOM_DB, 2)
    mv = _measure(src, ln, tp_aim)
    if mv is None:
        print("[finalize] master loudnorm: couldn't measure -> left at source level")
        return src
    af, note = master_af(mv, ln.i, tp_aim, ln.attenuate_only)
    print(f"[finalize] master: {note}")
    if af is None:
        return src
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
                        "-af", af, "-ar", "48000", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        str(out), "-y"],
                       capture_output=True)
    if r.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        print(f"[finalize] master loudnorm exit {r.returncode} -> left at source level")
        return src
    return _verdict(out, ln, tp_aim)


def _measure(path: Path, ln, tp_aim: float) -> dict | None:
    """One loudnorm print_format=json pass. `-map 0:a:0?`: unmapped, `-f null` decodes the whole VIDEO for
    an audio measure; on a silent master ffmpeg still exits nonzero, so rc is unchecked and None says it."""
    meas = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-map", "0:a:0?",
         "-af", f"loudnorm=I={ln.i}:TP={tp_aim}:LRA={ln.lra}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas.stderr, re.S)
    return json.loads(m.group(0)) if m else None


def _verdict(out: Path, ln, tp_aim: float) -> Path:
    """Measures what was actually ENCODED. The box's check_master sidecar is swept before anyone reads it,
    so this line is the durable proof; over the ceiling means the delivery chain failed and must not ship."""
    mv = _measure(out, ln, tp_aim)
    try:
        i, tp, lra = float(mv["input_i"]), float(mv["input_tp"]), float(mv["input_lra"])
    except (TypeError, KeyError, ValueError):
        print("[finalize] master: delivered level UNVERIFIED — post-encode measure unreadable")
        return out
    if tp > ln.tp:
        raise RuntimeError(
            f"[finalize] master: OFF-CONTRACT after limiter tp={tp} ceil={ln.tp} "
            f"(aim {tp_aim}, delivered lufs={i} lra={lra}) — refusing to ship a clipping master")
    print(f"[finalize] master: delivered lufs={i} tp={tp} lra={lra}")
    return out
