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
# TERMINAL encode (watermark pass): master rung, capped at the h1080-class ceiling — a pinned mirror of
# registry/encode.yaml `master` (no registry/ on this package). NVENC cq 19 ~= x264 crf 21 (Codex condition).
_FINAL_GPU = ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-cq", "19",
              "-rc", "vbr", "-b:v", "0", "-maxrate", "12M", "-bufsize", "24M",
              "-pix_fmt", "yuv420p", *_BT709]
_FINAL_CPU = ["-c:v", "libx264", "-crf", "21", "-preset", "medium",
              "-maxrate", "12M", "-bufsize", "24M", "-pix_fmt", "yuv420p", *_BT709]

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


def _run(cmd: list[str], what: str, *, timeout_s: int | None = None,
         check: bool = True, text: bool = False) -> subprocess.CompletedProcess:
    """Every ffmpeg wait in this module goes through here: a wedge or a crash names its STEP instead of
    surfacing as a bare TimeoutExpired. `check=False` keeps the passes whose own rc/parse is the verdict."""
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=text, timeout=timeout_s)
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

# loudnorm's TP is a PREDICTION it abandons (dynamic mode) rather than breach, so the alimiter below holds
# the ceiling; this headroom buys only the AAC encode's ~0.9 dBTP of codec peaks over limited PCM.
TP_HEADROOM_DB = 1.2
# slower than the weld's 5/50: a fast brickwall FLAT-TOPS the waveform, and an aac encode of flat tops
# overshot 2.4 dBTP in prod (job 06c0d02c) — well past any headroom this module could carry.
_LIMITER_ATTACK_MS = 15
_LIMITER_RELEASE_MS = 100
# alimiter only sees SAMPLE peaks; the true peak between two samples is what the encoder reconstructs, so
# the ceiling is applied 4x oversampled and brought back to the delivery rate afterwards.
_LIMITER_OVERSAMPLE_HZ = 192000
_MAX_DELIVERY_PASSES = 2
_RETRY_MARGIN_DB = 0.3
# the ceiling is the AIM of the loop; only real clipping (and an off-band level) may refuse a finished
# render, so the ceiling..0 dBTP window ships with a note instead of blocking the user.
_CLIP_TP_DBTP = 0.0
_LUFS_TOL_DB = 3.0       # the band the box's own check_master judges the delivered master by
# One full-length audio decode/encode runs far above realtime (render.py's _AUDIO_PASS_WALL_S, same class);
# a level pass that does not is wedged, and a wedge must fail loud rather than absorb the stage.
_AUDIO_PASS_WALL_S = 300
_MAKEUP_TRIGGER_LU = 1.0
# a residual this large means the integrated loudness was carried by transients the ceiling just removed;
# giving it back would only feed the limiter again, so it is a refusal, not a louder retry.
_MAKEUP_MAX_DB = 12.0


def limiter_af(tp_aim: float) -> str:
    """Brickwall at the TP aim. `limit` is linear amplitude, not dB; level=false keeps it a pure ceiling,
    so a master already under the aim comes out gain-identical."""
    return (f"aresample={_LIMITER_OVERSAMPLE_HZ},"
            f"alimiter=limit={round(10 ** (tp_aim / 20.0), 4)}"
            f":attack={_LIMITER_ATTACK_MS}:release={_LIMITER_RELEASE_MS}:level=false,"
            f"aresample={_DELIVERY_SAMPLE_RATE}")


def master_af(mv: dict, target: float, tp_aim: float, attenuate_only: bool) -> tuple[str | None, str]:
    """Delivery filter for the MEASURED master. Clean -> normalize to target; a source the planner
    flagged as clipping-hot and already at/under target -> ship as-is (boosting a hot mic only
    amplifies the crackle). Returns (af|None, note)."""
    in_i = float(mv["input_i"])
    # only INSIDE the band: a hot-flagged source that is also far under target would be refused by the very
    # box gate this shortcut skips, and the brickwall handles its crackle better than that refusal does
    if attenuate_only and target - _LUFS_TOL_DB <= in_i <= target:
        return None, f"hot source, {in_i} LUFS inside the band — shipped clean, no boost (crackle guard)"
    in_tp, in_lra = float(mv["input_tp"]), float(mv["input_lra"])
    # a non-finite measured_I is bit-exact silence across the whole master, not a quiet one — ffmpeg's loudnorm refuses it anyway, so name it here instead of the caller's generic "couldn't apply" fallback.
    if not (math.isfinite(in_i) and math.isfinite(in_tp) and math.isfinite(in_lra)):
        raise RuntimeError(
            f"master measure: no signal across the delivered master's measured span "
            f"(input_i={mv['input_i']!r} input_tp={mv['input_tp']!r} input_lra={mv['input_lra']!r}) — "
            f"refusing to feed a non-finite measured_I into the delivery loudnorm filter")
    gain = target - in_i
    pred_tp = in_tp + gain
    if pred_tp <= tp_aim:
        af = (f"loudnorm=I={target}:TP={tp_aim}:LRA=11:linear=true:measured_I={mv['input_i']}"
              f":measured_TP={mv['input_tp']}:measured_LRA={mv['input_lra']}"
              f":measured_thresh={mv['input_thresh']},{limiter_af(tp_aim)}")
        how = "linear loudnorm"
    else:
        # ffmpeg would DOWNGRADE this chain to dynamic mode (short of target, transients unhonoured), so
        # the gain is ours and the brickwall — not loudnorm's prediction — absorbs the overshoot.
        af = f"volume={gain:.2f}dB,{limiter_af(tp_aim)}"
        how = f"volume+limiter (linear infeasible: pred_tp {pred_tp:.2f} > aim {tp_aim})"
    verb = "attenuate" if in_i > target else "normalize"
    return af, (f"{in_i} -> {target} LUFS ({verb}{', hot-guarded' if attenuate_only else ''}) "
                f"via {how}")


def apply_loudnorm(fin, src: Path, out: Path) -> Path:
    """Level the master's audio in PCM, then mux it back over the copied video as ONE aac encode.
    A measurement that cannot be parsed leaves the master at source level rather than failing a finished
    render; a delivered master off the brand's band (too loud OR too quiet) raises instead of shipping."""
    ln = fin.loudnorm
    if ln is None:
        return src
    aim = round(ln.tp - TP_HEADROOM_DB, 2)
    mv = _measure(src, ln, aim)
    if mv is None:
        print("[finalize] master loudnorm: couldn't measure -> left at source level")
        return src
    for attempt in range(1, _MAX_DELIVERY_PASSES + 1):
        af, note = master_af(mv, ln.i, aim, ln.attenuate_only)
        print(f"[finalize] master: {note}")
        if af is None:
            return src
        if not _encode(src, out, af, ln, aim):
            return src
        got = _verdict(out, ln, aim, attempt)
        if got is None:
            return out
        i, tp, lra = got
        over = round(tp - ln.tp, 2)
        off_band = abs(i - ln.i) > _LUFS_TOL_DB
        if over <= 0 and not off_band:
            return out
        # a quiet master is the make-up's business and it already ran capped; only an OVERSHOOT (peak or
        # level) is something a tighter aim can still buy back, and only inside the pass budget.
        if attempt < _MAX_DELIVERY_PASSES and (over > 0 or i > ln.i + _LUFS_TOL_DB):
            aim = round(aim - (max(over, i - ln.i - _LUFS_TOL_DB, 0.0) + _RETRY_MARGIN_DB), 2)
            print(f"[finalize] master: encode overshoot {over:+.2f} dB past the ceiling -> retry at aim {aim}")
            continue
        if tp > _CLIP_TP_DBTP or off_band:
            _refuse(out, ln, i, tp, lra, aim)
        # ceiling..0 dBTP is the codec's overshoot on a limited waveform, not clipping — mastering converges,
        # it does not gate, and a user waiting on a finished render must not be blocked by this number.
        print(f"[finalize] master delivered over ceiling by {over:.2f} dB (codec overshoot) — shipped")
        return out
    return out


def _encode(src: Path, out: Path, af: str, ln, aim: float) -> bool:
    """Level the audio in PCM (chain + make-up), then mux it over the copied video as ONE aac encode."""
    lvl = out.with_name(out.stem + ".lvl.wav")
    mk = out.with_name(out.stem + ".mk.wav")
    try:
        if not _pcm_pass(src, lvl, af):
            print("[finalize] master loudnorm: level pass failed -> left at source level")
            return False
        audio = _make_up(lvl, mk, ln, aim)
        r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-i", str(audio),
                  "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                  "-ar", "48000", str(out), "-y"],
                 "master mux", timeout_s=_AUDIO_PASS_WALL_S, check=False)
        if r.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            print(f"[finalize] master loudnorm exit {r.returncode} -> left at source level")
            return False
    finally:
        lvl.unlink(missing_ok=True)
        mk.unlink(missing_ok=True)
    return True


def _pcm_pass(src: Path, dst: Path, af: str) -> bool:
    """One audio-only filter pass to 48k PCM. The level work happens BEFORE the single aac encode, so the
    make-up below can re-measure what the brickwall actually left instead of guessing through the codec."""
    r = _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn",
              "-af", af, "-ar", "48000", "-c:a", "pcm_s16le", str(dst), "-y"],
             "master level", timeout_s=_AUDIO_PASS_WALL_S, check=False)
    return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def _make_up(lvl: Path, mk: Path, ln, tp_aim: float) -> Path:
    """The brickwall removes exactly the transients a click-carried integrated loudness was made of, so
    the levelled PCM is re-measured and given back the residual — under the same ceiling, once."""
    d = _measure(lvl, ln, tp_aim)
    try:
        i, tp = float(d["input_i"]), float(d["input_tp"])
    except (TypeError, KeyError, ValueError):
        print("[finalize] master: levelled PCM UNVERIFIED — no make-up, the encode decides")
        return lvl
    residual = ln.i - i
    print(f"[finalize] master: levelled lufs={i} tp={tp} (residual {residual:+.2f} LU)")
    if residual <= _MAKEUP_TRIGGER_LU:
        return lvl
    if residual > _MAKEUP_MAX_DB:
        raise RuntimeError(
            f"[finalize] master: OFF-CONTRACT after limiter lufs={i} target={ln.i} "
            f"(residual {residual:+.2f} LU over the {_MAKEUP_MAX_DB} dB make-up cap) — a master whose "
            f"loudness lives only in the transients the ceiling removes is not a montage, refusing")
    if not _pcm_pass(lvl, mk, f"volume={residual:.2f}dB,{limiter_af(tp_aim)}"):
        print("[finalize] master: make-up pass failed -> shipping the levelled PCM")
        return lvl
    d2 = _measure(mk, ln, tp_aim)
    got = f"lufs={d2['input_i']} tp={d2['input_tp']}" if d2 else "UNVERIFIED"
    print(f"[finalize] master: make-up {residual:+.2f} dB -> {got}")
    return mk


def _measure(path: Path, ln, tp_aim: float) -> dict | None:
    """One loudnorm print_format=json pass. `-map 0:a:0?`: unmapped, `-f null` decodes the whole VIDEO for
    an audio measure; on a silent master ffmpeg still exits nonzero, so rc is unchecked and None says it."""
    meas = _run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-map", "0:a:0?",
         "-af", f"loudnorm=I={ln.i}:TP={tp_aim}:LRA={ln.lra}:print_format=json", "-f", "null", "-"],
        "master measure", timeout_s=_AUDIO_PASS_WALL_S, check=False, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas.stderr, re.S)
    return json.loads(m.group(0)) if m else None


def _verdict(out: Path, ln, aim: float, attempt: int) -> tuple[float, float, float] | None:
    """Measures what was actually ENCODED — the aac encode of a limited waveform moves the true peak, so
    nothing before the mux can answer this. None means unreadable, which is not evidence of a bad master."""
    mv = _measure(out, ln, aim)
    try:
        i, tp, lra = float(mv["input_i"]), float(mv["input_tp"]), float(mv["input_lra"])
    except (TypeError, KeyError, ValueError):
        print("[finalize] master: delivered level UNVERIFIED — post-encode measure unreadable")
        return None
    print(f"[finalize] master: delivered lufs={i} tp={tp} lra={lra} (pass {attempt}, aim {aim})")
    return i, tp, lra


def _refuse(out: Path, ln, i: float, tp: float, lra: float, aim: float) -> None:
    """The delivery address must not hold an off-contract master: whoever picks the file up next has no
    way to know the verdict raised, so the refusal takes the file with it."""
    out.unlink(missing_ok=True)
    why = (f"tp={tp} ceil={ln.tp} clip_at={_CLIP_TP_DBTP} (aim {aim}, delivered lufs={i} lra={lra}) — "
           f"refusing to ship a clipping master") if tp > _CLIP_TP_DBTP else (
        f"lufs={i} target={ln.i} tol={_LUFS_TOL_DB} (tp={tp} lra={lra}) — "
        f"the box refuses this level, refusing to ship it silently")
    raise RuntimeError(f"[finalize] master: OFF-CONTRACT after limiter {why}")
