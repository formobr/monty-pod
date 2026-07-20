"""Frame-accent filtergraph macros — the pod half of the delivery tail.

The planner decides WHICH accent fires WHERE and how hard (kind / at / intensity); this module only
turns those three numbers into an ffmpeg filtergraph. No thresholds, no selection, no rationale — the
accent list arrives fully resolved on `overlays.finalize.accents`.

Each builder emits a self-contained `[0:v]…[vout]` subgraph touching only its own short window, so
they CHAIN: accent k reads the previous [vout] and writes the next (`build_chain_filter`).

Design rules baked into these macros (they are load-bearing, not style):
  * Per-frame zoom/pan uses `zoompan`, NEVER `crop` — crop freezes w/h at frame 0.
  * Any `zoompan` that moves x/y renders at SS=2 then scales down, or it jitters from integer
    truncation.
  * Every accent is gated to its window (`enable=between(t,…)` or a sliced+concat span) so the rest
    of the clip is untouched — an accent is an INSTANT, not a wash.

These builders are MIRRORED by the planner (scripts/fx.py + scripts/transitions.py). The mirror is
not decorative: the planner-side copies drive local sample renders and the CLI. Both sides are held
byte-identical by a string-identity test over a parameter grid (tests/test_finalize_parity.py in the
engine repo) — a drift in either copy fails that test rather than silently shipping two different
masters from the two transports.
"""
from __future__ import annotations

import random
import re

SS = 2  # supersample factor for any zoompan x/y move
_PI = 3.14159265358979


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --- GPU translation ----------------------------------------------------------

def _sub_on(e: str) -> str:
    return re.sub(r"\bon\b", "n", str(e))


def lp_kenburns(z: str, x: str, y: str, out_w: int, out_h: int) -> str:
    """ONE libplacebo pass reproducing a zoompan(z, x, y -> out_w x out_h). `on`->`n`; the live `zoom`
    in x/y is the inlined z expression. zoompan CLAMPS x/y so the crop window stays inside the frame;
    libplacebo does NOT (it clamps the SAMPLER, so a window sticking out smears the edge row/column),
    so the window is clamped here to match."""
    zn = _sub_on(z)

    def tr(e: str) -> str:
        return re.sub(r"\bzoom\b", f"({zn})", _sub_on(e))

    cw, ch = f"iw/({zn})", f"ih/({zn})"
    cx = f"max(0,min({tr(x)},iw-{cw}))"
    cy = f"max(0,min({tr(y)},ih-{ch}))"
    return (f"libplacebo=w={out_w}:h={out_h}"
            f":crop_w='{cw}':crop_h='{ch}':crop_x='{cx}':crop_y='{cy}'"
            f":upscaler=spline36:downscaler=spline36")


# --- camera_shake -------------------------------------------------------------

def camera_shake_filter(at: float, *, intensity: float = 0.6, frames: int = 9, fps: float = 30.0,
                        w: int = 1080, h: int = 1920, gpu: bool = False) -> str:
    """A brief INTENTIONAL handheld shake centred on `at`: a fast, decaying x/y jitter, then the frame
    settles clean. intensity 0..1 -> amplitude 0.4%..2.5% of the dimension. Premium ONLY because it is
    SHORT and DECAYS; rendered at SS=2 so the sub-pixel offset stays smooth."""
    intensity = _clamp01(intensity)
    f = float(fps)
    n = max(1, frames)
    fdur = 1.0 / f
    t0 = max(0.0, at - (n * fdur) / 2.0)
    t1 = t0 + n * fdur
    amp_frac = 0.004 + 0.021 * intensity
    zoom = 1.0 + amp_frac * 2.2          # crop-in headroom so the jitter never bares an edge
    env = f"(1-on/{n})"
    ox = f"sin(on*2.7)*{env}*(iw*{amp_frac:.5f})"
    oy = f"cos(on*3.3)*{env}*(ih*{amp_frac:.5f})"
    x = f"(iw-iw/zoom)/2+{ox}"
    y = f"(ih-ih/zoom)/2+{oy}"
    sw, sh = w * SS, h * SS
    move = (f"hwupload,{lp_kenburns(f'{zoom:.4f}', x, y, sw, sh)},hwdownload,format=yuv420p"
            if gpu else
            f"zoompan=z='{zoom:.4f}':x='{x}':y='{y}':d=1:s={sw}x{sh}:fps={f:.4f}")
    return (
        f"[0:v]split=2[base][sh];"
        f"[sh]trim=start={t0:.4f}:end={t1:.4f},setpts=PTS-STARTPTS,"
        f"scale={sw}:{sh}:force_original_aspect_ratio=increase,crop={sw}:{sh},"
        f"{move},"
        f"scale={w}:{h},setpts=PTS-STARTPTS+{t0:.4f}/TB[shx];"
        f"[base][shx]overlay=enable='between(t,{t0:.4f},{t1:.4f})'[vout]"
    )


# --- grain --------------------------------------------------------------------

def grain_filter(at: float, *, intensity: float = 0.7, frames: int = 24, fps: float = 30.0,
                 w: int = 1080, h: int = 1920, gpu: bool = False) -> str:  # gpu: ignored (non-zoom)
    """Brief film-GRAIN burst on `at` — analog texture as an ACCENT, not a whole-video wash.
    intensity 0..1 -> luma-noise sigma 10..40, gated by overlay so the rest stays clean."""
    intensity = _clamp01(intensity)
    f = float(fps)
    fdur = 1.0 / f
    n = max(1, frames)
    t0 = max(0.0, at - (n * fdur) / 2.0)
    t1 = t0 + n * fdur
    strength = 10 + round(30 * intensity)
    return (
        f"[0:v]split=2[base][gr];"
        f"[gr]trim=start={t0:.4f}:end={t1:.4f},setpts=PTS-STARTPTS,"
        f"noise=c0s={strength}:c0f=t,eq=contrast={1.0 + 0.06 * intensity:.3f},"
        f"setpts=PTS-STARTPTS+{t0:.4f}/TB[grx];"
        f"[base][grx]overlay=enable='between(t,{t0:.4f},{t1:.4f})'[vout]"
    )


# --- zoom_punch ---------------------------------------------------------------

def zoom_punch_filter(at: float, *, intensity: float = 0.6, frames_in: int = 4,
                      frames_hold: int = 2, frames_out: int = 6, fps: float = 30.0,
                      w: int = 1080, h: int = 1920, gpu: bool = False) -> str:
    """A sharp scale PUNCH centred on `at`: scale 1.0 -> peak over `frames_in`, a tiny hold, then an
    eased settle back, with motion blur on the fast leg. intensity 0..1 -> peak +6%..+22%."""
    intensity = _clamp01(intensity)
    peak = 1.06 + 0.16 * intensity
    f = float(fps)
    t_in = max(1, frames_in) / f
    t_hold = max(0, frames_hold) / f
    t_out = max(1, frames_out) / f
    t0 = max(0.0, at - t_in)
    t_peak1 = at + t_hold
    t_end = t_peak1 + t_out
    n_in = max(1, frames_in)
    n_hold = max(0, frames_hold)
    n_out = max(1, frames_out)
    z = (
        f"if(lt(on,{n_in}),"
        f"1+({peak - 1:.4f})*(on/{n_in}),"
        f"if(lt(on,{n_in + n_hold}),{peak:.4f},"
        f"if(lt(on,{n_in + n_hold + n_out}),"
        f"{peak:.4f}-({peak - 1:.4f})*pow((on-{n_in + n_hold})/{n_out},2),"
        f"1)))"
    )
    blur = 2 + round(2 * intensity)
    cxe, cye = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    move = (f"hwupload,{lp_kenburns(z, cxe, cye, w, h)},hwdownload,format=yuv420p"
            if gpu else
            f"zoompan=z='{z}':d=1:s={w}x{h}:fps={f:.4f}:x='{cxe}':y='{cye}'")
    return (
        f"[0:v]split=3[pre][mid][post];"
        f"[pre]trim=end={t0:.4f},setpts=PTS-STARTPTS[a];"
        f"[mid]trim=start={t0:.4f}:end={t_end:.4f},setpts=PTS-STARTPTS,"
        f"{move},"
        f"tmix=frames={blur}:weights='{' '.join(['1'] * blur)}',"
        f"setpts=PTS-STARTPTS[b];"
        f"[post]trim=start={t_end:.4f},setpts=PTS-STARTPTS[c];"
        f"[a][b][c]concat=n=3:v=1:a=0[vout]"
    )


# --- glitch -------------------------------------------------------------------

def _glitch_steps(intensity: float, seed: int = 0) -> list[tuple[int, int, int, float]]:
    """Per-frame STROBE steps: (red_shift, blue_shift, vert_tear, mono). rgbashift/noise/desat take no
    per-frame expression, so this is a stepped, alternating, decaying sequence. DETERMINISTIC in
    (intensity, seed)."""
    rng = random.Random(seed)
    a = 2 + round(14 * intensity)
    t = 1 + round(7 * intensity)
    env = [1.0, -0.75, 0.9, -0.55, 0.65, -0.35, 0.18]
    mono_env = [0.9, 0.0, 1.0, 0.3, 0.7, 0.0, 0.4]
    steps = []
    for i, e in enumerate(env):
        je = e * rng.uniform(0.72, 1.28)
        rh = round(a * je)
        bh = round(-a * je * 0.8)
        rv = round(t * (1 if i % 2 else -1) * abs(je))
        mono = round(min(1.0, mono_env[i] * (0.4 + 0.6 * intensity)
                         * rng.uniform(0.8, 1.2)), 3)
        steps.append((rh, bh, rv, mono))
    return steps


def _glitch_roll_expr(intensity: float, seed: int, dur: float, h: int) -> str:
    """SMOOTH vertical frame-roll as a CONTINUOUS `crop y` expression in `t`: a damped sine with a
    faster tremble on top, wrapped with mod(...,H) over a vstacked pair so what rolls off one edge
    wraps onto the other."""
    rng = random.Random(seed ^ 0x9E3779B9)
    dur = max(dur, 1e-6)
    amp = rng.uniform(0.75, 1.0) * (22 + 130 * intensity)
    cycles = rng.uniform(1.2, 1.9)
    phase = rng.uniform(0.0, 6.283)
    decay = rng.uniform(2.2, 3.6)
    w1 = 2.0 * _PI * cycles / dur
    trem = 0.28 * amp
    w2 = 2.0 * _PI * rng.uniform(4.5, 6.5) / dur
    return (f"mod(("
            f"{amp:.1f}*sin({w1:.4f}*t+{phase:.3f})"
            f"+{trem:.1f}*sin({w2:.4f}*t))"
            f"*exp({-decay / dur:.4f}*t)"
            f"+{100 * h},{h})")


def glitch_filter(at: float, *, intensity: float = 0.5, frames: int = 7, fps: float = 30.0,
                  w: int = 1080, h: int = 1920, gpu: bool = False) -> str:  # gpu: ignored (non-zoom)
    """A brief digital GLITCH centred on `at`: stepped RGB channel split + a smooth vertical frame-roll
    + digital noise + a B/W flicker. The stepped amplitudes are seeded off `at`, so each glitch in a
    video looks a little different (deterministic per position)."""
    intensity = _clamp01(intensity)
    f = float(fps)
    fdur = 1.0 / f
    seed = int(round(at * 1000)) & 0xFFFFFFFF
    steps = _glitch_steps(intensity, seed)
    n = min(max(1, frames), len(steps))
    steps = steps[:n]
    t0 = max(0.0, at - (n * fdur) / 2.0)
    t1 = t0 + n * fdur
    nstr = round(14 + 34 * intensity)
    contrast = 1.0 + 0.25 * intensity

    parts = ["[0:v]split=2[base][gl]"]
    parts.append(f"[gl]trim=start={t0:.4f}:end={t1:.4f},setpts=PTS-STARTPTS,"
                 f"split={n}{''.join(f'[s{i}]' for i in range(n))}")
    glabels = []
    for i, (rh, bh, rv, mono) in enumerate(steps):
        a0, a1 = i * fdur, (i + 1) * fdur
        glabels.append(f"[g{i}]")
        ns = nstr + round(mono * 45)
        sat = round(max(0.0, 1.0 - mono), 3)
        parts.append(
            f"[s{i}]trim=start={a0:.4f}:end={a1:.4f},setpts=PTS-STARTPTS,"
            f"rgbashift=rh={rh}:bh={bh}:rv={rv}:bv={-rv}:edge=wrap,"
            f"noise=alls={ns}:allf=t,"
            f"eq=contrast={contrast:.3f}:saturation={sat:.3f}[g{i}]"
        )
    roll = _glitch_roll_expr(intensity, seed, n * fdur, h)
    parts.append(f"{''.join(glabels)}concat=n={n}:v=1:a=0[gcat]")
    parts.append(f"[gcat]split=2[rc0][rc1];[rc0][rc1]vstack=inputs=2,"
                 f"crop={w}:{h}:0:y='{roll}',"
                 f"setpts=PTS-STARTPTS+{t0:.4f}/TB[gx]")
    parts.append(f"[base][gx]overlay=enable='between(t,{t0:.4f},{t1:.4f})'[vout]")
    return ";".join(parts)


# --- rgb_split ----------------------------------------------------------------

def rgb_split_filter(at: float, *, intensity: float = 0.55, frames: int = 5, fps: float = 30.0,
                     w: int = 1080, h: int = 1920, gpu: bool = False) -> str:  # gpu: ignored (non-zoom)
    """Clean chromatic-aberration split on `at` — R/B tear apart then snap back, NO noise/roll/mono
    (that is `glitch`). intensity 0..1 -> peak split +-3..21 px."""
    intensity = _clamp01(intensity)
    f = float(fps)
    fdur = 1.0 / f
    n = max(1, frames)
    seed = int(round(at * 1000)) & 0xFFFFFFFF
    rng = random.Random(seed)
    amp = 3 + round(18 * intensity)
    env = ([1.0, 0.72, 0.46, 0.24, 0.10] + [0.05] * n)[:n]
    t0 = max(0.0, at - (n * fdur) / 2.0)
    t1 = t0 + n * fdur
    sat = 1.0 + 0.30 * intensity
    parts = ["[0:v]split=2[base][rs]"]
    parts.append(f"[rs]trim=start={t0:.4f}:end={t1:.4f},setpts=PTS-STARTPTS,"
                 f"split={n}{''.join(f'[p{i}]' for i in range(n))}")
    labels = []
    for i, e in enumerate(env):
        je = e * rng.uniform(0.82, 1.18)
        rh = round(amp * je)
        bh = round(-amp * je)
        rv = round(amp * je * 0.35)
        a0, a1 = i * fdur, (i + 1) * fdur
        labels.append(f"[r{i}]")
        parts.append(f"[p{i}]trim=start={a0:.4f}:end={a1:.4f},setpts=PTS-STARTPTS,"
                     f"rgbashift=rh={rh}:bh={bh}:rv={rv}:bv={-rv}:edge=smear,"
                     f"eq=saturation={sat:.3f}[r{i}]")
    parts.append(f"{''.join(labels)}concat=n={n}:v=1:a=0,"
                 f"setpts=PTS-STARTPTS+{t0:.4f}/TB[rsx]")
    parts.append(f"[base][rsx]overlay=enable='between(t,{t0:.4f},{t1:.4f})'[vout]")
    return ";".join(parts)


# --- zoom_blur ----------------------------------------------------------------

def zoom_blur_filter(at: float, *, intensity: float = 0.6, frames_in: int = 4, frames_out: int = 7,
                     fps: float = 30.0, w: int = 1080, h: int = 1920, gpu: bool = False) -> str:
    """Whip zoom-blur straddling a plan-change cut at `at`: blurred push IN to the cut, eased settle
    OUT. Bigger and blurrier than zoom_punch. intensity 0..1 -> peak +10%..+40%."""
    intensity = _clamp01(intensity)
    peak = 1.10 + 0.30 * intensity
    f = float(fps)
    n_in = max(1, frames_in)
    n_out = max(1, frames_out)
    t0 = max(0.0, at - n_in / f)
    t_end = at + n_out / f
    z = (
        f"if(lt(on,{n_in}),1+({peak - 1:.4f})*pow(on/{n_in},2),"
        f"if(lt(on,{n_in + n_out}),{peak:.4f}-({peak - 1:.4f})*pow((on-{n_in})/{n_out},1.6),1))"
    )
    blur = 3 + round(3 * intensity)
    cxe, cye = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    move = (f"hwupload,{lp_kenburns(z, cxe, cye, w, h)},hwdownload,format=yuv420p"
            if gpu else
            f"zoompan=z='{z}':d=1:s={w}x{h}:fps={f:.4f}:x='{cxe}':y='{cye}'")
    return (
        f"[0:v]split=3[pre][mid][post];"
        f"[pre]trim=end={t0:.4f},setpts=PTS-STARTPTS[a];"
        f"[mid]trim=start={t0:.4f}:end={t_end:.4f},setpts=PTS-STARTPTS,"
        f"{move},"
        f"tmix=frames={blur}:weights='{' '.join(['1'] * blur)}',"
        f"setpts=PTS-STARTPTS[b];"
        f"[post]trim=start={t_end:.4f},setpts=PTS-STARTPTS[c];"
        f"[a][b][c]concat=n=3:v=1:a=0[vout]"
    )


# --- pixelate -----------------------------------------------------------------

def pixelate_filter(at: float, *, intensity: float = 0.6, frames: int = 10, fps: float = 30.0,
                    w: int = 1080, h: int = 1920, gpu: bool = False) -> str:  # gpu: ignored (non-zoom)
    """Brief MOSAIC on `at` — the frame drops to blocks then snaps back (censorship / redacted /
    low-res meaning, distinct from glitch's signal failure). intensity 0..1 -> block ~8..36 px."""
    intensity = _clamp01(intensity)
    f = float(fps)
    fdur = 1.0 / f
    n = max(1, frames)
    t0 = max(0.0, at - (n * fdur) / 2.0)
    t1 = t0 + n * fdur
    block = 8 + round(28 * intensity)
    dw = max(2, round(w / block))
    dh = max(2, round(h / block))
    return (
        f"[0:v]split=2[base][px];"
        f"[px]trim=start={t0:.4f}:end={t1:.4f},setpts=PTS-STARTPTS,"
        f"scale={dw}:{dh}:flags=neighbor,scale={w}:{h}:flags=neighbor,"
        f"setpts=PTS-STARTPTS+{t0:.4f}/TB[pxx];"
        f"[base][pxx]overlay=enable='between(t,{t0:.4f},{t1:.4f})'[vout]"
    )


# --- chaining -----------------------------------------------------------------

BUILDERS = {
    "camera_shake": camera_shake_filter,
    "grain": grain_filter,
    "zoom_punch": zoom_punch_filter,
    "glitch": glitch_filter,
    "zoom_blur": zoom_blur_filter,
    "rgb_split": rgb_split_filter,
    "pixelate": pixelate_filter,
}
# The contract's accent-kind enum is exactly these keys; spec validation rejects anything else, so an
# unknown kind can never reach a builder (it dies at the seam, not as a silently-dropped accent).


def _namespace_labels(sub: str, tag: str, src_in: str, dst_out: str) -> str:
    """Rewrite a self-contained `[0:v]…[vout]` subgraph so it can be CHAINED: its input becomes
    `src_in`, its output `dst_out`, and every INTERNAL pad label gets a `tag` suffix so two subgraphs
    in one filtergraph never collide on a shared name (base/sh/pre/mid/…)."""
    internal = {m for m in re.findall(r"\[([A-Za-z][\w]*)\]", sub) if m not in ("0:v", "vout")}
    for lbl in internal:
        sub = re.sub(rf"\[{re.escape(lbl)}\]", f"[{lbl}_{tag}]", sub)
    return sub.replace("[0:v]", src_in).replace("[vout]", dst_out)


def build_chain_filter(accents, *, fps: float, w: int, h: int, gpu: bool = False) -> str | None:
    """Chain the resolved frame-accents into ONE `-filter_complex` over the whole clip ([0:v]->[vout]).
    Returns None when there is nothing to apply."""
    if not accents:
        return None
    chain, prev = [], "[0:v]"
    for k, a in enumerate(accents):
        out_lbl = "[vout]" if k == len(accents) - 1 else f"[fa{k}]"
        sub = BUILDERS[a.kind](float(a.at), intensity=float(a.intensity), fps=fps, w=w, h=h, gpu=gpu)
        chain.append(_namespace_labels(sub, f"a{k}", prev, out_lbl))
        prev = out_lbl
    return ";".join(chain)
