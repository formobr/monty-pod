"""ONE filter graph and ONE master encode for the BODY delivery tail — composite, mograph, captions,
accents, logo and watermark in a single pass. BUILT, NOT ROUTED: production still walks
render.render_spec's up-to-six conditional full-frame re-encodes; wiring this in is a later wave."""
from __future__ import annotations

import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from . import accents as _accents
from . import finalize as _finalize
from . import mograph as _mograph
from . import render as _render
from .cp import upload
from .models import RenderSpec
from .sanitize import safe_error

# The sync guard matches frames by argmin|ref-master| over GRAYSCALE at exactly this size
# (scripts/check_sync.py all_frames), so the reference is scaled and greyed INSIDE the graph: a
# compression delta between rungs would otherwise land straight in the frame match.
_REF_W, _REF_H = 80, 142
# libx264 at both rungs on purpose: h264_nvenc has a minimum encode width an 80-px frame is under,
# and at this size the encoder choice costs nothing anyway.
_REF_VIDEO = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]
_REF_AUDIO = ["-c:a", "aac", "-b:a", "128k"]
_MASTER_AUDIO = ["-c:a", "aac", "-b:a", "192k"]

# -stream_loop -1 on the idle input plus shortest=1 means `-t` is the ONLY thing that ends this
# process, so the wait is bounded by the WORK: a wedge is infinite, and any finite bound catches it.
_ENCODE_FLOOR_S = 300.0
_ENCODE_REALTIME_X = 60.0

# Merged link names. None may contain "__": that is the namespacing separator, so a builder's own
# internal pad can never collide with one of these no matter what it is called.
V_COMPOSITE, A_COMPOSITE = "vcomposite", "acomposite"
V_MOGRAPH, V_CAPTIONS = "vmograph", "vcaptions"
V_TAIL, V_REF_SRC, V_PRESYNC = "vtail", "vrefsrc", "vpresync"
A_TAIL, A_PRESYNC = "atail", "apresync"
V_ACCENT_IN, V_ACCENTS = "vaccentin", "vaccents"
V_LOGO = "vlogo"
V_WATERMARK, A_WATERMARK = "vwatermark", "awatermark"


@contextmanager
def _no_phase(_op: str):
    """Default for the event hook: the door must run identically with nobody listening."""
    yield


# --- the two composition primitives -------------------------------------------

_PAD = re.compile(r"\[([^\[\]]*)\]")
_EXTERNAL = re.compile(r"\d+:[av]")
_INTERNAL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def rewire(fragment: str, tag: str, subst: dict[str, str]) -> str:
    """ONE pass over the pad tokens: a declared boundary pad becomes its merged name, every other
    internal pad is namespaced by `tag`. A CHAIN of str.replace would rewrite text an earlier replace
    just inserted (accents._namespace_labels' defect), so this is a single re.sub and never that."""
    def one(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in subst:
            return f"[{subst[name]}]"
        if _EXTERNAL.fullmatch(name):
            raise ValueError(f"fragment {tag!r} reads input pad [{name}] that the allocator never handed out")
        if _INTERNAL.fullmatch(name):
            return f"[{name}__{tag}]"
        return m.group(0)
    return _PAD.sub(one, fragment)


class _Input(NamedTuple):
    path: Path
    flags: tuple[str, ...]


class Inputs:
    """The ONE index->path table of the merged argv, with each input's own decoder/loop flags. Every
    fragment's external pads are rewired to the index handed out here, so no builder's hardcoded
    [0:v]/[1:v] survives the merge."""

    def __init__(self) -> None:
        self._items: list[_Input] = []

    def add(self, path: Path, *flags: str) -> int:
        self._items.append(_Input(Path(path), tuple(flags)))
        return len(self._items) - 1

    def argv(self) -> list[str]:
        out: list[str] = []
        for item in self._items:
            out += [*item.flags, "-i", str(item.path)]
        return out


# --- 1. preflight -------------------------------------------------------------

def body_duration(spec: RenderSpec) -> float:
    """The ONE authoritative body length: the timeline's own arithmetic. There is no earlier pass left
    to ffprobe, and probing the first SOURCE (as the audio pad does today) measures the wrong thing."""
    return sum((s.out - s.in_) / s.speed for s in spec.timeline.segments)


def preflight(spec: RenderSpec) -> None:
    """Refuse every non-goal BEFORE any subprocess — same exception type and same timing as
    render.render_spec (render.py:552-568), so a v6 spec cannot half-render through this door either."""
    ov = spec.overlays if spec.mode == "final" else None
    if ov is None:
        return
    unimplemented = []
    if ov.trims:
        unimplemented.append("trims")
    if ov.opener is not None:
        unimplemented.append("opener")
    if ov.cover is not None:
        # Not in render.py's list because the multi-pass path DOES weld a cover; this graph does not,
        # and silently dropping a declared end-card is exactly the half-render that list exists to stop.
        unimplemented.append("cover")
    fin = ov.finalize
    if fin is not None and any(a.kind == "film_burn" for a in fin.accents):
        unimplemented.append("finalize.accents[kind=film_burn]")
    if unimplemented:
        raise NotImplementedError(
            f"body single-pass graph does not composite these yet (opener/junction waves): {unimplemented}")


def _check_assets(spec: RenderSpec, input_paths: dict) -> None:
    """Every asset the tail names must be a resolved inputs[] id, refused here rather than as a
    KeyError inside the assembler — same errors apply_logo/apply_watermark/_burn_captions raise."""
    ov = spec.overlays if spec.mode == "final" else None
    if ov is None:
        return
    mp = ov.motion_plan
    caps = mp.captions if mp is not None else None
    if caps is not None and caps.words and (not caps.font or caps.font not in input_paths):
        raise RuntimeError("captions present but no resolved font input on the spec")
    fin = ov.finalize
    if fin is None:
        return
    if fin.logo is not None and input_paths.get(fin.logo.asset) is None:
        raise RuntimeError(f"finalize.logo.asset {fin.logo.asset!r} is not a resolved inputs[] id")
    if fin.watermark is not None:
        for ref in (fin.watermark.sting, fin.watermark.idle):
            if ref not in input_paths:
                raise RuntimeError(f"finalize.watermark asset {ref!r} is not a resolved inputs[] id")


# --- 2. prepare ---------------------------------------------------------------

@dataclass(frozen=True)
class Prepared:
    """Everything `assemble` needs, with the parts that only exist after a subprocess already made."""

    spec: RenderSpec
    gpu: bool
    input_paths: dict[str, Path]
    duration: float
    master_out: Path
    presync_out: Path
    filter_script: Path
    bed: Path | None = None
    audio: object | None = None
    layers: tuple[dict, ...] = ()
    ass: Path | None = None
    font_dir: Path | None = None


@dataclass(frozen=True)
class Delivered:
    """What the door actually PUT. `master` is the loudnorm's output, which is a DIFFERENT file from
    the encode's — two-pass loudnorm measures a finished file, so it can never join the graph."""

    prepared: Prepared
    master: Path
    presync: Path | None
    outputs: list[str] = field(default_factory=list)


def _audio_mix(spec: RenderSpec, input_paths: dict, tmp: Path, dur: float, phase):
    """The voice+bed mix the composite consumes: measure the voice, pre-render the -33 LUFS bed. Both
    are render.py's own pre-passes, unchanged — the merge removes video encodes, not these."""
    ov = spec.overlays if spec.mode == "final" else None
    music = ov.music if ov is not None else None
    sfx = ov.sfx if ov is not None else None
    if music is None and not sfx:
        return None, None
    with phase("audio_prepare"):
        ids = _render.input_ids(spec)
        voice = input_paths[spec.timeline.segments[0].src]
        dirty = _render._voice_is_dirty(voice) and not spec.base_voice_rescued
        clean = "highpass=f=80" + (",afftdn=nr=8:nf=-30" if dirty else "")
        vln = _render._measure_loudnorm(voice, clean)
        bed, bed_idx = None, None
        if music is not None:
            bed = _render._prerender_bed(input_paths[music.track], music.start, dur, tmp)
            bed_idx = len(ids)
        mix = _render._AudioMix(
            voice_idx=ids.index(spec.timeline.segments[0].src), bed_idx=bed_idx, clean=clean,
            vln=vln, dur=dur, sfx=tuple((ids.index(s.sound), s.at, s.gain) for s in (sfx or [])))
    return mix, bed


def _write_ass(caps, motion_plan, input_paths: dict, out_dir: Path, w: int, h: int) -> tuple[Path, Path]:
    """The same ASS file render._burn_captions writes, minus its ffmpeg pass."""
    from .captions import build_ass
    font = input_paths[caps.font]
    fg, accent = _render._caption_colours(caps, motion_plan)
    ass = out_dir / "captions.ass"
    ass.write_text(build_ass([wd.model_dump() for wd in caps.words], font=font, w=w, h=h, fg=fg,
                             accent=accent,
                             center_y=caps.centerY if caps.centerY is not None else 0.76,
                             style=caps.style or "oneword"), encoding="utf-8")
    return ass, font.parent


def prepare(spec: RenderSpec, input_paths: dict, tmp: Path, gpu: bool, *,
            master_out: Path | None = None, presync_out: Path | None = None,
            phase=_no_phase) -> Prepared:
    """Run every preparation the current multi-pass path runs — mograph qtrle layers, voice/bed audio
    passes, the ASS file — and return them typed. Layers that produced no frames simply do not appear
    (mograph.py:143/275): a declared section with no bundle entry prints SKIP and the base stays bare."""
    _check_assets(spec, input_paths)
    tmp = Path(tmp)
    dur = body_duration(spec)
    ov = spec.overlays if spec.mode == "final" else None
    mp = ov.motion_plan if ov is not None else None
    audio, bed = _audio_mix(spec, input_paths, tmp, dur, phase)
    layers: tuple[dict, ...] = ()
    if mp is not None and mp.sections:
        with phase("mograph"):
            layers = tuple(_mograph._render_layers(
                mp.sections, mp.brand.model_dump() if mp.brand else None, input_paths, tmp,
                getattr(mp, "bundle", None)))
    ass = font_dir = None
    caps = mp.captions if mp is not None else None
    if caps is not None and caps.words:
        with phase("captions"):
            ass, font_dir = _write_ass(caps, mp, input_paths, tmp,
                                       spec.timeline.width, spec.timeline.height)
    return Prepared(
        spec=spec, gpu=gpu, input_paths=input_paths, duration=dur,
        master_out=master_out or tmp / "render.mp4",
        presync_out=presync_out or tmp / "render.presync.mp4",
        filter_script=tmp / "body_onepass.filter",
        bed=bed, audio=audio, layers=layers, ass=ass, font_dir=font_dir)


# --- 3. assemble --------------------------------------------------------------

def assemble(p: Prepared) -> tuple[str, list[str]]:
    """Pure: (filter-script text, argv). No I/O and no probe — every number was computed or crossed on
    the spec, which is what makes the graph a hashable text artifact the goldens can pin."""
    spec, gpu = p.spec, p.gpu
    ov = spec.overlays if spec.mode == "final" else None
    fin = ov.finalize if ov is not None else None
    w, h, fps = spec.timeline.width, spec.timeline.height, spec.timeline.fps

    inputs = Inputs()
    spec_pads: dict[str, str] = {}
    for iid in _render.input_ids(spec):
        n = inputs.add(p.input_paths[iid])
        spec_pads[f"{n}:v"] = f"{n}:v"
        spec_pads[f"{n}:a"] = f"{n}:a"
    if p.bed is not None:
        n = inputs.add(p.bed)
        spec_pads[f"{n}:a"] = f"{n}:a"

    chains = [rewire(_render.build_filtergraph(spec, gpu, p.audio), "cmp",
                     {**spec_pads, "vout": V_COMPOSITE, "aout": A_COMPOSITE})]
    vlink = V_COMPOSITE

    if p.layers:
        layers_v = [f"{inputs.add(Path(lay['mov']))}:v" for lay in p.layers]
        frag, last = _mograph.overlay_filtergraph(list(p.layers), base=vlink, layers_v=layers_v)
        chains.append(rewire(frag, "mog", {vlink: vlink, **{lv: lv for lv in layers_v}, last: V_MOGRAPH}))
        vlink = V_MOGRAPH

    if p.ass is not None:
        chains.append(f"[{vlink}]subtitles={p.ass}:fontsdir={p.font_dir}[{V_CAPTIONS}]")
        vlink = V_CAPTIONS

    # The reference is built only when the spec DECLARES somewhere to put it (final_spec.py:172-174
    # declares it with the tail and not otherwise); an undeclared one is a second full-body encode
    # thrown away. A filter link is single-use, so the fork is explicit, and it sits BEFORE the
    # accents — they are the very thing able to slide picture against sound.
    wants_ref = any(o.kind == "presync" for o in spec.outputs)
    alink, aref = A_COMPOSITE, None
    if wants_ref:
        chains.append(f"[{vlink}]split=2[{V_TAIL}][{V_REF_SRC}]")
        chains.append(f"[{V_REF_SRC}]scale={_REF_W}:{_REF_H},format=gray[{V_PRESYNC}]")
        chains.append(f"[{A_COMPOSITE}]asplit=2[{A_TAIL}][{A_PRESYNC}]")
        vlink, alink, aref = V_TAIL, A_TAIL, A_PRESYNC

    wm = fin.watermark if fin is not None else None

    accent_frag = (_accents.build_chain_filter(fin.accents, fps=fps, w=w, h=h, gpu=gpu)
                   if fin is not None and fin.accents else None)
    if accent_frag is not None:
        src = vlink
        if gpu:
            # The GPU accent branches issue a BARE hwupload (accents.py:81/145/300); the intermediate
            # encode that used to hand them yuv420p is the one this wave deletes.
            chains.append(f"[{vlink}]format=yuv420p[{V_ACCENT_IN}]")
            src = V_ACCENT_IN
        chains.append(rewire(accent_frag, "acc", {"0:v": src, "vout": V_ACCENTS}))
        vlink = V_ACCENTS

    if fin is not None and fin.logo is not None:
        lg = fin.logo
        logo_v = f"{inputs.add(p.input_paths[lg.asset])}:v"
        # body_end is the WHOLE body: apply_logo subtracts cover_hold because it probes a master that
        # already carries the welded end-card, and preflight refuses a cover here — subtracting it
        # would blank the logo over live body (engine gate: test_body_logo_runs_to_the_end_without_a_cover).
        frag = _finalize.body_logo_filter(lg.corner, lg.width, lg.opacity, lg.margin, p.duration,
                                          base_v=vlink, logo_v=logo_v, out_v=V_LOGO)
        chains.append(rewire(frag, "lgo", {vlink: vlink, logo_v: logo_v, V_LOGO: V_LOGO}))
        vlink = V_LOGO

    if wm is not None:
        # The alpha lives in a separate VP9 stream only libvpx-vp9 extracts, and the idle must outlast
        # the body it is overlaid onto — both are per-INPUT flags, so they ride their own -i (finalize.py:200).
        sting = inputs.add(p.input_paths[wm.sting], "-c:v", "libvpx-vp9")
        idle = inputs.add(p.input_paths[wm.idle], "-c:v", "libvpx-vp9", "-stream_loop", "-1")
        xy = (f"{wm.x}:{wm.y}" if wm.x is not None and wm.y is not None
              else _finalize._POS[wm.position].format(m=wm.margin))
        chime_a = f"{sting}:a" if wm.chime else None
        frag, _out_v, out_a = _finalize.watermark_filter(
            base_v=vlink, sting_v=f"{sting}:v", idle_v=f"{idle}:v", width=wm.width, overlay_xy=xy,
            base_a=alink, chime_a=chime_a, chime_vol=wm.chime_volume, delay=wm.delay,
            out_v=V_WATERMARK, out_a=A_WATERMARK)
        subst = {vlink: vlink, alink: alink, f"{sting}:v": f"{sting}:v", f"{idle}:v": f"{idle}:v",
                 V_WATERMARK: V_WATERMARK, A_WATERMARK: A_WATERMARK}
        if chime_a is not None:
            subst[chime_a] = chime_a
        chains.append(rewire(frag, "wmk", subst))
        # chime=false leaves the base audio UNTOUCHED (watermark_filter returns no audio label), so
        # the composite mix is mapped straight through. The old path emits -an there and ships a
        # silent master; that divergence is deliberate, not inherited.
        vlink, alink = V_WATERMARK, (out_a or alink)

    t = f"{p.duration:.3f}"
    cmd = ["ffmpeg", "-y", "-hide_banner"]
    if gpu:
        cmd += ["-init_hw_device", "vulkan"]  # libplacebo runs on a Vulkan device; hwupload derives from it
    cmd += inputs.argv()
    cmd += ["-filter_complex_script", str(p.filter_script)]
    # Output options bind to the output they PRECEDE, so each destination carries its whole clause;
    # -t bounds each (shortest=1 over a -stream_loop'ed idle is not a lifetime).
    cmd += ["-map", f"[{vlink}]", "-map", f"[{alink}]",
            *(_finalize._FINAL_GPU if gpu else _finalize._FINAL_CPU), *_MASTER_AUDIO,
            "-movflags", "+faststart", "-t", t, str(p.master_out)]
    if wants_ref:
        cmd += ["-map", f"[{V_PRESYNC}]", "-map", f"[{aref}]",
                *_REF_VIDEO, *_REF_AUDIO, "-t", t, str(p.presync_out)]
    return ";".join(chains), cmd


# --- the door -----------------------------------------------------------------

def encode_budget_s(duration: float) -> float:
    """The bound on the single encode: proportional to the WORK, because the failure it catches is a
    graph that never ends (a lost -t against a looped input), not a slow box."""
    return _ENCODE_FLOOR_S + _ENCODE_REALTIME_X * max(0.0, duration)


def _run(cmd: list[str], budget_s: float) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=budget_s)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-2000:]
        detail = tail.decode("utf-8", "replace") if isinstance(tail, bytes) else str(tail)
        raise RuntimeError(
            f"body single-pass ffmpeg exited {exc.returncode}: {safe_error(RuntimeError(detail))}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"body single-pass ffmpeg exceeded its {budget_s:.0f}s budget — a graph that cannot end "
            f"(check -t against the looped watermark idle)") from exc


def render_body(spec: RenderSpec, input_paths: dict, tmp: Path, gpu: bool, *,
                master_out: Path | None = None, presync_out: Path | None = None,
                phase=_no_phase) -> Delivered:
    """Refuse the non-goals, prepare, ONE ffmpeg for the delivery-rung master (+ the pre-accent sync
    reference when one is declared), THEN the delivery loudnorm, then PUT what was declared. `phase`
    is the render stage's per-op event seam (render.py:517-551), a no-op when nobody listens."""
    preflight(spec)
    p = prepare(spec, input_paths, tmp, gpu, master_out=master_out, presync_out=presync_out,
                phase=phase)
    graph, cmd = assemble(p)
    p.filter_script.write_text(graph, encoding="utf-8")
    with phase("ffmpeg"):
        _run(cmd, encode_budget_s(p.duration))
    fin = spec.overlays.finalize if (spec.mode == "final" and spec.overlays is not None) else None
    master = p.master_out
    if fin is not None:
        # The delivery level is a TWO-PASS loudnorm: it measures the finished file, so it cannot join
        # the graph and must not be skipped either — final_dispatch sets this block on every final
        # spec, and shipping the encode raw is every deliverable ~6 dB under the brand target.
        with phase("finalize"):
            master = _finalize.apply_loudnorm(fin, master, Path(tmp) / "fin_ln.mp4")
    done: list[str] = []
    with phase("upload"):
        for o in spec.outputs:
            if o.kind in ("cache", "cover"):
                continue
            if o.kind == "presync":
                upload(p.presync_out, o.put_url, "video/mp4")
            else:
                upload(master, o.put_url, "video/mp4")
            done.append(o.id)
    return Delivered(prepared=p, master=master,
                     presync=p.presync_out if any(o.kind == "presync" for o in spec.outputs) else None,
                     outputs=done)
