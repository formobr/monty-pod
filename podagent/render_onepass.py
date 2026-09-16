"""ONE filter graph and ONE master encode for the BODY delivery tail — composite, mograph, captions,
accents, logo and watermark in a single pass, the ONLY final encode core render.render_spec runs.
`refusals` names the non-goals this graph does not build; `preflight` hard-refuses them up front."""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from . import __version__
from . import accents as _accents
from . import finalize as _finalize
from . import mograph as _mograph
from . import render as _render
from .cp import upload
from .models import SPEC_VERSION, RenderSpec
from .render import body_duration
from .sanitize import safe_text

# The sync guard matches frames by argmin|ref-master| over GRAYSCALE at exactly this size
# (scripts/check_sync.py all_frames), so the reference is scaled and greyed INSIDE the graph: a
# compression delta between rungs would otherwise land straight in the frame match.
_REF_W, _REF_H = 80, 142
# libx264 at both rungs on purpose: h264_nvenc has a minimum encode width an 80-px frame is under,
# and at this size the encoder choice costs nothing anyway.
_REF_VIDEO = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", *_finalize._BT709]
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
V_MASTER = "vmaster"
V_TAP_LOGO = "vtaplogo"


@contextmanager
def _no_phase(_op: str):
    """Default for the event hook: the door must run identically with nobody listening."""
    yield


# --- the two composition primitives -------------------------------------------

_EXTERNAL = re.compile(r"\d+:[av]")
_INTERNAL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def rewire(fragment: str, tag: str, subst: dict[str, str]) -> str:
    """A declared boundary pad becomes its merged name, every other internal pad is namespaced by
    `tag`, and an input pad nobody allocated is an ERROR rather than another timeline's video. One
    pass, through the same primitive the accent chainer uses — see accents.substitute_pads."""
    def resolve(name: str) -> str:
        if name in subst:
            return subst[name]
        if _EXTERNAL.fullmatch(name):
            raise ValueError(f"fragment {tag!r} reads input pad [{name}] that the allocator never handed out")
        return f"{name}__{tag}" if _INTERNAL.fullmatch(name) else None
    return _accents.substitute_pads(fragment, resolve)


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

def refusals(spec: RenderSpec) -> list[str]:
    """The NAMED non-goals this graph hard-refuses, empty on accept — every named reason is a
    permanent non-goal, not a share to be measured toward a later go/no-go."""
    unimplemented = []
    if any(o.kind == "cover" for o in spec.outputs):
        # A declared cover OUTPUT with no overlays.cover block: this graph never writes a cover.png,
        # so the upload loop would silently skip the deliverable — the half-render this list stops.
        unimplemented.append("outputs[kind=cover]")
    kinds = [o.kind for o in spec.outputs]
    if "receipt" in kinds:
        if spec.mode != "final":
            unimplemented.append("outputs[kind=receipt] on mode=preview")
        elif "master" in kinds and kinds.index("receipt") > kinds.index("master"):
            # The receipt must be PUT before the master it accounts for; the upload loop walks
            # outputs in order, so a master listed first can land with no proof behind it.
            unimplemented.append("outputs[kind=receipt] declared after outputs[kind=master]")
    ov = spec.overlays if spec.mode == "final" else None
    if ov is None:
        return unimplemented
    if ov.trims:
        unimplemented.append("trims")
    if ov.opener is not None:
        unimplemented.append("opener")
    if ov.cover is not None:
        unimplemented.append("cover")
    return unimplemented


def preflight(spec: RenderSpec) -> None:
    """Refuse every non-goal BEFORE any subprocess — same exception type and same timing as
    render.render_spec, so a v6 spec cannot half-render through this door either."""
    _finalize.declared_grid(spec.timeline.fps)  # the ONLY refusal a lost render is worse than
    unimplemented = refusals(spec)
    if unimplemented:
        raise NotImplementedError(
            f"body single-pass graph does not composite these yet (opener/junction waves): {unimplemented}")


def _check_assets(spec: RenderSpec, input_paths: dict) -> None:
    """Every asset the tail names must be a resolved inputs[] id, refused here as a named RuntimeError
    rather than a KeyError deep inside the assembler once the graph is already being built."""
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
    for a in fin.accents:
        if a.kind == "film_burn" and input_paths.get(a.burn) is None:
            raise RuntimeError(f"film_burn accent burn input {a.burn!r} is not resolved")
    if fin.logo is not None and input_paths.get(fin.logo.asset) is None:
        raise RuntimeError(f"finalize.logo.asset {fin.logo.asset!r} is not a resolved inputs[] id")
    if fin.watermark is not None:
        for ref in (fin.watermark.sting, fin.watermark.idle):
            if ref not in input_paths:
                raise RuntimeError(f"finalize.watermark asset {ref!r} is not a resolved inputs[] id")


# Demuxer names ffprobe reports for a single-frame image — the declared kind can lie, this cannot.
_STILL_IMAGE_FORMATS = {"image2", "jpeg_pipe", "png_pipe", "webp_pipe", "bmp_pipe", "gif_pipe", "tiff_pipe"}
# Slack past the downstream trim's own [in_, in_+dur) read, not a real duration.
_LOOP_MARGIN_S = 0.5


def _is_still_image(path: Path) -> bool:
    """A bare `-i` on a single-frame source silently drops its cutaway (root cause) — this catches a mis-declared photo."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=format_name:stream=nb_frames",
             "-of", "default=nw=1", str(path)],
            capture_output=True, text=True, timeout=_finalize._PROBE_TIMEOUT_S).stdout
    except (subprocess.TimeoutExpired, OSError):
        return False
    fmt = nb = ""
    for line in out.splitlines():
        key, _, val = line.partition("=")
        if key == "format_name":
            fmt = val
        elif key == "nb_frames":
            nb = val
    if any(name in fmt.split(",") for name in _STILL_IMAGE_FORMATS):
        return True
    return nb.isdigit() and int(nb) == 1


def _broll_loop_bounds(spec: RenderSpec, input_paths: dict) -> dict[str, float]:
    """input id -> ffmpeg `-t` for every broll cutaway needing `-loop 1`, by declared or probed kind."""
    ov = spec.overlays if spec.mode == "final" else None
    if ov is None or ov.broll_final is None:
        return {}
    kinds = {inp.id: inp.kind for inp in spec.inputs}
    bounds: dict[str, float] = {}
    for c in ov.broll_final.broll:
        if c.dur is None:
            continue
        if kinds.get(c.clip) != "image" and not _is_still_image(input_paths[c.clip]):
            continue
        need = (c.in_ or 0.0) + c.dur + _LOOP_MARGIN_S
        bounds[c.clip] = max(bounds.get(c.clip, 0.0), need)
    return bounds


def tap_pads(spec: RenderSpec) -> list[str]:
    """Every pad this graph taps with a framemd5, in argv order: one per PLANNED cutaway, then the logo's
    own prepared frame. The list is the plan side of the receipt — it is built from the spec, never from
    what the encode happened to produce."""
    pads = list(_render.broll_tap_pads(spec))
    ov = spec.overlays if spec.mode == "final" else None
    fin = ov.finalize if ov is not None else None
    if fin is not None and fin.logo is not None:
        pads.append(V_TAP_LOGO)
    return pads


def tap_frames_expected(spec: RenderSpec) -> dict[str, int]:
    """Frames each tapped window PLANNED to deliver: a cutaway's own trim×fps, and exactly 1 for the logo
    (a still image with no loop flags — render_onepass never gives the logo any)."""
    expected: dict[str, int] = {}
    ov = spec.overlays if spec.mode == "final" else None
    if ov is not None and ov.broll_final is not None:
        fps = spec.timeline.fps
        for i, c in enumerate(ov.broll_final.broll):
            if c.dur is None:
                raise RuntimeError(f"final broll clip {c.clip!r} has no resolved dur")
            expected[f"vtap{i}"] = max(1, round(c.dur * fps))
    if ov is not None and ov.finalize is not None and ov.finalize.logo is not None:
        expected[V_TAP_LOGO] = 1
    return expected


def _check_inputs(spec: RenderSpec, input_paths: dict) -> None:
    """Refuse a resolved timeline/cutaway/burn input with no video stream before ffmpeg's graph init
    buries the same fact in a truncated "Stream specifier … matches no streams" (ticket b34ab41f); a
    missing mapping IS a refusal here, not a silent skip left for assembly's KeyError."""
    to_check: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def collect(rel: str, path: Path | None) -> None:
        if path is None or rel in seen:
            return
        seen.add(rel)
        to_check.append((rel, path))

    for seg in spec.timeline.segments:
        p = input_paths.get(seg.src)
        if p is None:
            raise RuntimeError(f"input {seg.src!r} has no input path resolved")
        collect(seg.src, p)
    ov = spec.overlays if spec.mode == "final" else None
    if ov is not None:
        if ov.broll_final is not None:
            for c in ov.broll_final.broll:
                p = input_paths.get(c.clip)
                if p is None:
                    raise RuntimeError(f"input {c.clip!r} has no input path resolved")
                collect(c.clip, p)
        fin = ov.finalize
        if fin is not None and any(a.kind == "film_burn" for a in fin.accents):
            plan = _accents.film_burn_plan(fin.accents)  # the shape refusal fires before any path/probe I/O
            collect(plan.burn, input_paths.get(plan.burn))
    if not to_check:
        return
    # Bounded by construction: n_unique_inputs × the per-file ffprobe bound — no separate aggregate cap
    # (a healthy 12×6s timeline is 72s of real inputs; capping the SUM below that refuses it for no fault).
    n = len(to_check)
    deadline = time.monotonic() + n * _finalize._PROBE_TIMEOUT_S
    last_probed: str | None = None
    for k, (rel, path) in enumerate(to_check, start=1):
        if not path.exists():
            raise RuntimeError(f"input {rel!r} does not resolve to a file on disk")
        if path.stat().st_size == 0:
            raise RuntimeError(f"input {rel!r} is zero bytes")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"probe budget exhausted after {k - 1} of {n} inputs; last probed {last_probed!r}")
        try:
            has_video = _finalize._has_video(path)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffprobe timed out on {rel!r}") from None
        if not has_video:
            raise RuntimeError(f"input {rel!r} carries no video stream")
        last_probed = rel


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
    flares: tuple[float, ...] = ()
    loop_bounds: dict[str, float] = field(default_factory=dict)
    # Empty unless the spec DECLARES a receipt: an old engine's spec must render byte-for-byte as before,
    # framemd5 files included (there are none).
    tap_md5: dict[str, Path] = field(default_factory=dict)
    receipt_out: Path | None = None


@dataclass(frozen=True)
class Delivered:
    """What the door actually PUT. `master` is the loudnorm's output, which is a DIFFERENT file from
    the encode's — two-pass loudnorm measures a finished file, so it can never join the graph.
    `defect` is the post-render grid verdict — None on a clean master, never a reason to withhold the PUT."""

    prepared: Prepared
    master: Path
    presync: Path | None
    outputs: list[str] = field(default_factory=list)
    defect: dict | None = None


def _voice_filters(spec: RenderSpec, input_paths: dict) -> tuple[str, str]:
    """The voice pre-filter and its measured loudnorm. Inherently serial inside: the dirty probe's
    verdict changes the chain the measure pass must run (the flag is read FIRST — a rescued voice
    skips the probe decode entirely)."""
    voice = input_paths[spec.timeline.segments[0].src]
    dirty = not spec.base_voice_rescued and _render._voice_is_dirty(voice)
    clean = "highpass=f=80" + (",afftdn=nr=8:nf=-30" if dirty else "")
    return clean, _render._measure_loudnorm(voice, clean)


# above the largest single child wall (mograph's 900s node) so a healthy run never trips it
_ARM_WALL_S = 960.0
_ARM_TICK_S = 15.0


def _drain(what: str, running: set, names: dict, stop: float) -> list[str]:
    """After a fault: hold the raise until every RUNNING arm lands, bounded by `stop`; returns the
    arms STILL unlanded at the wall — no outer wall can prove their children dead (a child timeout
    starts at ITS launch), so the caller must LEAK its tmp dir instead of tearing it down."""
    while running:
        left = stop - time.monotonic()
        if left <= 0:
            return sorted(names[f] for f in running)
        print(f"[render] {what}: failing — waiting for {', '.join(sorted(names[f] for f in running))} "
              f"to land first ({left:.0f}s of patience left)", file=sys.stderr, flush=True)
        done, running = cf.wait(running, timeout=min(_ARM_TICK_S, left),
                                return_when=cf.FIRST_COMPLETED)
        for f in done:
            f.exception(timeout=0)  # observed, deliberately dropped: the FIRST fault is the verdict
    return []


def _run_arms(arms: dict) -> dict:
    """Run named independent thunks concurrently → results by name. Bounded and narrated; the first
    arm to fail re-raises its error after unstarted siblings are cancelled and RUNNING ones are
    drained (_drain), so no child outlives the raise. No arms, no pool."""
    if not arms:
        return {}
    t0 = time.monotonic()
    stop = t0 + _ARM_WALL_S
    ex = cf.ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix="prepare")
    futs = {ex.submit(fn): name for name, fn in arms.items()}
    got, secs, pending = {}, {}, set(futs)
    try:
        while pending:
            left = stop - time.monotonic()
            if left <= 0:
                raise RuntimeError(f"prepare: out of patience after {_ARM_WALL_S:.0f}s with "
                                   + ", ".join(sorted(futs[f] for f in pending)) + " still out")
            done, pending = cf.wait(pending, timeout=min(_ARM_TICK_S, left),
                                    return_when=cf.FIRST_COMPLETED)
            for f in done:
                got[futs[f]] = f.result(timeout=0)  # an arm's exception re-raises HERE: first fault wins
                secs[futs[f]] = time.monotonic() - t0
            if pending:
                print(f"[render] prepare: {len(got)}/{len(futs)} arm(s) landed, "
                      f"{', '.join(sorted(futs[f] for f in pending))} still out, "
                      f"{max(0.0, stop - time.monotonic()):.0f}s of patience left",
                      file=sys.stderr, flush=True)
    except BaseException as exc:
        for f in pending:
            f.cancel()
        if left := _drain("prepare", {f for f in pending if not f.cancelled()}, futs, stop):
            exc.unlanded_arms = left  # read by render._job_tmpdir: leak the dir, never race a live child
        raise
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    print("[render] prepare arms: " + " ".join(f"{n}={secs[n]:.1f}s" for n in sorted(secs)),
          file=sys.stderr, flush=True)
    return got


def _write_ass(caps, motion_plan, input_paths: dict, out_dir: Path, w: int, h: int) -> tuple[Path, Path]:
    """The ASS text only — assemble() burns it inline via -vf subtitles=, so there is no ffmpeg pass here."""
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
    """Run every pre-pass the merged encode needs and return them typed. Independent pre-passes run
    as parallel ARMS under ONE `prepare` phase — the voice chain, the bed, the mograph layers and the
    flare scan read none of each other; layers with no frames simply do not appear (mograph.py:143/275)."""
    _check_assets(spec, input_paths)
    _check_inputs(spec, input_paths)
    tmp = Path(tmp)
    dur = body_duration(spec)
    ov = spec.overlays if spec.mode == "final" else None
    mp = ov.motion_plan if ov is not None else None
    music = ov.music if ov is not None else None
    sfx = ov.sfx if ov is not None else None
    caps = mp.captions if mp is not None else None
    fin = ov.finalize if ov is not None else None

    arms: dict = {}
    if music is not None or sfx:
        arms["voice"] = lambda: _voice_filters(spec, input_paths)
    if music is not None:
        arms["bed"] = lambda: _render._prerender_bed(input_paths[music.track], music.start, dur, tmp)
    if mp is not None and mp.sections:
        arms["mograph"] = lambda: tuple(_mograph._render_layers(
            mp.sections, mp.brand.model_dump() if mp.brand else None, input_paths, tmp,
            getattr(mp, "bundle", None)))
    if fin is not None and any(a.kind == "film_burn" for a in fin.accents):
        # The pure shape refusals (film_burn_plan) fire BEFORE the flare decode — cheap checks first.
        plan = _accents.film_burn_plan(fin.accents)
        arms["flares"] = lambda: tuple(_accents.detect_flares(input_paths[plan.burn]))

    with phase("prepare"):
        got = _run_arms(arms)
        ass = font_dir = None
        if caps is not None and caps.words:
            # pure text write, milliseconds — not worth an arm
            ass, font_dir = _write_ass(caps, mp, input_paths, tmp,
                                       spec.timeline.width, spec.timeline.height)

    audio = bed = None
    if "voice" in got:
        clean, vln = got["voice"]
        bed = got.get("bed")
        ids = _render.input_ids(spec)
        audio = _render._AudioMix(
            voice_idx=ids.index(spec.timeline.segments[0].src),
            bed_idx=len(ids) if bed is not None else None,
            clean=clean, vln=vln, dur=dur,
            sfx=tuple((ids.index(s.sound), s.at, s.gain) for s in (sfx or [])))
    tap_md5: dict[str, Path] = {}
    receipt_out: Path | None = None
    if spec.mode == "final" and any(o.kind == "receipt" for o in spec.outputs):
        receipt_out = tmp / "render.receipt.json"
        tap_md5 = {pad: tmp / f"tap_{pad}.framemd5" for pad in tap_pads(spec)}
    return Prepared(
        spec=spec, gpu=gpu, input_paths=input_paths, duration=dur,
        master_out=master_out or tmp / "render.mp4",
        presync_out=presync_out or tmp / "render.presync.mp4",
        filter_script=tmp / "body_onepass.filter",
        bed=bed, audio=audio, layers=got.get("mograph", ()), ass=ass, font_dir=font_dir,
        flares=got.get("flares", ()), loop_bounds=_broll_loop_bounds(spec, input_paths),
        tap_md5=tap_md5, receipt_out=receipt_out)


# --- 3. assemble --------------------------------------------------------------

def assemble(p: Prepared) -> tuple[str, list[str]]:
    """Pure: (filter-script text, argv). No I/O and no probe — every number was computed or crossed on
    the spec, which is what makes the graph a hashable text artifact the goldens can pin."""
    spec, gpu = p.spec, p.gpu
    ov = spec.overlays if spec.mode == "final" else None
    fin = ov.finalize if ov is not None else None
    w, h, fps = spec.timeline.width, spec.timeline.height, spec.timeline.fps
    grid = _finalize.declared_grid(fps)

    pads = tap_pads(spec) if p.tap_md5 else []
    if p.tap_md5 and set(p.tap_md5) != set(pads):
        raise RuntimeError(f"prepared framemd5 taps {sorted(p.tap_md5)} do not match the pads this graph "
                           f"plans to tap {pads}")
    broll_taps = [pad for pad in pads if pad != V_TAP_LOGO]

    inputs = Inputs()
    spec_pads: dict[str, str] = {}
    for iid in _render.input_ids(spec):
        bound = p.loop_bounds.get(iid)
        # A still image otherwise yields one frame at PTS 0 and EOFs before its overlay window opens.
        flags = ("-loop", "1", "-framerate", _render._num(fps), "-t", _render._num(bound)) if bound else ()
        n = inputs.add(p.input_paths[iid], *flags)
        spec_pads[f"{n}:v"] = f"{n}:v"
        spec_pads[f"{n}:a"] = f"{n}:a"
    if p.bed is not None:
        n = inputs.add(p.bed)
        spec_pads[f"{n}:a"] = f"{n}:a"

    # This graph is a reusable subgraph; stamp the final merged output once below, not the
    # [vout] boundary here (rewire would otherwise produce the same merged pad twice).
    composite = (_render.build_filtergraph(spec, gpu, p.audio, terminal_bt709=False, taps=True)
                 if broll_taps else _render.build_filtergraph(spec, gpu, p.audio, terminal_bt709=False))
    chains = [rewire(composite, "cmp",
                     {**spec_pads, "vout": V_COMPOSITE, "aout": A_COMPOSITE,
                      **{pad: pad for pad in broll_taps}})]
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

    if fin is not None and fin.accents:
        src = vlink
        if gpu:
            # The GPU accent branches issue a BARE hwupload (accents.py:81/145/300); the intermediate
            # encode that used to hand them yuv420p is the one this wave deletes.
            chains.append(f"[{vlink}]format=yuv420p[{V_ACCENT_IN}]")
            src = V_ACCENT_IN
        if any(a.kind == "film_burn" for a in fin.accents):
            plan = _accents.film_burn_plan(fin.accents)
            # Looped because add_filmburn trims the burn at flare offsets that may pass the clip's
            # end; -t already bounds the process (the same pairing finalize's own pass uses).
            burn = inputs.add(p.input_paths[plan.burn], "-stream_loop", "-1")
            parts: list[str] = []
            prev = f"[{src}]"
            if plan.singles:
                fc = _accents.build_chain_filter(plan.singles, fps=fps, w=w, h=h, gpu=gpu)
                prefix, terminal = fc.rsplit("[vout]", 1)
                parts = [prefix + "[preburn]" + terminal]
                prev = "[preburn]"
            parts, prev = _accents.add_offset_jump(parts, prev, plan.boundaries, w=w, h=h)
            parts, prev = _accents.add_filmburn(parts, prev, burn, plan.boundaries, list(p.flares),
                                                opacity=plan.opacity, w=w, h=h, fps=grid)
            chains.append(rewire(";".join(parts), "acc",
                                 {"0:v": src, src: src, f"{burn}:v": f"{burn}:v",
                                  prev[1:-1]: V_ACCENTS}))
        else:
            frag = _accents.build_chain_filter(fin.accents, fps=fps, w=w, h=h, gpu=gpu)
            chains.append(rewire(frag, "acc", {"0:v": src, "vout": V_ACCENTS}))
        vlink = V_ACCENTS

    if fin is not None and fin.logo is not None:
        lg = fin.logo
        logo_v = f"{inputs.add(p.input_paths[lg.asset])}:v"
        # body_end is the WHOLE body: cover_hold reserves the welded end-card's tail, and preflight
        # refuses a cover here, so there is no tail to reserve.
        tapped = {"tap_v": V_TAP_LOGO} if V_TAP_LOGO in pads else {}
        frag = _finalize.body_logo_filter(lg.corner, lg.width, lg.opacity, lg.margin, p.duration,
                                          base_v=vlink, logo_v=logo_v, out_v=V_LOGO, **tapped)
        subst = {vlink: vlink, logo_v: logo_v, V_LOGO: V_LOGO}
        if tapped:
            subst[V_TAP_LOGO] = V_TAP_LOGO
        chains.append(rewire(frag, "lgo", subst))
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
            grid=grid, sample_rate=48000, out_v=V_WATERMARK, out_a=A_WATERMARK)
        subst = {vlink: vlink, alink: alink, f"{sting}:v": f"{sting}:v", f"{idle}:v": f"{idle}:v",
                 V_WATERMARK: V_WATERMARK, A_WATERMARK: A_WATERMARK}
        if chime_a is not None:
            subst[chime_a] = chime_a
        chains.append(rewire(frag, "wmk", subst))
        # chime=false leaves the base audio UNTOUCHED (watermark_filter returns no audio label), so
        # the composite mix is mapped straight through. The old path emits -an there and ships a
        # silent master; that divergence is deliberate, not inherited.
        vlink, alink = V_WATERMARK, (out_a or alink)

    # Re-stamp AFTER every overlay stage: the merged graph can reset frame colour metadata downstream of the encoder's own context flags (check_master OFF-CONTRACT without this).
    chains.append(f"[{vlink}]{_finalize._BT709_SET_PARAMS}[{V_MASTER}]")
    vlink = V_MASTER

    t = f"{p.duration:.3f}"
    cmd = ["ffmpeg", "-y", "-hide_banner"]
    if gpu:
        cmd += ["-init_hw_device", "vulkan"]  # libplacebo runs on a Vulkan device; hwupload derives from it
    cmd += inputs.argv()
    cmd += ["-filter_complex_script", str(p.filter_script)]
    # Output options bind to the output they PRECEDE, so each destination carries its whole clause;
    # -t bounds each (shortest=1 over a -stream_loop'ed idle is not a lifetime).
    cmd += ["-map", f"[{vlink}]", "-map", f"[{alink}]",
            "-r", grid, "-fps_mode", "cfr",
            *(_finalize._FINAL_GPU if gpu else _finalize._FINAL_CPU),
            "-ar", "48000", *_MASTER_AUDIO,
            "-movflags", "+faststart", "-t", t, str(p.master_out)]
    if wants_ref:
        # check_sync.py all_frames matches ref against master by frame INDEX, so a different grid here
        # compares different instants — the ref carries the master's OWN -r, never its own.
        cmd += ["-map", f"[{V_PRESYNC}]", "-map", f"[{aref}]",
                "-r", grid, "-fps_mode", "cfr",
                *_REF_VIDEO, "-ar", "48000", *_REF_AUDIO, "-t", t, str(p.presync_out)]
    for pad in pads:
        # passthrough and NO -t: the tap must report the frames its own chain delivered, so a resampled
        # grid or a shared duration bound would answer with the master's timing instead of the overlay's.
        cmd += ["-map", f"[{pad}]", "-fps_mode", "passthrough", "-f", "framemd5", str(p.tap_md5[pad])]
    return ";".join(chains), cmd


# --- 4. the receipt: what this encode ACTUALLY delivered ----------------------

RECEIPT_SCHEMA = "monty.render.receipt/1"
# A build that cannot name itself signs an encode nobody can trace, so these are REFUSALS: a receipt whose
# producer is a placeholder is worse than no receipt, because the gate on the other side would trust it.
_PLACEHOLDER_IMAGE = {"dev", "unknown", "latest", "none", "null"}
_GIT_STAMP_TIMEOUT_S = 10
_SHA40 = re.compile(r"[0-9a-f]{40}")


def _worktree_stamp() -> str:
    """The in-process contour's identity: no image ran this encode, this checkout did."""
    pkg = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(pkg), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=_GIT_STAMP_TIMEOUT_S).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        sha = ""
    if not _SHA40.fullmatch(sha):
        raise RuntimeError(
            "receipt producer: POD_IMAGE_TAG is unset and this tree has no git revision to stamp — "
            "refusing to write a receipt no one can attribute (build with --build-arg IMAGE_TAG=<sha> "
            "or run from a checkout)")
    return f"worktree:{__version__}+{sha}"


def producer() -> dict:
    """WHO wrote this receipt: the image ref baked at build time, else the running checkout."""
    tag = os.environ.get("POD_IMAGE_TAG", "").strip()
    if tag and tag.lower() in _PLACEHOLDER_IMAGE:
        raise RuntimeError(
            f"receipt producer: POD_IMAGE_TAG={tag!r} is a placeholder, not an image identity — refusing "
            "to write a receipt that cannot name the build that encoded it")
    return {"image": tag or _worktree_stamp(), "podagent_version": __version__,
            "spec_version": SPEC_VERSION}


_TB_HEADER = re.compile(r"^#tb 0:\s*(\d+)/(\d+)$")
_FRAMEMD5_COLUMNS = 6


def read_framemd5(pad: str, path: Path, frames_expected: int) -> dict:
    """One tap row, counted from the framemd5 file THIS encode wrote. It reports, it never judges: a
    starved still reads 1 of N and a chain that never ran reads 0, both without raising."""
    if not path.exists():
        raise RuntimeError(f"tap [{pad}]: the encode declared {path} in its own argv and wrote no "
                           "framemd5 file")
    tb: float | None = None
    hashes: list[str] = []
    pts: list[int] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if (m := _TB_HEADER.match(line)) is not None:
                tb = int(m.group(1)) / int(m.group(2))
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != _FRAMEMD5_COLUMNS:
            raise RuntimeError(f"tap [{pad}]: framemd5 row carries {len(cols)} columns, not "
                               f"{_FRAMEMD5_COLUMNS} — the parsed format changed under us: {line!r}")
        try:
            pts.append(int(cols[2]))
        except ValueError:
            raise RuntimeError(f"tap [{pad}]: framemd5 pts column is not an integer: {line!r}") from None
        hashes.append(cols[5])
    if tb is None:
        raise RuntimeError(f"tap [{pad}]: framemd5 file has no '#tb 0: num/den' header, so its pts "
                           "columns name no unit of time")
    return {"pad": pad, "frames_expected": frames_expected, "frames_delivered": len(hashes),
            "pts_first_s": round(pts[0] * tb, 4) if pts else None,
            "pts_last_s": round(pts[-1] * tb, 4) if pts else None,
            "distinct_hashes": len(set(hashes)),
            "first_last_differ": bool(hashes) and hashes[0] != hashes[-1]}


_PROBE_ENTRIES = ("format=format_name,duration:"
                  "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames")


def _input_row(iid: str, path: Path) -> dict:
    """Header-only ffprobe row for one input. A probe failure is a FIELD, not an exception: the encode is
    already paid for by the time this runs (same contract as grid_verdict)."""
    row: dict = {"id": iid, "path": str(path)}
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", _PROBE_ENTRIES, "-of", "json",
                              str(path)], capture_output=True, text=True,
                             timeout=_finalize._PROBE_TIMEOUT_S)
        data = json.loads(out.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        row["probe_failed"] = safe_text(str(exc))
        return row
    fmt = data.get("format") or {}
    row["format_name"] = fmt.get("format_name")
    row["duration_s"] = fmt.get("duration")
    row["streams"] = [{k: s.get(k) for k in ("index", "codec_type", "codec_name", "width", "height",
                                             "r_frame_rate", "nb_frames")}
                      for s in (data.get("streams") or [])]
    return row


def _broll_rows(spec: RenderSpec, graph: str) -> list[dict]:
    """One row per PLANNED cutaway, each cross-checked against the graph the encoder got: a row whose
    chain or enable window is absent from that text would be a claim about an overlay that never rode."""
    ov = spec.overlays if spec.mode == "final" else None
    if ov is None or ov.broll_final is None:
        return []
    idx = {iid: n for n, iid in enumerate(_render.input_ids(spec))}
    rows: list[dict] = []
    for i, c in enumerate(ov.broll_final.broll):
        start, end = c.start, c.start + (c.dur or 0.0)
        enable = f"between(t,{start:.3f},{end:.3f})"
        # The label is read back AS MERGED (rewire namespaces a builder's internal pads), so the row
        # names the chain the encoder got rather than the one the builder meant to hand it.
        found = re.search(rf"\[(b{i}(?:__[A-Za-z0-9_]+)?)\]", graph)
        if found is None or enable not in graph:
            raise RuntimeError(f"planned cutaway {c.clip!r} has no chain [b{i}] enabled over "
                               f"[{start:.3f},{end:.3f}) in the filtergraph this encode ran")
        rows.append({"clip": c.clip, "input_index": idx[c.clip], "start": round(start, 3),
                     "end": round(end, 3), "enable": enable, "chain_label": found.group(1),
                     "tap_pad": f"vtap{i}"})
    return rows


def _logo_row(spec: RenderSpec, body_end: float, pads: list[str]) -> dict | None:
    ov = spec.overlays if spec.mode == "final" else None
    fin = ov.finalize if ov is not None else None
    if fin is None or fin.logo is None:
        return None
    return {"input_id": fin.logo.asset, "enable": f"lt(t,{body_end:.3f})",
            "tap": V_TAP_LOGO if V_TAP_LOGO in pads else None}


def _refuse_secret_leak(receipt: dict, spec: RenderSpec) -> None:
    """An output's PUT url is a credential: it rides the spec and nothing else, so it can never appear in
    the object the box stores and the logs print."""
    text = json.dumps(receipt)
    for o in spec.outputs:
        if "://" in o.put_url and o.put_url in text:
            raise RuntimeError(f"receipt would carry the signed put_url of output {o.id!r} — refusing")


def build_receipt(p: Prepared, graph: str, cmd: list[str], wall_s: float) -> dict:
    """The executed truth of ONE encode: the argv and graph that ran, the build that ran them, a row per
    planned overlay and the frames each tap actually delivered. The pod COUNTS; the verdict is the box's."""
    spec = p.spec
    expected = tap_frames_expected(spec)
    pads = tap_pads(spec)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "job_id": spec.job_id,
        "slug": spec.slug,
        "mode": spec.mode,
        "producer": producer(),
        "argv": list(cmd),
        "filtergraph": graph,
        # Every DECLARED input, not just the ones the graph decodes: the logo is the asset whose silent
        # absence this whole receipt exists to catch, and it is not a timeline source.
        "inputs": [_input_row(i.id, p.input_paths[i.id]) for i in spec.inputs
                   if i.id in p.input_paths],
        "overlays": {"broll": _broll_rows(spec, graph)},
        "logo": _logo_row(spec, p.duration, pads),
        "taps": [read_framemd5(pad, p.tap_md5[pad], expected[pad]) for pad in pads],
        "wall": round(wall_s, 3),
    }
    _refuse_secret_leak(receipt, spec)
    return receipt


# --- the door -----------------------------------------------------------------

def encode_budget_s(duration: float) -> float:
    """The bound on the single encode: proportional to the WORK, because the failure it catches is a
    graph that never ends (a lost -t against a looped input), not a slow box."""
    return _ENCODE_FLOOR_S + _ENCODE_REALTIME_X * max(0.0, duration)


def _speed_line(stderr: bytes) -> str | None:
    """Last ffmpeg progress line — capture_output eats stderr on success, and with it the only record of
    the single pass's actual throughput (speed=/fps=)."""
    for line in reversed((stderr or b"").splitlines()):
        if b"speed=" in line or b"fps=" in line:
            return line.decode("utf-8", "replace").strip()
    return None


_FAILURE_HEAD_MARKERS = ("Stream specifier", "Error initializing complex filters", "Invalid argument")


def _ffmpeg_failure_message(returncode: int, stderr: bytes | str | None) -> str:
    """Bound the failure: the tail keeps ffmpeg's terminal cause, the head keeps the specifier/label
    ffmpeg names FIRST — found anywhere in the FULL text, since only searching an already-cut tail
    lost a specifier sitting in a short stderr's middle (ticket b34ab41f)."""
    raw = stderr or b""
    rendered = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    # Scrub before bounding: a credential-bearing URL may begin before any cut, and cutting first
    # could leave its sensitive suffix behind.
    scrubbed = safe_text(rendered)
    prefix = f"body single-pass ffmpeg exited {returncode}: RuntimeError: "
    # main.py wraps this once more with safe_error(...), whose own 500-char cap must not trim the
    # terminal diagnostic we preserve here.
    room = max(0, 500 - len("RuntimeError: ") - len(prefix))
    if len(scrubbed) <= room:
        return prefix + scrubbed
    marker = " … "
    idx = -1
    for m in _FAILURE_HEAD_MARKERS:   # priority order, NOT earliest byte offset (Stream specifier wins)
        i = scrubbed.find(m)
        if i != -1:
            idx = i
            break
    if idx == -1:
        tail = scrubbed[-2000:]
        split_room = max(0, room - len(marker))
        head_room = split_room // 2
        return prefix + tail[:head_room] + marker + tail[-(split_room - head_room):]
    line_start = scrubbed.rfind("\n", 0, idx) + 1
    head_full = scrubbed[line_start:]
    budget = max(0, room - len(marker))
    head_budget = min(200, len(head_full), budget)
    tail_budget = min(250, len(scrubbed), max(0, budget - head_budget))
    head = head_full[:head_budget].strip()
    tail = scrubbed[-tail_budget:] if tail_budget else ""
    cleaned = f"{head}{marker}{tail}" if tail else head
    return prefix + cleaned


def _run(cmd: list[str], budget_s: float) -> None:
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, timeout=budget_s)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(_ffmpeg_failure_message(exc.returncode, exc.stderr)) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"body single-pass ffmpeg exceeded its {budget_s:.0f}s budget — a graph that cannot end "
            f"(check -t against the looped watermark idle)") from exc
    if (tail := _speed_line(proc.stderr)) is not None:
        print(f"[onepass] {tail}", file=sys.stderr, flush=True)


def run_encode(p: Prepared, phase=_no_phase) -> dict | None:
    """Write the assembled filter script and run the ONE bounded encode — the whole encode core
    behind the router, so render_spec and render_body cannot diverge on it. Returns the receipt of what
    that encode delivered when the spec declared one (also written to `p.receipt_out`), else None."""
    graph, cmd = assemble(p)
    p.filter_script.write_text(graph, encoding="utf-8")
    # Owner rule (all logs visible): the exact argv/filtergraph this encode runs, once, before it runs.
    print(f"[onepass] argv: {cmd}", file=sys.stderr, flush=True)
    print(f"[onepass] filter_complex: {graph}", file=sys.stderr, flush=True)
    with phase("ffmpeg"):
        t0 = time.monotonic()
        _run(cmd, encode_budget_s(p.duration))
        wall = time.monotonic() - t0
    if p.receipt_out is None:
        return None
    # Serialized only HERE, after _run returned: a receipt built any earlier could describe an encode
    # that never happened.
    receipt = build_receipt(p, graph, cmd, wall)
    p.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    taps = " ".join(f"{t['pad']}={t['frames_delivered']}/{t['frames_expected']}" for t in receipt["taps"])
    print(f"[onepass] receipt: image={receipt['producer']['image']} taps: {taps or 'none planned'}",
          file=sys.stderr, flush=True)
    return receipt


def render_body(spec: RenderSpec, input_paths: dict, tmp: Path, gpu: bool, *,
                master_out: Path | None = None, presync_out: Path | None = None,
                phase=_no_phase) -> Delivered:
    """Refuse the non-goals, prepare, ONE ffmpeg for the delivery-rung master (+ the pre-accent sync
    reference when one is declared), THEN the delivery loudnorm, then PUT what was declared. `phase`
    is the render stage's per-op event seam (render.py:517-551), a no-op when nobody listens."""
    preflight(spec)
    p = prepare(spec, input_paths, tmp, gpu, master_out=master_out, presync_out=presync_out,
                phase=phase)
    receipt = run_encode(p, phase=phase)
    fin = spec.overlays.finalize if (spec.mode == "final" and spec.overlays is not None) else None
    master = p.master_out
    if fin is not None:
        # The delivery level is a TWO-PASS loudnorm: it measures the finished file, so it cannot join
        # the graph and must not be skipped either — final_dispatch sets this block on every final
        # spec, and shipping the encode raw is every deliverable ~6 dB under the brand target.
        with phase("finalize"):
            master = _finalize.apply_loudnorm(fin, master, Path(tmp) / "fin_ln.mp4")
    # Header-only, after the paid-for encode and before the PUT: never withholds the master, only
    # reports on it (see grid_verdict's own contract for why it cannot raise).
    defect = _finalize.grid_verdict(master, spec.timeline.fps)
    if defect is not None:
        print(f"[render_onepass] grid verdict: declared vs measured mismatch {defect}", file=sys.stderr)
    done: list[str] = []
    with phase("upload"):
        for o in spec.outputs:
            if o.kind in ("cache", "cover"):
                continue
            if o.kind == "presync":
                upload(p.presync_out, o.put_url, "video/mp4")
            elif o.kind == "receipt":
                if receipt is None or p.receipt_out is None:
                    raise RuntimeError(f"output {o.id!r} kind=receipt has no producer on this run")
                upload(p.receipt_out, o.put_url, "application/json")
            else:
                upload(master, o.put_url, "video/mp4")
            done.append(o.id)
    return Delivered(prepared=p, master=master,
                     presync=p.presync_out if any(o.kind == "presync" for o in spec.outputs) else None,
                     outputs=done, defect=defect)
