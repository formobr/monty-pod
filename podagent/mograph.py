"""Mograph overlays on the pod: render motion_plan.sections via the (brand-agnostic) Remotion bundle the
job delivers — brand crosses through inputProps, role fonts + section media are staged into the bundle
public/. Each section packs to a transparent qtrle layer, overlaid onto the base gated to its
[start,start+dur] window. The bundle is a per-job input cached by content hash, not image ballast: see
bundle.py."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

FPS = 30
_STAGE_PREFIX = "mograph/"  # input id `mograph/<rel>` → staged into <bundle>/<rel> (public/ fonts+media, src/ bespoke)

# HEAD BELOW (any media riding over a slid-down head): the live head slides DOWN so its face clears the top-band
# picture/video, reversing at the beat end. Closed form of MontagePreview.headSettle (GLIDE spring) → master == preview.
# Bound COPIES of the SSOT in remotion/src/MontagePreview.tsx (HB_DROP_FRAC / HEAD_SETTLE_SEC) — the pod
# renders ffmpeg, not TS. The engine's tests/test_head_settle_ssot.py text-reads this file and reddens on drift.
HB_DROP_FRAC = 0.33          # frame-fraction the face travels down → seats at 0.75
HEAD_SETTLE_SEC = 0.4        # settle window at beat start; reversed at beat end
_HB_ZW0 = 15 / (2 * 0.9)     # GLIDE {damping:15, mass:0.9, stiffness:140} underdamped closed form (== engine)
_HB_WD = (140 / 0.9) ** 0.5 * (1 - (15 / (2 * (140 * 0.9) ** 0.5)) ** 2) ** 0.5
_HB_B = _HB_ZW0 / _HB_WD


def _hb_settle_y_expr(s: float, e: float) -> str:
    """ffmpeg overlay-y = 0.33·H · head-slide progress: springs down at s, reverses at e-HEAD_SETTLE_SEC. H is the
    base height at overlay time, so the drop is resolution-independent (== MontagePreview's 33% of frame)."""
    def spring(t0: float) -> str:
        tau = f"max(t-{t0:.5f}\\,0)"
        return f"(1-exp(-{_HB_ZW0:.5f}*{tau})*(cos({_HB_WD:.5f}*{tau})+{_HB_B:.5f}*sin({_HB_WD:.5f}*{tau})))"
    emf = e - HEAD_SETTLE_SEC
    up = f"if(gte(t\\,{s:.5f})\\,{spring(s)}\\,0)"
    down = f"if(gte(t\\,{emf:.5f})\\,{spring(emf)}\\,0)"
    return f"{HB_DROP_FRAC}*H*clip(({up})-({down})\\,0\\,1.2)"


def remotion_dir(ref, tmp: Path) -> Path:
    """A writable Remotion project for THIS job.

    The bundle is not baked into the image (504 MB most jobs never touch) — it arrives per job as a
    presigned tar, is cached by content hash, and each job renders out of its own workspace over that
    cache (bundle.workspace explains why the copy is not optional). MONTY_REMOTION_DIR still overrides
    with a local tree so `worker.py --local` runs against the repo's own remotion/ unchanged.
    """
    if d := os.environ.get("MONTY_REMOTION_DIR"):
        cand = Path(d)
        if not (cand / "render_batch.mjs").is_file():
            raise RuntimeError(f"MONTY_REMOTION_DIR={cand} holds no render_batch.mjs")
        return cand
    if ref is None:
        # Unreachable through the contract (SpecMotionPlan rejects sections without a bundle); kept so a
        # hand-built call fails with the reason rather than an AttributeError.
        raise RuntimeError("no Remotion bundle for this job and no MONTY_REMOTION_DIR — cannot render mograph")
    from . import bundle
    return bundle.workspace(bundle.ensure(ref), tmp / "remotion")


def _stage_public(input_paths: dict, rd: Path) -> None:
    """Copy every `mograph/<rel>` input into <bundle>/<rel> (public/ fonts+media, src/ bespoke .tsx+entry)."""
    for iid, path in input_paths.items():
        if iid.startswith(_STAGE_PREFIX):
            dest = rd / iid[len(_STAGE_PREFIX):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def _render_concurrency() -> int:
    """Chrome tabs to run in parallel. Fixed at 4 this box's size never mattered: a 28-core host idled at 14%
    while a 2-core one thrashed. Each tab needs ~2 GB, so RAM caps it as hard as cores do."""
    cores = os.cpu_count() or 4
    ram_gb = 0.0
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1 << 30)
    except (ValueError, OSError):
        pass
    by_ram = int(ram_gb // 2) if ram_gb else cores
    return max(2, min(cores - 2, by_ram, 16))


def _run_batch(rd: Path, items: list, spec_path: Path, entry_point: str | None) -> None:
    body = {"concurrency": _render_concurrency(), "items": items}
    if entry_point:
        body["entryPoint"] = entry_point
    spec_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run(["node", "render_batch.mjs", str(spec_path)], cwd=rd, capture_output=True)
    # JUDGE THE FRAMES, NOT THE EXIT CODE: headless Chrome aborts on teardown often enough (SIGABRT after every
    # sequence is already on disk) that a rc check throws away finished work and fails the whole master.
    missing = [str(i.get("comp")) for i in items if not any(Path(i["seqdir"]).glob("*.png"))]
    if not missing:
        if r.returncode:
            print(f"mograph: render_batch exited {r.returncode} AFTER writing every sequence "
                  f"(Chrome teardown) — keeping the frames", file=sys.stderr)
        return
    tail = ((r.stderr or b"") + (r.stdout or b""))[-3000:].decode("utf-8", "replace")
    raise RuntimeError(f"render_batch exited {r.returncode} with no frames for {', '.join(missing)}: {tail}")


def _pack(metas: list[dict], tmp: Path) -> list[dict]:
    layers = []
    for m in metas:
        pngs = sorted(m["seqdir"].glob("*.png"))
        if not pngs:
            continue
        mov = tmp / f"{m['seqdir'].name}.mov"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-pattern_type", "glob",
                        "-i", str(m["seqdir"] / "*.png"), "-c:v", "qtrle", str(mov)], check=True)
        layers.append({"mov": str(mov), "start": m["start"], "dur": len(pngs) / FPS,
                       "glass": m["glass"], "head_below": m.get("head_below", False),
                       "backing": m.get("backing")})
    return layers


def _render_layers(sections: list, brand: dict | None, input_paths: dict, tmp: Path,
                   bundle_ref=None) -> list[dict]:
    """Render sections to transparent qtrle layers: catalog comps in one bundle+Chrome batch; each Bespoke
    (LLM .tsx delivered + staged by the brain) via its own per-job entry. A missing bespoke entry = skip loud."""
    rd = remotion_dir(bundle_ref, tmp)
    _stage_public(input_paths, rd)
    tok = (brand or {}).get("tokens")
    fnt = (brand or {}).get("fonts")

    def _props(sec):
        p = dict(sec.props or {})
        if tok is not None:
            p["brandTokens"] = tok
        if fnt is not None:
            p["brandFonts"] = fnt
        return p

    cat_items, cat_metas, bespoke = [], [], []
    for i, sec in enumerate(sections):
        seqdir = tmp / f"seq{i}"
        seqdir.mkdir(parents=True, exist_ok=True)
        meta = {"seqdir": seqdir, "start": float(sec.start), "glass": bool(sec.glass),
                "head_below": bool((sec.props or {}).get("headBelow")),
                "backing": (sec.props or {}).get("backing")}
        item = {"comp": sec.comp, "props": _props(sec), "seqdir": str(seqdir)}
        if sec.comp.startswith("Bespoke"):
            entry = f"src/index.bespoke.{sec.comp}.tsx"
            if not (rd / entry).is_file():
                print(f"mograph: SKIP {sec.comp} @ {sec.start}s — no delivered entry", file=sys.stderr)
                continue
            bespoke.append((entry, item, meta))
        else:
            cat_items.append(item)
            cat_metas.append(meta)

    metas: list[dict] = []
    if cat_items:
        _run_batch(rd, cat_items, tmp / "batch_catalog.json", None)
        metas += cat_metas
    for n, (entry, item, meta) in enumerate(bespoke):
        _run_batch(rd, [item], tmp / f"batch_bespoke{n}.json", entry)
        metas.append(meta)
    return _pack(metas, tmp)


# ── MARK BACKING (the final half) ────────────────────────────────────────────────────────────────────────
# A brand mark gets its background under ITSELF, not under the whole frame: frosted glass
# scoped to the mark's box, or a shadow when the mark is light enough to read alone. The rect is NOT computed
# here — the planner bakes keyframes and both tiers interpolate them (anim_expr here, the same
# piecewise-linear rule in the browser), so the panel and the mark cannot land in different places.
# These are the ENGINE glass numbers reused verbatim (see the full-frame branch below) — one look, not a third.
_MARK_GLASS = {"sigma": 22, "brightness": -0.05}


# anim_expr reads only `.t` and `.rect`, and this rect is NOT models.MotionKeyframe's: a camera crop lives
# inside the frame (rect >= 0), while a mark ARRIVES from off-frame, so its x is negative for half a second.
_KF = namedtuple("_KF", "t rect")


def _kfs(backing: dict, start: float) -> list:
    return [_KF(float(k["t"]) + float(start), tuple(float(v) for v in k["rect"]))
            for k in (backing.get("motion") or [])]


def _esc(expr: str) -> str:
    """Commas inside a filter-arg expression must not be read as argument separators."""
    return expr.replace(",", "\\,")


def mark_glass_filters(i: int, src: str, backing: dict, start: float, end: float) -> tuple[list[str], str]:
    """Blur+darken ONLY the mark's own box, following it, gated to [start+from, end].

    Blur the whole frame once (gated to the window, so it costs nothing outside it), CROP the mark's moving box
    out of the blurred copy, and overlay that one box back onto the sharp frame at the same moving position.
    `from` is when the box has finished arriving on screen: a panel parked at the frame edge waiting for a mark
    that is still off-screen is the bug this avoids, and the browser gates on the very same number.

    The box SIZE is a constant of the backing (only its position animates) and is emitted as one — crop
    evaluates w/h once at configuration time, so an expression there would silently freeze at t=0 anyway."""
    from .render import anim_expr

    kfs = _kfs(backing, start)
    if not kfs:
        return [], src
    bw, bh = kfs[0].rect[2], kfs[0].rect[3]
    on = float(start) + float(backing.get("from") or 0.0)
    win = f"between(t,{on},{end})"
    # crop's box is in SOURCE pixels (in_w/in_h); overlay's in main-frame pixels (W/H). Same numbers, two
    # vocabularies — anim_expr takes the dimension expr, so each gets its own rather than a string rewrite.
    cx = _esc(f"clip({anim_expr(kfs, 0, 'linear', 'in_w')},0,in_w-out_w)")
    cy = _esc(f"clip({anim_expr(kfs, 1, 'linear', 'in_h')},0,in_h-out_h)")
    ox = _esc(f"clip({anim_expr(kfs, 0, 'linear', 'W')},0,W-w)")
    oy = _esc(f"clip({anim_expr(kfs, 1, 'linear', 'H')},0,H-h)")
    return ([f"[{src}]split[mk{i}a][mk{i}b]",
             f"[mk{i}b]gblur=sigma={_MARK_GLASS['sigma']}:enable='{win}',"
             f"eq=brightness={_MARK_GLASS['brightness']}:enable='{win}',"
             f"crop=w='in_w*{bw}':h='in_h*{bh}':x='{cx}':y='{cy}'[mk{i}c]",
             f"[mk{i}a][mk{i}c]overlay=x='{ox}':y='{oy}':enable='{win}'[mk{i}v]"], f"mk{i}v")


# The DISPATCH TABLE this tier draws a mark backing through. `None` = nothing to draw HERE because the shared
# Photo component already drew it into the alpha layer (a shadow follows the mark's silhouette and needs no
# knowledge of the frame behind it). The keys are the parity contract: the engine's gate demands they equal
# the player's table and the planner's vocabulary, so no treatment can be one-sided. Declared after the
# builder so the table holds the function object, not a name.
_MARK_BACKING = {"glass": lambda i, src, b, s, e: mark_glass_filters(i, src, b, s, e),
                 "shadow": None}


def overlay_filtergraph(layers: list[dict], *, base: str = "0:v",
                        layers_v: list[str] | None = None) -> tuple[str, str]:
    """Pure: (-filter_complex string, final video label) compositing alpha layers onto [base]. Each layer is
    shifted to its start and gated to [start,start+dur]; a glass layer blurs+darkens the frame behind it; a
    head_below layer slides the base head down first, so the layer's picture rides over the cleared top band."""
    # base/layers_v exist so a MERGED graph can hand real pad names: the defaults are this pass's own
    # -i order, which stops being 0/1/2… the moment the composite's inputs sit in front of the layers.
    filters, src = [], base
    for i, lay in enumerate(layers):
        s, e = lay["start"], lay["start"] + lay["dur"]
        idx = layers_v[i] if layers_v is not None else f"{i + 1}:v"
        if lay.get("glass"):
            # frosted-glass takeover: blur+darken the frame behind the card, gated to its window (parity with engine _composite).
            filters.append(f"[{src}]gblur=sigma=22:enable='between(t,{s},{e})',"
                           f"eq=brightness=-0.05:enable='between(t,{s},{e})'[g{i}]")
            src = f"g{i}"
        backing = lay.get("backing") or None
        if backing and _MARK_BACKING.get(str(backing.get("treatment"))) is not None:
            add, src = _MARK_BACKING[str(backing["treatment"])](i, src, backing, s, e)
            filters += add
        if lay.get("head_below"):
            # slide a copy of the base head DOWN over its window; the layer (SplitScreen alpha, overlaid next)
            # then covers the cleared top band. Mirrors MontagePreview / the retired engine _composite.
            yexpr = _hb_settle_y_expr(s, e)
            filters.append(f"[{src}]split[ha{i}][hb{i}];"
                           f"[hb{i}]trim=start={s}:end={e},setpts=PTS-STARTPTS+{s}/TB[hw{i}];"
                           f"[ha{i}][hw{i}]overlay=y='{yexpr}':enable='between(t,{s},{e})':eof_action=pass[hbv{i}]")
            src = f"hbv{i}"
        filters.append(f"[{idx}]setpts=PTS-STARTPTS+{s}/TB[o{i}];"
                       f"[{src}][o{i}]overlay=enable='between(t,{s},{e})':eof_action=pass[v{i}]")
        src = f"v{i}"
    return ";".join(filters), src


def _overlay(base: Path, layers: list[dict], out: Path, gpu: bool, enc) -> Path:
    """Composite alpha layers onto base in one ffmpeg pass (mirrors the engine _composite)."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-i", str(base)]
    for lay in layers:
        cmd += ["-i", lay["mov"]]
    fc, last = overlay_filtergraph(layers)
    from .render import _venc
    cmd += ["-filter_complex", fc, "-map", f"[{last}]", "-map", "0:a?"]
    cmd += _venc(enc, gpu) + ["-c:a", "copy", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-2000:]
        raise RuntimeError(f"mograph overlay ffmpeg exited {exc.returncode}: "
                           f"{tail.decode('utf-8', 'replace')}") from exc
    return out


def composite(motion_plan, base: Path, input_paths: dict, out: Path, gpu: bool, enc, tmp: Path) -> Path:
    """Render motion_plan.sections and overlay them onto `base`. Returns `base` unchanged if nothing rendered."""
    layers = _render_layers(motion_plan.sections, motion_plan.brand.model_dump() if motion_plan.brand else None,
                            input_paths, tmp, getattr(motion_plan, "bundle", None))
    return _overlay(base, layers, out, gpu, enc) if layers else base
