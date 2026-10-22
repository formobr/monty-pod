"""Mograph overlays on the pod: render motion_plan.sections via the (brand-agnostic) Remotion bundle the
job delivers — brand crosses through inputProps, role fonts + section media are staged into the bundle
public/. Each section packs to a transparent qtrle layer, overlaid onto the base gated to its
[start,start+dur) output-frame range. The bundle is a per-job input cached by content hash, not image ballast: see
bundle.py."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
from collections import namedtuple
from pathlib import Path

from . import frames as _frames

FPS = 30
_BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
_STAGE_PREFIX = "mograph/"  # input id `mograph/<rel>` → staged into <bundle>/<rel> (public/ fonts+media, src/ bespoke)

# HEAD BELOW (any media riding over a slid-down head): the live head slides DOWN so its face clears the top-band
# picture/video, reversing at the beat end. Closed form of MontagePreview.headSettle (GLIDE spring) → master == preview.
# Bound COPIES of the SSOT in the engine's MontagePreview component (HB_DROP_FRAC / HEAD_SETTLE_SEC) — the pod
# renders ffmpeg, not TS. The engine's own head-settle SSOT test text-reads this file and reddens on drift.
HB_DROP_FRAC = 0.33          # frame-fraction the face travels down → seats at 0.75
HEAD_SETTLE_SEC = 0.4        # settle window at beat start; reversed at beat end
# CRITICALLY damped {mass:0.9, stiffness:140, damping:2√(km)}: S(τ)=1-e^(-ω0τ)(1+ω0τ). The underdamped GLIDE
# passed rest by ~20% at BOTH edges and a run of mographs read as the head JUMPING (fbr ff22e5e4).
_HB_W0 = (140 / 0.9) ** 0.5


def _hb_spring(tau: float) -> float:
    """Settle 0→1 at local time tau≥0 seconds — monotone, never past 1 (no overshoot to read as a jump)."""
    if tau <= 0:
        return 0.0
    return 1 - math.exp(-_HB_W0 * tau) * (1 + _HB_W0 * tau)


def _hb_settle_progress(t: float, s: float, e: float) -> float:
    """Head-slide progress at absolute time t over beat [s,e]: springs down at s, reverses at e-HEAD_SETTLE_SEC.
    0 = full frame (face high), 1 = seated in the bottom band. Mirrors MontagePreview.headSettle."""
    return max(0.0, min(1.0, _hb_spring(t - s) - _hb_spring(t - (e - HEAD_SETTLE_SEC))))


def _hb_settle_y_expr(s: float, e: float) -> str:
    """ffmpeg overlay-y = 0.33·H · head-slide progress — the closed form of _hb_settle_progress in t. H is the
    base height at overlay time, so the drop is resolution-independent (== MontagePreview's 33% of frame)."""
    def spring(t0: float) -> str:
        tau = f"max(t-{t0:.5f}\\,0)"
        return f"(1-exp(-{_HB_W0:.5f}*{tau})*(1+{_HB_W0:.5f}*{tau}))"
    emf = e - HEAD_SETTLE_SEC
    up = f"if(gte(t\\,{s:.5f})\\,{spring(s)}\\,0)"
    down = f"if(gte(t\\,{emf:.5f})\\,{spring(emf)}\\,0)"
    return f"{HB_DROP_FRAC}*H*clip(({up})-({down})\\,0\\,1)"


def remotion_dir(ref, tmp: Path) -> Path:
    """A writable Remotion project for THIS job.

    The bundle is not baked into the image (504 MB most jobs never touch) — it arrives per job as a
    presigned tar, is cached by content hash, and each job renders out of its own workspace over that
    cache (bundle.workspace explains why the copy is not optional). MONTY_REMOTION_DIR still overrides
    with a local tree so the engine's own worker CLI run locally runs against its own Remotion tree unchanged.
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


class StagedInputNotAllowed(ValueError):
    """A staged mograph input id that would write outside this job's own render workspace."""


def _stage_dest(rd: Path, iid: str) -> Path:
    """<rd>/<rel> for input id `mograph/<rel>`, refusing anything that could escape `rd` — including through
    the `node_modules` symlink into bundle.py's SHARED cache (bundle.py:11-19,34,66-69), which a plain
    `resolve()` on the joined path still catches because it resolves that already-existing symlink too."""
    rel = iid[len(_STAGE_PREFIX):]
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        raise StagedInputNotAllowed(
            f"REFUSED staged input id {iid!r}: a staged input id must name a RELATIVE path under the "
            f"staging root — no absolute component, no `..` segment. A legitimate id looks like "
            f"mograph/public/_photo/x.jpg (the engine's final dispatch mints these).")
    dest = rd / rel
    rd_real, dest_real = rd.resolve(), dest.resolve()
    if dest_real != rd_real and not dest_real.is_relative_to(rd_real):
        raise StagedInputNotAllowed(
            f"REFUSED staged input id {iid!r}: {dest} resolves to {dest_real}, outside this job's staging "
            f"root {rd_real} — likely a symlink (node_modules) walked it into the shared bundle cache.")
    return dest


def _stage_public(input_paths: dict, rd: Path) -> None:
    """Copy every `mograph/<rel>` input into <bundle>/<rel> (public/ fonts+media, src/ bespoke .tsx+entry)."""
    for iid, path in input_paths.items():
        if iid.startswith(_STAGE_PREFIX):
            dest = _stage_dest(rd, iid)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


_CGROUP_MEM_LIMIT = (Path("/sys/fs/cgroup/memory.max"),                      # cgroup v2
                     Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))    # cgroup v1
_CGROUP_MEM_CURRENT = (Path("/sys/fs/cgroup/memory.current"),
                       Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
_OFFTHREAD_CACHE_ENV = "MONTY_OFFTHREAD_CACHE_MB"


def _cgroup_bytes(paths: tuple[Path, ...]) -> int | None:
    """First readable numeric value among `paths`; None = no limit set ("max") or nothing readable."""
    for p in paths:
        try:
            raw = p.read_text().strip()
        except OSError:
            continue
        if raw.isdigit():
            return int(raw)
    return None


def _effective_ram_bytes() -> int:
    """RAM THIS CONTAINER may actually use: the cgroup limit when it is below the machine total.
    sysconf answers for the whole rented host — sizing Chrome off it is how the kernel ends up
    OOM-killing the render on a memory-capped pod."""
    host = 0
    try:
        host = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    lim = _cgroup_bytes(_CGROUP_MEM_LIMIT)
    if lim is not None and (host <= 0 or lim < host):
        return lim
    return host


def _render_concurrency() -> int:
    """Chrome tabs to run in parallel, sized to the box: a 28-core host idled at 14% when this was fixed
    at 4, a 2-core one thrashed. Each tab needs ~2 GB, so RAM caps it as hard as cores do — container
    (cgroup/cpuset) numbers, not the host's."""
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 4
    ram_gb = _effective_ram_bytes() / (1 << 30)
    by_ram = int(ram_gb // 2) if ram_gb else cores
    return max(2, min(cores - 2, by_ram, 16))


def _offthread_cache_bytes() -> int:
    """Explicit OffthreadVideo frame-cache bound handed to render_batch.mjs. Unset, Remotion's compositor
    sizes it from host meminfo (cgroup-blind) — with a 4K source that alone can eat the container. A 4K
    RGBA frame is ~33 MB, so even the 512 MiB floor holds ~15 frames of lookahead."""
    if raw := os.environ.get(_OFFTHREAD_CACHE_ENV):
        try:
            if (mb := int(raw)) > 0:
                return mb << 20
        except ValueError:
            pass
    return max(512 << 20, min(2 << 30, _effective_ram_bytes() // 8))


def _oom_count() -> int | None:
    """The container's kernel oom_kill counter as a number, None when the cgroup will not say."""
    from . import main as _main   # lazy: keep mograph importable without the agent's boot wiring
    got = _main._oom_kills()
    return int(got) if got.isdigit() else None


def _box_facts(oom_before: int | None, conc: int, cache: int, tmp: Path) -> str:
    """The box's state at the moment a render died, compact enough to SURVIVE THE SEAM: safe_error cuts
    the message at 500 chars, so anything that must reach the control plane rides the front, not a tail."""
    after = _oom_count()
    if oom_before is None or after is None:
        oom = "oom=?"
    else:
        oom = f"oom+{after - oom_before}" if after > oom_before else "oom+0"
    cur = _cgroup_bytes(_CGROUP_MEM_CURRENT)
    lim = _effective_ram_bytes()
    mem = f"mem={'?' if cur is None else f'{cur / (1 << 30):.1f}'}/{lim / (1 << 30):.1f}GiB"
    try:
        du = shutil.disk_usage(tmp)
        disk = f"tmp={du.free / (1 << 30):.1f}GiB-free"
    except OSError:
        disk = "tmp=?"
    return f"{oom} {mem} {disk} tabs={conc} cache={cache >> 20}MiB"


_BATCH_WALL_S = 900  # bundle + every comp; each comp is already ceilinged by maxSeconds inside the batch


def _items_log(spec_path: Path) -> Path:
    return spec_path.with_suffix(".items.jsonl")


def read_item_log(path: Path) -> list[dict]:
    """render_batch.mjs's per-item lines ({tag, comp, start} then {tag, comp, ok, frames, s, …}); absent = []."""
    if not path.is_file():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def item_progress(items: list, rows: list[dict]) -> str:
    """Which TAGGED item (a v13 layer) the batch finished, failed, or still had in flight — the names a wall must carry."""
    done = {r["tag"]: r for r in rows if "ok" in r}
    started = [r["tag"] for r in rows if r.get("start")]
    in_flight = [t for t in started if t not in done]
    parts = [f"in flight: {', '.join(in_flight)}" if in_flight else ""]
    bad = [f"{t} ({r.get('error') or 'failed'})" for t, r in done.items() if not r.get("ok")]
    if bad:
        parts.append(f"failed: {'; '.join(bad)}")
    tagged = [it["tag"] for it in items if it.get("tag")]
    never = [t for t in tagged if t not in started]
    if never:
        parts.append(f"not reached: {len(never)} layer(s)")
    if done:
        parts.append(f"done: {sum(1 for r in done.values() if r.get('ok'))}")
    return " · ".join(p for p in parts if p)


BESPOKE_PARALLEL_WHY = """
EACH BESPOKE COMP IS ITS OWN ENTRY, SO ITS OWN WEBPACK BUNDLE — AND WEBPACK IS ONE CORE.

r4c final (pm2 pod-pool log, 2026-10-22): after the catalog batch, three Bespoke batches ran back to back,
each `bundled ONCE in 21.7-23.9s` then rendered ~9 s — 97 s of the mograph arm, ~70 s of it single-threaded
bundling on a 12-core box. Measured on the dev box (cold webpack cache, the pod's state): the three entries
bundle in 70.8 s one after another, 76.0 s as concurrent bundle() calls in ONE node process (no gain — the
compile is JS on one event loop), and 36.5 s as three node processes side by side.

So the Bespoke batches run as concurrent render_batch processes AFTER the catalog batch (never beside it: that
batch already holds every tab the box can afford). The Chrome tab budget is SPLIT between them, so the box
holds no more tabs than one batch did. Every comp still gets its own entry, its own bundle and its own seqdir;
`concurrency` only decides how many frames render at once, never what a frame is — the PNGs are the same.
"""
_BESPOKE_PARALLEL_MAX = 3
_TABS = threading.local()   # a bespoke worker's share of the tab budget; unset = the whole budget


def _batch_concurrency() -> int:
    share = getattr(_TABS, "share", None)
    return share if share is not None else _render_concurrency()


def _run_bespoke(rd: Path, bespoke: list, tmp: Path) -> None:
    """Every (entry, item, meta) of `bespoke` through its own render_batch, up to _BESPOKE_PARALLEL_MAX at
    once (BESPOKE_PARALLEL_WHY). All land before this returns; the FIRST failure in plan order is re-raised."""
    if not bespoke:
        return
    width = min(len(bespoke), _BESPOKE_PARALLEL_MAX)
    if width == 1:
        for n, (entry, item, _meta) in enumerate(bespoke):
            _run_batch(rd, [item], tmp / f"batch_bespoke{n}.json", entry)
        return
    share = max(2, _render_concurrency() // width)

    def one(n: int) -> None:
        entry, item, _meta = bespoke[n]
        _TABS.share = share
        try:
            _run_batch(rd, [item], tmp / f"batch_bespoke{n}.json", entry)
        finally:
            _TABS.share = None

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=width, thread_name_prefix="bespoke") as ex:
        futs = [ex.submit(one, n) for n in range(len(bespoke))]
        cf.wait(futs)          # every batch lands (or fails) before anything is raised — no orphan node
    for f in futs:
        f.result()             # first fault in plan order wins


def _run_batch(rd: Path, items: list, spec_path: Path, entry_point: str | None) -> None:
    conc = _batch_concurrency()
    cache = _offthread_cache_bytes()
    log = _items_log(spec_path)
    log.unlink(missing_ok=True)
    body = {"concurrency": conc, "offthreadVideoCacheSizeInBytes": cache, "items": items, "itemsLog": str(log)}
    if entry_point:
        body["entryPoint"] = entry_point
    spec_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    oom_before = _oom_count()
    try:
        r = subprocess.run(["node", "render_batch.mjs", str(spec_path)], cwd=rd, capture_output=True,
                           timeout=_BATCH_WALL_S)
    except subprocess.TimeoutExpired as exc:
        prog = item_progress(items, read_item_log(log))
        raise RuntimeError(f"render_batch ran out its {_BATCH_WALL_S}s wall "
                           f"[{_box_facts(oom_before, conc, cache, spec_path.parent)}]"
                           + (f" — {prog}" if prog else "")) from exc
    # Unconditional: the per-comp "[batch] <comp>: <frames> in <s>" line was DROPPED on the success path.
    batch_log = (r.stderr or b"").decode("utf-8", "replace")
    if batch_log.strip():
        sys.stderr.write(batch_log if batch_log.endswith("\n") else batch_log + "\n")
    # JUDGE THE FRAMES, NOT THE EXIT CODE: headless Chrome aborts on teardown often enough (SIGABRT after every
    # sequence is already on disk) that a rc check throws away finished work and fails the whole master.
    rows = read_item_log(log)
    for row in rows:   # the per-layer cost receipt, on the pod's own log
        if "ok" in row:
            print(f"mograph: layer {row['tag']} ({row.get('comp')}) {'ok' if row['ok'] else 'FAILED'} "
                  f"{row.get('frames')} frames in {row.get('s')}s" + ("" if row["ok"] else f" — {row.get('error')}"),
                  file=sys.stderr)
    # a v13 layer render_batch reported FAILED is refused even when a crash-retry left some PNGs in its seqdir:
    # packing those would ship a partial layer the preview never showed (stand bench r3b-056-bench-2: c2, c3)
    failed = [f"{row['tag']} ({row.get('error') or 'failed'})" for row in rows if row.get("ok") is False]
    if failed:
        raise RuntimeError(f"render_batch FAILED {len(failed)} v13 layer(s): {'; '.join(failed)} "
                           f"[{_box_facts(oom_before, conc, cache, spec_path.parent)}]")
    missing = [str(i.get("tag") or i.get("comp")) for i in items if not any(Path(i["seqdir"]).glob("*.png"))]
    if not missing:
        if r.returncode:
            print(f"mograph: render_batch exited {r.returncode} AFTER writing every sequence "
                  f"(Chrome teardown) — keeping the frames", file=sys.stderr)
        return
    full = (batch_log + (r.stdout or b"").decode("utf-8", "replace"))
    # Node error dumps carry message+stack at the TOP; a tail-only slice showed none of it. Head AND tail.
    snippet = full[:800] + ("…⋯…" + full[-400:] if len(full) > 800 else "")
    raise RuntimeError(f"render_batch exited {r.returncode} with no frames for {', '.join(missing)} "
                       f"[{_box_facts(oom_before, conc, cache, spec_path.parent)}]: {snippet}")


def _pack(metas: list[dict], tmp: Path) -> list[dict]:
    layers = []
    for m in metas:
        pngs = sorted(m["seqdir"].glob("*.png"))
        if not pngs:
            continue
        mov = tmp / f"{m['seqdir'].name}.mov"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-pattern_type", "glob",
                        "-i", str(m["seqdir"] / "*.png"), "-c:v", "qtrle", *_BT709, str(mov)], check=True)
        layers.append({"mov": str(mov), "start": m["start"], "dur": len(pngs) / FPS,
                       "glass": m["glass"], "head_below": m.get("head_below", False),
                       "backing": m.get("backing"), "z": m.get("z")})
    return layers


# SUPER-DIRECTOR v13 LAYOUT LAYERS (engine viewer_bridge.sd_layer_section): a `SdLayer` section whose props carry
# `sdFace: "base"` draws the SPEAKER inside its layout (side_shift / pip / vsplit …) — the clean head plate this job
# composites over. It is staged ONCE into the bundle's public/ and handed to the comp as faceSrc + faceStart (the
# section's start on the plate's own clock), so the layout frames the very pixels the base shows outside its window.
SD_FACE_KEY, SD_FACE_BASE = "sdFace", "base"
_SD_FACE_DIR = "_sd"
_SD_LAYER_COMP = "SdLayer"
_SD_LAYER_TAIL_S = 2.0       # a long (persistent) layer's own ceiling: its length + this, never under the default
# A v13 layer's render DEADLINE (render_batch.mjs `wallS`): a hang-ender, not a budget. Measured on the stand
# (sd_stand L4, a loaded 16-core box, 10 tabs): 0.14-0.4 s per frame for library / engine / layout layers; the
# deadline sits ~2.5x above the slowest of those plus the bundle-warm first item, so only a stuck item reaches it.
SD_ITEM_WALL_BASE_S = 60.0
SD_ITEM_WALL_PER_FRAME_S = 1.0


def sd_layer_tag(props: dict) -> str | None:
    """The v13 layer id a section renders (viewer_bridge: `sdLayer` on an engine section, `layer.id` on SdLayer)."""
    if props.get("sdLayer"):
        return str(props["sdLayer"])
    layer = props.get("layer")
    return str(layer["id"]) if isinstance(layer, dict) and layer.get("id") else None


def sd_layer_z(props: dict) -> int | None:
    """The v13 layer's stack z (sd_layers.LAYER_Z) a section renders: `layer.z` on SdLayer, `sdZ` on an engine section;
    None = a legacy mograph section (drawn above the person plane, as the preview draws plan.mograph)."""
    layer = props.get("layer")
    z = layer.get("z") if isinstance(layer, dict) else props.get("sdZ")
    return int(z) if isinstance(z, (int, float)) and not isinstance(z, bool) else None


def sd_item_wall_s(props: dict) -> float:
    frames = math.ceil(float(props.get("durationInSeconds") or 0.0) * FPS)
    return SD_ITEM_WALL_BASE_S + SD_ITEM_WALL_PER_FRAME_S * frames


def _stage_sd_face(rd: Path, base_path: Path) -> str:
    """public/_sd/base<suffix> → the public-relative name; a hard link when the filesystem allows one, else a copy
    (never a symlink: _stage_dest refuses anything resolving outside the workspace)."""
    rel = f"{_SD_FACE_DIR}/base{Path(base_path).suffix.lower() or '.mp4'}"
    dest = rd / "public" / rel
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(base_path, dest)
        except OSError:
            shutil.copy2(base_path, dest)
    return rel


def sd_face_props(props: dict, start: float, face_rel: str | None) -> dict:
    """Pure: a section's props with `sdFace` swapped for the staged plate. No plate → the request is refused by name
    (a layout with no speaker is a black panel the preview never showed)."""
    if props.get(SD_FACE_KEY) != SD_FACE_BASE:
        return props
    if face_rel is None:
        raise RuntimeError("mograph: an SdLayer layout asks for the head plate (sdFace=base) but no base input "
                           "path reached _render_layers — refusing a master with an empty speaker slot")
    out = {k: v for k, v in props.items() if k != SD_FACE_KEY}
    out["faceSrc"], out["faceStart"] = face_rel, float(start)
    return out


def _render_layers(sections: list, brand: dict | None, input_paths: dict, tmp: Path,
                   bundle_ref=None, max_seconds: float | None = None, base_path: Path | None = None) -> list[dict]:
    """Render sections to transparent qtrle layers: catalog comps in one bundle+Chrome batch; each Bespoke
    (LLM .tsx delivered + staged by the brain) via its own per-job entry. A missing bespoke entry = skip loud.
    `max_seconds` rides on every item as render_batch.mjs's per-item ceiling (independent of props).
    `base_path` = the head plate, staged only when a v13 layout layer asks for it (sd_face_props)."""
    rd = remotion_dir(bundle_ref, tmp)
    _stage_public(input_paths, rd)
    wants_face = any((sec.props or {}).get(SD_FACE_KEY) == SD_FACE_BASE for sec in sections)
    face_rel = _stage_sd_face(rd, base_path) if wants_face and base_path is not None else None
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
                "backing": (sec.props or {}).get("backing"), "z": sd_layer_z(sec.props or {})}
        item = {"comp": sec.comp, "props": sd_face_props(_props(sec), sec.start, face_rel), "seqdir": str(seqdir)}
        if max_seconds is not None:
            item["maxSeconds"] = max_seconds
        if (tag := sd_layer_tag(sec.props or {})) is not None:
            item["tag"], item["wallS"] = tag, sd_item_wall_s(sec.props or {})
        if sec.comp == _SD_LAYER_COMP:
            # a persistent layer can outlast render_batch's 120 s default; its own length is its ceiling
            need = float((sec.props or {}).get("durationInSeconds") or 0.0) + _SD_LAYER_TAIL_S
            item["maxSeconds"] = max(float(item.get("maxSeconds") or 120.0), need)
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
    _run_bespoke(rd, bespoke, tmp)
    metas += [meta for _entry, _item, meta in bespoke]
    return _pack(metas, tmp)


# ── PERSON PLANE (super-director v13 R4, spec overlays.person_matte) ─────────────────────────────────────
PERSON_Z = 3   # sd_layers.LAYER_Z["person"]: layers below it draw UNDER the person cut-out, the rest over it


def split_by_person(layers: list[dict]) -> tuple[list[dict], list[dict]]:
    """(below, above) the person plane, each in its own order: z < PERSON_Z below, everything else (z ≥ 3 and every
    legacy section, which the preview draws between its two layer bands) above."""
    below = [x for x in layers if x.get("z") is not None and int(x["z"]) < PERSON_Z]
    above = [x for x in layers if not (x.get("z") is not None and int(x["z"]) < PERSON_Z)]
    return below, above


def person_plane_chain(plate_v: str, alpha_v: str, segments: list[tuple[str, str, str]], w: int, h: int,
                       out: str) -> str:
    """Pure: [plate][alpha] → the person plane `out` (yuva420p), cut through the base's own segments (trim + speed +
    scale, the chain render.build_filtergraph runs on the plate) so frame i of the plane stays frame i of the base.
    `segments` = (in, out, speed) per timeline segment, ALREADY spelled by render._num (the base chain's own
    spelling — a re-formatted float would cut a different frame); the alpha is gray, merged before any rescale."""
    n = len(segments)
    parts = [f"[{plate_v}][{alpha_v}]alphamerge,format=yuva420p" + (f",split={n}" if n > 1 else "")
             + "".join(f"[pmsrc{k}]" for k in range(n))]
    pads = []
    for k, (a, b, sp) in enumerate(segments):
        parts.append(f"[pmsrc{k}]trim=start={a}:end={b},setpts=(PTS-STARTPTS)/{sp},"
                     f"scale={w}:{h}:flags=lanczos,setsar=1,format=yuva420p[pmseg{k}]")
        pads.append(f"[pmseg{k}]")
    parts.append(f"{''.join(pads)}concat=n={n}:v=1:a=0[{out}]" if n > 1 else f"[pmseg0]null[{out}]")
    return ";".join(parts)


def person_overlay(base: str, plane: str, windows: list[tuple[float, float]], fps: float, out: str) -> str:
    """Pure: the person plane over [base], shown only on the output frames of `windows` (frames.between — the same
    frame rounding every other window of this graph gets)."""
    enable = "+".join(_frames.between(*_frames.frame_range(a, b, fps), fps) for a, b in windows)
    return f"[{base}][{plane}]overlay=enable='{enable}':eof_action=pass:format=yuv420[{out}]"


def person_windows(windows: list[tuple[float, float]], cut: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """`windows` minus every `cut` span (the b-roll cutaways covering the head) — no cutaway between the plate and
    the person plane."""
    out = [(float(a), float(b)) for a, b in windows]
    for c0, c1 in cut:
        nxt: list[tuple[float, float]] = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return [(a, b) for a, b in out if b - a > 1e-6]


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


def mark_glass_filters(i: int, src: str, backing: dict, start: float, end: float,
                       fps: float) -> tuple[list[str], str]:
    """Blur+darken ONLY the mark's own box, following it, gated to the frame range [start+from, end).

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
    win = _frames.between(*_frames.frame_range(float(start) + float(backing.get("from") or 0.0), end, fps), fps)
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
_MARK_BACKING = {"glass": lambda i, src, b, s, e, fps: mark_glass_filters(i, src, b, s, e, fps),
                 "shadow": None}


def overlay_filtergraph(layers: list[dict], *, fps: float, base: str = "0:v",
                        layers_v: list[str] | None = None) -> tuple[str, str]:
    """Pure: (-filter_complex string, final video label) compositing alpha layers onto [base]. Each layer is
    seated on its start frame and gated to the output-frame range [start, start+dur) (frames.py — the same
    rounding a cutaway's window gets); a glass layer blurs+darkens the frame behind it; a head_below layer
    slides the base head down first, so the layer's picture rides over the cleared top band."""
    # base/layers_v exist so a MERGED graph can hand real pad names: the defaults are this pass's own
    # -i order, which stops being 0/1/2… the moment the composite's inputs sit in front of the layers.
    filters, src = [], base
    for i, lay in enumerate(layers):
        s, e = lay["start"], lay["start"] + lay["dur"]
        a, b = _frames.frame_range(s, e, fps)
        win, seat = _frames.between(a, b, fps), _frames.pts_at(a, fps)
        idx = layers_v[i] if layers_v is not None else f"{i + 1}:v"
        if lay.get("glass"):
            # frosted-glass takeover: blur+darken the frame behind the card, gated to its window (parity with engine _composite).
            filters.append(f"[{src}]gblur=sigma=22:enable='{win}',"
                           f"eq=brightness=-0.05:enable='{win}'[g{i}]")
            src = f"g{i}"
        backing = lay.get("backing") or None
        if backing and _MARK_BACKING.get(str(backing.get("treatment"))) is not None:
            add, src = _MARK_BACKING[str(backing["treatment"])](i, src, backing, s, e, fps)
            filters += add
        if lay.get("head_below"):
            # slide a copy of the base head DOWN over its window; the layer (SplitScreen alpha, overlaid next)
            # then covers the cleared top band. Mirrors MontagePreview / the retired engine _composite.
            yexpr = _hb_settle_y_expr(s, e)
            filters.append(f"[{src}]split[ha{i}][hb{i}];"
                           f"[hb{i}]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS+{seat}[hw{i}];"
                           f"[ha{i}][hw{i}]overlay=y='{yexpr}':enable='{win}':eof_action=pass[hbv{i}]")
            src = f"hbv{i}"
        filters.append(f"[{idx}]setpts=PTS-STARTPTS+{seat}[o{i}];"
                       f"[{src}][o{i}]overlay=enable='{win}':eof_action=pass[v{i}]")
        src = f"v{i}"
    return ";".join(filters), src
