"""Mograph overlays on the pod: render motion_plan.sections via the baked (brand-agnostic) Remotion bundle
— brand crosses through inputProps, role fonts + section media are staged into the bundle public/. Each
section packs to a transparent qtrle layer, overlaid onto the base gated to its [start,start+dur] window."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FPS = 30
_STAGE_PREFIX = "mograph/"  # input id `mograph/<rel>` → staged into <bundle>/<rel> (public/ fonts+media, src/ bespoke)


def remotion_dir() -> Path:
    """The baked Remotion project dir (render_batch.mjs + node_modules + src). MONTY_REMOTION_DIR overrides."""
    d = os.environ.get("MONTY_REMOTION_DIR")
    cand = Path(d) if d else Path(__file__).resolve().parents[2] / "remotion"
    if not (cand / "render_batch.mjs").is_file():
        raise RuntimeError(f"Remotion bundle not found at {cand} (set MONTY_REMOTION_DIR)")
    return cand


def _stage_public(input_paths: dict, rd: Path) -> None:
    """Copy every `mograph/<rel>` input into <bundle>/<rel> (public/ fonts+media, src/ bespoke .tsx+entry)."""
    for iid, path in input_paths.items():
        if iid.startswith(_STAGE_PREFIX):
            dest = rd / iid[len(_STAGE_PREFIX):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def _run_batch(rd: Path, items: list, spec_path: Path, entry_point: str | None) -> None:
    body = {"concurrency": 4, "items": items}
    if entry_point:
        body["entryPoint"] = entry_point
    spec_path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["node", "render_batch.mjs", str(spec_path)], cwd=rd, check=True, capture_output=True)


def _pack(metas: list[dict], tmp: Path) -> list[dict]:
    layers = []
    for m in metas:
        pngs = sorted(m["seqdir"].glob("*.png"))
        if not pngs:
            continue
        mov = tmp / f"{m['seqdir'].name}.mov"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-pattern_type", "glob",
                        "-i", str(m["seqdir"] / "*.png"), "-c:v", "qtrle", str(mov)], check=True)
        layers.append({"mov": str(mov), "start": m["start"], "dur": len(pngs) / FPS, "glass": m["glass"]})
    return layers


def _render_layers(sections: list, brand: dict | None, input_paths: dict, tmp: Path) -> list[dict]:
    """Render sections to transparent qtrle layers: catalog comps in one bundle+Chrome batch; each Bespoke
    (LLM .tsx delivered + staged by the brain) via its own per-job entry. A missing bespoke entry = skip loud."""
    rd = remotion_dir()
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
        meta = {"seqdir": seqdir, "start": float(sec.start), "glass": bool(sec.glass)}
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


def overlay_filtergraph(layers: list[dict]) -> tuple[str, str]:
    """Pure: (-filter_complex string, final video label) compositing alpha layers onto [0:v]. Each layer is
    shifted to its start and gated to [start,start+dur]; a glass layer blurs+darkens the frame behind it."""
    filters, src = [], "0:v"
    for i, lay in enumerate(layers):
        s, e, idx = lay["start"], lay["start"] + lay["dur"], i + 1
        if lay.get("glass"):
            # frosted-glass takeover: blur+darken the frame behind the card, gated to its window (parity with engine _composite).
            filters.append(f"[{src}]gblur=sigma=22:enable='between(t,{s},{e})',"
                           f"eq=brightness=-0.05:enable='between(t,{s},{e})'[g{i}]")
            src = f"g{i}"
        filters.append(f"[{idx}:v]setpts=PTS-STARTPTS+{s}/TB[o{i}];"
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
                            input_paths, tmp)
    return _overlay(base, layers, out, gpu, enc) if layers else base
