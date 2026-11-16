"""screen_quad inference: the 4 corners of a device's screen in a still picture, on the pod GPU — Florence-2 base phrase grounding
(`<device> screen`) → SAM 2.1 hiera base-plus box mask → an opencv 4-corner fit. The caller pins its own content onto those corners
(a perspective warp in its renderer). One receipt (screen_quad.schema.json) goes to put_url; nothing else is PUT.

Settings and their sources: the research receipt docs/research/pod-screen-quad.md (its «Runtime settings» and Conditions 1-11; the
2026-11-16 bench: 9 of 15 stock device faces found, ZERO wrong devices, ~0.5 s per picture warm).

  weights   a WeightsRef tar (weights.ensure): florence/ + sam/ (two HF snapshots) and config.json = the manifest
            {florence: {hf_repo, hf_revision}, sam: {hf_repo, hf_revision}, transformers, torch, torchvision,
            dtypes: {florence, sam}}. from_pretrained(<dir>, local_files_only=True), explicit processor classes, no
            trust_remote_code, no HF access (Conditions 6, 7).
  load      REFUSE when the manifest's torch / torchvision / transformers differ from this runtime's; both models load with
            output_loading_info and ANY missing / unexpected / mismatched key refuses the kind.
  intake    exif_transpose; real transparency composited onto white; RGB; long side capped at 2048 (LANCZOS). The capped, upright
            picture's (W, H) is what Florence's post-process, the quad fractions and the answer's w / h use (Condition 5).
  florence  `<CAPTION_TO_PHRASE_GROUNDING>` + `<device> screen`, device a closed enum, no «screen» fallback (Conditions 1, 9);
            generate(max_new_tokens=256, num_beams=3, do_sample=False) over the snapshot's generation_config; the EFFECTIVE
            config is logged and named in the receipt (Condition 8).
  boxes     drop a box over 80 % of the picture; drop a box that contains >= 2 other returned boxes (each >= 90 % inside it);
            largest first, at most 4.
  sam       ONE image embedding per picture, reused by every pass (Condition 10); per box 3 masks + iou_scores; a box whose best
            iou < 0.9 is skipped; a second pass from the best mask's bbox + one positive point at its median pixel.
  quad      largest external contour → convex hull → approxPolyN(4) → TL, TR, BR, BL; the smallest candidate mask whose quad
            fits it (polygon IoU >= 0.9) wins; corners are NOT clamped, a corner outside [-0.25, 1.25] is `screen_cut_by_frame`.
  named     per picture: found | none (with reasons — a normal answer) | unreadable. A degenerate mask is a per-item none,
            never a job error (Condition 4).

Placement is absolute: CUDA or refuse. A missing CUDA device, an out-of-memory forward, a manifest that is not this runtime's, a
checkpoint that loads with missing keys: each ends the job status=error WITH its reason — never a CPU run, never a fallback.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .cp import download, upload
from .models import (
    QUAD_FRACTION_MAX,
    QUAD_FRACTION_MIN,
    ScreenQuadOut,
    ScreenQuadParams,
    ScreenQuadPayload,
    ScreenQuadTimings,
)

LONG_SIDE_CAP = 2048          # a source bigger than this is shrunk (LANCZOS) first; the fractions are of THAT picture
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
GENERATION_OVERRIDES: Mapping[str, Any] = {"max_new_tokens": 256, "num_beams": 3, "do_sample": False}
# the thresholds below are calibrated on 15 stock pictures, not on an annotated set (the receipt's `verified`)
PICTURE_MAX_SHARE = 0.80      # a box (or a mask) covering more of the picture than this is the whole scene, not a screen
CONTAIN_SHARE = 0.90          # another box is «inside» a box when this share of it lies in it
CONTAIN_MIN_OTHERS = 2        # a box holding this many other returned boxes is a scene / desk, not a screen
MAX_BOXES = 4                 # per picture, largest first
SAM_IOU_MIN = 0.90            # a box whose best SAM iou_score is below this is skipped
QUAD_IOU_MIN = 0.90           # polygon IoU of the fitted quad against its mask
AREA_VS_BEST_MIN = 0.30       # a candidate mask is at least this share of the first pass's best mask
SECOND_PASS_AREA_MAX = 1.02   # a second-pass mask is at most this much of the best mask (it refines, never grows)
EDGE_RATIO_MIN = 0.05         # a quad whose shortest edge is under this share of its longest is a sliver
MANIFEST_KEYS = ("florence", "sam", "transformers", "torch", "torchvision", "dtypes")
_DTYPES = ("float16", "float32")
SURFACE = "device_face"


class ScreenQuadRefused(ValueError):
    """The request cannot be served as asked (no CUDA, out of memory, the weights were built for another runtime, a malformed
    manifest, a checkpoint that does not load cleanly). Ends the job status=error WITH this reason."""


class UnreadablePicture(ValueError):
    """The bytes of one source are not a picture; the receipt names that item `unreadable`, the others are unaffected."""


# ── the manifest and the runtime check (no torch needed) ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Manifest:
    florence_repo: str
    florence_revision: str
    sam_repo: str
    sam_revision: str
    transformers: str
    torch: str
    torchvision: str
    florence_dtype: str
    sam_dtype: str


def _nonempty_str(raw: Any, what: str) -> str:
    if not (isinstance(raw, str) and raw):
        raise ScreenQuadRefused(f"screen_quad manifest {what} is not a non-empty string")
    return raw


def read_manifest(wdir: Path) -> Manifest:
    """The weights directory's config.json and its two snapshot dirs. Anything else is a refusal that names what is wrong."""
    cfg = Path(wdir) / "config.json"
    if not cfg.is_file():
        raise ScreenQuadRefused(f"screen_quad weights hold no config.json manifest under {wdir}")
    try:
        raw = json.loads(cfg.read_text())
    except ValueError as e:
        raise ScreenQuadRefused(f"screen_quad manifest {cfg} is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ScreenQuadRefused(f"screen_quad manifest {cfg} is not a JSON object")
    missing = [k for k in MANIFEST_KEYS if k not in raw]
    if missing:
        raise ScreenQuadRefused(f"screen_quad manifest lacks {', '.join(missing)}")
    repos: dict[str, tuple[str, str]] = {}
    for name in ("florence", "sam"):
        block = raw[name]
        if not isinstance(block, dict):
            raise ScreenQuadRefused(f"screen_quad manifest {name} is not an object {{hf_repo, hf_revision}}")
        repos[name] = (_nonempty_str(block.get("hf_repo"), f"{name}.hf_repo"),
                       _nonempty_str(block.get("hf_revision"), f"{name}.hf_revision"))
        if not (Path(wdir) / name).is_dir():
            raise ScreenQuadRefused(f"screen_quad weights hold no {name}/ snapshot directory under {wdir}")
    dtypes = raw["dtypes"]
    if not (isinstance(dtypes, dict) and all(dtypes.get(k) in _DTYPES for k in ("florence", "sam"))):
        raise ScreenQuadRefused(f"screen_quad manifest dtypes {dtypes!r} must name florence and sam, each one of {list(_DTYPES)}")
    return Manifest(repos["florence"][0], repos["florence"][1], repos["sam"][0], repos["sam"][1],
                    _nonempty_str(raw["transformers"], "transformers"), _nonempty_str(raw["torch"], "torch"),
                    _nonempty_str(raw["torchvision"], "torchvision"), dtypes["florence"], dtypes["sam"])


def check_runtime(manifest: Manifest, torch_version: str, torchvision_version: str, transformers_version: str) -> None:
    """REFUSE weights built for a runtime that is not this one (Condition 7): the snapshots were verified under exactly that
    torch / torchvision / transformers, and a silent mismatch is a wrong quad, not an error. Names both triples."""
    if (manifest.torch, manifest.torchvision, manifest.transformers) != (torch_version, torchvision_version,
                                                                         transformers_version):
        raise ScreenQuadRefused(
            f"screen_quad weights were built for torch {manifest.torch} / torchvision {manifest.torchvision} / transformers "
            f"{manifest.transformers}, this pod runs torch {torch_version} / torchvision {torchvision_version} / transformers "
            f"{transformers_version} — refusing to load")


def check_loading_info(name: str, info: Mapping[str, Any]) -> None:
    """Condition 6: a snapshot that loads with ANY missing / unexpected / mismatched key is a wrong model that still runs."""
    bad = {k: sorted(info.get(k) or []) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        shown = "; ".join(f"{k}: {', '.join(str(x) for x in v[:5])}{' …' if len(v) > 5 else ''}" for k, v in bad.items())
        raise ScreenQuadRefused(f"screen_quad {name} snapshot did not load cleanly ({shown}) — refusing")


def effective_generation(base: Mapping[str, Any]) -> dict[str, Any]:
    """The snapshot's generation_config under this module's overrides, scalars only: what the receipt names (Condition 8)."""
    merged = {**base, **GENERATION_OVERRIDES}
    return {k: merged[k] for k in sorted(merged) if merged[k] is None or isinstance(merged[k], (bool, int, float, str))}


# ── intake ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def load_picture(path: Path) -> Any:
    """The source as an upright PIL RGB picture: exif_transpose, real transparency composited onto white, the long side capped
    at 2048 (LANCZOS). A source that is not a picture raises UnreadablePicture."""
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as opened:
            opened.load()
            img = ImageOps.exif_transpose(opened)
            img = img if img is not None else opened.copy()
            if img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in img.info:
                rgba = img.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                img = Image.alpha_composite(white, rgba).convert("RGB")
            else:
                img = img.convert("RGB")
    except Exception as e:  # PIL raises a zoo (OSError, DecompressionBombError, SyntaxError, ValueError)
        raise UnreadablePicture(f"{type(e).__name__}: {e}") from e
    w, h = img.size
    if max(w, h) > LONG_SIDE_CAP:
        scale = LONG_SIDE_CAP / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


# ── the pure geometry ───────────────────────────────────────────────────────────────────────────────────────────────


def _box_area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inside_share(inner: Sequence[float], outer: Sequence[float]) -> float:
    """The share of `inner`'s area that lies inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = _box_area(inner)
    return (ix * iy) / area if area > 0 else 0.0


def select_boxes(boxes: Sequence[Sequence[float]], w: int, h: int) -> tuple[list[tuple[int, list[float]]], list[str]]:
    """Florence's boxes (px) → [(index, box)] to try, largest first, at most MAX_BOXES, plus the reasons for the ones dropped.
    Florence's own order is not a ranking."""
    reasons: list[str] = []
    clean: dict[int, list[float]] = {}
    for i, raw in enumerate(boxes):
        x0, x1 = sorted((float(raw[0]), float(raw[2])))
        y0, y1 = sorted((float(raw[1]), float(raw[3])))
        box = [max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)]
        if _box_area(box) <= 0:
            reasons.append(f"box {i}: empty")
        else:
            clean[i] = box
    kept: list[tuple[int, list[float]]] = []
    for i, box in clean.items():
        if _box_area(box) > PICTURE_MAX_SHARE * w * h:
            reasons.append(f"box {i}: over_80_percent_of_picture")
            continue
        held = [j for j, other in clean.items() if j != i and _inside_share(other, box) >= CONTAIN_SHARE]
        if len(held) >= CONTAIN_MIN_OTHERS:
            reasons.append(f"box {i}: contains_{len(held)}_other_boxes")
            continue
        kept.append((i, box))
    kept.sort(key=lambda ib: _box_area(ib[1]), reverse=True)
    return kept[:MAX_BOXES], reasons


def order_quad(points: Any) -> Any:
    """4 points → TL, TR, BR, BL (angle around the centroid, rolled to start at the min(x + y) corner)."""
    import numpy as np

    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]    # y down: increasing angle runs clockwise on screen
    return np.roll(pts, -int(np.argmin(pts[:, 0] + pts[:, 1])), axis=0)


def fit_quad(mask: Any) -> tuple[Any, str]:
    """A bool [H,W] mask → ((4,2) float px quad TL, TR, BR, BL, "") or (None, why). Never raises on a degenerate mask."""
    import cv2
    import numpy as np

    try:
        contours, _ = cv2.findContours(np.ascontiguousarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "empty_mask"
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        if len(hull) < 4:
            return None, "hull_under_4_points"
        quad = np.asarray(cv2.approxPolyN(hull, 4, epsilon_percentage=-1, ensure_convex=True), dtype=np.float64)
    except cv2.error:
        return None, "cv2_error"
    if quad.size != 8:
        return None, "not_4_points"
    quad = order_quad(quad.reshape(-1, 2))
    edges = [float(np.hypot(*(quad[(i + 1) % 4] - quad[i]))) for i in range(4)]
    if max(edges) <= 0 or min(edges) < EDGE_RATIO_MIN * max(edges):
        return None, "quad_edge_under_5_percent_of_longest"
    return quad, ""


def polygon_iou(quad: Any, mask: Any) -> float:
    """IoU of the quad's polygon (rasterised at the mask's size) against the mask."""
    import cv2
    import numpy as np

    poly = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(poly, [np.round(np.asarray(quad)).astype(np.int32)], 1)
    a, b = poly.astype(bool), np.asarray(mask, dtype=bool)
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def fractions(quad: Any, w: int, h: int) -> list[list[float]]:
    return [[round(float(x) / w, 6), round(float(y) / h, 6)] for x, y in quad]


def within_frame_bounds(quad_fractions: Sequence[Sequence[float]]) -> bool:
    return all(QUAD_FRACTION_MIN <= v <= QUAD_FRACTION_MAX for p in quad_fractions for v in p)


def _median_pixel(mask: Any) -> tuple[float, float]:
    """The mask pixel nearest to the per-axis median of its pixels (the median itself may lie outside an L-shaped mask)."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    mx, my = float(np.median(xs)), float(np.median(ys))
    k = int(np.argmin((xs - mx) ** 2 + (ys - my) ** 2))
    return float(xs[k]), float(ys[k])


def _mask_bbox(mask: Any) -> list[float]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0]


# ── the two model seams ─────────────────────────────────────────────────────────────────────────────────────────────
#
#   grounder.boxes(img, device) -> [[x0, y0, x1, y1], ...] in px of `img`      (Florence-2; no score is returned)
#   segmenter.embed(img) -> an opaque embedding, computed ONCE per picture
#   segmenter.masks(embedding, box, point=None) -> ([3 bool [H,W] masks], [3 iou scores])      (SAM 2.1)
#
# The real ones are below; the tests stand in for both.


@dataclass
class ScreenAnswer:
    quad: list[list[float]] | None
    reasons: list[str] = field(default_factory=list)


def locate_screen(grounder: Any, segmenter: Any, img: Any, device: str) -> ScreenAnswer:
    """One upright RGB picture + the device's name → its screen's quad (fractions) or the reasons there is none."""
    w, h = img.size
    raw = grounder.boxes(img, device)
    boxes, reasons = select_boxes(raw, w, h)
    if not raw:
        return ScreenAnswer(None, ["florence_returned_no_box"])
    if not boxes:
        return ScreenAnswer(None, reasons)
    embedding = segmenter.embed(img)                      # ONE image embedding for every box and pass of this picture
    for i, box in boxes:
        masks, ious = segmenter.masks(embedding, box)
        best = max(range(len(masks)), key=lambda k: ious[k])
        if ious[best] < SAM_IOU_MIN:
            reasons.append(f"box {i}: sam_iou_{ious[best]:.2f}_under_{SAM_IOU_MIN:.2f}")
            continue
        best_area = int(masks[best].sum())
        if best_area == 0:
            reasons.append(f"box {i}: sam_mask_empty")
            continue
        second, _ = segmenter.masks(embedding, _mask_bbox(masks[best]), _median_pixel(masks[best]))
        candidates = [(m, False) for m in masks] + [(m, True) for m in second]
        fitted: list[tuple[int, Any]] = []
        why: list[str] = []
        for mask, refined in candidates:
            area = int(mask.sum())
            if area < AREA_VS_BEST_MIN * best_area or area >= PICTURE_MAX_SHARE * w * h:
                why.append("mask_area_out_of_range")
                continue
            if refined and area > SECOND_PASS_AREA_MAX * best_area:
                why.append("second_pass_mask_grew")
                continue
            quad, quad_why = fit_quad(mask)
            if quad is None:
                why.append(quad_why)
                continue
            if polygon_iou(quad, mask) < QUAD_IOU_MIN:
                why.append("quad_does_not_fit_mask")
                continue
            fitted.append((area, quad))
        if not fitted:
            reasons.append(f"box {i}: no_quad_fits_mask ({', '.join(sorted(set(why)))})")
            continue
        _, quad = min(fitted, key=lambda aq: aq[0])      # the smallest mask that has a quad: the screen, not the scene around it
        frac = fractions(quad, w, h)
        if not within_frame_bounds(frac):
            reasons.append(f"box {i}: screen_cut_by_frame")
            continue
        return ScreenAnswer(frac, reasons)
    return ScreenAnswer(None, reasons)


# ── the real models (CUDA only) ─────────────────────────────────────────────────────────────────────────────────────


class FlorenceGrounder:
    """Florence-2 base (florence-community snapshot, native in transformers 4.57.6), fp16 on CUDA."""

    def __init__(self, torch: Any, fdir: Path, dtype: str, device: str) -> None:
        from transformers import Florence2ForConditionalGeneration, Florence2Processor

        self.torch, self.device, self.dtype = torch, device, getattr(torch, dtype)
        # explicit classes, never AutoProcessor (the snapshot's preprocessor config carries a remote-code auto_map); NOT
        # use_fast=False (measured: it switches the tokenizer too and every box disappears)
        self.processor = Florence2Processor.from_pretrained(str(fdir), local_files_only=True)
        model, info = Florence2ForConditionalGeneration.from_pretrained(
            str(fdir), local_files_only=True, torch_dtype=self.dtype, output_loading_info=True)
        check_loading_info("florence", info)
        self.model = model.to(device).eval()
        self.generation = effective_generation(self.model.generation_config.to_dict())

    def boxes(self, img: Any, device: str) -> list[list[float]]:
        torch = self.torch
        w, h = img.size
        inputs = self.processor(text=f"{TASK}{device} screen", images=img, return_tensors="pt").to(self.device, self.dtype)
        try:
            with torch.inference_mode():
                ids = self.model.generate(**inputs, **GENERATION_OVERRIDES)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise ScreenQuadRefused(f"screen_quad ran out of GPU memory in Florence on a {w}x{h} picture "
                                    f"({str(e)[:160]}) — refusing, never a CPU run") from e
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=TASK, image_size=(w, h))
        return [[float(v) for v in b] for b in parsed[TASK]["bboxes"]]


class Sam2Segmenter:
    """SAM 2.1 hiera base-plus (Sam2Model, native in transformers 4.57.6), fp32 on CUDA."""

    def __init__(self, torch: Any, sdir: Path, dtype: str, device: str) -> None:
        from transformers import Sam2Model, Sam2Processor

        self.torch, self.device = torch, device
        self.processor = Sam2Processor.from_pretrained(str(sdir), local_files_only=True)
        # the config's model_type is sam2_video: the «sam2_video to instantiate sam2» warning is expected, the gate is the info
        model, info = Sam2Model.from_pretrained(str(sdir), local_files_only=True, torch_dtype=getattr(torch, dtype),
                                                output_loading_info=True)
        check_loading_info("sam", info)
        self.model = model.to(device).eval()

    def embed(self, img: Any) -> Any:
        torch = self.torch
        pre = self.processor(images=img, return_tensors="pt")
        try:
            with torch.inference_mode():
                feats = self.model.get_image_embeddings(pre["pixel_values"].to(self.device))
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise ScreenQuadRefused(f"screen_quad ran out of GPU memory in SAM on a {img.size[0]}x{img.size[1]} picture "
                                    f"({str(e)[:160]}) — refusing, never a CPU run") from e
        return feats, pre["original_sizes"]

    def masks(self, embedding: Any, box: Sequence[float], point: tuple[float, float] | None = None) -> tuple[list[Any], list[float]]:
        torch = self.torch
        feats, sizes = embedding
        prompt: dict[str, Any] = {"input_boxes": [[[float(v) for v in box]]]}
        if point is not None:
            prompt.update(input_points=[[[[float(point[0]), float(point[1])]]]], input_labels=[[[1]]])
        enc = self.processor(original_sizes=sizes, return_tensors="pt", **prompt).to(self.device)
        try:
            with torch.inference_mode():
                out = self.model(image_embeddings=feats, multimask_output=True,
                                 **{k: enc[k] for k in ("input_boxes", "input_points", "input_labels") if k in enc})
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise ScreenQuadRefused(f"screen_quad ran out of GPU memory in the SAM decoder ({str(e)[:160]}) — refusing, "
                                    f"never a CPU run") from e
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), sizes)[0][0]       # [3, H, W] bool
        return [m.numpy().astype(bool) for m in masks], [float(v) for v in out.iou_scores[0, 0].float().cpu()]


@dataclass
class ScreenQuadRun:
    infer_s: float
    timings: dict[str, float] = field(default_factory=dict)
    payload: ScreenQuadPayload | None = None


class ScreenQuadService:
    """Holds Florence-2 + SAM 2.1 on the card; serves every batch after that. `grounder` / `segmenter` are the two model seams:
    the pod never passes them (the real CUDA models are loaded), the tests inject stand-ins. `runtime` = (torch, torchvision,
    transformers) versions, None = the real ones."""

    def __init__(self, model_name: str, wdir: Path, weights_sha256: str, *, device: str = "cuda",
                 runtime: tuple[str, str, str] | None = None, grounder: Any = None, segmenter: Any = None) -> None:
        manifest = read_manifest(wdir)
        torch: Any = None
        if runtime is None:
            import torch
            import torchvision
            import transformers

            runtime = (torch.__version__, torchvision.__version__, transformers.__version__)
        check_runtime(manifest, *runtime)
        self.model_name, self.weights_sha256, self.manifest, self.runtime = model_name, weights_sha256, manifest, runtime
        self.real = grounder is None and segmenter is None
        if self.real:
            if torch is None:
                import torch
            if not device.startswith("cuda") or not torch.cuda.is_available():
                raise ScreenQuadRefused("screen_quad needs a CUDA device and this pod reports none — refusing, never CPU")
            grounder = FlorenceGrounder(torch, Path(wdir) / "florence", manifest.florence_dtype, device)
            segmenter = Sam2Segmenter(torch, Path(wdir) / "sam", manifest.sam_dtype, device)
        elif grounder is None or segmenter is None:
            raise ScreenQuadRefused("screen_quad takes both model seams or neither")
        self.torch, self.grounder, self.segmenter = torch, grounder, segmenter
        self.generation: dict[str, Any] = dict(getattr(grounder, "generation", None) or effective_generation({}))

    def run(self, params: ScreenQuadParams, put_url: str,
            note: Callable[[str], None] = lambda _s: None) -> ScreenQuadRun:
        outs: list[ScreenQuadOut] = []
        model_s = 0.0
        note(f"screen_quad generation config: {json.dumps(self.generation, sort_keys=True)}")
        with tempfile.TemporaryDirectory() as td:
            t_all = time.monotonic()
            for it in params.items:
                src = download(it.src_url, Path(td) / f"{it.id}.src")
                try:
                    img = load_picture(src)
                except UnreadablePicture as e:
                    outs.append(ScreenQuadOut(id=it.id, status="unreadable", device=it.device))
                    note(f"screen_quad {it.id}: unreadable — {e}")
                    continue
                w, h = img.size
                s = time.monotonic()
                answer = locate_screen(self.grounder, self.segmenter, img, it.device)
                model_s += time.monotonic() - s
                if answer.quad is None:
                    outs.append(ScreenQuadOut(id=it.id, status="none", device=it.device,
                                              reasons=answer.reasons or ["no_screen_found"]))
                    note(f"screen_quad {it.id}: none — {'; '.join(answer.reasons)}")
                else:
                    outs.append(ScreenQuadOut(id=it.id, status="found", device=it.device, quad=answer.quad, w=w, h=h,
                                              surface=SURFACE))
                    note(f"screen_quad {it.id}: found")
            wall_s = time.monotonic() - t_all
            payload = ScreenQuadPayload(
                model=self.model_name, model_sha256=self.weights_sha256, torch=self.runtime[0],
                torchvision=self.runtime[1], transformers=self.runtime[2], generation=self.generation, items=outs,
                timings=ScreenQuadTimings(model_s=round(model_s, 3), wall_s=round(wall_s, 3)))
            receipt = Path(td) / "screen_quad.json"
            receipt.write_text(payload.model_dump_json(exclude_none=True))
            upload(receipt, put_url, "application/json")
            found = sum(1 for o in outs if o.status == "found")
            note(f"screen_quad {found}/{len(outs)} found: model {model_s:.1f}s, wall {wall_s:.1f}s")
            if self.real:
                self.torch.cuda.empty_cache()             # the allocator's cache is not the weights: hand it back
            return ScreenQuadRun(infer_s=wall_s, payload=payload,
                                 timings={"infer_s": wall_s, "model_s": model_s, "found": float(found),
                                          "items": float(len(outs))})
