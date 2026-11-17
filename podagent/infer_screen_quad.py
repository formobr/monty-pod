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

The object_mask kind runs on the same pair: the caller's English noun phrase instead of `<device> screen`, the same boxes and SAM
passes, the kept mask the largest-iou candidate under PICTURE_MAX_SHARE (locate_object); no quad fit, no glass.

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
    ObjectMaskOut,
    ObjectMaskParams,
    ObjectMaskPayload,
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
# the glass inside the device face — docs/research/pod-screen-quad.md, «Amendment 2026-11-16 — the glass inside the device face»
GLASS_AREA_MIN = 0.30         # a glass candidate covers at least this share of the (rectified) face
GLASS_IOU_MIN = 0.70          # and has at least this SAM iou_score
GLASS_SHRINK_STEP = 0.0125    # the box prompt shrinks from 0 to GLASS_SHRINK_MAX of the face in these steps (each side)
GLASS_SHRINK_MAX = 0.15
GLASS_HOLE_MAX_SHARE = 0.02   # an interior hole under this share of the glass is filled in the mask PNG; a notch at the border never


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
#   grounder.boxes(img, phrase) ->[[x0, y0, x1, y1], ...] in px of `img`      (Florence-2; no score is returned)
#   segmenter.embed(img) -> an opaque embedding, computed ONCE per picture
#   segmenter.masks(embedding, box, point=None) -> ([3 bool [H,W] masks], [3 iou scores])      (SAM 2.1)
#
# The real ones are below; the tests stand in for both.


@dataclass
class ScreenAnswer:
    quad: list[list[float]] | None
    reasons: list[str] = field(default_factory=list)
    glass: list[list[float]] | None = None          # the glass inside the face, picture fractions
    glass_mask: Any = None                          # bool [H, W] of the picture, the glass (holes not yet filled)
    glass_reasons: list[str] = field(default_factory=list)


def _warp_to_face(quad_px: Any) -> tuple[Any, int, int] | None:
    """The perspective matrix from the picture to an upright face (size = the quad's max opposite edges), or None for a sliver."""
    import cv2
    import numpy as np

    q = np.asarray(quad_px, dtype=np.float32).reshape(4, 2)
    edge = lambda a, b: float(np.hypot(*(q[a] - q[b])))
    fw, fh = round(max(edge(0, 1), edge(3, 2))), round(max(edge(0, 3), edge(1, 2)))
    if fw < 2 or fh < 2:
        return None
    dst = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(q, dst), fw, fh


def _boundary_gradient(mask: Any, grad: Any) -> float:
    """Mean Sobel magnitude along the mask's 1-px morphological boundary (mask minus its erosion)."""
    import cv2
    import numpy as np

    m = np.asarray(mask, dtype=np.uint8)
    ring = (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)
    return float(grad[ring].mean()) if ring.any() else 0.0


def find_glass(segmenter: Any, img: Any, quad_px: Any) -> tuple[list[list[float]] | None, Any, list[str]]:
    """The glass inside the device face: → (4 picture fractions TL, TR, BR, BL, a bool [H, W] picture-size mask, []) or
    (None, None, [why]). The picture is warped upright by the face quad, ONE SAM embedding of that face, a box prompt shrunk
    0 → 15 % in 1.25 % steps (multimask); the kept candidate has the strongest boundary gradient."""
    import cv2
    import numpy as np
    from PIL import Image

    w, h = img.size
    warp = _warp_to_face(quad_px)
    if warp is None:
        return None, None, ["glass: face_quad_too_small"]
    matrix, fw, fh = warp
    face = cv2.warpPerspective(np.asarray(img), matrix, (fw, fh), flags=cv2.INTER_LINEAR)
    grey = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    embedding = segmenter.embed(Image.fromarray(face))
    best: tuple[float, Any] | None = None
    steps = round(GLASS_SHRINK_MAX / GLASS_SHRINK_STEP)
    for k in range(steps + 1):
        s = k * GLASS_SHRINK_STEP
        masks, ious = segmenter.masks(embedding, [s * fw, s * fh, (1 - s) * fw, (1 - s) * fh])
        for mask, iou in zip(masks, ious):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (fh, fw) or iou < GLASS_IOU_MIN or mask.sum() < GLASS_AREA_MIN * fw * fh:
                continue
            score = _boundary_gradient(mask, grad)
            if best is None or score > best[0]:
                best = (score, mask)
    if best is None:
        return None, None, ["glass: no_candidate_over_thresholds"]
    quad, why = fit_quad(best[1])
    if quad is None:
        return None, None, [f"glass: {why}"]
    back = np.linalg.inv(matrix)
    in_picture = cv2.perspectiveTransform(quad.astype(np.float32).reshape(-1, 1, 2), back).reshape(4, 2)
    frac = fractions(in_picture, w, h)
    if not within_frame_bounds(frac):
        return None, None, ["glass: glass_cut_by_frame"]
    mask_pic = cv2.warpPerspective(best[1].astype(np.uint8), back, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    return frac, mask_pic, []


def fill_small_holes(mask: Any) -> Any:
    """bool [H, W] → uint8 0/255 with the interior holes under GLASS_HOLE_MAX_SHARE of the glass filled. A hole is background NOT
    reachable from outside the mask (flood-fill from a padded border): a notch cut at the border, a thumb over the edge stay."""
    import cv2
    import numpy as np

    m = np.asarray(mask, dtype=np.uint8)
    glass = int(m.sum())
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    outside = padded.copy()
    ff = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(outside, ff, (0, 0), 2)
    holes = ((outside == 0)[1:-1, 1:-1]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    out = m.copy()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < GLASS_HOLE_MAX_SHARE * glass:
            out[labels == i] = 1
    return out * 255


def locate_screen(grounder: Any, segmenter: Any, img: Any, device: str) -> ScreenAnswer:
    """One upright RGB picture + the device's name → its screen's quad (fractions) or the reasons there is none."""
    w, h = img.size
    raw = grounder.boxes(img, f"{device} screen")
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
        glass, glass_mask, glass_reasons = find_glass(segmenter, img, [(x * w, y * h) for x, y in frac])
        return ScreenAnswer(frac, reasons, glass, glass_mask, glass_reasons)
    return ScreenAnswer(None, reasons)


@dataclass
class ObjectAnswer:
    mask: Any                                       # bool [H, W] of the picture, or None
    bbox: list[float] | None                        # [x, y, w, h] of the mask's bounding box, fractions of the picture
    reasons: list[str] = field(default_factory=list)


def locate_object(grounder: Any, segmenter: Any, img: Any, phrase: str) -> ObjectAnswer:
    """One upright RGB picture + the caller's English noun phrase → the object's mask and bbox fractions, or the reasons there is
    none. The same grounding task, box selection, ONE embedding and second pass as locate_screen; per box the kept mask is the
    candidate with the largest SAM iou_score that covers under PICTURE_MAX_SHARE of the picture."""
    w, h = img.size
    raw = grounder.boxes(img, phrase)
    boxes, reasons = select_boxes(raw, w, h)
    if not raw:
        return ObjectAnswer(None, None, ["florence_returned_no_box"])
    if not boxes:
        return ObjectAnswer(None, None, reasons)
    embedding = segmenter.embed(img)
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
        second, second_ious = segmenter.masks(embedding, _mask_bbox(masks[best]), _median_pixel(masks[best]))
        candidates = [(m, s, False) for m, s in zip(masks, ious)] + [(m, s, True) for m, s in zip(second, second_ious)]
        kept: list[tuple[float, Any]] = []
        for mask, score, refined in candidates:
            area = int(mask.sum())
            if area == 0 or area >= PICTURE_MAX_SHARE * w * h:
                continue
            if refined and area > SECOND_PASS_AREA_MAX * best_area:
                continue
            kept.append((score, mask))
        if not kept:
            reasons.append(f"box {i}: no_mask_under_80_percent_of_picture")
            continue
        _, mask = max(kept, key=lambda sm: sm[0])
        x0, y0, x1, y1 = _mask_bbox(mask)
        return ObjectAnswer(mask, [round(x0 / w, 6), round(y0 / h, 6), round((x1 - x0) / w, 6), round((y1 - y0) / h, 6)],
                            reasons)
    return ObjectAnswer(None, None, reasons)


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

    def boxes(self, img: Any, phrase: str) -> list[list[float]]:
        torch = self.torch
        w, h = img.size
        inputs = self.processor(text=f"{TASK}{phrase}", images=img, return_tensors="pt").to(self.device, self.dtype)
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


@dataclass
class ObjectMaskRun:
    infer_s: float
    timings: dict[str, float] = field(default_factory=dict)
    payload: ObjectMaskPayload | None = None


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

    def run_object_mask(self, params: ObjectMaskParams, put_url: str,
                        note: Callable[[str], None] = lambda _s: None) -> ObjectMaskRun:
        """The object_mask kind on this same pair: per item the cut RGBA PNG (the picture's pixels, alpha 255 inside the mask,
        0 outside) to cut_url, the 8-bit grey mask PNG to mask_url when given, and one receipt to put_url."""
        import numpy as np
        from PIL import Image

        outs: list[ObjectMaskOut] = []
        model_s = 0.0
        note(f"object_mask generation config: {json.dumps(self.generation, sort_keys=True)}")
        with tempfile.TemporaryDirectory() as td:
            t_all = time.monotonic()
            for it in params.items:
                src = download(it.src_url, Path(td) / f"{it.id}.src")
                try:
                    img = load_picture(src)
                except UnreadablePicture as e:
                    outs.append(ObjectMaskOut(id=it.id, status="unreadable"))
                    note(f"object_mask {it.id}: unreadable — {e}")
                    continue
                w, h = img.size
                s = time.monotonic()
                answer = locate_object(self.grounder, self.segmenter, img, it.phrase)
                model_s += time.monotonic() - s
                if answer.mask is None or answer.bbox is None:
                    outs.append(ObjectMaskOut(id=it.id, status="none", reasons=answer.reasons or ["no_object_found"]))
                    note(f"object_mask {it.id}: none — {'; '.join(answer.reasons)}")
                    continue
                inside = np.asarray(answer.mask, dtype=bool)
                alpha = Image.fromarray(inside.astype(np.uint8) * 255)
                cut = img.convert("RGBA")
                cut.putalpha(alpha)
                cut_png = Path(td) / f"{it.id}.cut.png"
                cut.save(cut_png, format="PNG")
                upload(cut_png, it.cut_url, "image/png")
                if it.mask_url:
                    mask_png = Path(td) / f"{it.id}.mask.png"
                    alpha.save(mask_png, format="PNG")
                    upload(mask_png, it.mask_url, "image/png")
                outs.append(ObjectMaskOut(id=it.id, status="found", w=w, h=h, bbox=answer.bbox))
                note(f"object_mask {it.id}: found")
            wall_s = time.monotonic() - t_all
            payload = ObjectMaskPayload(
                model=self.model_name, model_sha256=self.weights_sha256, torch=self.runtime[0],
                torchvision=self.runtime[1], transformers=self.runtime[2], generation=self.generation, items=outs,
                timings=ScreenQuadTimings(model_s=round(model_s, 3), wall_s=round(wall_s, 3)))
            receipt = Path(td) / "object_mask.json"
            receipt.write_text(payload.model_dump_json(exclude_none=True))
            upload(receipt, put_url, "application/json")
            found = sum(1 for o in outs if o.status == "found")
            note(f"object_mask {found}/{len(outs)} found: model {model_s:.1f}s, wall {wall_s:.1f}s")
            if self.real:
                self.torch.cuda.empty_cache()
            return ObjectMaskRun(infer_s=wall_s, payload=payload,
                                 timings={"infer_s": wall_s, "model_s": model_s, "found": float(found),
                                          "items": float(len(outs))})

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
                    put_mask = False
                    if answer.glass is not None and it.mask_url:
                        import cv2

                        png = Path(td) / f"{it.id}.mask.png"
                        cv2.imwrite(str(png), fill_small_holes(answer.glass_mask))
                        upload(png, it.mask_url, "image/png")
                        put_mask = True
                    outs.append(ScreenQuadOut(id=it.id, status="found", device=it.device, quad=answer.quad, w=w, h=h,
                                              surface=SURFACE, glass=answer.glass, mask=True if put_mask else None,
                                              reasons=answer.glass_reasons or None))
                    note(f"screen_quad {it.id}: found" + ("" if answer.glass is not None else
                                                          f" (no glass: {'; '.join(answer.glass_reasons)})"))
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
