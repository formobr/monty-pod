"""cutout inference: background removal on the pod GPU — BiRefNet_dynamic, traced to TorchScript, over a batch of pictures →
one RGBA PNG per picture (the source's own RGB + the model's alpha), each PUT to its own presigned URL; a receipt
(cutout.schema.json) goes to put_url. It replaces a paid background-removal call made from the control plane.

Settings and their sources: the research receipt docs/research/pod-cutout-birefnet.md (the model's own inference recipe,
the operator's 2026-11-06 bench: alpha IoU 0.994 against the paid cut).

  weights   a WeightsRef tar (weights.ensure): config.json = the manifest {hf_repo, hf_revision, input, dtype, mean, std,
            torch, torchvision} + the one traced .pt (+ the upstream MIT LICENSE). No trust_remote_code, no HF access.
  load      `import torchvision` (registers the torchvision::deform_conv2d TorchScript op the trace calls — without it
            torch.jit.load fails «Unknown builtin op»), REFUSE when the manifest's torch / torchvision differ from this
            runtime's (the .pt was traced by exactly that pair inside this image), then torch.jit.load(.pt, cuda).eval()
  pre       exif_transpose; long side capped at 2048 (LANCZOS); an input with real transparency is `already_cut` and the
            model does not run; RGB; PIL BILINEAR to the manifest's input size (antialiased on reduce, as benched — never
            F.interpolate without antialias); /255; (x - mean) / std; manifest dtype on CUDA
  post      the sigmoid map the traced wrapper returns, resized back to the picture's size bilinear (align_corners False)
            = the soft alpha; RGBA PNG = the picture's RGB + that alpha
  named     per picture: cut | already_cut | empty_cut (no alpha >= 24/255) | no_separation (>= 98 % of the frame >= 24/255)
            | unreadable (not a picture). Only `cut` is PUT to out_url; every other outcome is named in the receipt.

Placement is absolute: CUDA or refuse. A missing CUDA device, an out-of-memory forward, a runtime that is not the one the
weights were traced under: each ends the job status=error WITH its reason — never a CPU run, never a fallback, never a
pass-through reported as a success.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cp import download, upload
from .models import CutoutOut, CutoutParams, CutoutPayload, CutoutTimings

LONG_SIDE_CAP = 2048          # a source bigger than this is shrunk (LANCZOS) first; the cut PNG is that size
# alpha >= 24/255 is "the picture is there": THE SAME number as the engine's media_still mark_probe threshold, so the bbox and
# the empty / no-separation verdicts agree with how the engine later measures the cut PNG it gets back from us
ALPHA_VISIBLE = 24
NO_SEPARATION_SHARE = 0.98    # this share of the frame (or more) visible = the model found no subject apart from the background
TRANSPARENT_SHARE = 0.01      # an input with this share of its frame below ALPHA_VISIBLE already carries a cut (not a stray pixel)
MANIFEST_KEYS = ("hf_repo", "hf_revision", "input", "dtype", "mean", "std", "torch", "torchvision")
_DTYPES = {"float16": "float16", "float32": "float32"}


class CutoutRefused(ValueError):
    """The request cannot be served as asked (no CUDA, out of memory, the weights were traced under another runtime, a
    malformed manifest). Ends the job status=error WITH this reason."""


class UnreadablePicture(ValueError):
    """The bytes of one source are not a picture; the receipt names that item `unreadable`, the others are unaffected."""


# ── the manifest and the runtime check (no torch needed) ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Manifest:
    hf_repo: str
    hf_revision: str
    height: int
    width: int
    dtype: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    torch: str
    torchvision: str


def read_manifest(wdir: Path) -> tuple[Manifest, Path]:
    """The weights directory's config.json and its ONE traced .pt. Anything else is a refusal that names what is wrong."""
    cfg = Path(wdir) / "config.json"
    if not cfg.is_file():
        raise CutoutRefused(f"cutout weights hold no config.json manifest under {wdir}")
    try:
        raw = json.loads(cfg.read_text())
    except ValueError as e:
        raise CutoutRefused(f"cutout manifest {cfg} is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise CutoutRefused(f"cutout manifest {cfg} is not a JSON object")
    missing = [k for k in MANIFEST_KEYS if k not in raw]
    if missing:
        raise CutoutRefused(f"cutout manifest lacks {', '.join(missing)}")
    shape = raw["input"]
    if not (isinstance(shape, list) and len(shape) == 4 and shape[0] == 1 and shape[1] == 3
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in shape)):
        raise CutoutRefused(f"cutout manifest input {shape!r} is not [1, 3, H, W]")
    if raw["dtype"] not in _DTYPES:
        raise CutoutRefused(f"cutout manifest dtype {raw['dtype']!r} is not one of {sorted(_DTYPES)}")
    triples: dict[str, tuple[float, float, float]] = {}
    for key in ("mean", "std"):
        v = raw[key]
        if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                           for x in v)):
            raise CutoutRefused(f"cutout manifest {key} {v!r} is not three numbers")
        triples[key] = (float(v[0]), float(v[1]), float(v[2]))
    if any(s <= 0 for s in triples["std"]):
        raise CutoutRefused("cutout manifest std must be positive")
    for key in ("hf_repo", "hf_revision", "torch", "torchvision"):
        if not (isinstance(raw[key], str) and raw[key]):
            raise CutoutRefused(f"cutout manifest {key} is not a non-empty string")
    pts = sorted(Path(wdir).glob("*.pt"))
    if len(pts) != 1:
        raise CutoutRefused(f"cutout weights hold {len(pts)} traced .pt files under {wdir}, expected exactly one")
    return Manifest(raw["hf_repo"], raw["hf_revision"], shape[2], shape[3], raw["dtype"], triples["mean"], triples["std"],
                    raw["torch"], raw["torchvision"]), pts[0]


def check_runtime(manifest: Manifest, torch_version: str, torchvision_version: str) -> None:
    """REFUSE a traced model under a runtime that is not the one it was traced with: a TorchScript archive pins the op set of
    its torch / torchvision, and a silent mismatch is a wrong picture, not an error. Names both pairs."""
    if manifest.torch != torch_version or manifest.torchvision != torchvision_version:
        raise CutoutRefused(f"cutout weights were traced under torch {manifest.torch} / torchvision {manifest.torchvision}, "
                            f"this pod runs torch {torch_version} / torchvision {torchvision_version} — refusing to load")


# ── the pure pixel steps ────────────────────────────────────────────────────────────────────────────────────────────


def bbox_fractions(visible: Any) -> list[float]:
    """[x0, y0, x1, y1] as fractions of the frame, over a [H,W] bool mask with at least one True."""
    h, w = visible.shape
    rows, cols = visible.any(axis=1), visible.any(axis=0)
    y0, x0 = int(rows.argmax()), int(cols.argmax())
    y1, x1 = h - int(rows[::-1].argmax()), w - int(cols[::-1].argmax())
    return [round(x0 / w, 6), round(y0 / h, 6), round(x1 / w, 6), round(y1 / h, 6)]


def classify(alpha: Any) -> tuple[str, list[float] | None]:
    """A model alpha [H,W] uint8 → (status, bbox): `empty_cut` (nothing visible), `no_separation` (the frame is nearly all
    visible) or `cut`."""
    visible = alpha >= ALPHA_VISIBLE
    if not visible.any():
        return "empty_cut", None
    bbox = bbox_fractions(visible)
    if float(visible.mean()) >= NO_SEPARATION_SHARE:
        return "no_separation", bbox
    return "cut", bbox


def load_picture(path: Path) -> tuple[Any, Any]:
    """The source as (PIL RGB image, None) — or (PIL RGBA image, alpha ndarray) when it carries real transparency. exif_transpose
    first, then the long-side cap; a source that is not a picture raises UnreadablePicture."""
    import numpy as np
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as opened:
            opened.load()
            img = ImageOps.exif_transpose(opened)
            img = img if img is not None else opened.copy()
            has_alpha = img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in img.info
            img = img.convert("RGBA" if has_alpha else "RGB")
    except Exception as e:  # PIL raises a zoo (OSError, DecompressionBombError, SyntaxError, ValueError)
        raise UnreadablePicture(f"{type(e).__name__}: {e}") from e
    w, h = img.size
    if max(w, h) > LONG_SIDE_CAP:
        scale = LONG_SIDE_CAP / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    if not has_alpha:
        return img, None
    alpha = np.asarray(img)[:, :, 3]
    if float((alpha < ALPHA_VISIBLE).mean()) >= TRANSPARENT_SHARE:
        return img, alpha
    return img.convert("RGB"), None                       # an opaque RGBA: the alpha channel says nothing, cut it


def rgba_png(rgb: Any, alpha: Any) -> bytes:
    """The source's own RGB pixels + `alpha` ([H,W] uint8) as a PNG."""
    import io

    import numpy as np
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.dstack([np.asarray(rgb), alpha])).save(out, format="PNG", compress_level=3)
    return out.getvalue()


@dataclass
class CutoutRun:
    infer_s: float
    timings: dict[str, float] = field(default_factory=dict)
    payload: CutoutPayload | None = None


class CutoutService:
    """Holds the traced BiRefNet on the card; serves every batch after that. `device` is 'cuda' in production — the tests'
    CPU stand-in is injected, the pod never passes it. `runtime` = (torch, torchvision) versions, None = the real ones
    (and then torchvision is imported, which registers the op the trace calls)."""

    def __init__(self, model_name: str, wdir: Path, weights_sha256: str, *, device: str = "cuda",
                 runtime: tuple[str, str] | None = None) -> None:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise CutoutRefused("cutout needs a CUDA device and this pod reports none — refusing, never CPU")
        manifest, pt = read_manifest(wdir)
        if runtime is None:
            import torchvision  # noqa: F401 — registers torchvision::deform_conv2d BEFORE torch.jit.load reads the archive

            runtime = (torch.__version__, torchvision.__version__)
        check_runtime(manifest, *runtime)
        self.torch, self.runtime = torch, runtime
        self.model_name, self.weights_sha256, self.device, self.manifest = model_name, weights_sha256, device, manifest
        self.dtype = getattr(torch, _DTYPES[manifest.dtype])
        try:
            self.model = torch.jit.load(str(pt), map_location=device).eval()
        except RuntimeError as e:
            raise CutoutRefused(f"cutout traced model {pt.name} failed to load: {str(e)[:300]}") from e

    def alpha_map(self, rgb: Any) -> Any:
        """A PIL RGB picture → its alpha [H,W] uint8 at the picture's own size (pre/post of the module docstring)."""
        import numpy as np
        from PIL import Image

        torch, m = self.torch, self.manifest
        w, h = rgb.size
        small = np.asarray(rgb.resize((m.width, m.height), Image.BILINEAR), dtype=np.float32) / 255.0
        norm = (small - np.asarray(m.mean, dtype=np.float32)) / np.asarray(m.std, dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(norm.transpose(2, 0, 1)))[None].to(device=self.device, dtype=self.dtype)
        oom = getattr(torch.cuda, "OutOfMemoryError", ())
        try:
            with torch.inference_mode():
                pred = self.model(x)
                if not torch.is_tensor(pred) or tuple(pred.shape) != (1, 1, m.height, m.width):
                    raise CutoutRefused(f"cutout model returned {getattr(pred, 'shape', type(pred).__name__)}, "
                                        f"expected a [1, 1, {m.height}, {m.width}] map")
                soft = torch.nn.functional.interpolate(pred.float(), size=(h, w), mode="bilinear", align_corners=False)
                return (soft[0, 0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
        except oom as e:      # type: ignore[misc]
            torch.cuda.empty_cache()
            raise CutoutRefused(f"cutout ran out of GPU memory on a {w}x{h} picture ({str(e)[:160]}) — refusing, "
                                f"never a CPU run") from e

    def run(self, params: CutoutParams, put_url: str,
            note: Callable[[str], None] = lambda _s: None) -> CutoutRun:
        outs: list[CutoutOut] = []
        files: list[tuple[Path, str]] = []
        model_s = 0.0
        with tempfile.TemporaryDirectory() as td:
            t_all = time.monotonic()
            for it in params.items:
                src = download(it.src_url, Path(td) / f"{it.id}.src")
                try:
                    img, alpha = load_picture(src)
                except UnreadablePicture as e:
                    outs.append(CutoutOut(id=it.id, status="unreadable"))
                    note(f"cutout {it.id}: unreadable — {e}")
                    continue
                w, h = img.size
                if alpha is not None:                     # real transparency already: no model run, nothing PUT
                    status, bbox = classify(alpha)
                    status = "empty_cut" if status == "empty_cut" else "already_cut"
                    outs.append(CutoutOut(id=it.id, status=status, **({} if bbox is None else {"w": w, "h": h, "bbox": bbox})))
                    note(f"cutout {it.id}: {status} (no model run)")
                    continue
                s = time.monotonic()
                alpha = self.alpha_map(img)
                model_s += time.monotonic() - s
                status, bbox = classify(alpha)
                if status == "empty_cut":
                    outs.append(CutoutOut(id=it.id, status=status))
                elif status == "no_separation":
                    outs.append(CutoutOut(id=it.id, status=status, w=w, h=h, bbox=bbox))
                else:
                    png = rgba_png(img, alpha)
                    path = Path(td) / f"{it.id}.png"
                    path.write_bytes(png)
                    outs.append(CutoutOut(id=it.id, status="cut", w=w, h=h, bbox=bbox, bytes=len(png),
                                          sha256=hashlib.sha256(png).hexdigest()))
                    files.append((path, it.out_url))
                note(f"cutout {it.id}: {status}")
            wall_s = time.monotonic() - t_all
            payload = CutoutPayload(model=self.model_name, model_sha256=self.weights_sha256, torch=self.runtime[0],
                                    torchvision=self.runtime[1], items=outs,
                                    timings=CutoutTimings(model_s=round(model_s, 3), wall_s=round(wall_s, 3)))
            for path, url in files:
                upload(path, url, "image/png")
            receipt = Path(td) / "cutout.json"
            receipt.write_text(payload.model_dump_json(exclude_none=True))
            upload(receipt, put_url, "application/json")
            note(f"cutout {len(files)}/{len(outs)} cut: model {model_s:.1f}s, wall {wall_s:.1f}s")
            if self.device.startswith("cuda"):
                self.torch.cuda.empty_cache()             # the allocator's cache is not the weights: hand it back
            return CutoutRun(infer_s=wall_s, payload=payload,
                             timings={"infer_s": wall_s, "model_s": model_s, "cut": float(len(files)),
                                      "items": float(len(outs))})
