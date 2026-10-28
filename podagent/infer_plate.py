"""clean_plate inference (super-director v13 vertigo class): the background of a video WITHOUT the person, one plate per
window, from the person matte the pod already made for the same video.

  1. sample   each window [t0, t1] is sampled at `samples` evenly spread frames of the plate video, and the same frames of
              the matte (cutout WebM: the alpha plane of the VP9-alpha stream; alpha Matroska: the gray stream itself)
  2. median   the person (matte alpha > person_alpha, grown by dilate_px) is excluded; every pixel is the per-channel
              median of the samples where it was NOT the person; a pixel seen as background in fewer than min_visible
              samples is a HOLE (the speaker never moved off it), and so is one whose visible samples DISAGREE by more than
              stable_range (a static background agrees with itself — a spread means the person's edge leaked in)
  3. inpaint  the hole, grown by 3 px, is filled by LaMa (big-lama TorchScript, baked in the image, CUDA only) at a long side
              of at most inpaint_long_side; the output is upsampled and blended into the hole ONLY, so a real background
              pixel is never replaced by the network's guess
  4. refuse   a window whose hole is more than max_hole_fraction of the frame has no plate — the receipt names it in
              `refused`, the engine drops that layer by name and the other windows are unaffected; a plate is never a
              hallucination of most of the picture, and never a copy of the speaker

Settings and their sources: the clean-plate research receipt. Placement is absolute: CUDA or refuse, never CPU.
The median runs on the card too (the same device LaMa uses), in row bands so the working set stays a few hundred MiB.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import io
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .cp import download, upload
from .infer_matte import probe
from .models import CleanPlateOut, CleanPlateParams, CleanPlatePayload, CleanPlateRefusal, CleanPlateTimings
from .sanitize import safe_text

# THE model: the TorchScript export of the original advimman/lama «big-lama» checkpoint (Apache-2.0). The id is what a request
# must name; the digest pins the baked bytes (OUR measurement of the release asset; the upstream publishes none).
LAMA_MODEL_ID = "big_lama_torchscript"
LAMA_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
LAMA_ENV = "MODEL_LAMA"
LAMA_DEFAULT_PATH = "/opt/models/big-lama.pt"

HOLE_GROW_PX = 3          # the LaMa mask is the hole grown by this: the seam is drawn by the network, not left raw
FEATHER_PX = 5            # the blend ramp between the median and the network's pixels
BAND_ROWS = 128           # the median's row band (bounds the card working set)
MIN_FRAMES = 3            # fewer decoded frames than this is no median


class PlateRefused(ValueError):
    """The request cannot be served as asked (wrong model, unreadable input, too much hole, no CUDA). Ends the job
    status=error WITH this reason — a smeared plate behind a graphic is a visible defect."""


class WindowRefused(PlateRefused):
    """ONE window has no plate (too much of it is never uncovered, or it lies outside the video). The receipt names it in
    `refused`; the other windows of the request are still drawn."""


class Inpainter(Protocol):
    """LaMa, injectable so the pipeline is testable without the weights (tests pass a fake)."""

    def fill(self, rgb: Any, hole: Any, long_side: int) -> Any:
        """rgb [H,W,3] uint8, hole [H,W] bool (True = to fill) → [H,W,3] uint8 at the SAME size."""


def model_path() -> Path:
    return Path(os.environ.get(LAMA_ENV) or LAMA_DEFAULT_PATH)


def read_verified(path: Path) -> bytes:
    """The baked bytes ARE the model: read the file ONCE, refuse a missing or different one, and hand back THOSE bytes —
    torch.jit.load unpickles the archive, so it must never see a byte the digest did not cover."""
    if not Path(path).is_file():
        raise PlateRefused(f"clean_plate model file {path} is not on this pod ({LAMA_ENV}); the image bakes it")
    data = Path(path).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != LAMA_SHA256:
        raise PlateRefused(f"clean_plate model file {path} has sha256 {got[:12]}…, pinned {LAMA_SHA256[:12]}…")
    return data


class LamaTorch:
    """big-lama on CUDA. CUDA or nothing: placement is absolute, a CPU inpaint would be a silent crawl."""

    def __init__(self, path: Path) -> None:
        import torch

        if not torch.cuda.is_available():
            raise PlateRefused("clean_plate needs a CUDA device and this pod reports none — refusing, never CPU")
        data = read_verified(path)
        self.torch = torch
        self.model = torch.jit.load(io.BytesIO(data), map_location="cuda").eval()

    def fill(self, rgb: Any, hole: Any, long_side: int) -> Any:
        torch = self.torch
        F = torch.nn.functional
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        scale = min(1.0, long_side / max(h, w))
        hh, ww = max(8, round(h * scale)), max(8, round(w * scale))
        with torch.inference_mode():
            img = rgb.permute(2, 0, 1)[None].float() / 255.0                      # [1,3,H,W] RGB 0..1
            msk = hole[None, None].float()
            if (hh, ww) != (h, w):
                img = F.interpolate(img, size=(hh, ww), mode="area")
                # a hole pixel must survive the shrink wherever it falls between the sampled points (a thin edge band of the
                # person is exactly that): area-average, then `> 0` — never nearest, which drops it
                msk = F.interpolate(msk, size=(hh, ww), mode="area")
            msk = (msk > 0).float()                                               # binary, as IOPaint's (mask > 0)
            img = img * (1.0 - msk)                                               # what the model sees there is nothing, never the median's leak
            ph, pw = (-hh) % 8, (-ww) % 8
            if ph or pw:
                img = F.pad(img, (0, pw, 0, ph), mode="reflect")
                msk = F.pad(msk, (0, pw, 0, ph), mode="reflect")
            out = self.model(img, msk)[:, :, :hh, :ww]
            if (hh, ww) != (h, w):
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
            return (out[0].permute(1, 2, 0).clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)


# ── the pure pixel steps (device-agnostic: the pod runs them on the card, the tests on the CPU) ───────────────────────


def sample_fps(t0: float, t1: float, n: int) -> float:
    """The `fps` filter rate that spreads n frames evenly over [t0, t1)."""
    return n / max(1e-3, t1 - t0)


def person_mask(alpha: Any, person_alpha: int, dilate_px: int) -> Any:
    """alpha [K,H,W] uint8 → [K,H,W] bool: the person, grown by dilate_px (a max-pool = a square dilation)."""
    import torch

    p = (alpha > person_alpha).float()[:, None]
    if dilate_px > 0:
        p = torch.nn.functional.max_pool2d(p, kernel_size=2 * dilate_px + 1, stride=1, padding=dilate_px)
    return p[:, 0] > 0


def masked_median(frames: Any, hidden: Any, band: int = BAND_ROWS) -> tuple[Any, Any, Any]:
    """frames [K,H,W,3] uint8, hidden [K,H,W] bool (True = the person, not background) → (median [H,W,3] uint8,
    count [H,W] int16 = samples where the pixel WAS background, spread [H,W] uint8 = the largest channel range
    (max − min) over those samples). The median is the lower middle of the visible samples, per channel; a pixel with no
    visible sample is 0 (its count says so). A static background AGREES WITH ITSELF over the window: a large spread means
    the person's halo or edge leaked into the samples, which the caller treats as unreliable."""
    import torch

    k, h, w, _ = frames.shape
    med = torch.zeros((h, w, 3), dtype=torch.uint8, device=frames.device)
    cnt = torch.zeros((h, w), dtype=torch.int16, device=frames.device)
    spr = torch.zeros((h, w), dtype=torch.uint8, device=frames.device)
    for r0 in range(0, h, band):
        f = frames[:, r0:r0 + band].to(torch.float32)
        hid = hidden[:, r0:r0 + band]
        n = (~hid).sum(0)
        f = torch.where(hid[..., None], torch.full_like(f, float("inf")), f)
        srt = f.sort(dim=0).values
        idx = ((n - 1).clamp(min=0) // 2).long()
        pick = srt.gather(0, idx[None, :, :, None].expand(1, -1, -1, 3))[0]
        pick = torch.where(n[..., None] > 0, pick, torch.zeros_like(pick))
        top = srt.gather(0, (n - 1).clamp(min=0).long()[None, :, :, None].expand(1, -1, -1, 3))[0]
        rng = torch.where(n[..., None] > 0, top - srt[0], torch.zeros_like(top)).amax(-1)
        med[r0:r0 + band] = (pick + 0.5).clamp(0, 255).to(torch.uint8)
        cnt[r0:r0 + band] = n.to(torch.int16)
        spr[r0:r0 + band] = rng.clamp(0, 255).to(torch.uint8)
    return med, cnt, spr


def grow(mask: Any, px: int) -> Any:
    """[H,W] bool grown by px (square)."""
    import torch

    if px <= 0:
        return mask
    m = torch.nn.functional.max_pool2d(mask.float()[None, None], kernel_size=2 * px + 1, stride=1, padding=px)
    return m[0, 0] > 0


def blend(med: Any, filled: Any, region: Any, feather_px: int = FEATHER_PX) -> Any:
    """filled over med inside `region` ([H,W] bool), with a feathered edge so the seam never shows. Pixels outside the
    region's feather are exactly med's."""
    import torch

    soft = region.float()[None, None]
    if feather_px > 1:
        soft = torch.nn.functional.avg_pool2d(soft, kernel_size=feather_px, stride=1, padding=feather_px // 2,
                                              count_include_pad=False)
        soft = torch.where(region[None, None], torch.ones_like(soft), soft)       # the hole itself is all network
    a = soft[0, 0][..., None]
    out = med.float() * (1 - a) + filled.float() * a
    return (out + 0.5).clamp(0, 255).to(torch.uint8)


@dataclass(frozen=True)
class PlateResult:
    image: Any            # [H,W,3] uint8 on the working device
    covered: float        # share of the frame that is the median of real background
    inpainted: float      # share that is the network's (the hole)
    samples: int


def build_plate(frames: Any, alpha: Any, params: CleanPlateParams, inpainter: Inpainter | None,
                timings: dict[str, float] | None = None) -> PlateResult:
    """One window: frames [K,H,W,3] uint8, alpha [K,H,W] uint8 (same device) → the plate. `inpainter` None is only
    legal when the window has no hole."""
    t = timings if timings is not None else {}
    s = time.monotonic()
    hidden = person_mask(alpha, params.person_alpha, params.dilate_px)
    med, cnt, spr = masked_median(frames, hidden)
    hole = (cnt < params.min_visible) | (spr > params.stable_range)
    t["median_s"] = t.get("median_s", 0.0) + (time.monotonic() - s)
    frac = float(hole.float().mean())
    if frac > params.max_hole_fraction:
        raise WindowRefused(f"{frac:.0%} of the frame is never uncovered in this window (limit {params.max_hole_fraction:.0%}) — "
                            f"too much of the picture would be drawn by the network, so there is no plate")
    if frac == 0.0:
        return PlateResult(med, 1.0, 0.0, int(frames.shape[0]))
    if inpainter is None:
        raise PlateRefused("the window has a hole to inpaint and this service holds no inpainting model")
    s = time.monotonic()
    region = grow(hole, HOLE_GROW_PX)
    filled = inpainter.fill(med, region, params.inpaint_long_side)
    out = blend(med, filled, region)
    t["inpaint_s"] = t.get("inpaint_s", 0.0) + (time.monotonic() - s)
    return PlateResult(out, 1.0 - frac, frac, int(frames.shape[0]))


# ── decode ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _decode(argv: list[str], size: int, what: str) -> np.ndarray:
    err = tempfile.TemporaryFile()
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=err, timeout=600)
    if proc.returncode != 0:
        err.seek(0)
        raise PlateRefused(f"clean_plate {what} decode failed: "
                           + safe_text(err.read().decode(errors="replace").strip()[-300:], 300))
    buf = proc.stdout
    if len(buf) % size:            # no bytes at all is a window past the end of the video: the caller refuses THAT window
        raise PlateRefused(f"clean_plate {what} decode delivered {len(buf)} bytes, not a whole number of {size}-byte frames")
    return np.frombuffer(buf, np.uint8)


def video_argv(src: Path, t0: float, t1: float, n: int) -> list[str]:
    return ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(src),
            "-map", "0:v:0", "-vf", f"fps={sample_fps(t0, t1, n):.6f}", "-frames:v", str(n),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def matte_argv(src: Path, kind: str, t0: float, t1: float, n: int) -> list[str]:
    # the cutout is a VP9-alpha WebM: ffmpeg's native vp9 decoder drops the alpha plane, libvpx-vp9 keeps it
    dec = ["-c:v", "libvpx-vp9"] if kind == "cutout" else []
    plane = "alphaextract," if kind == "cutout" else ""
    return ["ffmpeg", "-v", "error", "-nostdin", *dec, "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(src),
            "-map", "0:v:0", "-vf", f"fps={sample_fps(t0, t1, n):.6f},{plane}format=gray", "-frames:v", str(n),
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]


def png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="PNG", compress_level=3)
    return out.getvalue()


@dataclass
class PlateRun:
    infer_s: float
    timings: dict[str, float] = field(default_factory=dict)
    payload: CleanPlatePayload | None = None


class PlateService:
    """Holds big-lama on the card (load ~2 s); serves every video after that. `inpainter` is injectable: tests pass a
    fake, the pod builds LamaTorch from the baked file. `device` is where the median runs ('cuda' in production)."""

    def __init__(self, model_name: str, path: Path | None = None, *, inpainter: Inpainter | None = None,
                 device: str = "cuda") -> None:
        if model_name != LAMA_MODEL_ID:
            raise PlateRefused(f"clean_plate serves {LAMA_MODEL_ID!r} only; the request named {model_name!r}")
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise PlateRefused("clean_plate needs a CUDA device and this pod reports none — refusing, never CPU")
        self.model_name, self.device = model_name, device
        self.inpainter = inpainter if inpainter is not None else LamaTorch(path or model_path())

    def run(self, params: CleanPlateParams, put_url: str,
            note: Callable[[str], None] = lambda _s: None) -> PlateRun:
        import torch

        with tempfile.TemporaryDirectory() as td:
            video = download(params.video_url, Path(td) / "video")
            matte = download(params.matte_url, Path(td) / "matte")
            t_all = time.monotonic()
            info = probe(video)
            minfo = probe(matte)
            if minfo.fps != info.fps:
                raise PlateRefused(f"clean_plate matte runs at {minfo.fps}, the video at {info.fps} — the matte is "
                                   f"frame-for-frame with its video, so these are not a pair")
            if (minfo.width, minfo.height) != (info.width, info.height):
                raise PlateRefused(f"clean_plate matte is {minfo.width}x{minfo.height}, the video {info.width}x{info.height} "
                                   f"— the matte would not register over it")
            timings: dict[str, float] = {"median_s": 0.0, "inpaint_s": 0.0}
            decode_s = 0.0
            outs: list[CleanPlateOut] = []
            refused: list[CleanPlateRefusal] = []
            files: list[tuple[Path, str]] = []
            for w in params.windows:
                s = time.monotonic()
                fsz, asz = info.width * info.height * 3, info.width * info.height
                # the two decodes are independent processes: run them side by side, the window waits for the slower
                with cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="plate-decode") as ex:
                    fv = ex.submit(_decode, video_argv(video, w.t0, w.t1, params.samples), fsz, f"video window {w.id}")
                    fa = ex.submit(_decode, matte_argv(matte, params.matte_kind, w.t0, w.t1, params.samples), asz,
                                   f"matte window {w.id}")
                    raw_v, raw_a = fv.result(), fa.result()
                decode_s += time.monotonic() - s
                nv, na = raw_v.size // fsz, raw_a.size // asz
                if nv != na:
                    raise PlateRefused(f"clean_plate window {w.id}: the video gave {nv} samples and the matte {na} — "
                                       f"they do not share one timeline")
                if nv < max(MIN_FRAMES, params.min_visible):
                    refused.append(CleanPlateRefusal(
                        id=w.id, reason=f"[{w.t0:.2f}, {w.t1:.2f}] gave {nv} frames — the window lies (partly) outside the "
                                        f"video, or is too short for {params.min_visible} samples"))
                    continue
                frames = torch.from_numpy(raw_v.reshape(nv, info.height, info.width, 3).copy()).to(self.device)
                alpha = torch.from_numpy(raw_a.reshape(nv, info.height, info.width).copy()).to(self.device)
                try:
                    res = build_plate(frames, alpha, params, self.inpainter, timings)
                except WindowRefused as e:
                    refused.append(CleanPlateRefusal(id=w.id, reason=f"[{w.t0:.2f}, {w.t1:.2f}] {e}"))
                    note(f"clean_plate {w.id}: refused — {e}")
                    continue
                png = png_bytes(res.image.cpu().numpy())
                path = Path(td) / f"{w.id}.png"
                path.write_bytes(png)
                outs.append(CleanPlateOut(id=w.id, bytes=len(png), sha256=hashlib.sha256(png).hexdigest(),
                                          samples=res.samples, covered=round(res.covered, 4),
                                          inpainted=round(res.inpainted, 4)))
                files.append((path, w.put_url))
                note(f"clean_plate {w.id}: {res.samples} samples, {res.inpainted:.0%} inpainted")
            wall_s = time.monotonic() - t_all
            payload = CleanPlatePayload(
                model=self.model_name, model_sha256=LAMA_SHA256, width=info.width, height=info.height, plates=outs,
                refused=refused,
                timings=CleanPlateTimings(decode_s=round(decode_s, 3), median_s=round(timings["median_s"], 3),
                                          inpaint_s=round(timings["inpaint_s"], 3), wall_s=round(wall_s, 3)))
            for path, url in files:
                upload(path, url, "image/png")
            receipt = Path(td) / "clean_plate.json"
            receipt.write_text(payload.model_dump_json(exclude_none=True))
            upload(receipt, put_url, "application/json")
            note(f"clean_plate {len(outs)} plates {info.width}x{info.height}: decode {decode_s:.1f}s, "
                 f"median {timings['median_s']:.1f}s, inpaint {timings['inpaint_s']:.1f}s, wall {wall_s:.1f}s")
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()      # the allocator's cache is not the weights: hand it back between requests
            return PlateRun(infer_s=wall_s, payload=payload,
                            timings={"infer_s": wall_s, "decode_s": decode_s, "median_s": timings["median_s"],
                                     "inpaint_s": timings["inpaint_s"], "plates": float(len(outs))})
