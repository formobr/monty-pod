"""clip_rank inference: both SigLIP towers over the request's (intent, images) groups → cosine
relevance + L2-normalized image embeddings, packed as clip_rank.schema.json and PUT to the request's
presigned URL. Nothing is ranked here — the reorder, the relevance floor and the MMR dedup stay upstream.

The cosine runs HERE, inside the same fp16/no_grad block as the towers, so one forward yields both
numbers the planner needs and only the numbers cross back.

THE ENVELOPE IS NOT GPU WORK — IT IS A DOWNLOAD WITH A FORWARD ON THE END (see LANE_SIZING_WHY): tiles go to
one BOX-wide fetch pool (latency-bound, wide), the towers stay on the lane (VRAM-bound, narrow), so group k
forwards while k+1.. are still on the wire."""
from __future__ import annotations

import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .cp import download, upload
from .models import ClipRankGroup, ClipRankGroupResult, ClipRankParams, ClipRankPayload
from .sanitize import safe_endpoint, safe_text

_MISS = -1.0   # unreadable image, or an embed-only group where a score has nothing to mean
_DP = 4        # cosine error is ~1e-3; 4dp keeps the payload ~¼ the size

# Chunk the image tower so ONE lane's VRAM is a known number, not a function of a beat's candidate count —
# the cap engine vram_budget.SIGLIP_TILE_BATCH declares and this arm had dropped (per-image embeds are same).
TILE_BATCH = 12

LANE_SIZING_WHY = """
THE clip_rank ENVELOPE IS ~80% NETWORK, AND IT WAS QUEUED AS IF IT WERE A RENDER.

Everything model-shaped went through ONE pod-side worker: align, face_probe, clip_rank AND render. Measured,
the SigLIP forward is 0.847 s for 14 tiles on an RTX 2060, while pulling those 14 tiles costs ~14 × 211 ms of
round-trip latency (the prod box's measured per-tile transfer) — and the fetch ran tile-by-tile INSIDE each
group's turn, so the card idled through most of the envelope. A one-worker GPU queue was protecting VRAM
against work that barely touches the card, and a render blocked every ranking envelope of a b-roll wave.

TWO PHASES, TWO WIDTHS. Tiles are latency-bound, so their width is a count of round trips in flight
(_FETCH_WORKERS_DEFAULT, the same 8 the engine's tile PUTs use). ONE pool for the whole process: a
per-envelope pool would multiply with the lanes, which is the box-wide-budget lesson ops/runner.step_slots()
already learned. The towers are VRAM-bound and keep their own narrow lane.

THE LANE WIDTH IS DERIVED FROM THE CARD THIS AGENT BOOTED ON, NEVER FROM A CONSTANT. The weights load ONCE
and every lane shares them (main.py caches the service by weights hash), so they are a fixed toll, not a
per-lane cost; what a lane costs is one capped forward's activations. Both numbers are measured on an RTX
2060 with nvidia-smi and declared in the engine's vram_budget.py: 2322 MiB for the fp16 weights + CUDA
context, then 2736 MiB at 12 tiles, 3144 at 24, 3782 at 48 — ~34 MiB per tile, so one TILE_BATCH chunk is
2736 − 2322 = 414 MiB. The dev 2060 (3254 MiB free under a desktop) therefore gets exactly 1 lane, which is
the truth about that card and not a regression.

THE SECOND BOUND IS THE HOST AND IT IS NOT MEASURED — say so rather than hide it. A lane decodes and
preprocesses its own tiles on the CPU (PIL + the processor's resize/normalize) before anything reaches the
card, so past one lane per hardware thread the box is the bottleneck and the VRAM headroom buys nothing. It
is also what stops a 48 GB card from being handed 100 lanes on evidence that measured none of them.

A CARD THAT REPORTS NOTHING GETS 1, NOT A CPU. No nvidia-smi, no parse, no number => run the towers NARROW,
on the GPU, and say so. Ranking on the CPU because a QUERY failed would move the work across the placement
axis, which is the fallback the ruling forbids.
"""

_VRAM_WEIGHTS_MB = 2322.0
_VRAM_PER_LANE_MB = 414.0
_VRAM_RESERVE_MB = 512.0
_LANES_ENV = "CLIP_RANK_LANES"
_FETCH_WORKERS_ENV = "CLIP_RANK_FETCH_WORKERS"
_FETCH_WORKERS_DEFAULT = 8


@dataclass(frozen=True)
class ClipRankRun:
    """One request's wall plus phase walls; values overlap only where the implementation overlaps work."""

    infer_s: float
    timings: dict[str, float]


def _log(msg: str) -> None:
    print(f"[clip_rank] {safe_text(msg)}", file=sys.stderr, flush=True)


def _free_vram_mb() -> float | None:
    """Free VRAM on the card the towers will run on, or None if this machine cannot say."""
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    vals = [float(ln.strip()) for ln in r.stdout.splitlines() if ln.strip().replace(".", "", 1).isdigit()]
    return max(vals) if vals else None


def vram_total_mb() -> float | None:
    """Installed VRAM, or None if unreadable — never a coerced 0. Same probe as _free_vram_mb, PUBLIC because
    main.capacity_payload is a cross-module caller, self-contained because that caller has no guard of its own."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    vals = [float(ln.strip()) for ln in r.stdout.splitlines() if ln.strip().replace(".", "", 1).isdigit()]
    return max(vals) if vals else None


def _host_threads() -> int:
    """Hardware threads this process may use — a lane also decodes and preprocesses its tiles on the CPU."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)


def _cpu_quota_cores() -> float | None:
    """Cgroup CPU quota in cores, or None when unlimited/unreadable — a co-tenant box can hand us the host's
    full affinity mask while cpu.max caps the actual cycles, so affinity alone overstates what we can use."""
    try:
        raw = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="ascii").split()  # v2: "<quota|max> <period>"
        if len(raw) == 2 and raw[0] != "max":
            quota, period = float(raw[0]), float(raw[1])
            return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        pass
    try:  # v1 fallback
        quota = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="ascii"))
        period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="ascii"))
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


def usable_cores() -> int:
    """CPU cores this pod can actually schedule: min(affinity mask, cgroup quota). The broker admits offers
    by ADVERTISED cores; this is the measured after-claim truth the box gates the lease on (min_cpu_cores).
    The quota ROUNDS (7.9 cores is an 8-core box): truncation would condemn healthy hosts on the box side."""
    cores = _host_threads()
    quota = _cpu_quota_cores()
    if quota is not None:
        cores = min(cores, max(1, round(quota)))
    return max(1, int(cores))


def _width_from(free_mb: float | None, threads: int, env_raw: str | None) -> tuple[int, str]:
    """The lane width and the sentence that explains it (LANE_SIZING_WHY). Pure — readings are injected."""
    if env_raw:
        try:
            return max(1, int(env_raw)), f"env={env_raw}"
        except ValueError:
            # A typo in an operator's env may not kill a box that is already being billed: be loud, derive.
            _log(f"⚠ {_LANES_ENV}={env_raw!r} is not an integer — ignoring it and sizing from the card")
    if free_mb is None:
        return 1, "the card reports no free VRAM — one lane, still the GPU, never the CPU"
    by_vram = max(1, int((free_mb - _VRAM_RESERVE_MB - _VRAM_WEIGHTS_MB) // _VRAM_PER_LANE_MB))
    return min(by_vram, max(1, threads)), (f"{free_mb:.0f} MiB free − {_VRAM_RESERVE_MB:.0f} reserve − "
                                           f"{_VRAM_WEIGHTS_MB:.0f} weights → vram-bound={by_vram} "
                                           f"({_VRAM_PER_LANE_MB:.0f} MiB per lane), host-bound={threads}")


def lane_width() -> int:
    """How many clip_rank envelopes may run at once on THIS card. An operator's number still wins."""
    env_raw = os.environ.get(_LANES_ENV)
    try:
        free = None if env_raw else _free_vram_mb()
    except (OSError, ValueError) as e:
        _log(f"⚠ could not ask the card for free VRAM ({type(e).__name__}: {e}) — one lane, still the GPU")
        return 1
    n, why = _width_from(free, _host_threads(), env_raw)
    _log(f"{n} concurrent clip_rank lane(s): {why}")
    return n


def fetch_width() -> int:
    """The process-wide tile-fetch width, resolved without constructing its pool."""
    try:
        return max(1, int(os.environ.get(_FETCH_WORKERS_ENV, "") or _FETCH_WORKERS_DEFAULT))
    except ValueError:
        return _FETCH_WORKERS_DEFAULT


_fetch_pool_lock = threading.Lock()
_FETCH_POOL: "cf.ThreadPoolExecutor | None" = None


def _fetch_pool() -> "cf.ThreadPoolExecutor":
    """The ONE tile-fetch pool of this process — per-envelope pools would multiply with the lanes."""
    global _FETCH_POOL
    with _fetch_pool_lock:
        if _FETCH_POOL is None:
            _FETCH_POOL = cf.ThreadPoolExecutor(max_workers=fetch_width(), thread_name_prefix="tile")
        return _FETCH_POOL


def _redacted_tile_url(url: str) -> str:
    """Diagnostic tile address with its capability query removed, while retaining the URL scheme."""
    try:
        parts = urlsplit(url)
        if not parts.scheme:
            return safe_endpoint(url)
        return f"{parts.scheme}://{safe_endpoint(url)}"
    except ValueError:
        return "[redacted-url]"


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) if response is not None else None
    if status is None:
        status = getattr(exc, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _fetch_tile(url: str, dest: Path):
    """One tile: presigned GET → decode, returning image or a secret-safe failure record."""
    from PIL import Image

    redacted_url = _redacted_tile_url(url)
    try:
        return Image.open(download(url, dest)).convert("RGB"), None, redacted_url
    except Exception as exc:  # noqa: BLE001 — a broken tile is data, not a pipeline fault
        status = _http_status(exc)
        reason = type(exc).__name__ + (f" (HTTP {status})" if status is not None else "")
        return None, reason, redacted_url


def _submit_tiles(urls: list[str], workdir: Path) -> list:
    """Put every tile on the wire NOW. Submitting FLAT — never a pool task that itself submits — is what
    keeps a bounded shared pool from deadlocking on its own children."""
    pool = _fetch_pool()
    by_url = {u: pool.submit(_fetch_tile, u, workdir / f"{i}.img")
              for i, u in enumerate(dict.fromkeys(urls))}
    return [by_url[u] for u in urls]


def _gather(futs: list, cells=None) -> tuple[list, list[int]]:
    """The decoded images and REQUEST indices; failed tiles stay data, with sheet failures named loudly."""
    images: list = []
    ok: list[int] = []
    for i, f in enumerate(futs):
        fetched = f.result()
        if isinstance(fetched, tuple) and len(fetched) == 3:
            img, reason, redacted_url = fetched
        elif isinstance(fetched, tuple) and len(fetched) == 2:
            img, reason = fetched
            redacted_url = "[redacted-url]"
        else:  # Compatibility for direct/unit-test futures made before failure records existed.
            img, reason, redacted_url = fetched, None, "[redacted-url]"
        if reason is not None:
            print(f"[clip_rank] tile fetch failed — {reason} — {redacted_url}",
                  file=sys.stderr, flush=True)
        cell = cells[i] if cells is not None else None
        if cell is not None:
            if img is None:
                raise ValueError(f"clip_rank sheet for cell {i} could not be fetched or decoded — "
                                 f"{reason or 'unknown fetch/decode failure'} — {redacted_url}")
            x, y, w, h, sheet_w, sheet_h = (int(v) for v in cell)
            if min(x, y) < 0 or min(w, h, sheet_w, sheet_h) <= 0:
                raise ValueError(f"clip_rank cell {i} has non-positive geometry")
            if img.size != (sheet_w, sheet_h):
                raise ValueError(f"clip_rank cell {i} expected sheet {sheet_w}x{sheet_h}, got "
                                 f"{img.width}x{img.height}")
            if x + w > sheet_w or y + h > sheet_h:
                raise ValueError(f"clip_rank cell {i} lies outside its declared sheet")
            img = img.crop((x, y, x + w, y + h))
        if img is not None:
            images.append(img)
            ok.append(i)
    return images, ok


class ClipRankService:
    """Loads SigLIP once (the dominant cost); serves every group batch after that.

    Loads from a LOCAL directory the pod fetched and hash-verified — never from a hub id. `model_id` is
    provenance only (it rides into the payload), so a request string can never steer what gets loaded.
    """

    def __init__(self, model_id: str, weights_dir: Path, *, parallel: int = 1,
                 slots: threading.BoundedSemaphore | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_id = model_id
        self.parallel = max(1, int(parallel))
        self._slots = slots or threading.BoundedSemaphore(self.parallel)
        self.model = AutoModel.from_pretrained(
            str(weights_dir), dtype=self.dtype, local_files_only=True).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(str(weights_dir), local_files_only=True)

    def run(self, params: ClipRankParams, put_url: str,
            progress: "Callable[[str], None] | None" = None) -> ClipRankRun:
        """Returns total inference wall plus measured gather/forward/serialization/upload phase walls."""
        t0 = time.monotonic()
        n = len(params.groups)
        gather_s = forward_s = 0.0
        with tempfile.TemporaryDirectory() as td:
            # score_many sends several phrasings over the SAME pixels. Download/decode/image-forward that exact
            # URL tuple once, then reuse its image embeddings for every text tower. Distinct shortlists occupy
            # the already-sized rank lanes concurrently; result assembly remains in request order.
            keys = [(tuple(g.image_urls), tuple(g.image_cells) if g.image_cells is not None else None)
                    for g in params.groups]
            unique = list(dict.fromkeys(keys))
            pending = {key: _submit_tiles(list(key[0]), Path(td) / f"g{i}")
                       for i, key in enumerate(unique)}
            if progress is not None:
                progress(f"clip_rank preparing {len(unique)} unique image set(s) for {n} group(s)")

            def prepare(key):
                tg = time.monotonic()
                images, ok = _gather(pending[key], key[1])
                gs = time.monotonic() - tg
                if not self._slots.acquire(timeout=60.0):
                    raise TimeoutError("clip_rank image forward waited 60s for a card lane")
                try:
                    tf = time.monotonic()
                    prepared = self._prepare(images, ok)
                    return prepared, gs, time.monotonic() - tf
                finally:
                    self._slots.release()

            with cf.ThreadPoolExecutor(max_workers=min(self.parallel, max(1, len(unique))),
                                       thread_name_prefix="clip-rank-group") as ex:
                prepared = {key: ex.submit(prepare, key) for key in unique}
                ready = {}
                for key in unique:
                    ready[key], gs, fs = prepared[key].result()
                    gather_s += gs
                    forward_s += fs

            groups = []
            for i, (g, key) in enumerate(zip(params.groups, keys)):
                if progress is not None:
                    progress(f"clip_rank group {i + 1}/{n} ({len(g.image_urls)} tiles)")
                if not self._slots.acquire(timeout=60.0):
                    raise TimeoutError("clip_rank text forward waited 60s for a card lane")
                tf = time.monotonic()
                try:
                    groups.append(self._score_prepared(g, *ready[key]))
                    forward_s += time.monotonic() - tf
                finally:
                    self._slots.release()
            tp = time.monotonic()
            payload = ClipRankPayload(model=self.model_id, groups=groups)
            out = Path(td) / "clip_rank.json"
            out.write_text(payload.model_dump_json())
            payload_s = time.monotonic() - tp
            infer_s = time.monotonic() - t0
            if progress is not None:
                progress(f"clip_rank {n} groups done in {infer_s:.0f}s, uploading")
            tu = time.monotonic()
            upload(out, put_url, "application/json")
            upload_s = time.monotonic() - tu
        return ClipRankRun(
            infer_s=infer_s,
            timings={
                "infer_s": round(infer_s, 3),
                "tile_gather_work_s": round(gather_s, 3),
                "forward_work_s": round(forward_s, 3),
                "payload_s": round(payload_s, 3),
                "upload_s": round(upload_s, 3),
                "work_s": round(time.monotonic() - t0, 3),
                "unique_image_sets_n": float(len(unique)),
                "reused_groups_n": float(n - len(unique)),
                "parallel_width_n": float(self.parallel),
            },
        )

    def _run_group(self, group: ClipRankGroup, workdir: Path) -> ClipRankGroupResult:
        """One group, both phases — run() pipelines them instead, so the fetch of k+1 overlaps k's forward."""
        return self._score(group, *self._fetch(group.image_urls, workdir))

    def _score(self, group: ClipRankGroup, images: list, ok: list[int]) -> ClipRankGroupResult:
        return self._score_prepared(group, *self._prepare(images, ok))

    def _prepare(self, images: list, ok: list[int]):
        if not images:
            return None, [], ok
        with self.torch.no_grad():
            ie = self._image_features(images)
            embeds = [[round(x, _DP) for x in e] for e in ie.float().cpu().tolist()]
        return ie, embeds, ok

    def _score_prepared(self, group: ClipRankGroup, ie, embeds: list, ok: list[int]) -> ClipRankGroupResult:
        n = len(group.image_urls)
        if ie is None:
            return ClipRankGroupResult(scores=[_MISS] * n, embeds=[None] * n)
        scores = self._text_scores(group.intent, ie)
        # re-align onto the REQUESTED order: an image we could not fetch keeps -1.0/None so it sorts last
        by_idx = dict(zip(ok, scores))
        emb_by_idx = dict(zip(ok, embeds))
        return ClipRankGroupResult(
            scores=[by_idx.get(i, _MISS) for i in range(n)],
            embeds=[emb_by_idx.get(i) for i in range(n)],
        )

    def _text_scores(self, intent: str, ie) -> list[float]:
        if not intent:
            return [_MISS] * len(getattr(ie, "rows", ie))
        torch = self.torch
        with torch.no_grad():
            tin = self.proc(text=[intent], return_tensors="pt", padding="max_length", truncation=True)
            te = torch.nn.functional.normalize(
                self._feat(self.model.get_text_features(**tin.to(self.device))), dim=-1)
            sims = (ie @ te.T).squeeze(-1).float().cpu().tolist()
        return [round(s, _DP) for s in sims]

    @staticmethod
    def _feat(out):
        """The pooled embedding, whatever get_image/text_features returned. Some transformers versions hand
        back a bare tensor; others a BaseModelOutputWithPooling whose pooled embed is `.pooler_output`. Extract
        the tensor either way — F.normalize calls `.norm()` on its input, so a model-output object crashes with
        `'BaseModelOutputWithPooling' object has no attribute 'norm'`. The Dockerfile pins transformers, but a
        `pip install --no-deps .` once shipped an unpinned one past the pyproject pin, so guard in code too."""
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def _image_features(self, images: list):
        """L2-normalized image vectors, CHUNKED at TILE_BATCH — one forward over a whole sheet made a lane's
        VRAM a function of a beat's candidate count, which is not a number a lane can be sized against. The
        chunks are concatenated on the DEVICE so the cosine below is bit-for-bit the one-forward answer."""
        torch = self.torch
        chunks = []
        for a in range(0, len(images), TILE_BATCH):
            iin = self.proc(images=images[a:a + TILE_BATCH], return_tensors="pt").to(self.device)
            iin = {k: (v.to(self.dtype) if v.dtype == torch.float32 else v) for k, v in iin.items()}
            chunks.append(torch.nn.functional.normalize(
                self._feat(self.model.get_image_features(**iin)), dim=-1))
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

    def _forward(self, intent: str, images: list) -> tuple[list[float], list[list[float]]]:
        ie, embeds, _ = self._prepare(images, list(range(len(images))))
        return self._text_scores(intent, ie), embeds

    @staticmethod
    def _fetch(urls: list[str], workdir: Path) -> tuple[list, list[int]]:
        """Download+decode every url CONCURRENTLY; returns the decoded images and their REQUEST indices."""
        return _gather(_submit_tiles(urls, workdir))
