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
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import requests

from . import originrule
from .cp import _TIMEOUT as _TILE_TIMEOUT_S
from .cp import TransferTimeout, UrlNotAllowed, download, upload
from .models import (ClipRankGroup, ClipRankGroupResult, ClipRankParams, ClipRankPayload, ClipRankTileError,
                     ClipRankTileMeasure)
from .sanitize import safe_endpoint, safe_error, safe_text

# NO SENTINEL SCORE. A tile with nothing to score is `score: null` + a named error, never -1.0: in prod every
# b-roll tile was refused by the fetch guard (UrlNotAllowed) for 11 days, each came back a well-formed -1.0,
# and the planner read eleven days of failures as eleven days of irrelevant footage.
_EMPTY_INTENT = ClipRankTileError(kind="empty_intent",
                                  detail="embed-only group: the intent is empty, so no tile has a score")
# An origin tile (a stock provider's thumbnail) is fetched through originrule, never cp.download: it is an
# address a third party handed us, not a capability we minted. Its body cap is the engine's own
# provider-thumbnail cap (THUMB_MAX_BYTES, in its preview-link resolver) — what the box accepts off a
# provider; its per-read timeout is the same `cp._TIMEOUT` an image_urls tile's download runs under.
_ORIGIN_MAX_BYTES = 8_000_000
_DP = 4        # cosine error is ~1e-3; 4dp keeps the payload ~¼ the size

# Chunk the image tower so ONE lane's VRAM is a known number, not a function of a beat's candidate count —
# the cap the engine's own VRAM budget module declares and this arm had dropped (per-image embeds are same).
TILE_BATCH = 12

LANE_SIZING_WHY = """
THE clip_rank ENVELOPE IS ~80% NETWORK, AND IT WAS QUEUED AS IF IT WERE A RENDER.

Everything model-shaped went through ONE pod-side worker: align, face_probe, clip_rank AND render. Measured,
the SigLIP forward is 0.847 s for 14 tiles on an RTX 2060, while pulling those 14 tiles costs ~14 × 211 ms of
round-trip latency (the prod box's measured per-tile transfer) — and the fetch ran tile-by-tile INSIDE each
group's turn, so the card idled through most of the envelope. A one-worker GPU queue was protecting VRAM
against work that barely touches the card, and a render blocked every ranking envelope of a b-roll wave.

TWO PHASES, TWO WIDTHS. Tiles are latency-bound, so their width is a count of round trips in flight
(_FETCH_WORKERS, the same 8 the engine's tile PUTs use). ONE pool for the whole process: a
per-envelope pool would multiply with the lanes, which is the box-wide-budget lesson ops/runner.step_slots()
already learned. The towers are VRAM-bound and keep their own narrow lane.

THE LANE WIDTH IS DERIVED FROM THE CARD THIS AGENT BOOTED ON, NEVER FROM A CONSTANT. The weights load ONCE
and every lane shares them (main.py caches the service by weights hash), so they are a fixed toll, not a
per-lane cost; what a lane costs is one capped forward's activations. Both numbers are measured on an RTX
2060 with nvidia-smi and declared in the engine's VRAM budget module: 2322 MiB for the fp16 weights + CUDA
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
# No env lever for either width (MISC-333 §5): neither CLIP_RANK_LANES nor CLIP_RANK_FETCH_WORKERS was set on
# prod, so the lane count is the card's derivation and the fetch width is this constant.
_FETCH_WORKERS = 8


@dataclass(frozen=True)
class ClipRankRun:
    """One request's wall plus phase walls; values overlap only where the implementation overlaps work."""

    infer_s: float
    timings: dict[str, float]


def _log(msg: str) -> None:
    print(f"[clip_rank] {safe_text(msg)}", file=sys.stderr, flush=True)


def _free_vram_mb() -> float | None:
    """Free VRAM on the card the towers will run on, or None if this machine cannot say. Bounded: main's boot
    preflight reads it before the pod reports ready, and a hung nvidia-smi must not hang the boot."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
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


def _width_from(free_mb: float | None, threads: int) -> tuple[int, str]:
    """The lane width and the sentence that explains it (LANE_SIZING_WHY). Pure — readings are injected."""
    if free_mb is None:
        return 1, "the card reports no free VRAM — one lane, still the GPU, never the CPU"
    by_vram = max(1, int((free_mb - _VRAM_RESERVE_MB - _VRAM_WEIGHTS_MB) // _VRAM_PER_LANE_MB))
    return min(by_vram, max(1, threads)), (f"{free_mb:.0f} MiB free − {_VRAM_RESERVE_MB:.0f} reserve − "
                                           f"{_VRAM_WEIGHTS_MB:.0f} weights → vram-bound={by_vram} "
                                           f"({_VRAM_PER_LANE_MB:.0f} MiB per lane), host-bound={threads}")


def lane_width() -> int:
    """How many clip_rank envelopes may run at once on THIS card — derived from the card, never an env."""
    try:
        free = _free_vram_mb()
    except (OSError, ValueError) as e:
        _log(f"⚠ could not ask the card for free VRAM ({type(e).__name__}: {e}) — one lane, still the GPU")
        return 1
    n, why = _width_from(free, _host_threads())
    _log(f"{n} concurrent clip_rank lane(s): {why}")
    return n


def fetch_width() -> int:
    """The process-wide tile-fetch width, resolved without constructing its pool."""
    return _FETCH_WORKERS


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


def _finite(score: float | None, embed: list[float] | None) -> bool:
    return all(math.isfinite(x) for x in ([score] if score is not None else []) + (embed or []))


def _fetch_kind(exc: BaseException) -> str:
    if isinstance(exc, UrlNotAllowed):
        return "url_not_allowed"
    if isinstance(exc, (TransferTimeout, requests.Timeout, TimeoutError)):
        return "fetch_timeout"
    return "fetch_failed"


def _tile_error(kind: str, exc: BaseException, redacted_url: str) -> ClipRankTileError:
    status = _http_status(exc)
    head = safe_error(exc, 300) + (f" (HTTP {status})" if status is not None else "")
    return ClipRankTileError(kind=kind, detail=f"{head} — {redacted_url}")


def _fetch_tile(url: str, dest: Path):
    """One tile: GET → decode, returning the image or a secret-safe ClipRankTileError naming which stage
    failed and why (fetch refused / timed out / failed, or bytes that would not decode)."""
    from PIL import Image

    redacted_url = _redacted_tile_url(url)
    try:
        path = download(url, dest)
    except Exception as exc:  # noqa: BLE001 — a broken tile is data, not a pipeline fault; it is NAMED below
        return None, _tile_error(_fetch_kind(exc), exc, redacted_url), redacted_url
    try:
        return Image.open(path).convert("RGB"), None, redacted_url
    except Exception as exc:  # noqa: BLE001 — undecodable bytes are this tile's error, not the batch's
        return None, _tile_error("decode_error", exc, redacted_url), redacted_url


def _fetch_origin_tile(url: str, dest: Path):
    """One stock-provider tile: GET through the origin guard → workdir file → decode, exactly like an image_urls
    tile after its download. Returns (image, error, redacted url, measure); the measure is the fetched bytes
    and decoded size, None for a tile that did not decode. A guard refusal is `origin_refused`."""
    from PIL import Image

    redacted_url = _redacted_tile_url(url)
    try:
        data = originrule.fetch(url, max_bytes=_ORIGIN_MAX_BYTES, timeout_s=_TILE_TIMEOUT_S)
    except originrule.OriginRefused as exc:
        return None, _tile_error("origin_refused", exc, redacted_url), redacted_url, None
    except Exception as exc:  # noqa: BLE001 — a broken tile is data, not a pipeline fault; it is NAMED below
        return None, _tile_error(_fetch_kind(exc), exc, redacted_url), redacted_url, None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        img = Image.open(dest).convert("RGB")
    except Exception as exc:  # noqa: BLE001 — undecodable bytes are this tile's error, not the batch's
        return None, _tile_error("decode_error", exc, redacted_url), redacted_url, None
    return img, None, redacted_url, ClipRankTileMeasure(bytes=len(data), w=img.width, h=img.height)


def _submit_origins(urls: list[str], workdir: Path, shared: dict) -> list:
    """Put every origin tile on the wire, ONE fetch per url per REQUEST: `shared` (url → future) spans every
    origin group of the request, so a thumbnail two beats both shortlisted is fetched once."""
    pool = _fetch_pool()
    for u in urls:
        if u not in shared:
            shared[u] = pool.submit(_fetch_origin_tile, u, workdir / f"{len(shared)}.img")
    return [shared[u] for u in urls]


def _gather_origins(futs: list):
    """An origin group's decoded images, their indices, per-index errors and measures, and the first failed
    tile's redacted url (None when every tile decoded)."""
    images: list = []
    ok: list[int] = []
    errors: list[ClipRankTileError | None] = []
    measures: list[ClipRankTileMeasure | None] = []
    first_failed: str | None = None
    for i, f in enumerate(futs):
        img, reason, redacted_url, measure = f.result()
        if reason is not None:
            print(f"[clip_rank] tile {reason.kind} — {reason.detail}", file=sys.stderr, flush=True)
            first_failed = first_failed or redacted_url
        else:
            images.append(img)
            ok.append(i)
        errors.append(reason)
        measures.append(measure)
    return images, ok, errors, measures, first_failed


def _origin_report(i: int, n: int, result: ClipRankGroupResult, first_failed: str | None) -> str:
    """The one progress line an origin group earns: its failure count per kind and the first failed tile's
    redacted url — stderr alone never reaches the box, and a failure the box cannot see reads as data."""
    kinds = Counter(e.kind for e in result.errors if e is not None)
    line = f"clip_rank origin group {i + 1}/{n}: {sum(kinds.values())}/{len(result.errors)} tile(s) failed"
    if kinds:
        line += " (" + ", ".join(f"{k}={c}" for k, c in sorted(kinds.items())) + ")"
    if first_failed is not None:
        line += f", first {first_failed}"
    return line


def _submit_tiles(urls: list[str], workdir: Path) -> list:
    """Put every tile on the wire NOW. Submitting FLAT — never a pool task that itself submits — is what
    keeps a bounded shared pool from deadlocking on its own children."""
    pool = _fetch_pool()
    by_url = {u: pool.submit(_fetch_tile, u, workdir / f"{i}.img")
              for i, u in enumerate(dict.fromkeys(urls))}
    return [by_url[u] for u in urls]


def _gather(futs: list, cells=None) -> tuple[list, list[int], list[ClipRankTileError | None]]:
    """The decoded images, their REQUEST indices, and per request index the error that kept a tile out (None
    for a tile that decoded). A failed tile stays data — a NAMED error in the result; sheet failures raise."""
    images: list = []
    ok: list[int] = []
    errors: list[ClipRankTileError | None] = []
    for i, f in enumerate(futs):
        fetched = f.result()
        if isinstance(fetched, tuple) and len(fetched) == 3:
            img, reason, redacted_url = fetched
        elif isinstance(fetched, tuple) and len(fetched) == 2:
            img, reason = fetched
            redacted_url = "[redacted-url]"
        else:  # Compatibility for direct/unit-test futures made before failure records existed.
            img, reason, redacted_url = fetched, None, "[redacted-url]"
        if img is None and reason is None:
            reason = ClipRankTileError(kind="fetch_failed",
                                       detail=f"unknown fetch/decode failure — {redacted_url}")
        elif isinstance(reason, str):
            reason = ClipRankTileError(kind="fetch_failed", detail=f"{reason} — {redacted_url}")
        if reason is not None:
            # stderr is the operator's view; the SAME reason also rides in the result (errors[i]) — a failure
            # reported only to a log is a failure the planner reads as data.
            print(f"[clip_rank] tile {reason.kind} — {reason.detail}", file=sys.stderr, flush=True)
        cell = cells[i] if cells is not None else None
        if cell is not None:
            if img is None:
                raise ValueError(f"clip_rank sheet for cell {i} could not be fetched or decoded — "
                                 f"{reason.kind}: {reason.detail}")
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
        errors.append(reason if img is None else None)
    return images, ok, errors


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
            keys = [(tuple(g.tile_urls), tuple(g.image_cells) if g.image_cells is not None else None,
                     g.origin_urls is not None)
                    for g in params.groups]
            unique = list(dict.fromkeys(keys))
            shared_origins: dict = {}
            pending = {key: (_submit_origins(list(key[0]), Path(td) / "origins", shared_origins) if key[2]
                             else _submit_tiles(list(key[0]), Path(td) / f"g{i}"))
                       for i, key in enumerate(unique)}
            origin_extra: dict = {}
            if progress is not None:
                progress(f"clip_rank preparing {len(unique)} unique image set(s) for {n} group(s)")

            def prepare(key):
                tg = time.monotonic()
                if key[2]:
                    images, ok, errors, measures, first_failed = _gather_origins(pending[key])
                    origin_extra[key] = (measures, first_failed)
                else:
                    images, ok, errors = _gather(pending[key], key[1])
                gs = time.monotonic() - tg
                if not self._slots.acquire(timeout=60.0):
                    raise TimeoutError("clip_rank image forward waited 60s for a card lane")
                try:
                    tf = time.monotonic()
                    prepared = (*self._prepare(images, ok), errors)
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
                    progress(f"clip_rank group {i + 1}/{n} ({len(g.tile_urls)} tiles)")
                if not self._slots.acquire(timeout=60.0):
                    raise TimeoutError("clip_rank text forward waited 60s for a card lane")
                tf = time.monotonic()
                try:
                    measures, first_failed = origin_extra.get(key, (None, None))
                    groups.append(self._score_prepared(g, *ready[key], measures=measures))
                    forward_s += time.monotonic() - tf
                finally:
                    self._slots.release()
                if key[2] and progress is not None:
                    progress(_origin_report(i, n, groups[-1], first_failed))
            tp = time.monotonic()
            payload = ClipRankPayload(model=self.model_id, groups=groups)
            out = Path(td) / "clip_rank.json"
            out.write_text(payload.model_dump_json(exclude_none=True))
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
        if group.origin_urls is not None:
            images, ok, errors, measures, _ = _gather_origins(_submit_origins(group.origin_urls, workdir, {}))
            return self._score_prepared(group, *self._prepare(images, ok), errors, measures=measures)
        return self._score(group, *self._fetch(group.tile_urls, workdir))

    def _score(self, group: ClipRankGroup, images: list, ok: list[int],
               errors: list[ClipRankTileError | None]) -> ClipRankGroupResult:
        return self._score_prepared(group, *self._prepare(images, ok), errors)

    def _prepare(self, images: list, ok: list[int]):
        if not images:
            return None, [], ok
        with self.torch.no_grad():
            ie = self._image_features(images)
            embeds = [[round(x, _DP) for x in e] for e in ie.float().cpu().tolist()]
        return ie, embeds, ok

    def _score_prepared(self, group: ClipRankGroup, ie, embeds: list, ok: list[int],
                        errors: list[ClipRankTileError | None],
                        measures: list[ClipRankTileMeasure | None] | None = None) -> ClipRankGroupResult:
        """`measures` rides only on an origin group (parallel to its tiles); an image_urls group carries none."""
        n = len(group.tile_urls)
        if ie is None:
            # NO tile produced anything: the group itself is the error, carrying its first tile's reason.
            first = next(e for e in errors if e is not None)
            return ClipRankGroupResult(status="error", error=first, scores=[None] * n, embeds=[None] * n,
                                       errors=list(errors), measures=measures)
        scores = self._text_scores(group.intent, ie)
        # re-align onto the REQUESTED order: an image we could not fetch keeps score null + its named error
        by_idx = dict(zip(ok, scores))
        emb_by_idx = dict(zip(ok, embeds))
        out_scores: list[float | None] = []
        out_embeds: list[list[float] | None] = []
        out_errors: list[ClipRankTileError | None] = []
        for i in range(n):
            score, embed, err = by_idx.get(i), emb_by_idx.get(i), errors[i]
            if err is None and not _finite(score, embed):
                # A NaN/inf (a degenerate fp16 row) is a failure too: JSON would carry it as an UNNAMED null.
                err = ClipRankTileError(kind="non_finite",
                                        detail=f"the forward produced a non-finite score/embedding for tile {i}")
                score = embed = None
            elif err is None and score is None:
                err = _EMPTY_INTENT
            out_scores.append(score)
            out_embeds.append(embed)
            out_errors.append(err)
        if all(s is None and e is None for s, e in zip(out_scores, out_embeds)):
            first = next(e for e in out_errors if e is not None)
            return ClipRankGroupResult(status="error", error=first, scores=out_scores, embeds=out_embeds,
                                       errors=out_errors, measures=measures)
        return ClipRankGroupResult(status="ok", scores=out_scores, embeds=out_embeds, errors=out_errors,
                                   measures=measures)

    def _text_scores(self, intent: str, ie) -> list[float | None]:
        if not intent:
            return [None] * len(getattr(ie, "rows", ie))
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

    def _forward(self, intent: str, images: list) -> tuple[list[float | None], list[list[float]]]:
        ie, embeds, _ = self._prepare(images, list(range(len(images))))
        return self._text_scores(intent, ie), embeds

    @staticmethod
    def _fetch(urls: list[str], workdir: Path) -> tuple[list, list[int], list[ClipRankTileError | None]]:
        """Download+decode every url CONCURRENTLY; returns the decoded images, their REQUEST indices and the
        per-index errors."""
        return _gather(_submit_tiles(urls, workdir))
