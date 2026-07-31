"""Entrypoint. The agent's whole runtime contract is two environment variables (CP_URL,
JOB_TOKEN) — everything else this process does arrives as data from the control plane.
Run: python -m podagent.main"""
from __future__ import annotations

import concurrent.futures as cf
import os
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests
from pydantic import ValidationError

from .cp import ControlPlane
from .models import SPEC_VERSION, InferRequest, InferResult, InferTiming, PodJob, RenderSpec

if TYPE_CHECKING:
    from .infer_align import AlignService
    from .infer_cliprank import ClipRankService
    from .infer_probe import ProbeService

INFER_KINDS = ("align", "face_probe", "clip_rank")

# /pod/job is a LONG poll: an empty answer is supposed to cost the server-side block, not a round trip. When
# it does not (CP restart, a short block, a 204 storm) a bare `continue` hammers the control plane from a
# RENTED box at an unbounded rate. Floor one poll per second; a real long-poll never notices it.
_MIN_POLL_INTERVAL_S = 1.0

# The claim loop DISPATCHES; it does not execute. It used to run each envelope inline, which made this box the
# one place the whole pipeline serialises: the brain fans b-roll ranking out ≤6 rank chains wide and awaits
# them by corr_id (op_backend deliberately dropped its per-jid lane lock for exactly this), and every one of
# them then queued behind the previous chain here. Ops go to a pool bounded by OPS_MAX_CHAINS, whose real
# budget is the runner's box-wide step semaphore; render/infer keep a pool of ONE, so the GPU and the weight
# caches stay single-threaded as before while a long render no longer blocks the CLAIM of an ops chain.
_OPS_MAX_CHAINS_ENV = "OPS_MAX_CHAINS"
_OPS_MAX_CHAINS_DEFAULT = 8

BOOT_T0 = time.monotonic()


def ops_chain_pool_size() -> int:
    """How many ops chains may be IN FLIGHT. Not a work budget — the runner's step slots are that; this only
    bounds how many workspaces (tmpdirs, in-flight fetches) exist at once."""
    try:
        return max(1, int(os.environ.get(_OPS_MAX_CHAINS_ENV, "") or _OPS_MAX_CHAINS_DEFAULT))
    except ValueError:
        return _OPS_MAX_CHAINS_DEFAULT


def _log(msg: str) -> None:
    print(f"[podagent] {msg}", file=sys.stderr, flush=True)


def _env_or_exit(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        _log(f"missing required environment variable {name}")
        sys.exit(2)
    return val


def _setup_vulkan_icd() -> None:
    # driver's default ICD points at libGLX_nvidia (X11 front); headless pod has no X11 → loader finds no driver
    # → libplacebo SILENTLY CPU-falls-back. libEGL_nvidia is the headless ICD; ffmpeg children inherit the env.
    import subprocess
    if os.environ.get("VK_ICD_FILENAMES"):
        return
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    lib = next((ln.split()[-1] for ln in out.splitlines() if "libEGL_nvidia.so.0" in ln), None)
    if not lib:
        _log("WARNING no libEGL_nvidia.so.0 — Vulkan/libplacebo will CPU-fall-back (slow, 0% GPU)")
        return
    icd = "/tmp/nvidia_egl_icd.json"
    Path(icd).write_text(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"%s","api_version":"1.4.0"}}\n' % lib)
    os.environ["VK_ICD_FILENAMES"] = icd
    _log(f"Vulkan ICD → {lib}")


def _log_gpu_status() -> None:
    # LOUD at boot: we rent a GPU to compute on it, not to crawl on CPU. Surface the torch arch so a host our
    # torch can't run shows immediately, not as a mystery-slow job.
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            _log(f"GPU {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} torch={torch.__version__}")
        else:
            _log("WARNING torch sees NO CUDA device — align will run on CPU (slow)")
    except Exception as e:  # noqa: BLE001 — a diagnostic must never block boot
        _log(f"WARNING GPU status check failed: {e}")


def _run_infer(
    raw: dict[str, Any],
    cp: ControlPlane,
    align_cache: dict[str, "AlignService"],
    probe_cache: dict[tuple[Path, str], "ProbeService"],
    rank_cache: dict[str, "ClipRankService"],
    yunet_path: Path,
    boot_reported: bool,
    corr_id: str | None = None,
) -> bool:
    """Runs one infer job, reports the result, and returns the updated boot_reported flag.

    corr_id is the claimed envelope's correlation id, echoed back on the posted result (pool demux)."""
    job_id = raw.get("job_id", "unknown")
    kind = raw.get("kind") if raw.get("kind") in INFER_KINDS else "align"
    # The brain awaits by result_key, and a status=error result is forbidden to carry one — so keep it here
    # to route the last-resort event, which is otherwise unclaimable and starves the awaiter anyway.
    wake_key = urlparse(str(raw.get("put_url", ""))).path.lstrip("/") or None

    def note(step: str) -> None:
        # No job_id: an infer request's job_id is a per-request id, NOT the token's lane, so sending it made
        # the CP's spoof guard 403 every progress event. Omitted → attributed to the token's own job.
        ev: dict[str, Any] = {"stage": "infer", "status": "step", "step": f"{job_id}: {step}"}
        cp.note(_tag(ev, corr_id, None))

    note(f"claimed {kind}")
    try:
        req = InferRequest.model_validate(raw)
        # Services are cached by the weights CONTENT hash, not by the model name: two requests naming the
        # same model but carrying different checkpoints must not share a loaded model, and a warm pod that
        # sees the same hash again skips both the fetch and the (dominant) load.
        if req.kind == "align":
            from .infer_align import AlignService
            from .weights import ensure

            assert req.align is not None and req.weights is not None
            align_svc = align_cache.get(req.weights.sha256)
            if align_svc is None:
                wdir = ensure(req.weights, req.model, note)
                note(f"loading {req.model}")
                align_svc = align_cache[req.weights.sha256] = AlignService(req.model, wdir)
            infer_s = align_svc.run(req.align, req.put_url, note)
        elif req.kind == "clip_rank":
            from .infer_cliprank import ClipRankService
            from .weights import ensure

            assert req.clip_rank is not None and req.weights is not None
            rank_svc = rank_cache.get(req.weights.sha256)
            if rank_svc is None:
                wdir = ensure(req.weights, req.model, note)
                note(f"loading {req.model}")
                rank_svc = rank_cache[req.weights.sha256] = ClipRankService(req.model, wdir)
            infer_s = rank_svc.run(req.clip_rank, req.put_url, note)
        else:
            from .infer_probe import ProbeService

            assert req.face_probe is not None
            key = (yunet_path, req.model)
            probe_svc = probe_cache.get(key)
            if probe_svc is None:
                probe_svc = probe_cache[key] = ProbeService(yunet_path, req.model)
            infer_s = probe_svc.run(req.face_probe, req.put_url)

        boot_s = None if boot_reported else time.monotonic() - BOOT_T0
        result = InferResult(
            infer_version=req.infer_version,
            job_id=req.job_id,
            kind=req.kind,
            status="ok",
            result_key=urlparse(req.put_url).path.lstrip("/"),
            corr_id=corr_id,
            timing=InferTiming(infer_s=infer_s, boot_s=boot_s),
        )
        cp.report_infer_result(result.model_dump(exclude_none=True))
        return True
    # BaseException, not Exception: a CUDA abort, a SystemExit or an interrupted download must still leave a
    # terminal behind — an unreported one strands the brain for INFER_TIMEOUT_S and reads as a dead host.
    except BaseException as e:
        _log(f"infer job {job_id} failed: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        error_result = InferResult(
            infer_version=SPEC_VERSION,
            job_id=str(job_id),
            kind=kind,
            status="error",
            corr_id=corr_id,
            error=f"{type(e).__name__}: {e}"[:500],
        )
        cp.report_infer_result(error_result.model_dump(exclude_none=True), wake_key)
        if not isinstance(e, Exception):
            raise
        return boot_reported


def _run_render(raw: dict[str, Any], cp: ControlPlane, corr_id: str | None = None,
                session_id: str | None = None) -> None:
    job_id = raw.get("job_id", "unknown")
    try:
        spec = RenderSpec.model_validate(raw)
        from .render import render_spec  # posts its own events; heavy deps stay out until a render job lands

        render_spec(spec, cp, corr_id=corr_id, session_id=session_id)
    except Exception as e:
        ev = {"job_id": job_id, "stage": "render", "status": "error", "error": str(e)[:500]}
        _tag(ev, corr_id, session_id)
        cp.post_event(ev)


def _tag(ev: dict[str, Any], corr_id: str | None, session_id: str | None) -> dict[str, Any]:
    """Stamp pool correlation onto an event/terminal — echoed from the claimed envelope, dropped when absent."""
    if corr_id is not None:
        ev["corr_id"] = corr_id
    if session_id is not None:
        ev["session_id"] = session_id
    return ev


def _run_ops(chain: Any, cp: ControlPlane, corr_id: str | None = None,
             session_id: str | None = None) -> None:
    """Run an op chain. There is NO per-op branch here and there must never be one: the op names its
    handler in contracts/ops/<op>.json, the pack provides it, and dispatch is a registry LOOKUP. Adding a
    tool costs a declaration and a handler — this file is not one of the files that changes.
    tests/test_ops_dispatch_is_a_lookup.py keeps it that way."""
    from .ops.runner import run_chain

    try:
        run_chain(chain, cp, corr_id=corr_id, session_id=session_id)
    except Exception as e:
        ev = {"job_id": chain.job_id, "stage": "ops", "status": "error", "error": str(e)[:500]}
        _tag(ev, corr_id, session_id)
        cp.post_event(ev)


def _report_boot(cp: ControlPlane) -> None:
    """One event before the first poll. A keyless pod that cannot reach the CP has NO other voice: the box
    boots, bills and stays silent, which reads exactly like a dead host. This beacon turns that into a
    fact on the wire — its presence proves the pod reached us, its absence indicts the network."""
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda"
    except Exception:  # noqa: BLE001 — the beacon must never be what kills a boot
        gpu = "no-torch"
    cp.note({"stage": "boot", "status": "step",
             "step": f"agent up · gpu={gpu} · {time.monotonic() - BOOT_T0:.1f}s"})


def _guarded(fn: Any, cp: ControlPlane) -> None:
    """Run a dispatched envelope, reporting instead of dying. Off the claim loop the old outer `except` no
    longer covers these, and a worker thread that raises is a job the brain waits out in silence."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — a worker must never take the agent down
        _log(f"dispatch failed: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        cp.note({"stage": "dispatch", "status": "error", "error": f"{type(e).__name__}: {e}"[:500]})


def main() -> None:
    cp_url = _env_or_exit("CP_URL")
    job_token = _env_or_exit("JOB_TOKEN")
    cp = ControlPlane(cp_url, job_token)
    _setup_vulkan_icd()   # before any ffmpeg child so libplacebo/the motion filters run on GPU, not a CPU crawl
    _log_gpu_status()
    _report_boot(cp)

    yunet_path = Path(os.environ.get("MODEL_YUNET", "/opt/models/yunet.onnx"))
    align_cache: dict[str, "AlignService"] = {}
    probe_cache: dict[tuple[Path, str], "ProbeService"] = {}
    rank_cache: dict[str, "ClipRankService"] = {}
    boot: list[bool] = [False]   # a 1-wide pool owns this, so the flag stays effectively single-threaded

    def _heavy(pod_job: PodJob) -> None:
        if pod_job.type == "infer":
            assert pod_job.request is not None
            request_raw = pod_job.request.model_dump(by_alias=True, mode="json")
            boot[0] = _run_infer(request_raw, cp, align_cache, probe_cache, rank_cache,
                                 yunet_path, boot[0], corr_id=pod_job.corr_id)
        else:
            assert pod_job.spec is not None
            spec_raw = pod_job.spec.model_dump(by_alias=True, mode="json")
            _run_render(spec_raw, cp, corr_id=pod_job.corr_id, session_id=pod_job.session_id)

    with cf.ThreadPoolExecutor(max_workers=ops_chain_pool_size(), thread_name_prefix="ops") as ops_pool, \
            cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu") as heavy_pool:
        _dispatch_loop(cp, ops_pool, heavy_pool, _heavy)


def _dispatch_loop(cp: ControlPlane, ops_pool: Any, heavy_pool: Any, heavy: Any,
                   once: bool = False) -> None:
    """Claim envelopes and hand them to a pool. `once` is the test seam — the production loop never returns."""
    while True:
        try:
            t_poll = time.monotonic()
            job = cp.poll_job()
            if job is None:
                idle = time.monotonic() - t_poll
                if idle < _MIN_POLL_INTERVAL_S:
                    time.sleep(_MIN_POLL_INTERVAL_S - idle)
                if once:
                    return
                continue

            try:
                pod_job = PodJob.model_validate(job)
            except ValidationError as e:
                cp.post_event({"stage": "dispatch", "status": "error", "error": str(e)[:500]})
                if once:
                    return
                continue

            if pod_job.type == "ops":
                assert pod_job.chain is not None
                chain, corr, sid = pod_job.chain, pod_job.corr_id, pod_job.session_id
                ops_pool.submit(_guarded,
                                lambda c=chain, r=corr, s=sid: _run_ops(c, cp, corr_id=r, session_id=s), cp)
            else:
                heavy_pool.submit(_guarded, lambda j=pod_job: heavy(j), cp)
            if once:
                return
        except requests.RequestException as e:
            _log(f"control-plane request failed: {e}")
            time.sleep(5)
        # Anything else used to unwind main() and end the process with the claimed job unreported: the pod
        # stayed rented, the brain waited out its timeout, and nothing on the wire said why.
        except Exception as e:
            _log(f"dispatch failed: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            cp.note({"stage": "dispatch", "status": "error", "error": f"{type(e).__name__}: {e}"[:500]})
            time.sleep(5)
        if once:
            return


if __name__ == "__main__":
    main()
