"""Entrypoint. The agent's whole runtime contract is two environment variables (CP_URL,
JOB_TOKEN) — everything else this process does arrives as data from the control plane.
Run: python -m podagent.main"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import json
import os
import shutil
import signal
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from pydantic import ValidationError

from .cp import ControlPlane
from .event_stream import DeliveryPending, TransportUnhealthy
from .models import SPEC_VERSION, InferRequest, InferResult, InferTiming, PodJob, RenderSpec
from .sanitize import safe_error, safe_text, safe_traceback

if TYPE_CHECKING:
    from .infer_align import AlignService
    from .infer_cliprank import ClipRankService
    from .infer_probe import ProbeService

INFER_KINDS = ("align", "face_probe", "clip_rank")

# THERE IS NO POLL ANY MORE. Work arrives on the socket the pod already holds (podagent.event_stream); the
# floor below survives it for one reason only — a socket that will NOT open returns immediately, and a bare
# `continue` around that would spin a rented box at an unbounded rate against a control plane that is down.
# On a healthy connection nothing ever waits here: the claim itself blocks on the wire.
_MIN_POLL_INTERVAL_S = 1.0

# The claim loop DISPATCHES; it does not execute. It used to run each envelope inline, which made this box the
# one place the whole pipeline serialises: the brain fans b-roll ranking out ≤6 rank chains wide and awaits
# them by corr_id (op_backend deliberately dropped its per-jid lane lock for exactly this), and every one of
# them then queued behind the previous chain here. Ops go to a pool bounded by OPS_MAX_CHAINS, whose real
# budget is the runner's box-wide step semaphore; clip_rank gets its OWN card-sized lane (~80% network, and
# it queued behind renders — infer_cliprank.LANE_SIZING_WHY); align/face_probe/render keep the pool of ONE.
_OPS_MAX_CHAINS_ENV = "OPS_MAX_CHAINS"
_OPS_MAX_CHAINS_DEFAULT = 8
_OPS_CLAIM_MAX = 4

# Twin of podagent.render.VULKAN_PROBE and scripts/montyops/camera_apply.py.VULKAN_PROBE; pin all three
# copies in the superproject parity test.
VULKAN_PROBE = (
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-init_hw_device", "vulkan",
    "-f", "lavfi", "-i", "testsrc=duration=0.1:size=64x64:rate=10",
    "-vf", "format=yuv420p,hwupload,libplacebo=w=32:h=32,hwdownload,format=yuv420p",
    "-c:v", "h264_nvenc", "-f", "null", "-",
)

# Two pools now run infer and both can miss the SAME weights hash at the same instant — a second 4.6 GB
# SigLIP is an OOM, not a cache miss.
_SVC_LOAD_LOCK = threading.Lock()

BOOT_T0 = time.monotonic()

# Minted once per process START, never per request: execv() replaces the process image and re-imports this
# module, which is what lets a receipt reader tell "still the incarnation that warmed" from "restarted since".
BOOT_ID = uuid.uuid4().hex


def ops_chain_pool_size() -> int:
    """How many ops chains may be IN FLIGHT. Not a work budget — the runner's step slots are that; this only
    bounds how many workspaces (tmpdirs, in-flight fetches) exist at once."""
    try:
        return max(1, int(os.environ.get(_OPS_MAX_CHAINS_ENV, "") or _OPS_MAX_CHAINS_DEFAULT))
    except ValueError:
        return _OPS_MAX_CHAINS_DEFAULT


CAPACITY_VRAM_WHY = """
CLAIM_CAPACITY IS BOUNDED BY CLASS, NOT BY OPS_MAX_CHAINS ALONE: the API admits ops/rank/heavy separately so
the ops pool can hold several workspaces without multiplying serial render/align work or the card-sized rank
lane. Four is the transport/OOM ceiling even when OPS_MAX_CHAINS is larger; the runner's step semaphore stays
the lower-level execution bound.

VRAM_TOTAL_MB IS A FIXED FACT ABOUT THE RENTED CARD, read the same way infer_cliprank._free_vram_mb reads
free VRAM — nvidia-smi, never a second device-reading tool.

VRAM_PEAK_USED_MB IS DELIBERATELY ABSENT. nvidia-smi's memory.used is an instantaneous snapshot — "allocated
by active contexts" right now — and this whole declaration runs ONCE, at boot, before any model has loaded.
Publishing that snapshot under a name that promises a peak would be the exact "0.0 reads like somebody
looked" failure rent_receipt.py exists to catch: a genuine peak needs a source that TRACKS one over the pod's
life, and nvidia-smi has no such query. So: a real total, and a named absence rather than a mislabeled instant.
"""


def capacity_payload(*, rank_lanes: int, fetch_workers: int,
                     vram_total_mb: float | None = None,
                     vulkan: bool = False) -> dict[str, Any]:
    """Advertise diagnostics, bounded per-class credits, and the card's VRAM total when known (CAPACITY_VRAM_WHY)."""
    payload: dict[str, Any] = {
        "rank_lanes": int(rank_lanes),
        "fetch_workers": int(fetch_workers),
        "claim_capacity": {
            "ops": min(ops_chain_pool_size(), _OPS_CLAIM_MAX),
            "rank": 1,
            "heavy": 1,
        },
        "boot_id": BOOT_ID,
        "vulkan": bool(vulkan),
    }
    if vram_total_mb is not None:
        payload["vram_total_mb"] = float(vram_total_mb)
    return payload


def _log(msg: str) -> None:
    print(f"[podagent] {safe_text(msg)}", file=sys.stderr, flush=True)


def _lifecycle(cp: "ControlPlane", *, phase: str, job_id: str = "unknown",
               session_id: str | None = None, corr_id: str | None = None,
               stage: str = "dispatch", op: str | None = None,
               timings: dict[str, float] | None = None,
               **extra: Any) -> None:
    """One structured lifecycle event. Progress is durable-before-return but never waits on its ACK."""
    ev: dict[str, Any] = {
        "job_id": job_id,
        "stage": stage,
        "status": "step",
        "phase": phase,
    }
    if session_id:
        ev["session_id"] = session_id
    if corr_id:
        ev["corr_id"] = corr_id
    if op:
        ev["op"] = op
    if timings:
        ev["timings"] = {k: round(float(v), 3) for k, v in timings.items()}
    ev.update({k: v for k, v in extra.items() if v is not None})
    cp.note(ev)


def _raw_job_meta(raw: dict[str, Any]) -> dict[str, str | None]:
    block = raw.get("request") or raw.get("spec") or raw.get("chain") or {}
    kind = block.get("kind") if isinstance(block, dict) else None
    return {
        "job_id": str(block.get("job_id") or "unknown") if isinstance(block, dict) else "unknown",
        "session_id": str(raw.get("session_id")) if raw.get("session_id") else None,
        "corr_id": str(raw.get("corr_id")) if raw.get("corr_id") else None,
        "stage": str(raw.get("type") or "dispatch"),
        "op": f"{raw.get('type')}:{kind}" if kind else str(raw.get("type") or "unknown"),
    }


def _with_lifecycle(cp: "ControlPlane", meta: dict[str, str | None],
                    queued_at: float, fn: Any) -> None:
    """Own admission/queue lifecycle only; the stage runner owns its single work_finished terminal."""
    started = time.monotonic()
    _lifecycle(cp, phase="started", timings={"pod_queue_s": started - queued_at}, **meta)
    fn()


def _dispatch_validation_terminal(cp: "ControlPlane", meta: dict[str, str | None], error: BaseException) -> None:
    """Release a claimed credit when envelope validation fails before PodJob exists.

    The raw envelope is the only authority available on this path, so preserve
    its corr/session/product job ids in both event and typed terminal. If one
    required address is absent, a result cannot be routed safely; the event
    still names the defect loudly instead of fabricating a corr-less terminal.
    """
    detail = safe_error(error)
    event = {
        "stage": "dispatch",
        "status": "error",
        "phase": "work_finished",
        "outcome": "error",
        "error_type": type(error).__name__,
        "error": detail,
        "job_id": meta.get("job_id") or "unknown",
    }
    for field in ("session_id", "corr_id"):
        if meta.get(field):
            event[field] = meta[field]
    cp.send_event(event)
    if meta.get("corr_id") and meta.get("session_id"):
        cp.send_result({
            "job_id": meta.get("job_id") or "unknown",
            "stage": "dispatch",
            "status": "error",
            "corr_id": meta["corr_id"],
            "session_id": meta["session_id"],
            "error": detail,
        })
    else:
        _log("validation failure had no complete corr/session authority; typed terminal was not fabricated")


def _env_or_exit(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        _log(f"missing required environment variable {name}")
        sys.exit(2)
    return val


@contextlib.contextmanager
def _llm_correlation(corr_id: str | None):
    """Delegate one product corr to the keyless control-plane LLM proxy.

    The pod's ``JOB_TOKEN`` authenticates the physical worker, not a tenant.  The API therefore requires
    the claimed corr on worker LLM requests.  Keep the binding in ``scripts.llm``'s ContextVar so concurrent
    chains cannot race through a process-wide ``MONTY_CORR_ID`` environment variable.  The import is lazy:
    the public pod image may run an ops pack with no engine LLM module at all.
    """
    try:
        import llm  # type: ignore[import-not-found]
    except ImportError:
        yield
        return
    scope = getattr(llm, "correlation", None)
    if scope is None:
        yield
        return
    with scope(corr_id):
        yield


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
    _log_nvidia_runtime()
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            _log(f"GPU {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} torch={torch.__version__}")
        else:
            _log("WARNING torch sees NO CUDA device — align will run on CPU (slow)")
    except Exception as e:  # noqa: BLE001 — a diagnostic must never block boot
        _log(f"WARNING GPU status check failed: {e}")


def _log_nvidia_runtime() -> None:
    """Name the injected device/driver independently of Torch without making this diagnostic a new gate."""
    import subprocess
    probe = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    try:
        r = subprocess.run(probe, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        _log(f"WARNING nvidia-smi unavailable: {safe_error(e)}")
        return
    raw = r.stdout if r.returncode == 0 else (r.stderr or r.stdout)
    detail = _bounded_stderr(raw or b"", edge=240) or "no diagnostic output"
    if r.returncode == 0:
        _log(f"nvidia-smi OK · {detail}")
    else:
        _log(f"WARNING nvidia-smi failed exit {r.returncode}: {detail}")


def _bounded_stderr(stderr: bytes, *, edge: int = 500) -> str:
    """A secret-safe root-cause line plus tail, without turning an ffmpeg dump into an event payload."""
    text = safe_text(stderr.decode("utf-8", "replace").strip())
    if len(text) <= edge * 2:
        return text
    omitted = len(text) - edge * 2
    return f"{text[:edge].rstrip()}\n... {omitted} chars omitted ...\n{text[-edge:].lstrip()}"


def _nvenc_or_refuse(cp: "ControlPlane") -> None:
    """Encode ONE frame before claiming anything, and die loudly if the card will not take it. The driver the
    provider boots is not ours to choose, so a working encoder is a per-POD fact, not an image-build one."""
    import subprocess
    probe = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=c=black:s=256x256:d=0.1", "-c:v", "h264_nvenc", "-preset", "p5",
             "-frames:v", "1", "-f", "null", "-"]
    try:
        r = subprocess.run(probe, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        detail = f"{type(e).__name__}: {e}"
    else:
        if r.returncode == 0:
            _log("nvenc probe OK — h264_nvenc opens on this host")
            return
        detail = f"exit {r.returncode}: {_bounded_stderr(r.stderr or b'')}"
    # Refused HERE or discovered 157 s into ingest, with b-roll already paid for on the same run.
    msg = f"REFUSING work: h264_nvenc will not open on this pod — {detail}"
    _log(msg)
    try:
        cp.send_event(
            {"stage": "boot", "status": "error", "phase": "work_finished", "step": msg},
            wait=True,
        )
    except Exception as e:  # noqa: BLE001 — the capability verdict stands even if its report cannot land
        _log(f"NVENC refusal delivery failed: {safe_error(e)}")
    _mark_stopped()
    sys.exit(3)


def _vulkan_preflight(cp: "ControlPlane") -> bool:
    """Return Vulkan verdict; an absent GPU is a routable fact, not a boot refusal."""
    import subprocess
    # Twin of podagent.render.VULKAN_PROBE and scripts/montyops/camera_apply.py.VULKAN_PROBE.
    try:
        r = subprocess.run(VULKAN_PROBE, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        detail = f"{type(e).__name__}: {e}"
    else:
        if r.returncode == 0:
            _log("vulkan/libplacebo probe OK — motion path opens on this host")
            return True
        detail = f"exit {r.returncode}: {_bounded_stderr(r.stderr or b'')}"
    if summary := _vulkaninfo_summary():
        detail = f"{detail} · {summary}"
    warning = f"WARNING VULKAN UNAVAILABLE: {safe_text(detail)}"
    _log(warning)
    try:
        cp.send_event({"stage": "boot", "status": "step", "phase": "work_finished", "step": warning},
                      wait=True)
    except Exception as e:  # noqa: BLE001 — the capability verdict stands even if its report cannot land
        _log(f"Vulkan warning delivery failed: {safe_error(e)}")
    return False


def _vulkaninfo_summary() -> str:
    """Return one useful vulkaninfo line, if the image carries the diagnostic binary."""
    import subprocess
    exe = shutil.which("vulkaninfo")
    if not exe:
        return ""
    try:
        r = subprocess.run([exe, "--summary"], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    raw = r.stdout or r.stderr or b""
    lines = [safe_text(line.strip()) for line in raw.decode("utf-8", "replace").splitlines() if line.strip()]
    if not lines:
        return ""
    line = next((line for line in lines if any(k in line for k in ("deviceName", "GPU", "driverName"))), lines[-1])
    return f"vulkaninfo: {line}"


def _report_ready(cp: "ControlPlane", *, capacity: dict[str, Any] | None = None) -> None:
    """Open admission only after the capability verdict itself is durably acknowledged by the box."""
    event: dict[str, Any] = {
        "stage": "boot",
        "status": "step",
        "phase": "ready",
        "step": "capability preflight passed",
    }
    if capacity is not None:
        event["capacity"] = dict(capacity)
    accepted = cp.send_event(event, wait=True)
    if not accepted:
        # EventStream already latched DeliveryPending before returning False. Raising here keeps main from
        # reaching capacity or the dispatch loop even if a test double (or future transport) only returns it.
        raise DeliveryPending("boot readiness ACK remains ambiguous; refusing job admission")


def _run_infer(
    raw: dict[str, Any],
    cp: ControlPlane,
    align_cache: dict[str, "AlignService"],
    probe_cache: dict[tuple[Path, str], "ProbeService"],
    rank_cache: dict[str, "ClipRankService"],
    yunet_path: Path,
    boot_reported: bool,
    corr_id: str,
    session_id: str,
    rank_parallel: int = 1,
    rank_slots: threading.BoundedSemaphore | None = None,
) -> bool:
    """Runs one infer job, reports the result, and returns the updated boot_reported flag.

    corr_id is the claimed envelope's correlation id, echoed back on the posted result (pool demux)."""
    job_id = raw.get("job_id", "unknown")
    kind = raw.get("kind") if raw.get("kind") in INFER_KINDS else "align"

    @contextmanager
    def phase(name: str):
        phase_started = time.monotonic()
        _lifecycle(
            cp, phase=f"{name}_started", job_id=str(job_id), session_id=session_id,
            corr_id=corr_id, stage="infer", op=kind)
        try:
            yield
        except BaseException as exc:
            _lifecycle(
                cp, phase=f"{name}_error", job_id=str(job_id), session_id=session_id,
                corr_id=corr_id, stage="infer", op=kind, outcome="error",
                error_type=type(exc).__name__, error=safe_error(exc),
                timings={"phase_s": time.monotonic() - phase_started})
            raise
        else:
            _lifecycle(
                cp, phase=f"{name}_finished", job_id=str(job_id), session_id=session_id,
                corr_id=corr_id, stage="infer", op=kind, outcome="ok",
                timings={"phase_s": time.monotonic() - phase_started})

    def note(step: str) -> None:
        ev: dict[str, Any] = {
            "stage": "infer", "status": "step", "phase": "progress",
            "op": kind, "step": f"{job_id}: {step}",
            "job_id": str(job_id),
        }
        cp.note(_tag(ev, corr_id, session_id))

    note(f"claimed {kind}")
    work_started = time.monotonic()
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
                with phase("weights_fetch"):
                    wdir = ensure(req.weights, req.model, note)
                with phase("model_load"):
                    note(f"loading {req.model}")
                    align_svc = align_cache[req.weights.sha256] = AlignService(req.model, wdir)
            infer_s = align_svc.run(req.align, req.put_url, note)
        elif req.kind == "clip_rank":
            from .infer_cliprank import ClipRankService
            from .weights import ensure

            assert req.clip_rank is not None and req.weights is not None
            rank_svc = rank_cache.get(req.weights.sha256)
            if rank_svc is None:
                # Lanes race here; the SECOND one must wait for the load, not start its own.
                with _SVC_LOAD_LOCK:
                    rank_svc = rank_cache.get(req.weights.sha256)
                    if rank_svc is None:
                        with phase("weights_fetch"):
                            wdir = ensure(req.weights, req.model, note)
                        with phase("model_load"):
                            note(f"loading {req.model}")
                            rank_svc = rank_cache[req.weights.sha256] = ClipRankService(
                                req.model, wdir, parallel=rank_parallel, slots=rank_slots)
            rank_run = rank_svc.run(req.clip_rank, req.put_url, note)
            infer_s = rank_run.infer_s
            work_timings = rank_run.timings
        else:
            from .infer_probe import ProbeService

            assert req.face_probe is not None
            key = (yunet_path, req.model)
            probe_svc = probe_cache.get(key)
            if probe_svc is None:
                with phase("model_load"):
                    probe_svc = probe_cache[key] = ProbeService(yunet_path, req.model)
            infer_s = probe_svc.run(req.face_probe, req.put_url)
        if req.kind != "clip_rank":
            work_timings = {
                "infer_s": infer_s,
                "work_s": time.monotonic() - work_started,
            }

        boot_s = None if boot_reported else time.monotonic() - BOOT_T0
        _lifecycle(
            cp,
            phase="work_finished",
            job_id=str(job_id),
            session_id=session_id,
            corr_id=corr_id,
            stage="infer",
            op=req.kind,
            timings=work_timings,
            outcome="ok",
        )
        result = InferResult(
            infer_version=req.infer_version,
            job_id=req.job_id,
            kind=req.kind,
            status="ok",
            result_key=corr_id,
            corr_id=corr_id,
            timing=InferTiming(infer_s=infer_s, boot_s=boot_s),
        )
        wire_result = result.model_dump(exclude_none=True)
        wire_result["session_id"] = session_id
        cp.report_infer_result(wire_result)
        return True
    # BaseException, not Exception: a CUDA abort, a SystemExit or an interrupted download must still leave a
    # terminal behind — an unreported one strands the brain for INFER_TIMEOUT_S and reads as a dead host.
    except BaseException as e:
        _log(f"infer job {job_id} failed: {safe_error(e)}")
        print(safe_traceback(e), file=sys.stderr, flush=True)
        if isinstance(e, TransportUnhealthy):
            raise
        _lifecycle(
            cp,
            phase="work_finished",
            job_id=str(job_id),
            session_id=session_id,
            corr_id=corr_id,
            stage="infer",
            op=kind,
            timings={"work_s": time.monotonic() - work_started},
            outcome="error",
            error_type=type(e).__name__,
            error=safe_error(e),
        )
        error_result = InferResult(
            infer_version=SPEC_VERSION,
            job_id=str(job_id),
            kind=kind,
            status="error",
            corr_id=corr_id,
            error=safe_error(e),
        )
        wire_error = error_result.model_dump(exclude_none=True)
        wire_error["session_id"] = session_id
        cp.report_infer_result(wire_error)
        if not isinstance(e, Exception):
            raise
        return boot_reported


def _run_render(raw: dict[str, Any], cp: ControlPlane, corr_id: str, session_id: str) -> None:
    job_id = raw.get("job_id", "unknown")
    try:
        spec = RenderSpec.model_validate(raw)
        from .render import render_spec  # posts its own events; heavy deps stay out until a render job lands

        render_spec(spec, cp, corr_id=corr_id, session_id=session_id)
    except BaseException as e:
        if isinstance(e, TransportUnhealthy):
            raise
        ev = {"job_id": job_id, "stage": "render", "status": "error", "error": safe_error(e)}
        _tag(ev, corr_id, session_id)
        ev.update({"phase": "work_finished", "outcome": "error", "error_type": type(e).__name__})
        cp.send_event(ev)
        cp.send_result({
            "job_id": job_id,
            "stage": "render",
            "status": "error",
            "corr_id": corr_id,
            "error": safe_error(e),
            **({"session_id": session_id} if session_id is not None else {}),
        })
        if not isinstance(e, Exception):
            raise


def _tag(ev: dict[str, Any], corr_id: str | None, session_id: str | None) -> dict[str, Any]:
    """Stamp pool correlation onto an event/terminal — echoed from the claimed envelope, dropped when absent."""
    if corr_id is not None:
        ev["corr_id"] = corr_id
    if session_id is not None:
        ev["session_id"] = session_id
    return ev


def _run_ops(chain: Any, cp: ControlPlane, corr_id: str, session_id: str,
             coordinator: "RestartCoordinator | None" = None) -> None:
    """Run an op chain. Dispatch is a registry LOOKUP, never a per-op branch (tests/test_ops_dispatch_is_a_lookup.py).
    A RESTART_REQUIRED return means the pack generation flipped; coordinator hears about it, no terminal is built."""
    from .ops.runner import RESTART_REQUIRED, run_chain

    try:
        with _llm_correlation(corr_id):
            result = run_chain(chain, cp, corr_id=corr_id, session_id=session_id)
        if result is RESTART_REQUIRED:
            reason = f"chain {chain.job_id} names ops-pack {chain.pack.sha256[:12]}, this process activated a different one"
            if coordinator is None:
                raise RuntimeError(f"pack generation change with no restart coordinator to tell: {reason}")
            coordinator.request_restart(reason, target_pack=chain.pack.sha256)
            _log(f"restart requested: {reason}")
        return
    except BaseException as e:
        if isinstance(e, TransportUnhealthy):
            raise
        ev = {"job_id": chain.job_id, "stage": "ops", "status": "error", "error": safe_error(e)}
        _tag(ev, corr_id, session_id)
        ev.update({"phase": "work_finished", "outcome": "error", "error_type": type(e).__name__})
        cp.send_event(ev)
        cp.send_result({
            "job_id": chain.job_id,
            "stage": "ops",
            "status": "error",
            "corr_id": corr_id,
            "error": safe_error(e),
            **({"session_id": session_id} if session_id is not None else {}),
        })
        if not isinstance(e, Exception):
            raise


POST_MORTEM_WHY = """
A SECOND BOOT IS A DEATH NOBODY WROTE DOWN.

MEASURED on a preview run: this agent stopped talking at run-offset 62.7 s and reported `agent up` again at
114.3 s — two boots on ONE rent, so the process died inside the container and the supervisor restarted it.
Fifty-one seconds of silence, twelve envelopes dropped by a control plane whose socket had gone, and a box
that convicted two CDNs for it. When the question "why did it die" was finally asked, the answer was already
gone: the container log dies with the pod, and by then the pod had been terminated to stop the billing.

A crash cannot report itself. SIGKILL — which is what an OOM is — runs no handler, and a process that segv's
has no voice at all. So the report is made by the NEXT incarnation, out of state that outlives a process but
not a container:

  · a liveness mark, written at boot and REMOVED on a clean exit. Finding it means the last agent did not
    leave, it was taken.
  · the cgroup's own `memory.events` oom_kill counter, which the kernel increments when it kills something in
    this container. It is the difference between "we were too big" and "we broke".

NEITHER IS ASSUMED WHEN UNREADABLE. A cgroup this kernel will not show is reported as unknown, never as zero:
"nobody was OOM-killed" and "we could not ask" are different findings, and only one of them exonerates the
memory budget.
"""

# NOT AN ENV KNOB. It was one for a minute, and `test_no_knob_the_pod_reads_is_written_by_nobody` was right
# to refuse it: nothing on the box would ever have set it, and a knob no writer turns is a knob whose only
# effect is to look configurable. The tests that need another path set THIS name.
_LIVE_MARK = Path("/tmp/podagent.alive")


_OOM_FILES = (Path("/sys/fs/cgroup/memory.events"),                 # cgroup v2
              Path("/sys/fs/cgroup/memory/memory.oom_control"))     # v1, older hosts

# A generation-flip execv() leaves _LIVE_MARK standing (same PID, no _mark_stopped call) — an env var, not
# a file, so a FAILED exec can never leave a stale "planned" signal for a later, unrelated process to read.
_PLANNED_RESTART_ENV = "PODAGENT_PLANNED_RESTART"


def _oom_from(text: str) -> str | None:
    """The `oom_kill` count in one cgroup file's body, or None if this file does not carry one. Pure, so the
    parse is testable without a kernel that happens to expose the right cgroup version."""
    for line in text.splitlines():
        key, _, val = line.partition(" ")
        if key == "oom_kill" and val.strip().isdigit():
            return val.strip()
    return None


def _oom_kills() -> str:
    """How many times the kernel OOM-killed something in THIS container, or `unknown` (POST_MORTEM_WHY)."""
    for p in _OOM_FILES:
        try:
            got = _oom_from(p.read_text())
        except OSError:
            continue
        if got is not None:
            return got
    return "unknown"


def _post_mortem() -> str:
    """What happened to the PREVIOUS agent in this container, as a phrase for the boot beacon."""
    oom = _oom_kills()
    planned = _consume_planned_restart_mark()
    if planned is not None:
        return f"prev=PLANNED-RESTART ({planned}) · oom_kills={oom}"
    try:
        died = _LIVE_MARK.read_text().strip()
    except OSError:
        return f"prev=none (first boot in this container) · oom_kills={oom}"
    return f"prev=UNCLEAN (mark from {died} survived — the last agent was taken, not stopped) · oom_kills={oom}"


def _consume_planned_restart_mark() -> str | None:
    """Read-then-clear THIS process's own environment — consumed on the boot that inherited it, so a
    later crash+respawn (fresh env, no execv) can never see a stale "planned" signal."""
    return os.environ.pop(_PLANNED_RESTART_ENV, None)


def _mark_alive() -> None:
    """Claim the liveness mark for this incarnation. Best effort: a mark we cannot write costs the NEXT
    post-mortem its evidence, and that is not a reason to refuse the boot the pod is already billing for."""
    try:
        _LIVE_MARK.parent.mkdir(parents=True, exist_ok=True)
        _LIVE_MARK.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except OSError as e:
        _log(f"⚠ could not write the liveness mark at {_LIVE_MARK} ({type(e).__name__}: {e}) — a crash of "
             f"THIS agent will be indistinguishable from a clean stop (POST_MORTEM_WHY)")


def _mark_stopped() -> None:
    """Drop the mark on a deliberate exit, so the next boot does not report a death that never happened."""
    try:
        _LIVE_MARK.unlink(missing_ok=True)
    except OSError as e:
        _log(f"⚠ could not clear the liveness mark ({type(e).__name__}: {e}) — the next boot will read this "
             f"clean stop as a crash")


def _mark_planned_restart(reason: str, abandoned: list[str]) -> None:
    """Set in THIS process's own environment, right before execv() — inherited by the next incarnation
    ONLY if the exec itself succeeds (os.environ, not a file: see _PLANNED_RESTART_ENV)."""
    body = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} · {safe_text(reason)}"
    if abandoned:
        body += f" · abandoned={abandoned}"
    os.environ[_PLANNED_RESTART_ENV] = body


def _stop_and_exit(signum: int, _frame: Any) -> None:
    """A signalled stop is deliberate: clear the mark, say which signal, and leave by the signal's own code."""
    _log(f"signal {signum} — stopping; the liveness mark is cleared so the next boot reads a STOP, not a death")
    _mark_stopped()
    sys.exit(128 + int(signum))


def _report_boot(cp: ControlPlane) -> None:
    """One event before the first poll. A keyless pod that cannot reach the CP has NO other voice: the box
    boots, bills and stays silent, which reads exactly like a dead host. This beacon turns that into a
    fact on the wire — its presence proves the pod reached us, its absence indicts the network.

    It also carries the PREVIOUS incarnation's post-mortem, because a second boot on one rent is the only
    place a crash that ran no handler can still be described (POST_MORTEM_WHY)."""
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda"
    except Exception:  # noqa: BLE001 — the beacon must never be what kills a boot
        gpu = "no-torch"
    post = _post_mortem()
    _mark_alive()
    cp.note({"stage": "boot", "status": "step", "phase": "started",
             "step": f"agent up · gpu={gpu} · {time.monotonic() - BOOT_T0:.1f}s · {post}"})


def _capability_preflight(cp: ControlPlane, *, capacity: dict[str, Any] | None = None) -> None:
    """Report the boot, prove the encoder, then make readiness an ACKed admission barrier."""
    _report_boot(cp)
    _nvenc_or_refuse(cp)
    if capacity is not None:
        capacity["vulkan"] = _vulkan_preflight(cp)
    else:
        _vulkan_preflight(cp)
    _report_ready(cp, capacity=capacity)


def _guarded(fn: Any, cp: ControlPlane, meta: dict[str, str | None] | None = None) -> None:
    """Run a dispatched envelope, reporting instead of dying. Off the claim loop the old outer `except` no
    longer covers these, and a worker thread that raises is a job the brain waits out in silence."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — a worker must never take the agent down
        _log(f"dispatch failed: {safe_error(e)}")
        print(safe_traceback(e), file=sys.stderr, flush=True)
        if isinstance(e, TransportUnhealthy):
            raise
        if meta is not None:
            _dispatch_validation_terminal(cp, meta, e)
        else:
            cp.note({"stage": "dispatch", "status": "error", "phase": "work_finished",
                     "error": safe_error(e)})


# DERIVED: above the largest known single-arm budget (montyops/patience.py ARM_S, cut_apply.py
# CUT_APPLY_SEG_S — both 900s), capped at the brain's own "ops" patience (pod_lane.DEFAULT_WORK_BUDGET_S=1800s).
_RESTART_DRAIN_S_ENV = "POD_RESTART_DRAIN_S"
_RESTART_DRAIN_S_DEFAULT = 1800.0

# A rate limit on the flip ITSELF, durable across execv: the drain-wall size above closes the common
# cause; 3 generations in one hour is the backstop reading as ping-pong, not convergence.
_FLIP_HISTORY_MARK = Path("/tmp/podagent.pack-flip-history.json")
_FLIP_MAX_PER_HOUR = 3
_FLIP_WINDOW_S = 3600.0


class RestartRefused(RuntimeError):
    """Too many ops-pack generation flips in too short a window — this is ping-pong, not convergence."""


def _restart_drain_s() -> float:
    try:
        v = float(os.environ.get(_RESTART_DRAIN_S_ENV, "") or _RESTART_DRAIN_S_DEFAULT)
    except ValueError:
        return _RESTART_DRAIN_S_DEFAULT
    return v if v > 0 else _RESTART_DRAIN_S_DEFAULT


def _flip_history(now: float) -> list[dict[str, Any]]:
    try:
        raw = json.loads(_FLIP_HISTORY_MARK.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict) and isinstance(row.get("t"), (int, float))
            and now - float(row["t"]) < _FLIP_WINDOW_S]


def _record_flip_or_refuse(target_pack: str, *, now: float) -> None:
    """Append this restart to the durable trailing-hour history, or raise RestartRefused instead."""
    history = _flip_history(now)
    if len(history) >= _FLIP_MAX_PER_HOUR:
        raise RestartRefused(
            f"{len(history)} ops-pack generation flip(s) in the last hour (max {_FLIP_MAX_PER_HOUR}) — "
            f"refusing another restart toward {target_pack[:12] or 'unknown'}")
    history.append({"t": now, "pack": target_pack[:12]})
    try:
        _FLIP_HISTORY_MARK.parent.mkdir(parents=True, exist_ok=True)
        _FLIP_HISTORY_MARK.write_text(json.dumps(history))
    except OSError as e:
        _log(f"⚠ could not persist the flip-history marker ({type(e).__name__}: {e}) — the flip-count "
             f"guard may undercount across this restart")


def _kill_orphan_children() -> None:
    """Never exec with live child processes: best effort, /proc-only. montyops.patience.run starts each op
    subprocess in its own session, so killing its GROUP takes any grandchildren it spawned too."""
    try:
        raw = Path(f"/proc/self/task/{os.getpid()}/children").read_text()
    except OSError:
        return
    for pid_s in raw.split():
        try:
            pid = int(pid_s)
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            continue


class RestartCoordinator:
    """One process-wide latch a pack-generation flip trips, plus the future registry the drain waits on."""

    _POOLS = ("ops", "heavy", "rank")

    def __init__(self) -> None:
        self._latch = threading.Event()
        self._lock = threading.Lock()
        self._reason = ""
        self._target_pack = ""
        # Keyed by the future's OWN identity — a label (corr_id) can repeat and a second future silently
        # overwriting the first's slot would drop it from `pending` early, not just at completion.
        self._futures: dict[str, dict[int, tuple[str, "cf.Future[Any]"]]] = {p: {} for p in self._POOLS}

    def request_restart(self, reason: str, *, target_pack: str = "") -> None:
        with self._lock:
            if not self._latch.is_set():
                self._reason = reason
                self._target_pack = target_pack
        self._latch.set()

    def restart_requested(self) -> bool:
        return self._latch.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def target_pack(self) -> str:
        with self._lock:
            return self._target_pack

    def track(self, pool: str, label: str, future: "cf.Future[Any]") -> None:
        """A future drops itself out on completion — a long-lived process must never outgrow this registry."""
        key = id(future)
        with self._lock:
            self._futures[pool][key] = (label, future)

        def _untrack(_f: "cf.Future[Any]", pool: str = pool, key: int = key) -> None:
            with self._lock:
                self._futures[pool].pop(key, None)

        future.add_done_callback(_untrack)

    def pending(self) -> dict[str, list[str]]:
        with self._lock:
            return {pool: sorted(label for label, _fut in entries.values())
                    for pool, entries in self._futures.items() if entries}


def _drain_and_restart(cp: ControlPlane, coordinator: RestartCoordinator, *,
                       drain_s: float | None = None, sleep: Any = time.sleep,
                       execv: Any = os.execv, now: Any = time.time,
                       kill_orphans: Any = None) -> None:
    """flip-guard → close_stream() → bounded drain → kill orphans → mark → execv(), inside main()'s `with`
    so it precedes the pools' unconditional wait=True shutdown. RestartRefused and a failed execv both
    propagate loud on purpose — neither is a restart."""
    reason = coordinator.reason
    target = coordinator.target_pack
    _record_flip_or_refuse(target, now=now())
    _log(f"ops-pack generation changed ({reason}) — closing admission and draining before restart")
    cp.close_stream()
    deadline = time.monotonic() + (drain_s if drain_s is not None else _restart_drain_s())
    abandoned: list[str] = []
    while True:
        pending = coordinator.pending()
        if not pending:
            break
        if time.monotonic() >= deadline:
            abandoned = [f"{pool}:{key}" for pool, keys in pending.items() for key in keys]
            _log(f"restart drain deadline expired with {len(abandoned)} chain(s) still in flight: {abandoned}")
            break
        sleep(0.1)
    (kill_orphans or _kill_orphan_children)()
    _mark_planned_restart(reason, abandoned)
    argv = [sys.executable, "-m", "podagent.main"]
    try:
        execv(sys.executable, argv)
    except OSError as e:
        os.environ.pop(_PLANNED_RESTART_ENV, None)
        _log(f"⚠ execv failed ({type(e).__name__}: {e}) — this incarnation cannot restart; exiting "
             f"immediately, not waiting on any executor")
        os._exit(1)


def main() -> None:
    cp_url = _env_or_exit("CP_URL")
    job_token = _env_or_exit("JOB_TOKEN")
    # Shared-fleet workers hold one physical bearer.  It is deliberately not copied into a product-session
    # field; the control plane resolves the product tenant from corr attribution for every LLM call.
    os.environ["MONTY_JOB_TOKEN"] = job_token
    cp = ControlPlane(cp_url, job_token)
    # A STOP IS NOT A DEATH, and the next boot may not report it as one (POST_MORTEM_WHY). The teardown
    # sends SIGTERM; anything that does NOT reach here — SIGKILL, a segv, an OOM — leaves the mark standing,
    # which is exactly the signal.
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _stop_and_exit)
    _setup_vulkan_icd()   # before any ffmpeg child so libplacebo/the motion filters run on GPU, not a CPU crawl
    _log_gpu_status()
    from .infer_cliprank import fetch_width, lane_width, vram_total_mb
    rank_width = lane_width()
    capacity = capacity_payload(rank_lanes=rank_width, fetch_workers=fetch_width(),
                                vram_total_mb=vram_total_mb())
    _capability_preflight(cp, capacity=capacity)
    # On reconnect the API resets credit. Replay the ACKed readiness verdict
    # together with capacity; this event is installed only after preflight.
    cp.set_bootstrap_event({
        "stage": "boot",
        "status": "step",
        "phase": "ready",
        "step": "capability preflight passed",
        "capacity": capacity,
    })

    yunet_path = Path(os.environ.get("MODEL_YUNET", "/opt/models/yunet.onnx"))
    align_cache: dict[str, "AlignService"] = {}
    probe_cache: dict[tuple[Path, str], "ProbeService"] = {}
    rank_cache: dict[str, "ClipRankService"] = {}
    rank_slots = threading.BoundedSemaphore(rank_width)
    boot: list[bool] = [False]
    boot_lock = threading.Lock()

    def _claim_boot(taken: bool = True) -> bool:
        # boot_s is BILLED once, so two lanes racing the first result may not both carry it — nor may an
        # envelope that ERRORED keep a report it never sent.
        with boot_lock:
            first, boot[0] = not boot[0], taken
            return first

    def _heavy(pod_job: PodJob) -> None:
        if pod_job.type == "infer":
            assert pod_job.request is not None
            request_raw = pod_job.request.model_dump(by_alias=True, mode="json")
            mine = _claim_boot()
            if not _run_infer(request_raw, cp, align_cache, probe_cache, rank_cache,
                              yunet_path, not mine, corr_id=pod_job.corr_id,
                              session_id=pod_job.session_id, rank_parallel=rank_width,
                              rank_slots=rank_slots) and mine:
                _claim_boot(taken=False)
        else:
            assert pod_job.spec is not None
            spec_raw = pod_job.spec.model_dump(by_alias=True, exclude_none=True, mode="json")
            _run_render(spec_raw, cp, corr_id=pod_job.corr_id, session_id=pod_job.session_id)

    coordinator = RestartCoordinator()
    with cf.ThreadPoolExecutor(max_workers=ops_chain_pool_size(), thread_name_prefix="ops") as ops_pool, \
            cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu") as heavy_pool, \
            cf.ThreadPoolExecutor(max_workers=rank_width, thread_name_prefix="rank") as rank_pool:
        _dispatch_loop(cp, ops_pool, heavy_pool, rank_pool, _heavy, coordinator=coordinator)
        if coordinator.restart_requested():
            # Runs INSIDE this `with` on purpose — see _drain_and_restart's docstring.
            _drain_and_restart(cp, coordinator)


def _is_clip_rank(pod_job: PodJob) -> bool:
    """Does this envelope belong on the rank lane? Kind, never type — an `align` is weights on the card."""
    return pod_job.type == "infer" and pod_job.request is not None and pod_job.request.kind == "clip_rank"


def _dispatch_loop(cp: ControlPlane, ops_pool: Any, heavy_pool: Any, rank_pool: Any, heavy: Any,
                   once: bool = False, coordinator: "RestartCoordinator | None" = None) -> None:
    """Claim envelopes and hand them to a pool. `once` is the test seam — the production loop never returns
    on its own; a set restart latch is the one other way out, checked before every new claim wait."""
    while True:
        if coordinator is not None and coordinator.restart_requested():
            return
        try:
            t_poll = time.monotonic()
            job = cp.poll_job()          # a wait on the live socket, not a request (cp.poll_job WHY)
            if job is None:
                idle = time.monotonic() - t_poll
                if idle < _MIN_POLL_INTERVAL_S:
                    time.sleep(_MIN_POLL_INTERVAL_S - idle)
                if once:
                    return
                continue

            received_at = time.monotonic()
            raw_meta = _raw_job_meta(job)
            _lifecycle(cp, phase="received", **raw_meta)
            try:
                pod_job = PodJob.model_validate(job)
            except ValidationError as e:
                _dispatch_validation_terminal(cp, raw_meta, e)
                if once:
                    return
                continue

            queued_at = time.monotonic()
            _lifecycle(cp, phase="queued", timings={"dispatch_s": queued_at - received_at}, **raw_meta)
            if pod_job.type == "ops":
                assert pod_job.chain is not None
                chain, corr, sid = pod_job.chain, pod_job.corr_id, pod_job.session_id
                ops_kwargs = {"coordinator": coordinator} if coordinator is not None else {}
                fut = ops_pool.submit(_guarded,
                                lambda c=chain, r=corr, s=sid, m=raw_meta, q=queued_at, k=ops_kwargs:
                                _with_lifecycle(
                                    cp, m, q, lambda: _run_ops(
                                        c, cp, corr_id=r, session_id=s, **k)), cp, raw_meta)
                if coordinator is not None:
                    coordinator.track("ops", str(corr or raw_meta.get("job_id") or "?"), fut)
            else:
                pool = rank_pool if _is_clip_rank(pod_job) else heavy_pool
                fut = pool.submit(_guarded,
                            lambda j=pod_job, m=raw_meta, q=queued_at:
                            _with_lifecycle(cp, m, q, lambda: heavy(j)), cp, raw_meta)
                if coordinator is not None:
                    pool_name = "rank" if pool is rank_pool else "heavy"
                    coordinator.track(pool_name, str(raw_meta.get("corr_id") or raw_meta.get("job_id") or "?"), fut)
            if once:
                return
        except requests.RequestException as e:
            _log(f"control-plane request failed: {safe_error(e)}")
            time.sleep(5)
        except TransportUnhealthy:
            raise
        # Anything else used to unwind main() and end the process with the claimed job unreported: the pod
        # stayed rented, the brain waited out its timeout, and nothing on the wire said why.
        except Exception as e:
            _log(f"dispatch failed: {safe_error(e)}")
            print(safe_traceback(e), file=sys.stderr, flush=True)
            cp.note({"stage": "dispatch", "status": "error", "phase": "work_finished",
                     "error": safe_error(e)})
            time.sleep(5)
        if once:
            return


if __name__ == "__main__":
    main()
