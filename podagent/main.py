"""Entrypoint. The agent's whole runtime contract is two environment variables (CP_URL,
JOB_TOKEN) — everything else this process does arrives as data from the control plane.
Run: python -m podagent.main"""
from __future__ import annotations

import concurrent.futures as cf
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests
from pydantic import ValidationError

from .cp import ControlPlane
from .event_stream import FrameRejected
from .models import SPEC_VERSION, InferRequest, InferResult, InferTiming, PodJob, RenderSpec

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

# Two pools now run infer and both can miss the SAME weights hash at the same instant — a second 4.6 GB
# SigLIP is an OOM, not a cache miss.
_SVC_LOAD_LOCK = threading.Lock()

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
    started = time.monotonic()
    _lifecycle(cp, phase="started", timings={"pod_queue_s": started - queued_at}, **meta)
    try:
        fn()
    except BaseException as e:
        _lifecycle(
            cp,
            phase="work_finished",
            timings={"work_s": time.monotonic() - started},
            outcome="error",
            error_type=type(e).__name__,
            error=str(e)[:500],
            **meta,
        )
        raise


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
        detail = f"exit {r.returncode}: {(r.stderr or b'')[-600:].decode('utf-8', 'replace').strip()}"
    # Refused HERE or discovered 157 s into ingest, with b-roll already paid for on the same run.
    msg = f"REFUSING work: h264_nvenc will not open on this pod — {detail}"
    _log(msg)
    try:
        cp.note({"stage": "boot", "status": "error", "phase": "work_finished", "step": msg})
    except Exception:  # noqa: BLE001 — the refusal stands whether or not the beacon lands
        pass
    sys.exit(3)


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
) -> bool:
    """Runs one infer job, reports the result, and returns the updated boot_reported flag.

    corr_id is the claimed envelope's correlation id, echoed back on the posted result (pool demux)."""
    job_id = raw.get("job_id", "unknown")
    kind = raw.get("kind") if raw.get("kind") in INFER_KINDS else "align"
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
                # Lanes race here; the SECOND one must wait for the load, not start its own.
                with _SVC_LOAD_LOCK:
                    rank_svc = rank_cache.get(req.weights.sha256)
                    if rank_svc is None:
                        wdir = ensure(req.weights, req.model, note)
                        note(f"loading {req.model}")
                        rank_svc = rank_cache[req.weights.sha256] = ClipRankService(req.model, wdir)
            rank_run = rank_svc.run(req.clip_rank, req.put_url, note)
            infer_s = rank_run.infer_s
            work_timings = rank_run.timings
        else:
            from .infer_probe import ProbeService

            assert req.face_probe is not None
            key = (yunet_path, req.model)
            probe_svc = probe_cache.get(key)
            if probe_svc is None:
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
        if isinstance(e, FrameRejected):
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
            error=str(e)[:500],
        )
        error_result = InferResult(
            infer_version=SPEC_VERSION,
            job_id=str(job_id),
            kind=kind,
            status="error",
            corr_id=corr_id,
            error=f"{type(e).__name__}: {e}"[:500],
        )
        cp.report_infer_result(error_result.model_dump(exclude_none=True))
        if not isinstance(e, Exception):
            raise
        return boot_reported


def _run_render(raw: dict[str, Any], cp: ControlPlane, corr_id: str, session_id: str) -> None:
    job_id = raw.get("job_id", "unknown")
    try:
        spec = RenderSpec.model_validate(raw)
        from .render import render_spec  # posts its own events; heavy deps stay out until a render job lands

        render_spec(spec, cp, corr_id=corr_id, session_id=session_id)
    except Exception as e:
        if isinstance(e, FrameRejected):
            raise
        ev = {"job_id": job_id, "stage": "render", "status": "error", "error": str(e)[:500]}
        _tag(ev, corr_id, session_id)
        ev.update({"phase": "work_finished", "outcome": "error", "error_type": type(e).__name__})
        cp.send_event(ev)
        cp.send_result({
            "job_id": job_id,
            "stage": "render",
            "status": "error",
            "corr_id": corr_id,
            "error": str(e)[:500],
            **({"session_id": session_id} if session_id is not None else {}),
        })


def _tag(ev: dict[str, Any], corr_id: str | None, session_id: str | None) -> dict[str, Any]:
    """Stamp pool correlation onto an event/terminal — echoed from the claimed envelope, dropped when absent."""
    if corr_id is not None:
        ev["corr_id"] = corr_id
    if session_id is not None:
        ev["session_id"] = session_id
    return ev


def _run_ops(chain: Any, cp: ControlPlane, corr_id: str, session_id: str) -> None:
    """Run an op chain. There is NO per-op branch here and there must never be one: the op names its
    handler in contracts/ops/<op>.json, the pack provides it, and dispatch is a registry LOOKUP. Adding a
    tool costs a declaration and a handler — this file is not one of the files that changes.
    tests/test_ops_dispatch_is_a_lookup.py keeps it that way."""
    from .ops.runner import run_chain

    try:
        run_chain(chain, cp, corr_id=corr_id, session_id=session_id)
    except Exception as e:
        if isinstance(e, FrameRejected):
            raise
        ev = {"job_id": chain.job_id, "stage": "ops", "status": "error", "error": str(e)[:500]}
        _tag(ev, corr_id, session_id)
        ev.update({"phase": "work_finished", "outcome": "error", "error_type": type(e).__name__})
        cp.send_event(ev)
        cp.send_result({
            "job_id": chain.job_id,
            "stage": "ops",
            "status": "error",
            "corr_id": corr_id,
            "error": str(e)[:500],
            **({"session_id": session_id} if session_id is not None else {}),
        })


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
    try:
        died = _LIVE_MARK.read_text().strip()
    except OSError:
        return f"prev=none (first boot in this container) · oom_kills={oom}"
    return f"prev=UNCLEAN (mark from {died} survived — the last agent was taken, not stopped) · oom_kills={oom}"


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


def _guarded(fn: Any, cp: ControlPlane) -> None:
    """Run a dispatched envelope, reporting instead of dying. Off the claim loop the old outer `except` no
    longer covers these, and a worker thread that raises is a job the brain waits out in silence."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — a worker must never take the agent down
        _log(f"dispatch failed: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)
        cp.note({"stage": "dispatch", "status": "error", "phase": "work_finished",
                 "error": f"{type(e).__name__}: {e}"[:500]})


def main() -> None:
    cp_url = _env_or_exit("CP_URL")
    job_token = _env_or_exit("JOB_TOKEN")
    cp = ControlPlane(cp_url, job_token)
    # A STOP IS NOT A DEATH, and the next boot may not report it as one (POST_MORTEM_WHY). The teardown
    # sends SIGTERM; anything that does NOT reach here — SIGKILL, a segv, an OOM — leaves the mark standing,
    # which is exactly the signal.
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _stop_and_exit)
    _setup_vulkan_icd()   # before any ffmpeg child so libplacebo/the motion filters run on GPU, not a CPU crawl
    _log_gpu_status()
    _report_boot(cp)
    _nvenc_or_refuse(cp)   # after the beacon, so a refusal reaches the box instead of dying mute

    yunet_path = Path(os.environ.get("MODEL_YUNET", "/opt/models/yunet.onnx"))
    align_cache: dict[str, "AlignService"] = {}
    probe_cache: dict[tuple[Path, str], "ProbeService"] = {}
    rank_cache: dict[str, "ClipRankService"] = {}
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
                              session_id=pod_job.session_id) and mine:
                _claim_boot(taken=False)
        else:
            assert pod_job.spec is not None
            spec_raw = pod_job.spec.model_dump(by_alias=True, mode="json")
            _run_render(spec_raw, cp, corr_id=pod_job.corr_id, session_id=pod_job.session_id)

    from .infer_cliprank import fetch_width, lane_width

    rank_width = lane_width()
    cp.note({
        "stage": "infer",
        "status": "step",
        "phase": "capacity",
        "op": "clip_rank",
        "capacity": {
            "rank_lanes": rank_width,
            "fetch_workers": fetch_width(),
        },
    })
    with cf.ThreadPoolExecutor(max_workers=ops_chain_pool_size(), thread_name_prefix="ops") as ops_pool, \
            cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu") as heavy_pool, \
            cf.ThreadPoolExecutor(max_workers=rank_width, thread_name_prefix="rank") as rank_pool:
        _dispatch_loop(cp, ops_pool, heavy_pool, rank_pool, _heavy)


def _is_clip_rank(pod_job: PodJob) -> bool:
    """Does this envelope belong on the rank lane? Kind, never type — an `align` is weights on the card."""
    return pod_job.type == "infer" and pod_job.request is not None and pod_job.request.kind == "clip_rank"


def _dispatch_loop(cp: ControlPlane, ops_pool: Any, heavy_pool: Any, rank_pool: Any, heavy: Any,
                   once: bool = False) -> None:
    """Claim envelopes and hand them to a pool. `once` is the test seam — the production loop never returns."""
    while True:
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
                cp.send_event({"stage": "dispatch", "status": "error", "phase": "work_finished",
                               "error": str(e)[:500]})
                if once:
                    return
                continue

            queued_at = time.monotonic()
            _lifecycle(cp, phase="queued", timings={"dispatch_s": queued_at - received_at}, **raw_meta)
            if pod_job.type == "ops":
                assert pod_job.chain is not None
                chain, corr, sid = pod_job.chain, pod_job.corr_id, pod_job.session_id
                ops_pool.submit(_guarded,
                                lambda c=chain, r=corr, s=sid, m=raw_meta, q=queued_at:
                                _with_lifecycle(
                                    cp, m, q, lambda: _run_ops(c, cp, corr_id=r, session_id=s)), cp)
            else:
                pool = rank_pool if _is_clip_rank(pod_job) else heavy_pool
                pool.submit(_guarded,
                            lambda j=pod_job, m=raw_meta, q=queued_at:
                            _with_lifecycle(cp, m, q, lambda: heavy(j)), cp)
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
            cp.note({"stage": "dispatch", "status": "error", "phase": "work_finished",
                     "error": f"{type(e).__name__}: {e}"[:500]})
            time.sleep(5)
        if once:
            return


if __name__ == "__main__":
    main()
