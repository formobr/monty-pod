"""Entrypoint. The agent's whole runtime contract is two environment variables (CP_URL,
JOB_TOKEN) — everything else this process does arrives as data from the control plane.
Run: python -m podagent.main"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests

from .cp import ControlPlane
from .models import InferRequest, InferResult, InferTiming, RenderSpec

if TYPE_CHECKING:
    from .infer_align import AlignService
    from .infer_probe import ProbeService

BOOT_T0 = time.monotonic()


def _log(msg: str) -> None:
    print(f"[podagent] {msg}", file=sys.stderr, flush=True)


def _env_or_exit(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        _log(f"missing required environment variable {name}")
        sys.exit(2)
    return val


def _run_infer(
    raw: dict[str, Any],
    cp: ControlPlane,
    align_cache: dict[str, "AlignService"],
    probe_cache: dict[tuple[Path, str], "ProbeService"],
    yunet_path: Path,
    boot_reported: bool,
) -> bool:
    """Runs one infer job, reports the result, and returns the updated boot_reported flag."""
    job_id = raw.get("job_id", "unknown")
    kind = raw.get("kind") if raw.get("kind") in ("align", "face_probe") else "align"
    try:
        req = InferRequest.model_validate(raw)
        if req.kind == "align":
            from .infer_align import AlignService

            assert req.align is not None
            align_svc = align_cache.get(req.model)
            if align_svc is None:
                align_svc = align_cache[req.model] = AlignService(req.model)
            infer_s = align_svc.run(req.align, req.put_url)
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
            timing=InferTiming(infer_s=infer_s, boot_s=boot_s),
        )
        cp.post_infer_result(result.model_dump(exclude_none=True))
        return True
    except Exception as e:
        _log(f"infer job {job_id} failed: {e}")
        error_result = InferResult(
            infer_version=1,
            job_id=str(job_id),
            kind=kind,
            status="error",
            error=str(e)[:500],
        )
        cp.post_infer_result(error_result.model_dump(exclude_none=True))
        return boot_reported


def _run_render(raw: dict[str, Any], cp: ControlPlane) -> None:
    job_id = raw.get("job_id", "unknown")
    try:
        spec = RenderSpec.model_validate(raw)
        from .render import render_spec  # posts its own events; heavy deps stay out until a render job lands

        render_spec(spec, cp)
    except Exception as e:
        cp.post_event({"job_id": job_id, "stage": "render", "status": "error", "error": str(e)[:500]})


def main() -> None:
    cp_url = _env_or_exit("CP_URL")
    job_token = _env_or_exit("JOB_TOKEN")
    cp = ControlPlane(cp_url, job_token)

    yunet_path = Path(os.environ.get("MODEL_YUNET", "/opt/models/yunet.onnx"))
    align_cache: dict[str, "AlignService"] = {}
    probe_cache: dict[tuple[Path, str], "ProbeService"] = {}
    boot_reported = False

    while True:
        try:
            job = cp.poll_job()
            if job is None:
                continue

            # envelope shape is transitional until the control-plane API freezes
            jtype = job.get("type")
            if jtype == "infer":
                boot_reported = _run_infer(job.get("request", {}), cp, align_cache, probe_cache, yunet_path, boot_reported)
            elif jtype == "render":
                _run_render(job.get("spec", {}), cp)
            else:
                cp.post_event({"stage": "dispatch", "status": "error", "error": f"unknown job type {jtype!r}"})
        except requests.RequestException as e:
            _log(f"control-plane request failed: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
