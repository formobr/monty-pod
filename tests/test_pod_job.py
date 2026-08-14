"""Direct unit coverage for PodJob, on top of the golden round-trips in
test_contracts_goldens.py — pins the dispatch contract main.py relies on."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from podagent.models import PodJob, RenderSpec

_ALIGN_REQUEST = {
    "infer_version": 6,
    "job_id": "j",
    "kind": "align",
    "model": "m",
    "put_url": "p",
    "align": {"audio_url": "u", "windows": [[0.0, 10.0]]},
    "weights": {"url": "https://r2.example/models/x.tar", "sha256": "a" * 64},
}

_PREVIEW_SPEC = {
    "spec_version": 6,
    "job_id": "j",
    "slug": "s",
    "mode": "preview",
    "inputs": [{"id": "src", "kind": "video", "sha256": "0" * 64, "url": "u"}],
    "timeline": {"fps": 30, "width": 2, "height": 2, "segments": [{"src": "src", "in": 0, "out": 1, "speed": 1}]},
    "encode": {"video": "libx264", "preset": "p4", "cq": 29, "pix_fmt": "yuv420p", "audio": "aac", "audio_bitrate": "192k"},
    "outputs": [{"id": "proxy", "kind": "proxy", "put_url": "p"}],
}


_FINAL_SPEC_WITH_ACCENT = {
    **_PREVIEW_SPEC,
    "mode": "final",
    "outputs": [{"id": "master", "kind": "master", "put_url": "p"}],
    "overlays": {"finalize": {"accents": [{"kind": "zoom_punch", "at": 1.0, "intensity": 0.7}]}},
}


def test_infer_job_valid() -> None:
    job = PodJob.model_validate({
        "type": "infer", "session_id": "s", "corr_id": "c", "request": _ALIGN_REQUEST})
    assert job.type == "infer"
    assert job.request is not None and job.spec is None


def test_render_job_valid() -> None:
    job = PodJob.model_validate({
        "type": "render", "session_id": "s", "corr_id": "c", "spec": _PREVIEW_SPEC})
    assert job.type == "render"
    assert job.spec is not None and job.request is None


@pytest.mark.parametrize("missing", ["session_id", "corr_id"])
def test_missing_routing_id_is_rejected_before_work(missing: str) -> None:
    raw = {
        "type": "infer", "session_id": "s", "corr_id": "c", "request": _ALIGN_REQUEST}
    raw.pop(missing)
    with pytest.raises(ValidationError):
        PodJob.model_validate(raw)


def test_pool_ids_carried() -> None:
    job = PodJob.model_validate(
        {"type": "render", "spec": _PREVIEW_SPEC, "session_id": "tenant_1", "corr_id": "abc123",
         "target_worker_id": "fleet-worker-7"})
    assert (job.session_id, job.corr_id, job.target_worker_id) == ("tenant_1", "abc123", "fleet-worker-7")


def test_target_worker_id_is_strictly_bounded() -> None:
    with pytest.raises(ValidationError):
        PodJob.model_validate({
            "type": "infer", "session_id": "s", "corr_id": "c",
            "target_worker_id": "worker with spaces", "request": _ALIGN_REQUEST})


def test_mismatched_block_rejected() -> None:
    with pytest.raises(ValidationError):
        PodJob.model_validate({
            "type": "infer", "session_id": "s", "corr_id": "c",
            "request": _ALIGN_REQUEST, "spec": _PREVIEW_SPEC})


def test_missing_block_rejected() -> None:
    with pytest.raises(ValidationError):
        PodJob.model_validate({"type": "render", "session_id": "s", "corr_id": "c"})


def test_dispatch_redump_of_a_zoom_punch_accent_carries_no_burn_or_clicks_keys() -> None:
    """NEGATIVE: main._heavy re-dumps pod_job.spec before _run_render re-validates it. A dump that
    resurrects the unset burn/clicks defaults as explicit nulls makes the pod refuse its own valid
    envelope ("accent.burn must not be null"), so the dispatch dump must exclude_none."""
    job = PodJob.model_validate({
        "type": "render", "session_id": "s", "corr_id": "c", "spec": _FINAL_SPEC_WITH_ACCENT})
    assert job.spec is not None
    naive = job.spec.model_dump(by_alias=True, mode="json")
    assert naive["overlays"]["finalize"]["accents"][0]["burn"] is None
    with pytest.raises(ValidationError, match="must not be null"):
        RenderSpec.model_validate(naive)
    raw = job.spec.model_dump(by_alias=True, exclude_none=True, mode="json")
    accent = raw["overlays"]["finalize"]["accents"][0]
    assert "burn" not in accent and "clicks" not in accent
    assert RenderSpec.model_validate(raw).overlays is not None


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        PodJob.model_validate({
            "type": "upscale", "session_id": "s", "corr_id": "c", "request": _ALIGN_REQUEST})
