"""Direct unit coverage for PodJob, on top of the golden round-trips in
test_contracts_goldens.py — pins the dispatch contract main.py relies on."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from podagent.models import PodJob

_ALIGN_REQUEST = {
    "infer_version": 5,
    "job_id": "j",
    "kind": "align",
    "model": "m",
    "put_url": "p",
    "align": {"audio_url": "u", "windows": [[0.0, 10.0]]},
    "weights": {"url": "https://r2.example/models/x.tar", "sha256": "a" * 64},
}

_PREVIEW_SPEC = {
    "spec_version": 5,
    "job_id": "j",
    "slug": "s",
    "mode": "preview",
    "inputs": [{"id": "src", "kind": "video", "sha256": "0" * 64, "url": "u"}],
    "timeline": {"fps": 30, "width": 2, "height": 2, "segments": [{"src": "src", "in": 0, "out": 1, "speed": 1}]},
    "encode": {"video": "libx264", "preset": "p4", "cq": 29, "pix_fmt": "yuv420p", "audio": "aac", "audio_bitrate": "192k"},
    "outputs": [{"id": "proxy", "kind": "proxy", "put_url": "p"}],
}


def test_infer_job_valid() -> None:
    job = PodJob.model_validate({"type": "infer", "request": _ALIGN_REQUEST})
    assert job.type == "infer"
    assert job.request is not None and job.spec is None


def test_render_job_valid() -> None:
    job = PodJob.model_validate({"type": "render", "spec": _PREVIEW_SPEC})
    assert job.type == "render"
    assert job.spec is not None and job.request is None


def test_mismatched_block_rejected() -> None:
    with pytest.raises(ValidationError):
        PodJob.model_validate({"type": "infer", "request": _ALIGN_REQUEST, "spec": _PREVIEW_SPEC})


def test_missing_block_rejected() -> None:
    with pytest.raises(ValidationError):
        PodJob.model_validate({"type": "render"})


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        PodJob.model_validate({"type": "upscale", "request": _ALIGN_REQUEST})
