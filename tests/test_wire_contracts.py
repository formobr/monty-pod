from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from podagent.models import PodJob
from podagent.stream_models import (
    JobAckPayload,
    PodStreamFrame,
    PodStreamServerFrame,
    StreamAck,
    StreamEvent,
    StreamResult,
)
from podagent.wire_generated import WIRE_BUNDLE_SHA256, WIRE_CONTRACT_VERSION
from wire_fixtures import DELETE, invalid_wire, valid_wire

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "vendor" / "monty-contracts" / "wire_bundle.json"
BUNDLE_DOCUMENT = json.loads(BUNDLE.read_text(encoding="utf-8"))


def _cases(classification: str) -> list[tuple[str, str]]:
    return [
        (name, str(entry["surface"]))
        for name, entry in sorted(BUNDLE_DOCUMENT["fixtures"].items())
        if entry["surface"] in {"pod_stream", "pod_stream_server"}
        and entry["classification"] == classification
    ]


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _model(surface: str):
    return PodStreamFrame if surface == "pod_stream" else PodStreamServerFrame


def _generator():
    path = ROOT / "tools" / "gen_wire_models.py"
    spec = importlib.util.spec_from_file_location("_pod_wire_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pod_job_for_shared_golden() -> dict:
    job = json.loads((ROOT / "contracts" / "examples" / "pod_job.ops.json").read_text(encoding="utf-8"))
    job["corr_id"] = "clip-rank-12"
    job = PodJob.model_validate(job).model_dump(exclude_none=True, mode="json")
    job["chain"]["ops"] = DELETE
    return job


def test_vendored_bundle_digest_pin_and_generated_output_are_current() -> None:
    bundle = BUNDLE_DOCUMENT
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    assert hashlib.sha256(_canonical(unsigned)).hexdigest() == bundle["bundle_sha256"]
    assert WIRE_BUNDLE_SHA256 == bundle["bundle_sha256"]
    assert WIRE_CONTRACT_VERSION == bundle["contract_version"]
    pin = (ROOT / "vendor" / "monty-contracts" / "PIN").read_text(encoding="utf-8").strip()
    assert len(pin) == 40 and all(char in "0123456789abcdef" for char in pin)
    subprocess.run([sys.executable, "tools/gen_wire_models.py", "--check"], cwd=ROOT, check=True)


def test_a_corrupt_vendored_bundle_stamp_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = _generator()
    bad = dict(BUNDLE_DOCUMENT)
    bad["bundle_sha256"] = "0" * 64
    path = tmp_path / "wire_bundle.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(generator, "BUNDLE", path)
    with pytest.raises(ValueError, match="bundle digest mismatch"):
        generator.produce()


def test_stale_generated_output_is_a_red_regen_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = _generator()
    output = tmp_path / "wire_generated.py"
    output.write_text("# stale\n", encoding="utf-8")
    monkeypatch.setattr(generator, "OUTPUT", output)
    assert generator.main(["--check"]) == 1


@pytest.mark.parametrize(
    "name,surface",
    _cases("valid"),
)
def test_shared_valid_goldens_round_trip_generated_fields_and_semantic_facade(name: str, surface: str) -> None:
    model = _model(surface)
    patch = {"job": _pod_job_for_shared_golden()} if name == "pod_stream_server.job" else None
    document = valid_wire(name, patch, consumer_validator=model.model_validate)
    assert model.model_validate(document).model_dump(exclude_none=True, mode="json") == document


@pytest.mark.parametrize(
    "name,surface",
    _cases("invalid"),
)
def test_shared_schema_invalid_goldens_are_rejected(name: str, surface: str) -> None:
    document = invalid_wire(name, reason="producer marks this shared-wire shape invalid")
    with pytest.raises(ValidationError):
        _model(surface).model_validate(document)


@pytest.mark.parametrize(
    "name,surface",
    _cases("model-invalid"),
)
def test_shared_model_invalid_goldens_are_rejected_by_facade(name: str, surface: str) -> None:
    model = _model(surface)
    patch = {"job": _pod_job_for_shared_golden()} if surface == "pod_stream_server" else None
    document = invalid_wire(
        name,
        reason="cross-field identity cannot be expressed by JSON Schema",
        patch=patch,
        consumer_validator=model.model_validate,
    )
    with pytest.raises(ValidationError):
        model.model_validate(document)


def test_local_render_infer_contract_version_remains_independent() -> None:
    assert (ROOT / "contracts" / "VERSION").read_text(encoding="utf-8").strip() == "5"


def test_generated_models_preserve_the_established_v12_encoder_order() -> None:
    event = StreamEvent.model_validate({
        "stage": "ops", "status": "step", "job_id": "j", "session_id": "s", "corr_id": "c",
    }).model_dump(exclude_none=True, mode="json")
    result = StreamResult.model_validate({
        "job_id": "j", "session_id": "s", "corr_id": "c", "status": "ok", "stage": "ops",
    }).model_dump(exclude_none=True, mode="json")
    job_ack = JobAckPayload.model_validate({
        "delivery_id": "c", "corr_id": "c", "attempt_id": "a" * 32, "client_recv_mono_ns": 1,
    }).model_dump(mode="json")
    ack = StreamAck.model_validate({
        "type": "ack", "stream_id": "s", "seq": 1, "status": 202,
        "server_recv_unix_ns": 1, "server_send_unix_ns": 2,
    }).model_dump(exclude_none=True, mode="json")
    assert list(event) == ["stage", "status", "job_id", "session_id", "corr_id"]
    assert list(result) == ["job_id", "session_id", "corr_id", "status", "stage"]
    assert list(job_ack) == ["delivery_id", "corr_id", "attempt_id", "client_recv_mono_ns"]
    assert list(ack) == [
        "type", "stream_id", "seq", "status", "server_recv_unix_ns", "server_send_unix_ns",
    ]
