"""Tripwire: podagent/models.py must mirror contracts/*.schema.json exactly.

Every golden in contracts/examples/ is re-validated against BOTH the JSON Schema
(the SSOT) and the pydantic mirror; every golden in contracts/examples/invalid/
must be rejected by both. A model round-trip (model_dump back through the schema)
catches mirrors that accept more than the schema does.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from podagent.models import (
    SPEC_VERSION,
    ClipRankPayload,
    FaceProbePayload,
    InferRequest,
    InferResult,
    PodJob,
    RenderSpec,
)

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
EXAMPLES = CONTRACTS / "examples"

SCHEMAS = {p.name.split(".")[0]: json.loads(p.read_text()) for p in CONTRACTS.glob("*.schema.json")}

MODELS = {
    "spec": RenderSpec,
    "infer_request": InferRequest,
    "infer_result": InferResult,
    "face_probe": FaceProbePayload,
    "clip_rank": ClipRankPayload,
    "pod_job": PodJob,
}

VALID_EXAMPLES = sorted(EXAMPLES.glob("*.json"))
INVALID_EXAMPLES = sorted((EXAMPLES / "invalid").glob("*.json"))


def _prefix(path: Path) -> str:
    return path.name.split(".")[0]


def _validator(prefix: str) -> Draft202012Validator:
    return Draft202012Validator(SCHEMAS[prefix])


@pytest.mark.parametrize("path", VALID_EXAMPLES, ids=lambda p: p.name)
def test_valid_example_matches_schema_and_model(path: Path) -> None:
    data = json.loads(path.read_text())
    prefix = _prefix(path)
    validator = _validator(prefix)

    errors = sorted(validator.iter_errors(data), key=str)
    assert not errors, f"{path.name}: schema rejected a valid example: {errors[0].message}"

    model_cls = MODELS[prefix]
    instance = model_cls.model_validate(data)

    dumped = instance.model_dump(by_alias=True, exclude_none=True, mode="json")
    dump_errors = sorted(validator.iter_errors(dumped), key=str)
    assert not dump_errors, (
        f"{path.name}: model round-trip dump no longer matches schema: {dump_errors[0].message}"
    )


@pytest.mark.parametrize("path", INVALID_EXAMPLES, ids=lambda p: p.name)
def test_invalid_example_rejected_by_schema_and_model(path: Path) -> None:
    data = json.loads(path.read_text())
    prefix = _prefix(path)
    validator = _validator(prefix)

    assert next(validator.iter_errors(data), None) is not None, (
        f"{path.name}: schema accepted an example meant to be invalid"
    )

    model_cls = MODELS[prefix]
    with pytest.raises(ValidationError):
        model_cls.model_validate(data)


def test_top_level_required_parity() -> None:
    for prefix, model_cls in MODELS.items():
        model_required = {
            (field.alias or name)
            for name, field in model_cls.model_fields.items()
            if field.is_required()
        }
        schema_required = set(SCHEMAS[prefix].get("required", []))
        assert model_required == schema_required, (
            f"{prefix}: model-required {model_required} != schema-required {schema_required}"
        )


def test_version_pin() -> None:
    version = int((CONTRACTS / "VERSION").read_text().strip())
    assert version == SPEC_VERSION
    assert SCHEMAS["spec"]["properties"]["spec_version"]["const"] == version
    assert SCHEMAS["infer_request"]["properties"]["infer_version"]["const"] == version
    assert SCHEMAS["infer_result"]["properties"]["infer_version"]["const"] == version
