"""Canonical test fixtures for the pod's two shared EventStream surfaces."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "vendor" / "monty-contracts" / "wire_bundle.json"
ConsumerValidator = Callable[[dict[str, Any]], Any]


class FixtureError(ValueError):
    pass


class _Delete:
    pass


DELETE = _Delete()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bundle() -> dict[str, Any]:
    document = json.loads(BUNDLE.read_text(encoding="utf-8"))
    claimed = document.get("bundle_sha256")
    unsigned = {key: value for key, value in document.items() if key != "bundle_sha256"}
    actual = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if claimed != actual:
        raise FixtureError(f"wire bundle digest mismatch: claimed {claimed!r}, computed {actual}")
    return document


def _patch(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if value is DELETE:
            if key not in target:
                raise FixtureError(f"cannot delete absent fixture field {key!r}")
            del target[key]
        elif isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _patch(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _fixture(name: str, patch: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = _bundle()
    try:
        entry = bundle["fixtures"][name]
        schema = bundle["schemas"][entry["surface"]]
    except KeyError as exc:
        raise FixtureError(f"unknown wire fixture {name!r}") from exc
    if entry["surface"] not in {"pod_stream", "pod_stream_server"}:
        raise FixtureError(f"fixture {name!r} is outside the pod EventStream surfaces")
    document = copy.deepcopy(entry["document"])
    if patch:
        _patch(document, patch)
    return document, entry, schema


def _model_accepts(document: dict[str, Any], validator: ConsumerValidator | None) -> bool | None:
    if validator is None:
        return None
    try:
        validator(copy.deepcopy(document))
    except Exception:
        return False
    return True


def valid_wire(
    name: str,
    patch: Mapping[str, Any] | None = None,
    *,
    consumer_validator: ConsumerValidator | None = None,
) -> dict[str, Any]:
    document, entry, schema = _fixture(name, patch)
    if entry["classification"] != "valid":
        raise FixtureError(f"{name!r} is not a valid fixture")
    error = next(Draft202012Validator(schema).iter_errors(document), None)
    if error is not None:
        raise FixtureError(f"patched valid fixture is schema-invalid: {error.message}")
    if _model_accepts(document, consumer_validator) is False:
        raise FixtureError(f"patched valid fixture {name!r} is rejected by the consumer")
    return document


def invalid_wire(
    name: str,
    *,
    reason: str,
    patch: Mapping[str, Any] | None = None,
    consumer_validator: ConsumerValidator | None = None,
) -> dict[str, Any]:
    if not reason.strip():
        raise FixtureError("invalid_wire requires a reason")
    document, entry, schema = _fixture(name, patch)
    schema_accepts = next(Draft202012Validator(schema).iter_errors(document), None) is None
    model_accepts = _model_accepts(document, consumer_validator)
    classification = entry["classification"]
    if classification == "invalid" and schema_accepts:
        raise FixtureError(f"invalid fixture {name!r} is accepted by the schema")
    if classification == "model-invalid":
        if not schema_accepts or consumer_validator is None or model_accepts is not False:
            raise FixtureError(f"model-invalid fixture {name!r} did not prove its consumer-only failure")
    if classification == "valid" and schema_accepts and model_accepts is not False:
        raise FixtureError(f"patched valid fixture {name!r} is still accepted")
    return document
