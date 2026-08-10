#!/usr/bin/env python3
"""Generate pod EventStream field mixins from the vendored shared-wire bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "vendor" / "monty-contracts" / "wire_bundle.json"
OUTPUT = ROOT / "podagent" / "wire_generated.py"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bundle() -> dict[str, Any]:
    document = json.loads(BUNDLE.read_text(encoding="utf-8"))
    claimed = document.get("bundle_sha256")
    unsigned = {key: value for key, value in document.items() if key != "bundle_sha256"}
    actual = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if claimed != actual:
        raise ValueError(f"wire bundle digest mismatch: claimed {claimed!r}, computed {actual}")
    if not {"pod_stream", "pod_stream_server"} <= set(document.get("schemas", {})):
        raise ValueError("wire bundle is missing the pod EventStream surfaces")
    return document


@dataclass(frozen=True)
class ModelSpec:
    name: str
    surface: str
    schema: dict[str, Any]


_REF_NAMES = {
    ("pod_stream", "streamId"): "str",
    ("pod_stream", "attemptId"): "str",
    ("pod_stream", "monotonicNs"): "int",
    ("pod_stream", "podEvent"): "StreamEvent",
    ("pod_stream", "podResult"): "StreamResult",
    ("pod_stream", "jobAck"): "JobAckPayload",
    ("pod_stream_server", "streamId"): "str",
    ("pod_stream_server", "attemptId"): "str",
    ("pod_stream_server", "unixNs"): "int",
    ("pod_stream_server", "podJob"): "PodJob",
    ("pod_stream_server", "jobTimeline"): "JobTimeline",
}

# JSON object order is semantically irrelevant, but the durable outbox replays the exact bytes it first
# wrote.  The producer bundle is canonicalized with sorted keys, so preserve the established v12 encoder
# order explicitly instead of letting a source-file formatting choice rewrite every new frame.
_WIRE_ORDER = {
    "StreamEventFields": (
        "stage", "status", "job_id", "session_id", "corr_id", "step", "op", "phase", "outcome",
        "optional", "steps", "skipped", "outputs", "error", "error_type", "timings", "timeline",
        "capacity", "ts",
    ),
    "StreamResultFields": (
        "job_id", "session_id", "corr_id", "status", "kind", "stage", "result_key", "timing",
        "timings", "error", "timeline",
    ),
    "StreamEventFrameFields": ("type", "stream_id", "seq", "client_send_mono_ns", "event"),
    "StreamResultFrameFields": (
        "type", "stream_id", "seq", "client_send_mono_ns", "attempt_id", "result",
    ),
    "JobAckPayloadFields": ("delivery_id", "corr_id", "attempt_id", "client_recv_mono_ns"),
    "StreamJobAckFrameFields": ("type", "stream_id", "seq", "client_send_mono_ns", "job_ack"),
    "StreamAckFields": (
        "type", "stream_id", "seq", "status", "server_recv_unix_ns", "server_send_unix_ns",
        "duplicate", "error",
    ),
    "JobTimelineFields": (
        "enqueue_min_unix_ns", "enqueue_max_unix_ns", "claim_min_unix_ns", "claim_max_unix_ns",
        "socket_write_min_unix_ns",
    ),
    "StreamJobFields": (
        "type", "delivery_id", "attempt_id", "stream_id", "seq", "replayed", "timeline", "job",
    ),
}


def _ref_name(surface: str, ref: str) -> str:
    return _REF_NAMES[(surface, ref.rsplit("/", 1)[-1])]


def _literal(values: list[Any]) -> str:
    return "Literal[" + ", ".join(repr(value) for value in values) + "]"


def _annotation(surface: str, schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        return _ref_name(surface, str(schema["$ref"]))
    if "const" in schema:
        return _literal([schema["const"]])
    if "enum" in schema:
        return _literal(list(schema["enum"]))
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        members = [item for item in raw_type if item != "null"]
        base = _annotation(surface, {**schema, "type": members[0]}) if members else "None"
        return f"{base} | None"
    if raw_type == "string":
        return "datetime" if schema.get("format") == "date-time" else "str"
    if raw_type == "integer":
        return "int"
    if raw_type == "number":
        return "float"
    if raw_type == "boolean":
        return "bool"
    if raw_type == "array":
        return f"list[{_annotation(surface, schema.get('items', {}))}]"
    if raw_type == "object":
        return "dict[str, Any]"
    return "Any"


def _resolved(surface: str, schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    if "$ref" not in schema:
        return schema
    key = str(schema["$ref"]).rsplit("/", 1)[-1]
    if _REF_NAMES.get((surface, key)) in {"str", "int"}:
        return root["$defs"][key]
    return schema


def _constraints(schema: dict[str, Any]) -> list[str]:
    names = {
        "minLength": "min_length",
        "maxLength": "max_length",
        "pattern": "pattern",
        "minimum": "ge",
        "maximum": "le",
    }
    return [f"{target}={schema[source]!r}" for source, target in names.items() if source in schema]


def _allows_null(schema: dict[str, Any]) -> bool:
    raw = schema.get("type")
    return schema == {} or (isinstance(raw, list) and "null" in raw)


def _field_line(spec: ModelSpec, field: str, schema: dict[str, Any], required: set[str], root: dict[str, Any]) -> str:
    resolved = _resolved(spec.surface, schema, root)
    annotation = _annotation(spec.surface, schema)
    if field not in required and not annotation.endswith(" | None"):
        annotation += " | None"
    constraints = _constraints(resolved)
    if field not in required:
        args = ", ".join(["default=None", *constraints])
        return f"    {field}: {annotation} = Field({args})" if constraints else f"    {field}: {annotation} = None"
    return f"    {field}: {annotation} = Field({', '.join(constraints)})" if constraints else f"    {field}: {annotation}"


def _model(spec: ModelSpec, roots: dict[str, dict[str, Any]]) -> str:
    root = roots[spec.surface]
    required = set(spec.schema.get("required", []))
    extra = "forbid" if spec.schema.get("additionalProperties") is False else "allow"
    properties = spec.schema.get("properties", {})
    order = _WIRE_ORDER[spec.name]
    if set(order) != set(properties):
        raise ValueError(
            f"{spec.name}: compatibility order differs from bundle fields: "
            f"missing={sorted(set(properties) - set(order))}, stale={sorted(set(order) - set(properties))}"
        )
    optional_non_null = tuple(
        name for name in order
        for schema in (properties[name],)
        if name not in required and not _allows_null(_resolved(spec.surface, schema, root))
    )
    lines = [f"class {spec.name}:", f"    model_config = ConfigDict(extra={extra!r})"]
    if optional_non_null:
        lines.extend([
            f"    _wire_non_null_optional: ClassVar[frozenset[str]] = frozenset({optional_non_null!r})",
            "",
            "    @model_validator(mode=\"before\")",
            "    @classmethod",
            "    def _wire_optional_fields_are_absent_not_null(cls, value: Any) -> Any:",
            "        if isinstance(value, dict):",
            "            nulls = sorted(key for key in cls._wire_non_null_optional if value.get(key, _MISSING) is None)",
            "            if nulls:",
            "                raise ValueError(f\"optional wire fields must be omitted, not null: {nulls}\")",
            "        return value",
        ])
    if properties:
        lines.append("")
    lines.extend(
        _field_line(spec, field, properties[field], required, root)
        for field in order
    )
    return "\n".join(lines)


def _specs(schemas: dict[str, dict[str, Any]]) -> list[ModelSpec]:
    client = schemas["pod_stream"]
    server = schemas["pod_stream_server"]
    return [
        ModelSpec("StreamEventFields", "pod_stream", client["$defs"]["podEvent"]),
        ModelSpec("StreamResultFields", "pod_stream", client["$defs"]["podResult"]),
        ModelSpec("StreamEventFrameFields", "pod_stream", client["$defs"]["eventFrame"]),
        ModelSpec("StreamResultFrameFields", "pod_stream", client["$defs"]["resultFrame"]),
        ModelSpec("JobAckPayloadFields", "pod_stream", client["$defs"]["jobAck"]),
        ModelSpec("StreamJobAckFrameFields", "pod_stream", client["$defs"]["jobAckFrame"]),
        ModelSpec("StreamAckFields", "pod_stream_server", server["$defs"]["ackFrame"]),
        ModelSpec("JobTimelineFields", "pod_stream_server", server["$defs"]["jobTimeline"]),
        ModelSpec("StreamJobFields", "pod_stream_server", server["$defs"]["jobFrame"]),
    ]


def produce() -> str:
    bundle = _bundle()
    header = f'''# @generated by tools/gen_wire_models.py from vendor/monty-contracts/wire_bundle.json -- DO NOT EDIT.
from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Final, Literal

from pydantic import ConfigDict, Field, model_validator

_MISSING = object()
WIRE_BUNDLE_FORMAT: Final = {int(bundle["bundle_format"])}
WIRE_BUNDLE_SHA256: Final = {str(bundle["bundle_sha256"])!r}
WIRE_CONTRACT_VERSION: Final = {int(bundle["contract_version"])}
'''
    blocks = [header.rstrip(), *(_model(spec, bundle["schemas"]) for spec in _specs(bundle["schemas"]))]
    exports = [spec.name for spec in _specs(bundle["schemas"])] + [
        "WIRE_BUNDLE_FORMAT", "WIRE_BUNDLE_SHA256", "WIRE_CONTRACT_VERSION",
    ]
    blocks.append("__all__ = " + repr(sorted(exports)))
    return "\n\n\n".join(blocks).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        rendered = produce()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"wire models: {exc}", file=sys.stderr)
        return 1
    if args.write:
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(ROOT)}")
        return 0
    try:
        current = OUTPUT.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"wire models: cannot read generated output: {exc}", file=sys.stderr)
        return 1
    if current != rendered:
        print("wire models: generated output is stale; run tools/gen_wire_models.py --write", file=sys.stderr)
        return 1
    print(f"wire models current: {json.loads(BUNDLE.read_text())['bundle_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
