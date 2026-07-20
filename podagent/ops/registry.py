"""The op registry: ONE declaration set, read by both sides.

`contracts/ops/*.json` is the SSOT. The control plane reads it to validate params and decide placement;
the pod reads the SAME files out of the image to validate what it was sent and to find the handler. For
`params` there is therefore nothing to mirror and nothing to drift — the alternative (a hand-written
Pydantic model per op) is the thing this design exists to avoid.

VERSION SPLIT. `contracts/VERSION` pins the ENVELOPE/transport; each op carries its own `version`. A new
op is purely additive — no envelope bump, no Go change, no image rebuild (the declaration ships in the
image, but a control plane pinning an op the pod's image predates simply gets `unknown op`, loudly). That
split is what makes adding a tool cost two files instead of a release.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

# /app/podagent/ops/registry.py -> /app/contracts ; pod-agent/podagent/ops/registry.py -> pod-agent/contracts
CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
OPS_DIR = CONTRACTS / "ops"

NEEDS_VOCAB = frozenset({"video_encode", "model_weights", "browser", "net", "keys", "llm"})
PARITY_MODES = frozenset({"bit_exact", "perceptual", "numeric"})


class OpError(RuntimeError):
    """A registry/param/placement violation. Distinct from a handler's own failure: this one means the
    CALL was malformed, so it must fail before any pixel is produced."""


@dataclass(frozen=True)
class Port:
    id: str
    kind: str
    optional: bool = False
    durable: bool = False


@dataclass(frozen=True)
class Op:
    op: str
    version: int
    summary: str
    needs: frozenset[str]
    judgement: bool
    parity_mode: str
    parity_tol: float | None
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    params_schema: dict[str, Any]
    handler: str

    @property
    def durable_outputs(self) -> tuple[Port, ...]:
        """Outputs that MUST reach R2. Everything else stays on the pod's local disk between ops in the
        same job — see runner.Workspace for why that is the whole performance story."""
        return tuple(p for p in self.outputs if p.durable)


def _port(raw: dict[str, Any]) -> Port:
    return Port(id=raw["id"], kind=raw["kind"],
                optional=bool(raw.get("optional", False)),
                durable=bool(raw.get("durable", False)))


def _load_one(path: Path) -> Op:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Structural validation is the meta-schema's job (contracts/validate.py runs it over every
    # declaration). What we re-check HERE is the handful of invariants a JSON Schema cannot express and
    # that the rest of this module then assumes.
    unknown = set(raw.get("needs", [])) - NEEDS_VOCAB
    if unknown:
        raise OpError(f"{path.name}: needs outside the closed vocabulary: {sorted(unknown)}")
    parity = raw.get("parity") or {}
    if parity.get("mode") not in PARITY_MODES:
        raise OpError(f"{path.name}: parity.mode must be one of {sorted(PARITY_MODES)}")
    if raw["op"] != path.stem:
        raise OpError(f"{path.name}: declares op={raw['op']!r} but is filed as {path.stem!r}")
    ps = raw["params"]
    if ps.get("type") != "object" or ps.get("additionalProperties") is not False:
        # An open params bag is argv wearing a hat: it re-admits code across the seam, re-publishes the
        # algorithm, and blinds the placement gate. Refuse at load, not at call.
        raise OpError(f"{path.name}: params must be a CLOSED object (additionalProperties:false)")
    return Op(
        op=raw["op"], version=int(raw["version"]), summary=raw["summary"],
        needs=frozenset(raw.get("needs", [])), judgement=bool(raw["judgement"]),
        parity_mode=parity["mode"], parity_tol=parity.get("tol"),
        inputs=tuple(_port(p) for p in raw.get("inputs", [])),
        outputs=tuple(_port(p) for p in raw["outputs"]),
        params_schema=ps, handler=raw["handler"],
    )


@lru_cache(maxsize=1)
def all_ops() -> dict[str, Op]:
    if not OPS_DIR.is_dir():
        raise OpError(f"no op registry at {OPS_DIR}")
    ops = {}
    for path in sorted(OPS_DIR.glob("*.json")):
        op = _load_one(path)
        ops[op.op] = op
    return ops


def get(name: str) -> Op:
    ops = all_ops()
    if name not in ops:
        raise OpError(f"unknown op {name!r}; registry holds {sorted(ops)}")
    return ops[name]


def validate_params(name: str, params: dict[str, Any]) -> None:
    """Check params against the declaration's schema fragment. Same call on both sides, same file, so a
    payload the control plane accepts is one the pod accepts."""
    op = get(name)
    try:
        from jsonschema import Draft202012Validator
    except ImportError as e:   # pragma: no cover - jsonschema is a runtime dep of both sides
        raise OpError("jsonschema is required to validate op params") from e
    errs = sorted(Draft202012Validator(op.params_schema).iter_errors(params), key=str)
    if errs:
        raise OpError(f"{name}: invalid params: {errs[0].message}")


def assert_pod_safe(name: str) -> None:
    """THE constructional gate. A judgement op names a DECISION (which keeps, which drops, pacing, tempo,
    shot planning, the b-roll judge, thresholds) and is control-plane-only — not by policy, by refusal.

    This is called on the POD, in the dispatcher, before a handler is looked up. It is deliberately
    redundant with the control plane's own placement check: the whole point of `judgement` is that the cut
    algorithm cannot reach a pod even if some future stage table says it may, and a check that lives only
    on the side doing the routing is a check the routing bug disables.
    """
    op = get(name)
    if op.judgement:
        raise OpError(
            f"op {name!r} is judgement:true — it decides, it does not execute, and a pod must never run "
            f"it. This refusal is the executable form of decision `pods-run-all-heavy-work` (the brain "
            f"stays on the control plane); if you are seeing it, the ROUTING is wrong, not this check.")
    if "keys" in op.needs or "llm" in op.needs:
        raise OpError(
            f"op {name!r} needs {sorted(op.needs & {'keys', 'llm'})} — the pod is KEYLESS (its entire "
            f"credential surface is CP_URL + JOB_TOKEN) and cannot be handed an op that wants a secret.")
