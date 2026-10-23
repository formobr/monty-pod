#!/usr/bin/env python3
"""Golden tripwire for the SSOT: examples/*.json must validate, examples/invalid/*.json
must be REJECTED. Schema picked by filename prefix against every *.schema.json here.
Run: python validate.py (needs jsonschema)."""
import json
import pathlib
import sys

from jsonschema import Draft202012Validator

ROOT = pathlib.Path(__file__).parent
SCHEMAS = {p.name.split(".")[0]: json.loads(p.read_text()) for p in ROOT.glob("*.schema.json")}


def _schema_for(path: pathlib.Path) -> Draft202012Validator:
    key = path.name.split(".")[0]
    if key not in SCHEMAS:
        raise SystemExit(f"no schema for prefix {key!r} ({path})")
    return Draft202012Validator(SCHEMAS[key])


def _reject_non_finite(token: str) -> None:
    # json.loads's own extension lets 'NaN'/'Infinity'/'-Infinity' through as Python floats; standard
    # JSON has no such literal at all. Raising here is what makes _as_json below reject them.
    raise ValueError(f"{token} is not valid JSON (no NaN/Infinity literal)")


def _as_json(data: object) -> object:
    """`data` round-tripped through the JSON text it would actually travel as. A bare Python object can
    carry a float('nan') that no JSON document ever can — and Draft202012Validator's own numeric keywords
    (type, minimum) do not catch it: `nan < 0` and `nan >= 0` are both False, so `minimum: 0` lets it
    through silently. Raises ValueError for anything that could not have arrived as JSON text."""
    return json.loads(json.dumps(data, allow_nan=True), parse_constant=_reject_non_finite)


def accepts(prefix: str, data: object) -> bool:
    """The schema verdict for one document, picked by the same filename prefix the goldens use."""
    if prefix not in SCHEMAS:
        raise SystemExit(f"no schema for prefix {prefix!r}")
    try:
        as_json = _as_json(data)
    except ValueError:
        return False
    return next(Draft202012Validator(SCHEMAS[prefix]).iter_errors(as_json), None) is None


def main() -> int:
    fails: list[str] = []

    for path in sorted((ROOT / "examples").glob("*.json")):
        try:
            data = _as_json(json.loads(path.read_text()))
        except ValueError as exc:
            fails.append(f"VALID example is not valid JSON: {path.name} -> {exc}")
            continue
        errs = sorted(_schema_for(path).iter_errors(data), key=str)
        if errs:
            fails.append(f"VALID example rejected: {path.name} -> {errs[0].message}")

    for path in sorted((ROOT / "examples" / "invalid").glob("*.json")):
        try:
            data = _as_json(json.loads(path.read_text()))
        except ValueError:
            continue  # not valid JSON at all is correctly rejected before the schema even runs
        if next(_schema_for(path).iter_errors(data), None) is None:
            fails.append(f"INVALID example accepted: {path.name} (schema let it through)")

    n = len(list((ROOT / "examples").glob("*.json"))) + len(list((ROOT / "examples" / "invalid").glob("*.json")))
    if fails:
        print(f"✗ pod-contracts: {len(fails)}/{n} golden checks failed", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"✓ pod-contracts: {n} golden checks pass across {len(SCHEMAS)} schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
