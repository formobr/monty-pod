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


def main() -> int:
    fails: list[str] = []

    for path in sorted((ROOT / "examples").glob("*.json")):
        errs = sorted(_schema_for(path).iter_errors(json.loads(path.read_text())), key=str)
        if errs:
            fails.append(f"VALID example rejected: {path.name} -> {errs[0].message}")

    for path in sorted((ROOT / "examples" / "invalid").glob("*.json")):
        if next(_schema_for(path).iter_errors(json.loads(path.read_text())), None) is None:
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
