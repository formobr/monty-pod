# Shared wire vendor

`wire_bundle.json` is the deterministic bundle from `monty-contracts` at the
exact commit recorded in `PIN`.  The pod consumes only the `pod_stream` and
`pod_stream_server` surfaces from it.  The independent render/infer/op domain
remains under `../contracts/` at version 5.

Refresh this directory from a reviewed producer commit, then run:

```bash
python3 tools/gen_wire_models.py --write
pytest -q tests/test_wire_contracts.py tests/test_wire_fixture_gate.py
```

Do not copy the producer schemas or examples back into `contracts/`; that
would restore the second source of truth this vendor replaces.
