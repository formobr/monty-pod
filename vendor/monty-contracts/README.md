# Shared wire vendor

`wire_bundle.json` is the `monty-contracts` bundle at the exact commit
recorded in `PIN`, **trimmed to the two surfaces the pod actually consumes**:
`pod_stream` and `pod_stream_server` (`schemas`, `fixtures`,
`registry.surfaces`). The independent render/infer/op domain remains under
`../contracts/` at version 5.

The trim is not cosmetic: this is a PUBLIC repo
(tests/test_public_repo_names_no_engine_internals.py), and a surface the pod
never reads can carry the producer's own internal prose (another provider's
enum value, a design-doc name, a brand) that this pod has no business
tracking. Vendoring only what is consumed means that prose never reaches
this tree in the first place, so there is nothing here to excuse.

Refresh this directory from a reviewed producer commit:

1. Copy the producer's built bundle in, as `wire_bundle.json`.
2. Drop every `schemas` / `fixtures` / `registry.surfaces` entry whose
   surface is not `pod_stream` or `pod_stream_server`; trim `source_sha256`
   to the files the kept surfaces still reference.
3. Recompute `bundle_sha256` over the trimmed document (see
   `tests/test_wire_contracts.py::test_vendored_bundle_digest_pin_and_generated_output_are_current`
   for the exact canonicalization) and bump `PIN` to the producer commit.
4. Run:

```bash
python3 tools/gen_wire_models.py --write
pytest -q tests/test_wire_contracts.py tests/test_wire_fixture_gate.py \
        tests/test_public_repo_names_no_engine_internals.py
```

Do not copy the producer schemas or examples back into `contracts/`; that
would restore the second source of truth this vendor replaces.
