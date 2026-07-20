"""The operation seam, pod side.

Every test here is a NEGATIVE test in the sense docs/TESTING.md means: each asserts that a specific
wrong thing is REFUSED, and each was watched fail with its guard reverted before being committed.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from podagent.models import OpChain, PodJob
from podagent.ops import pack, registry

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"


# ── the registry itself ──────────────────────────────────────────────────────────────────────────

def test_every_declaration_loads():
    ops = registry.all_ops()
    assert "media.scale" in ops and "mograph.render" in ops


def test_declarations_validate_against_the_meta_schema():
    """op.schema.json is the SSOT for what a declaration may say. A declaration that drifts from it is
    caught here rather than at dispatch on a rented box."""
    jsonschema = pytest.importorskip("jsonschema")
    meta = json.loads((CONTRACTS / "op.schema.json").read_text())
    v = jsonschema.Draft202012Validator(meta)
    for p in sorted((CONTRACTS / "ops").glob("*.json")):
        errs = sorted(v.iter_errors(json.loads(p.read_text())), key=str)
        assert not errs, f"{p.name}: {errs[0].message}"


def test_needs_vocabulary_is_closed(tmp_path, monkeypatch):
    """A `needs` outside the closed vocabulary must not load. An open vocabulary would let a handler
    invent a capability the placement gate has never heard of and therefore cannot refuse."""
    d = tmp_path / "ops"
    d.mkdir()
    decl = json.loads((CONTRACTS / "ops" / "media.scale.json").read_text())
    decl["needs"] = ["teleportation"]
    (d / "media.scale.json").write_text(json.dumps(decl))
    monkeypatch.setattr(registry, "OPS_DIR", d)
    registry.all_ops.cache_clear()
    with pytest.raises(registry.OpError, match="closed vocabulary"):
        registry.all_ops()
    registry.all_ops.cache_clear()


def test_open_params_are_refused(tmp_path, monkeypatch):
    """`additionalProperties:false` is not a style preference. An open params bag re-admits argv across
    the seam — and argv is code: it publishes the algorithm verbatim and blinds the placement gate."""
    d = tmp_path / "ops"
    d.mkdir()
    decl = json.loads((CONTRACTS / "ops" / "media.scale.json").read_text())
    decl["params"]["additionalProperties"] = True
    (d / "media.scale.json").write_text(json.dumps(decl))
    monkeypatch.setattr(registry, "OPS_DIR", d)
    registry.all_ops.cache_clear()
    with pytest.raises(registry.OpError, match="CLOSED"):
        registry.all_ops()
    registry.all_ops.cache_clear()


def test_params_are_validated_against_the_declaration():
    registry.validate_params("media.scale", {"height": 960, "encode_profile": "proxy"})
    with pytest.raises(registry.OpError):          # unknown key — closed schema
        registry.validate_params("media.scale", {"height": 960, "encode_profile": "proxy", "argv": ["x"]})
    with pytest.raises(registry.OpError):          # odd height — encoders need even dimensions
        registry.validate_params("media.scale", {"height": 961, "encode_profile": "proxy"})
    with pytest.raises(registry.OpError):          # profile is a NAMED tier, not free text
        registry.validate_params("media.scale", {"height": 960, "encode_profile": "-crf 18"})


# ── the constructional placement gate ────────────────────────────────────────────────────────────

def test_judgement_op_is_refused_on_the_pod(tmp_path, monkeypatch):
    """THE invariant. A judgement op DECIDES rather than executes, and is control-plane-only.

    Stage-level placement could not express this: a stage that is allowed on a pod carries every task in
    it, including the ones that decide. Per-op `judgement: true` replaces that with a refusal at the point
    of execution, so a routing table cannot authorise what this check forbids."""
    d = tmp_path / "ops"
    d.mkdir()
    decl = json.loads((CONTRACTS / "ops" / "media.scale.json").read_text())
    decl.update({"op": "cut.decide", "judgement": True})
    (d / "cut.decide.json").write_text(json.dumps(decl))
    monkeypatch.setattr(registry, "OPS_DIR", d)
    registry.all_ops.cache_clear()
    with pytest.raises(registry.OpError, match="judgement"):
        registry.assert_pod_safe("cut.decide")
    registry.all_ops.cache_clear()


def test_key_needing_op_is_refused_on_the_pod(tmp_path, monkeypatch):
    """The pod is KEYLESS — its entire credential surface is CP_URL + JOB_TOKEN. An op declaring it needs
    a secret cannot be handed to it, whatever the routing says."""
    d = tmp_path / "ops"
    d.mkdir()
    decl = json.loads((CONTRACTS / "ops" / "mograph.render.json").read_text())
    decl.update({"op": "llm.author", "needs": ["llm", "keys"]})
    (d / "llm.author.json").write_text(json.dumps(decl))
    monkeypatch.setattr(registry, "OPS_DIR", d)
    registry.all_ops.cache_clear()
    with pytest.raises(registry.OpError, match="KEYLESS"):
        registry.assert_pod_safe("llm.author")
    registry.all_ops.cache_clear()


def test_shipped_ops_are_all_pod_safe():
    for name in registry.all_ops():
        registry.assert_pod_safe(name)


# ── the envelope ─────────────────────────────────────────────────────────────────────────────────

_PACK = {"url": "https://x/p.tar", "sha256": "a" * 64, "size": 10}


def _step(sid, **kw):
    base = {"id": sid, "op": "media.scale", "needs": [],
            "params": {"height": 960, "encode_profile": "proxy"},
            "inputs": [{"port": "src", "url": "https://x/in.mp4"}],
            "outputs": [{"port": "dst", "url": "https://x/out.mp4"}]}
    base.update(kw)
    return base


def test_chain_rejects_a_cycle():
    """A cycle would deadlock the runner. On a rented box that bills by the second, a hang is far more
    expensive than a validation error that names the steps."""
    with pytest.raises(ValidationError, match="cycle"):
        OpChain(job_id="j", pack=_PACK, steps=[
            _step("a", needs=["b"]), _step("b", needs=["a"])])


def test_reading_a_step_requires_depending_on_it():
    """Binding `from_step` without listing it in `needs` lets the runner schedule reader and writer
    concurrently — the reader then races a half-written file. That is a race, not an optimisation."""
    with pytest.raises(ValidationError, match="race"):
        OpChain(job_id="j", pack=_PACK, steps=[
            _step("a"),
            _step("b", needs=[], inputs=[{"port": "src", "from_step": "a"}])])


def test_binding_names_exactly_one_source():
    with pytest.raises(ValidationError, match="exactly one"):
        OpChain(job_id="j", pack=_PACK, steps=[
            _step("a", inputs=[{"port": "src", "url": "https://x/i", "from_step": "b"}])])
    with pytest.raises(ValidationError, match="exactly one"):
        OpChain(job_id="j", pack=_PACK, steps=[_step("a", inputs=[{"port": "src"}])])


def test_pod_job_ops_envelope_is_exclusive():
    """The envelope stays closed and one-block-per-type as `ops` joins infer/render."""
    chain = OpChain(job_id="j", pack=_PACK, steps=[_step("a")]).model_dump(mode="json")
    PodJob(type="ops", chain=chain)
    with pytest.raises(ValidationError):
        PodJob(type="ops")                                   # missing its block
    with pytest.raises(ValidationError):
        PodJob(type="render", chain=chain)                   # wrong block for the type


def test_a_new_op_adds_no_model_to_the_envelope():
    """`params` is a plain dict validated against the registry, so adding an op does NOT add a Pydantic
    model here. If this ever fails, someone started mirroring op surfaces into the envelope."""
    import podagent.models as m

    per_op = [n for n in dir(m)
              if n.endswith("Params") and n not in {"ClipRankParams", "AlignParams", "FaceProbeParams"}]
    assert per_op == [], f"per-op envelope models appeared: {per_op}"


# ── the pack ─────────────────────────────────────────────────────────────────────────────────────

def _make_pack(tmp_path: Path, body: str) -> tuple[object, Path]:
    src = tmp_path / "montyops"
    src.mkdir(parents=True, exist_ok=True)
    (src / "__init__.py").write_text("")
    (src / "demo.py").write_text(body)
    tar = tmp_path / "pack.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(src, arcname="montyops")
    sha = hashlib.sha256(tar.read_bytes()).hexdigest()

    class Ref:
        url = tar.as_uri()
        sha256 = sha
        size = tar.stat().st_size
    return Ref(), tar


def test_pack_is_verified_by_digest(tmp_path, monkeypatch):
    """A tar whose bytes do not match the declared sha256 must never be extracted. The pod is handed a URL
    by a control plane it trusts, but 'verify then extract' is what makes a swapped or truncated object a
    loud failure instead of mystery-wrong output."""
    ref, tar = _make_pack(tmp_path, "def run(**kw): pass\n")
    monkeypatch.setenv(pack.PACK_CACHE_ENV, str(tmp_path / "cache"))
    tar.write_bytes(tar.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        pack.ensure(ref)


def test_pack_activates_and_resolves_a_handler(tmp_path, monkeypatch):
    pack.reset_for_tests()
    ref, _ = _make_pack(tmp_path, "def run(*, params, inputs, outputs):\n    return params\n")
    monkeypatch.setenv(pack.PACK_CACHE_ENV, str(tmp_path / "cache"))
    pack.activate(ref)
    fn = pack.resolve("montyops.demo:run")
    assert fn(params={"k": 1}, inputs={}, outputs={}) == {"k": 1}
    pack.reset_for_tests()


def test_switching_packs_in_one_process_is_refused(tmp_path, monkeypatch):
    """importlib caches by module NAME, so a second pack would be shadowed by the first and the job would
    run the wrong handlers while reporting success. Refuse loudly instead."""
    pack.reset_for_tests()
    monkeypatch.setenv(pack.PACK_CACHE_ENV, str(tmp_path / "cache"))
    a, _ = _make_pack(tmp_path / "a", "def run(**kw): return 'a'\n")
    b, _ = _make_pack(tmp_path / "b", "def run(**kw): return 'b'\n")
    pack.activate(a)
    with pytest.raises(pack.PackError, match="already activated"):
        pack.activate(b)
    pack.reset_for_tests()


# ── dispatch is a lookup ─────────────────────────────────────────────────────────────────────────

def test_dispatch_has_no_per_op_branch():
    """Adding a tool must not cost an `if/elif` in main.py. The op names its handler in the registry, the
    pack provides it, dispatch is a lookup — so no op NAME may appear in the dispatcher at all."""
    src = (Path(__file__).resolve().parents[1] / "podagent" / "main.py").read_text()
    body = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    for name in registry.all_ops():
        assert name not in body, f"main.py branches on op {name!r} — dispatch must stay a registry lookup"


# ── envelope schema is the SSOT, and `ops` was added ADDITIVELY ──────────────────────────────────

_POD_JOB_SCHEMA = json.loads((CONTRACTS / "pod_job.schema.json").read_text())


def test_envelope_schema_knows_every_type_the_model_accepts():
    """The SCHEMA is the SSOT (it ships to the public repo and Go validates against it); the Pydantic
    model is its mirror. They drifted once already: the model learned `ops` while the schema still
    enumerated only infer/render, so a real ops envelope would have been rejected upstream while every
    model-level test stayed green. Derive both sides and compare instead of trusting either."""
    import typing

    schema_types = set(_POD_JOB_SCHEMA["properties"]["type"]["enum"])
    model_types = set(typing.get_args(PodJob.model_fields["type"].annotation))
    assert schema_types == model_types, f"envelope type drift: schema={schema_types} model={model_types}"


def test_every_type_has_its_block_in_the_schema():
    """A type in the enum with no matching property is the same drift one step later."""
    props = set(_POD_JOB_SCHEMA["properties"])
    for t, block in (("infer", "request"), ("render", "spec"), ("ops", "chain")):
        assert t in _POD_JOB_SCHEMA["properties"]["type"]["enum"]
        assert block in props, f"type {t!r} has no {block!r} property in the envelope schema"


def test_adding_ops_did_not_change_any_pre_existing_envelope():
    """THE justification for leaving contracts/VERSION at 5. `ops` is additive only if every envelope that
    validated before still validates BYTE-FOR-BYTE unchanged — so assert it against the goldens that
    predate this change rather than asserting it in a comment."""
    jsonschema = pytest.importorskip("jsonschema")
    v = jsonschema.Draft202012Validator(_POD_JOB_SCHEMA)
    for name in ("pod_job.infer.json", "pod_job.render.json"):
        doc = json.loads((CONTRACTS / "examples" / name).read_text())
        errs = sorted(v.iter_errors(doc), key=str)
        assert not errs, f"{name} stopped validating when `ops` was added — that is a BREAKING change: {errs[0].message}"
    # …and the exclusivity that made those goldens meaningful must still bite.
    bad = json.loads((CONTRACTS / "examples" / "invalid" / "pod_job.infer-carries-spec.json").read_text())
    assert next(v.iter_errors(bad), None) is not None, "envelope exclusivity regressed"


def test_ops_envelope_golden_round_trips_through_the_model():
    """Schema-valid must mean model-valid. A golden the schema accepts and the model rejects is the same
    drift wearing the other hat."""
    doc = json.loads((CONTRACTS / "examples" / "pod_job.ops.json").read_text())
    job = PodJob.model_validate(doc)
    assert job.type == "ops" and job.chain is not None
    assert job.chain.steps[0].op == "media.scale"
    registry.validate_params(job.chain.steps[0].op, job.chain.steps[0].params)
