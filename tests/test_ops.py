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


def test_handler_budget_vocabulary_is_closed(tmp_path, monkeypatch):
    """A scheduling class is a resource claim, not an arbitrary per-op knob: typos must fail at image load."""
    d = tmp_path / "ops"
    d.mkdir()
    decl = json.loads((CONTRACTS / "ops" / "media.fetch.json").read_text())
    decl["budget"] = "unbounded"
    (d / "media.fetch.json").write_text(json.dumps(decl))
    monkeypatch.setattr(registry, "OPS_DIR", d)
    registry.all_ops.cache_clear()
    with pytest.raises(registry.OpError, match="budget outside the closed vocabulary"):
        registry.all_ops()
    registry.all_ops.cache_clear()


def test_media_fetch_and_fused_image_ops_are_transport_budgeted_shipped_ops():
    ops = registry.all_ops()
    assert ops["media.fetch"].budget == "transport"
    assert ops["media.image_tile"].budget == "transport"
    assert ops["media.image_filmstrip"].budget == "transport"
    assert all(op.budget == "cpu" for name, op in ops.items()
               if name not in {"media.fetch", "media.image_tile", "media.image_filmstrip"})


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
    PodJob(type="ops", session_id="s", corr_id="c", chain=chain)
    with pytest.raises(ValidationError):
        PodJob(type="ops", session_id="s", corr_id="c")      # missing its block
    with pytest.raises(ValidationError):
        PodJob(type="render", session_id="s", corr_id="c", chain=chain)  # wrong block for the type


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


# ── an OPTIONAL output is a verdict the runner must let through ──────────────────────────────────

def test_a_missing_optional_output_is_allowed_and_a_missing_required_one_is_not(tmp_path, monkeypatch):
    """The runner enforces the port's own `optional` flag, and an op that reports through a sidecar depends
    on BOTH halves of that. NEGATIVE: drop the `optional` check in runner._run_step and a handler that
    legitimately produced nothing (a host that cannot do the work, art that needed none) kills the chain
    instead of returning the verdict that says which — and a caller that is only told 'no file' cannot tell
    'this host cannot' from 'there was nothing to do'.
    """
    from podagent.ops import runner

    op = next(o for o in registry.all_ops().values()
              if any(p.optional for p in o.outputs) and any(not p.optional for p in o.outputs))
    required = next(p.id for p in op.outputs if not p.optional)
    optional = next(p.id for p in op.outputs if p.optional)

    src = tmp_path / "in.bin"
    src.write_bytes(b"x")
    step = type("S", (), {
        "id": "s", "op": op.op, "params": {}, "needs": [],
        "inputs": [type("B", (), {"port": p.id, "url": None, "from_step": None, "path": str(src)})()
                   for p in op.inputs],
        "outputs": [type("B", (), {"port": required, "url": None, "urls": None})()]})()
    ws = runner.Workspace(tmp_path)

    def _handler(*, params, inputs, outputs):
        outputs[required].write_text("{}")            # the optional one deliberately stays absent

    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler)
    out = runner._run_step(step, ws, {})
    assert not out[optional].exists(), "the optional port must be allowed to stay empty"

    monkeypatch.setattr(runner.pack, "resolve", lambda h: (lambda **kw: None))
    with pytest.raises(runner.ChainError, match=required):
        runner._run_step(step, runner.Workspace(tmp_path / "second"), {})


# ── one keep-list, two renders, one timeline ─────────────────────────────────────────────────────

def test_the_two_cut_renders_agree_on_every_param_they_share():
    """`cut.audio` composes the decision axis and `cut.apply` composes the picture — from the SAME keep-list,
    on the SAME frame grid, with the SAME joins. Every offset measured on the audio is later compared against
    the picture, so the two declarations may not disagree about what a param means or is allowed to be.

    NEGATIVE: let one side widen `xfade`, or default `speed` differently, and one keep-list renders two
    timelines that differ by a fade. Nothing errors — the caller simply measures its cut on a clock the film
    does not use, and every remap that rides those offsets lands late.
    """
    def props(name: str) -> dict:
        return json.loads((CONTRACTS / "ops" / name).read_text())["params"]["properties"]

    audio, picture = props("cut.audio.json"), props("cut.apply.json")
    shared = set(audio) & set(picture)
    assert {"keep", "fps_grid", "speed", "xfade"} <= shared, \
        "the two renders must share the keep-list, the grid and the joins"
    for key in sorted(shared):
        strip = lambda d: {k: v for k, v in d.items() if k != "description"}   # noqa: E731
        assert strip(audio[key]) == strip(picture[key]), f"the two cut renders disagree about {key!r}"
    # the audio axis carries no pixels, so a picture-only knob there would be a knob on nothing
    assert not ({"crf", "cpu", "no_hwaccel", "max_h"} & set(audio))


def test_neither_cut_render_may_claim_judgement():
    """Both take an ALREADY-COMPOSED keep-list. A cut op that declared judgement would be refused the pod by
    assert_pod_safe — and the mechanics genuinely belong beside the bytes; it is the list that must not."""
    ops = registry.all_ops()
    for name in ("cut.audio", "cut.apply"):
        assert ops[name].judgement is False, f"{name} decides nothing; it applies a decision"


# ── the claim: an unrunnable chain must cost NOTHING but the claim ───────────────────────────────

def _older_image(monkeypatch, without: str):
    """The registry of an image that predates `without` — the exact state of the box in the incident."""
    kept = {k: v for k, v in registry.all_ops().items() if k != without}
    monkeypatch.setattr(registry, "all_ops", lambda: kept)
    return kept


def test_an_op_this_image_lacks_is_refused_before_any_step_runs(monkeypatch):
    """THE INCIDENT. A chain whose LAST step names an op this image does not carry ran its earlier steps
    first — 236 s of transcode on a rented box — and only then reported `unknown op 'media.audio'`.

    NEGATIVE: drop `preflight_chain` from run_chain and the first handler below runs, which is the whole
    cost this refusal exists to avoid. Everything knowable at claim is decided at claim.
    """
    from podagent.ops import runner

    _older_image(monkeypatch, "media.audio")
    ran: list[str] = []
    monkeypatch.setattr(runner.pack, "activate", lambda p: ran.append("pack"))
    monkeypatch.setattr(runner.pack, "resolve", lambda h: (lambda **kw: ran.append("handler")))
    chain = OpChain(job_id="j", pack=_PACK, steps=[
        _step("scale"),
        _step("audio", op="media.audio", needs=["scale"], params={},
              outputs=[{"port": "mp3", "url": "https://x/a.mp3"}])])

    cp = type("CP", (), {
        "send_event": staticmethod(lambda _ev, wait=False: True),
        "send_result": staticmethod(lambda _result, wait=True: True),
    })()
    with pytest.raises(runner.ChainError, match="media.audio"):
        runner.run_chain(chain, cp=cp)
    assert ran == [], "nothing may be fetched, decoded or even unpacked for a chain this image cannot run"


def test_the_refusal_names_the_image_and_what_it_does_hold(monkeypatch):
    """`unknown op X` alone is unactionable: the reader's next question is always WHICH IMAGE is this, and
    the answer decides whether to bump the pin or ship the op."""
    from podagent.ops import runner

    kept = _older_image(monkeypatch, "media.audio")
    monkeypatch.setenv("POD_IMAGE_TAG", "v0.6.0")
    chain = OpChain(job_id="j", pack=_PACK, steps=[_step("a", op="media.audio", params={},
                                                         outputs=[{"port": "mp3", "url": "https://x/a.mp3"}])])
    with pytest.raises(runner.ChainError) as e:
        runner.preflight_chain(chain)
    said = str(e.value)
    assert "v0.6.0" in said and "media.audio" in said
    assert str(sorted(kept)) in said, "the registry it DOES hold is what tells you which side moved"


def test_bad_params_are_refused_at_claim_too(monkeypatch):
    """Same principle, one layer down: a param the declaration forbids is knowable before the first byte
    moves, so it may not be discovered by the step that finally validates it."""
    from podagent.ops import runner

    pytest.importorskip("jsonschema")
    chain = OpChain(job_id="j", pack=_PACK, steps=[
        _step("a"), _step("b", needs=["a"], params={"height": 960, "encode_profile": "not-a-profile"})])
    with pytest.raises(registry.OpError, match="invalid params"):
        runner.preflight_chain(chain)


def test_a_chain_this_image_can_run_passes_preflight():
    """The gate must not be a wall: every op the shipped chains name is in this image, so preflight is a
    no-op on a correct pairing."""
    from podagent.ops import runner

    runner.preflight_chain(OpChain(job_id="j", pack=_PACK, steps=[_step("a"), _step("b", needs=["a"])]))


# ── ONE step, N addressable outputs (runner.ARITY_WHY) ───────────────────────────────────────────

def _frames_step(sid="g", n=3, **kw):
    base = {"id": sid, "op": "media.frames", "needs": [],
            "params": {"positions": [i / n for i in range(n)], "width": 384, "height": 384},
            "inputs": [{"port": "src", "url": "https://x/in.mp4"}],
            "outputs": [{"port": "frames", "urls": [f"https://x/g{i}.png" for i in range(n)]}]}
    base.update(kw)
    return base


def test_a_list_port_binds_one_address_per_file():
    """The arity of a step was the arity of its transport: one path per declared port, so N frames of ONE
    decode needed N steps and therefore N decodes. NEGATIVE: bind `urls` to a port the op declares as ONE
    file and the addresses name nothing — that must be refused, not silently truncated to the first."""
    from podagent.ops import runner

    chain = OpChain(job_id="j", pack=_PACK, steps=[_frames_step()])
    runner.preflight_chain(chain)

    single = OpChain(job_id="j", pack=_PACK, steps=[
        _step("a", outputs=[{"port": "dst", "urls": ["https://x/1.mp4", "https://x/2.mp4"]}])])
    with pytest.raises(runner.ChainError, match="ONE file"):
        runner.preflight_chain(single)


def test_a_list_port_bound_with_a_single_url_is_refused_at_claim():
    """NEGATIVE, and the reason it is at claim: the count of files comes from the BINDING, so a `url` where
    `urls` belongs is a decode whose results have nowhere to go — knowable before the fetch."""
    from podagent.ops import runner

    chain = OpChain(job_id="j", pack=_PACK, steps=[
        _frames_step(outputs=[{"port": "frames", "url": "https://x/one.png"}])])
    with pytest.raises(runner.ChainError, match="must bind `urls`"):
        runner.preflight_chain(chain)


def test_a_list_port_nobody_addressed_is_refused():
    """NEGATIVE: unbound, nothing says HOW MANY — and a decode whose output nothing reads is rent spent on
    nothing. The arity is the binding's, never the handler's guess."""
    from podagent.ops import runner

    chain = OpChain(job_id="j", pack=_PACK, steps=[_frames_step(outputs=[])])
    with pytest.raises(runner.ChainError, match="must be bound with `urls`"):
        runner.preflight_chain(chain)


def test_a_later_step_may_not_read_a_list_port():
    """NEGATIVE: `from_step` names a port, not an element, so 'which of the N' is a question the binding
    cannot ask — and a runner that guessed would hand a later step the wrong picture, silently."""
    from podagent.ops import runner

    chain = OpChain(job_id="j", pack=_PACK, steps=[
        _frames_step("g"),
        _step("s", needs=["g"], op="media.sheet",
              params={"cols": 1, "cell_w": 384, "cell_h": 384, "gap": 0, "head": 0, "caption_h": 0,
                      "plate": True, "bg": [18, 18, 18], "captions": [[]]},
              inputs=[{"port": "tile0", "from_step": "g", "from_port": "frames"}],
              outputs=[{"port": "sheet", "url": "https://x/s.png"}])])
    with pytest.raises(runner.ChainError, match="LIST port"):
        runner.preflight_chain(chain)


def test_an_input_may_not_bind_urls():
    """An input is ONE file: N addresses on one input port have no defined order to be read in, and the
    runner would have to invent one. N inputs are N ports."""
    with pytest.raises(ValidationError, match="only an output may"):
        OpChain(job_id="j", pack=_PACK, steps=[
            _step("a", inputs=[{"port": "src", "urls": ["https://x/1.mp4", "https://x/2.mp4"]}])])


def test_a_binding_still_names_exactly_one_source():
    with pytest.raises(ValidationError, match="exactly one of url/urls/from_step/path"):
        OpChain(job_id="j", pack=_PACK, steps=[
            _step("a", outputs=[{"port": "dst", "url": "https://x/1.mp4", "urls": ["https://x/2.mp4"]}])])


def test_every_element_of_a_list_port_is_moved_to_its_own_address(tmp_path, monkeypatch):
    """The whole point, end to end through the runner: ONE handler call, N destination paths, N uploads —
    index i to address i. NEGATIVE: hand the handler a single path and the step can deliver one frame."""
    from podagent.ops import runner

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(runner, "upload", lambda src, url: sent.append((src.name, url)))
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)

    seen: dict = {}

    def _handler(*, params, inputs, outputs):
        seen["paths"] = list(outputs["frames"])
        for i, p in enumerate(outputs["frames"]):
            p.write_text(f"frame {i}")

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    step = OpChain(job_id="j", pack=_PACK, steps=[
        _frames_step(n=4, inputs=[{"port": "src", "path": str(src)}])]).steps[0]

    out = runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})
    assert len(seen["paths"]) == 4 and len(set(seen["paths"])) == 4, "the handler got one path per address"
    assert [u for _n, u in sent] == [f"https://x/g{i}.png" for i in range(4)]
    assert [Path(p).read_text() for p in out["frames"]] == [f"frame {i}" for i in range(4)]


def test_an_element_the_handler_did_not_write_is_simply_not_moved(tmp_path, monkeypatch):
    """`frames` is optional PER ELEMENT: one frame that will not render must cost that frame, not the batch
    — the same polarity the single-file strip already had. NEGATIVE: require every element and one dead
    position takes its siblings down."""
    from podagent.ops import runner

    sent: list[str] = []
    monkeypatch.setattr(runner, "upload", lambda src, url: sent.append(url))
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: (
        lambda *, params, inputs, outputs: [p.write_text("f") for i, p in enumerate(outputs["frames"])
                                            if i != 1]))
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    step = OpChain(job_id="j", pack=_PACK, steps=[
        _frames_step(n=3, inputs=[{"port": "src", "path": str(src)}])]).steps[0]

    out = runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})
    assert sent == ["https://x/g0.png", "https://x/g2.png"], "a missing element must not raise, and must "\
                                                             "not shift the addresses of the others"
    assert not Path(out["frames"][1]).exists()


def test_an_element_that_will_not_upload_fails_the_step_and_names_which(tmp_path, monkeypatch):
    """The failure polarity a two-port step already had: `upload` retries and then raises, and the raise ends
    the STEP. NEGATIVE — the two halves that matter: (1) what already landed STAYS landed, so a caller that
    addresses its files by content re-cuts only the holes; (2) elements after the failure are NOT attempted,
    because an address the store refused three times with backoff will refuse the next one too and a rented
    box may not spend its lease proving it."""
    from podagent.ops import runner

    sent: list[str] = []

    def _upload(src, url):
        if url.endswith("g2.png"):
            raise RuntimeError("503 from the store")
        sent.append(url)

    monkeypatch.setattr(runner, "upload", _upload)
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: (
        lambda *, params, inputs, outputs: [p.write_text("f") for p in outputs["frames"]]))
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    step = OpChain(job_id="j", pack=_PACK, steps=[
        _frames_step(n=5, inputs=[{"port": "src", "path": str(src)}])]).steps[0]

    with pytest.raises(runner.ChainError, match=r"frames'\[2\] of 5"):
        runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})
    assert sent == ["https://x/g0.png", "https://x/g1.png"], "what landed before the failure must stay, and "\
                                                             "nothing after it may be attempted"
