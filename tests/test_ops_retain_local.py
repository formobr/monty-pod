"""`retain: local` — the master that stops being PUT so the same box can read it back
(runner.RETAIN_WHY, inputcache.RETENTION_WHY).

Every test here is NEGATIVE in the sense docs/TESTING.md means: each was watched fail with the mechanism it
asserts removed. The two that matter most are the refusals — a retained object exists on exactly ONE worker,
and the whole licence for skipping the upload is that the two ways to be wrong about that (a 404 minutes from
its cause, a silent re-render of work already done) are replaced by a sentence naming both workers.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from podagent import identity
from podagent.models import OpChain, OpsPackRef
from podagent.ops import inputcache, runner

_PACK = OpsPackRef(url="https://x/p.tgz", sha256="a" * 64)
MASTER = "https://r2.example/monty/jobs/j/slug.cut.mp4?X-Amz-Signature=aaa"
MASTER_REMINTED = "https://r2.example/monty/jobs/j/slug.cut.mp4?X-Amz-Signature=zzz&X-Amz-Expires=900"
ME = "fleet-worker-7"
SOMEONE_ELSE = "fleet-worker-9"


def _bearer(worker: str) -> str:
    """A JOB_TOKEN shaped exactly like the minted one, signature deliberately junk: naming yourself is not
    authorising yourself, and nothing on this path verifies (podagent.identity)."""
    body = base64.urlsafe_b64encode(
        json.dumps({"jid": worker, "tid": "t", "sid": "fleet", "exp": 1}).encode()).rstrip(b"=").decode()
    return f"{body}.not-a-real-signature"


@pytest.fixture(autouse=True)
def _one_pod(tmp_path, monkeypatch):
    """One worker identity and one private cache root per test. The identity arrives the way it does in
    production — as the bearer's `jid` claim — because a test seam of its own would be the second vocabulary
    this whole mechanism exists to avoid."""
    monkeypatch.setenv(inputcache.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.delenv(inputcache.DISABLE_ENV, raising=False)
    monkeypatch.setenv("JOB_TOKEN", _bearer(ME))
    monkeypatch.delenv("MONTY_JOB_TOKEN", raising=False)
    identity._reset()
    inputcache._locks.clear()
    inputcache._slot_locks.clear()
    yield
    identity._reset()


def _cut_step(sid="cut", *, retain: bool):
    out = {"port": "dst", "url": MASTER}
    if retain:
        out["retain"] = "local"
    return {"id": sid, "op": "media.scale", "needs": [],
            "params": {"height": 960, "encode_profile": "browser"},
            "inputs": [{"port": "src", "url": "https://x/in.mp4"}],
            "outputs": [out]}


def _reader_step(sid="read", *, retained_on: str | None):
    src: dict = {"port": "src", "url": MASTER_REMINTED}
    if retained_on is not None:
        src["retained_on"] = retained_on
    return {"id": sid, "op": "media.scale", "needs": [],
            "params": {"height": 480, "encode_profile": "browser"},
            "inputs": [src],
            "outputs": [{"port": "dst", "url": "https://x/small.mp4"}]}


@pytest.fixture()
def wired(monkeypatch):
    """A handler that writes every declared output, and a store that RECORDS instead of transferring."""
    puts: list[str] = []
    gets: list[str] = []
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.registry, "assert_pod_safe", lambda *a, **k: None)
    monkeypatch.setattr(runner.pack, "resolve", lambda _h: (
        lambda *, params, inputs, outputs: [p.write_bytes(b"master-bytes") for p in outputs.values()]))
    monkeypatch.setattr(runner, "upload", lambda src, url, ct=None: puts.append(url))

    def _download(url: str, dest: Path) -> None:
        gets.append(url)
        dest.write_bytes(b"from-the-store")

    monkeypatch.setattr(runner, "download", _download)
    runner._reset_step_slots()
    yield puts, gets
    runner._reset_step_slots()


# ── the deliverable ──────────────────────────────────────────────────────────────────────────────

def test_a_retained_output_is_not_uploaded_and_the_next_chain_reads_it_off_this_box(tmp_path, wired):
    """THE deliverable, both halves at once: nothing crosses on the way out, and a LATER chain — its own
    workspace, its own re-minted presign — binds the same object and gets the produced bytes without a GET.
    Drop the retain branch from `_output_arms` and the PUT reappears; drop `fetch_retained` and the second
    chain downloads whatever the store happens to hold."""
    puts, gets = wired
    producer = OpChain(job_id="j", pack=_PACK, steps=[_cut_step(retain=True)]).steps[0]
    runner._run_step(producer, runner.Workspace(tmp_path / "chain1"), {})
    assert puts == [], f"a retained output crossed the wire: {puts}"

    reader = OpChain(job_id="j2", pack=_PACK, steps=[_reader_step(retained_on=ME)]).steps[0]
    seen: dict = {}

    def _handler(*, params, inputs, outputs):
        seen["src"] = inputs["src"].read_bytes()
        outputs["dst"].write_bytes(b"x")

    runner.pack.resolve = lambda _h: _handler          # noqa: E731 — restored by the `wired` fixture teardown
    runner._run_step(reader, runner.Workspace(tmp_path / "chain2"), {})
    assert seen["src"] == b"master-bytes", "the next chain did not read the retained master"
    assert gets == ["https://x/in.mp4"], f"the retained master was fetched from the store: {gets}"


def test_a_step_with_no_retain_flag_uploads_exactly_as_it_always_did(tmp_path, wired):
    """NEGATIVE for the final tier, which is every tier that did not ask. The flag is the caller's claim and
    nothing here infers it: an unmarked output must still pay its PUT."""
    puts, _gets = wired
    step = OpChain(job_id="j", pack=_PACK, steps=[_cut_step(retain=False)]).steps[0]
    runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})
    assert puts == [MASTER]
    assert inputcache.retained_holder(MASTER) is None, "an uploaded object must stay evictable"


# ── the refusals, which are what makes skipping the PUT legal at all ──────────────────────────────

def test_a_bind_on_another_workers_retained_object_is_refused_by_name_before_anything_runs(tmp_path):
    """THE refusal. Both workers are NAMED and the chain dies at PREFLIGHT — before the rent, the pull and
    the decode that a 404 at bind time would have charged for. Remove the preflight check and the same chain
    reaches `_bind_inputs`; remove that one too and it 404s with no idea whose box the bytes are on."""
    chain = OpChain(job_id="j", pack=_PACK, steps=[_reader_step(retained_on=SOMEONE_ELSE)])
    with pytest.raises(runner.ChainError) as e:
        runner.preflight_chain(chain)
    said = str(e.value)
    assert f"retained on worker {SOMEONE_ELSE}" in said and f"claimed on worker {ME}" in said
    assert "route to the same worker" in said


def test_the_same_worker_with_the_entry_gone_is_also_refused_rather_than_re_rendered(tmp_path, wired):
    """NEGATIVE: identity alone is not the proof — a restarted agent or a wiped cache is the same worker
    with none of the bytes. Nothing uploaded this object, so the only alternatives to a named refusal are a
    404 or paying for the render twice."""
    reader = OpChain(job_id="j", pack=_PACK, steps=[_reader_step(retained_on=ME)]).steps[0]
    with pytest.raises(runner.ChainError, match="no longer holds"):
        runner._run_step(reader, runner.Workspace(tmp_path / "ws"), {})
    _puts, gets = wired
    assert gets == [], "a missing retained object must not fall back to a download"


def test_a_retention_that_cannot_be_adopted_fails_the_step_instead_of_going_quiet(tmp_path, monkeypatch,
                                                                                 wired):
    """NEGATIVE: adoption is the ONLY thing keeping a retained object in existence. If it fails and the step
    still reports ok, the chain has delivered a file that is nowhere — the exact class of swallowed error
    the loud-failure law exists for."""
    monkeypatch.setenv(inputcache.DISABLE_ENV, "1")
    step = OpChain(job_id="j", pack=_PACK, steps=[_cut_step(retain=True)]).steps[0]
    with pytest.raises(runner.ChainError, match="retain=local"):
        runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})


# ── the cache side: a retained entry is the truth, not a copy of it ───────────────────────────────

def test_the_pruner_never_evicts_what_nothing_uploaded(tmp_path):
    """NEGATIVE: eviction is cheap because a miss costs one download. For a retained object it costs the
    object. Drop BOTH RETAINED checks (`_entries` and the re-check under the slot lock, which cover each
    other) and a 24 GB cap deletes the master mid-lease."""
    src = tmp_path / "m.bin"
    src.write_bytes(b"m" * 4096)
    inputcache.adopt_local(MASTER, src, holder=ME)
    other = tmp_path / "o.bin"
    other.write_bytes(b"o" * 4096)
    inputcache.get("https://r2.example/other.bin", lambda _u, d: d.write_bytes(b"o" * 4096),
                   lease=tmp_path / "lease")

    inputcache.prune(keep_bytes=0)
    assert inputcache.retained_holder(MASTER) == ME, "the retained master was evicted"
    assert inputcache.retained_holder("https://r2.example/other.bin") is None


def test_a_real_put_over_a_retained_key_hands_eviction_back(tmp_path):
    """Retention is a claim about THIS object's only copy. Once something genuinely uploads it the claim is
    over, and leaving the marker behind would make the slot unevictable forever."""
    src = tmp_path / "m.bin"
    src.write_bytes(b"m" * 32)
    inputcache.adopt_local(MASTER, src, holder=ME)
    inputcache.upload_and_adopt(MASTER_REMINTED, src, lambda _s, _u: None)
    assert inputcache.retained_holder(MASTER) is None


# ── the envelope ─────────────────────────────────────────────────────────────────────────────────

def test_retention_is_an_output_word_and_reading_one_is_an_input_word():
    """The two halves cannot be swapped: an input the handler never wrote has nothing to keep, and an output
    that has not been produced yet cannot have been retained anywhere."""
    with pytest.raises(ValidationError, match="only an output"):
        OpChain(job_id="j", pack=_PACK, steps=[{
            **_cut_step(retain=False),
            "inputs": [{"port": "src", "url": MASTER, "retain": "local"}]}])
    with pytest.raises(ValidationError, match="never where it is read"):
        OpChain(job_id="j", pack=_PACK, steps=[{
            **_cut_step(retain=False),
            "outputs": [{"port": "dst", "url": MASTER, "retained_on": ME}]}])


def test_retention_keys_on_one_object_address_and_refuses_to_key_on_none():
    """Both fields hang off the object key a single `url` carries: `urls` is N keys with one flag between
    them, and from_step/path name no key at all."""
    with pytest.raises(ValidationError, match="binds no `url`"):
        OpChain(job_id="j", pack=_PACK, steps=[{
            **_cut_step(retain=False),
            "outputs": [{"port": "dst", "urls": [MASTER], "retain": "local"}]}])
    with pytest.raises(ValidationError, match="one binding is one end"):
        OpChain(job_id="j", pack=_PACK, steps=[{
            **_cut_step(retain=False),
            "outputs": [{"port": "dst", "url": MASTER, "retain": "local", "retained_on": ME}]}])
    with pytest.raises(ValidationError):
        OpChain(job_id="j", pack=_PACK, steps=[{
            **_cut_step(retain=False),
            "outputs": [{"port": "dst", "url": MASTER, "retain": "forever"}]}])


def test_the_flag_is_additive_so_the_envelope_pin_does_not_move():
    """THE justification for leaving `contracts/VERSION` at 5, asserted rather than claimed in a comment: an
    envelope written before retention existed still validates, byte for byte unchanged."""
    contracts = Path(__file__).resolve().parents[1] / "contracts"
    doc = json.loads((contracts / "examples" / "pod_job.ops.json").read_text())
    OpChain.model_validate(doc["chain"])
    assert (contracts / "VERSION").read_text().strip() == "5"
    plain = OpChain(job_id="j", pack=_PACK, steps=[_cut_step(retain=False)])
    dumped = plain.model_dump(mode="json", exclude_none=True)
    assert "retain" not in json.dumps(dumped), "an absent flag must not appear on an old-shaped envelope"


# ── who this box is ──────────────────────────────────────────────────────────────────────────────

def test_the_worker_names_itself_the_way_the_control_plane_names_it(monkeypatch):
    """The API fences a targeted delivery on the bearer's `jid` claim and writes that same string into the
    corr→owner index. A hostname here would be a SECOND vocabulary: every cross-worker check would either
    always match or never, and both are silent wrongness."""
    monkeypatch.setenv("JOB_TOKEN", _bearer(SOMEONE_ELSE))
    identity._reset()
    assert identity.worker_identity() == SOMEONE_ELSE


def test_an_unreadable_bearer_is_not_an_identity_and_never_an_exception(monkeypatch):
    """NEGATIVE: this answers 'who am I', so it may not be the thing that takes an agent down. A dev contour
    with no minted token still gets a name, and the name is MARKED local — a retained artifact claimed under
    it can only ever be read back on the same machine, which is exactly what local means."""
    import socket

    monkeypatch.setenv("JOB_TOKEN", "not-a-token-at-all")
    monkeypatch.setattr(socket, "gethostname", lambda: "devbox")
    identity._reset()
    assert identity.worker_identity() == "local:devbox"


def test_there_is_no_override_knob_for_the_worker_name(monkeypatch):
    """NEGATIVE, and it is the pod-orphan gate's finding kept as a test: the box writes the pod's environment
    WHOLE, so an escape hatch nothing in the seam assigns can only ever be its default — a switch nobody can
    throw that LOOKS like a way to repair a mis-routed retention. Re-add one and this goes red."""
    monkeypatch.setenv("POD_WORKER_ID", "wishful-worker")
    monkeypatch.setenv("JOB_TOKEN", _bearer(ME))
    identity._reset()
    assert identity.worker_identity() == ME
