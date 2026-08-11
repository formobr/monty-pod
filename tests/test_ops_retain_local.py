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


def test_retained_bytes_are_counted_by_the_pruner_even_though_they_are_never_evicted(tmp_path):
    """THE COUNTEREXAMPLE, kept: 25 runs of a 1 GB master under 25 distinct work prefixes used to leave the
    candidate list EMPTY, so `prune` summed a total of zero, compared clean against the cap and freed nothing
    while the disk filled. Never-evicted is not the same as not-counted. Drop `held` from the total in
    `prune` and this goes red — and the real failure it stands for is ENOSPC, not an eviction."""
    for run in range(5):
        src = tmp_path / f"m{run}.bin"
        src.write_bytes(b"m" * 1000)
        inputcache.adopt_local(f"https://r2.example/monty/jobs/run{run}/other{run}.cut.mp4", src, holder=ME)
    assert inputcache.retained_bytes() == 5000

    cached = tmp_path / "c.bin"
    cached.write_bytes(b"c" * 4000)
    inputcache.get("https://r2.example/plain.bin", lambda _u, d: d.write_bytes(b"c" * 4000),
                   lease=tmp_path / "lease")

    # A cap BELOW what retention alone holds: the retained set stays and the ordinary entry is what pays.
    freed = inputcache.prune(keep_bytes=4500)
    assert freed == 4000, "the pruner did not evict against the retained weight"
    assert inputcache.retained_bytes() == 5000, "a retained object was evicted"


def test_a_newer_runs_adoption_releases_the_generation_it_supersedes(tmp_path):
    """A re-run mints a NEW work prefix, so the previous run's master is bytes with no address pointing at
    them: its readers bind their own run's key. Without this release, every re-run of a slug adds a whole
    master to a set nothing ever frees. The prefix is read off the object key's own shape — this box still
    learns no slug and no job id."""
    first, second = (f"https://r2.example/monty/jobs/{r}/slug.cut.mp4" for r in ("run-a", "run-b"))
    for url in (first, second):
        src = tmp_path / "m.bin"
        src.write_bytes(b"m" * 1000)
        inputcache.adopt_local(url, src, holder=ME)
    assert inputcache.retained_holder(first) is None, "the superseded run's master was kept"
    assert inputcache.retained_holder(second) == ME
    assert inputcache.retained_bytes() == 1000


def test_a_different_artifact_of_the_same_run_is_not_superseded(tmp_path):
    """NEGATIVE on the release rule: it keys on the BASENAME under a different prefix. Match on the prefix
    alone and one chain's second retained output would delete its own sibling."""
    src = tmp_path / "m.bin"
    src.write_bytes(b"m" * 500)
    a = "https://r2.example/monty/jobs/run-a/slug.cut.mp4"
    b = "https://r2.example/monty/jobs/run-a/slug.audio.m4a"
    inputcache.adopt_local(a, src, holder=ME)
    inputcache.adopt_local(b, src, holder=ME)
    assert inputcache.retained_holder(a) == ME and inputcache.retained_holder(b) == ME


def test_a_marker_from_a_dead_lease_is_released_rather_than_held_forever(tmp_path):
    """A retained entry whose holder is not this process is unclaimable — a reader naming that worker is
    refused here anyway — so holding its bytes is pure loss. This is the restart/re-lease case."""
    src = tmp_path / "m.bin"
    src.write_bytes(b"m" * 800)
    stale = "https://r2.example/monty/jobs/old-run/slug.cut.mp4"
    inputcache.adopt_local(stale, src, holder=SOMEONE_ELSE)
    assert inputcache.retained_bytes() == 800

    inputcache.adopt_local("https://r2.example/monty/jobs/new-run/unrelated.mp4", src, holder=ME)
    assert inputcache.retained_holder(stale) is None, "a dead lease's retained bytes were kept"
    assert inputcache.retained_bytes() == 800, "only the live entry should remain"


def test_the_ceiling_refuses_a_new_adoption_LOUDLY_instead_of_filling_the_disk(tmp_path, monkeypatch):
    """THE bound. A step that fails is recoverable; a full box takes every chain on it down with an ENOSPC
    that names nothing. The refusal quotes the tally and the ceiling, because a budget you cannot read is a
    budget you cannot be refused against. Remove the check and the disk is the only limit left."""
    monkeypatch.setenv(inputcache.MAX_GB_ENV, "0.00001")     # 10 kB cap -> 5 kB retained ceiling
    assert inputcache.retained_cap() == 5000
    src = tmp_path / "m.bin"
    src.write_bytes(b"m" * 4000)
    inputcache.adopt_local("https://r2.example/monty/jobs/run-a/one.mp4", src, holder=ME)

    big = tmp_path / "big.bin"
    big.write_bytes(b"b" * 4000)
    with pytest.raises(inputcache.RetentionUnavailable) as e:
        inputcache.adopt_local("https://r2.example/monty/jobs/run-a/two.mp4", big, holder=ME)
    said = str(e.value)
    assert "ceiling" in said and "REFUSING" in said and "without retain" in said
    assert inputcache.retained_bytes() == 4000, "a refused adoption must leave nothing behind"


def test_the_ceiling_refusal_reaches_the_caller_as_a_failed_step(tmp_path, monkeypatch, wired):
    """The refusal is only a bound if it STOPS the work: the output was not uploaded, so a step that reports
    ok after a refused adoption has delivered a file that is nowhere."""
    monkeypatch.setenv(inputcache.MAX_GB_ENV, "0.0000000001")
    step = OpChain(job_id="j", pack=_PACK, steps=[_cut_step(retain=True)]).steps[0]
    with pytest.raises(runner.ChainError, match="retain=local"):
        runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})
    puts, _gets = wired
    assert puts == [], "a refused retention must not silently fall back to an upload nobody asked for"


def test_replacing_the_SAME_object_does_not_count_twice_against_the_ceiling(tmp_path, monkeypatch):
    """NEGATIVE on the accounting: a step re-run under the same address replaces its payload in place. Count
    the incoming bytes without discounting what that key already holds and the second attempt at an
    unchanged master is refused for being its own weight."""
    monkeypatch.setenv(inputcache.MAX_GB_ENV, "0.00001")     # 5 kB retained ceiling
    src = tmp_path / "m.bin"
    src.write_bytes(b"m" * 4000)
    url = "https://r2.example/monty/jobs/run-a/one.mp4"
    inputcache.adopt_local(url, src, holder=ME)
    inputcache.adopt_local(url, src, holder=ME)
    assert inputcache.retained_bytes() == 4000


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


def test_the_terminal_names_the_holder_exactly_when_a_step_retained(tmp_path, wired):
    """The consumer half's premise: the box learns WHERE the bytes stayed only from the terminal. A retain
    whose terminal stays silent is a master nobody can bind — the engine refuses the whole preview cut on it
    (op_chains.require_retained_worker), so this seam is load-bearing on every preview."""
    _puts, _gets = wired
    kept = OpChain(job_id="j", pack=_PACK, steps=[_cut_step(retain=True)]).steps[0]
    sink_kept: list[runner.StepTiming] = []
    runner._run_step(kept, runner.Workspace(tmp_path / "a"), {}, sink_kept)
    assert sink_kept and sink_kept[0].retained_ports == ["dst"]
    assert runner.terminal_retained_on(sink_kept) == ME

    plain = OpChain(job_id="j2", pack=_PACK, steps=[_cut_step(sid="cut2", retain=False)]).steps[0]
    sink_plain: list[runner.StepTiming] = []
    runner._run_step(plain, runner.Workspace(tmp_path / "b"), {}, sink_plain)
    assert sink_plain and sink_plain[0].retained_ports == []
    assert runner.terminal_retained_on(sink_plain) is None
