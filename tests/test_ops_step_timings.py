"""THE TERMINAL MUST CARRY WHAT ONLY THIS BOX CAN SEE: the seconds each step spent.

Measured on the control plane 2026-08-04/05: a b-roll fetch leg cost 4.13 s at the median and 33.61 s at the
slowest one that DELIVERED, for the same ~5-7 MB interior window of a 4K master. 7 MB in 33 s is 0.2 MB/s,
which no CDN does — so the seconds were probably not the wire. The box could not say, because it books one
span per chain WALL and everything from its own enqueue to the terminal's trip home lives inside it. The
split exists ONLY here, so it is measured here and it rides the terminal home.

Every test is a NEGATIVE test in the sense docs/TESTING.md means: each was watched fail with its mechanism
reverted. Hermetic — no control plane, no network, no pack fetch (runner.STEP_TIMING_WHY).
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

# The agent's own suite runs with `cd pod-agent`; the engine's one door (scripts/test.sh) runs from the
# superproject. Spelled here so this file is runnable from either, exactly as the engine-side tests that
# reach into the submodule already do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from podagent.ops import pack, registry, runner  # noqa: E402


class _CP:
    """Collects the events the runner posts. The terminal is the last one."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.results: list[dict] = []

    def send_event(self, payload: dict, *, wait: bool = False) -> bool:
        self.events.append(payload)
        return True

    def send_result(self, payload: dict, *, wait: bool = True) -> bool:
        self.results.append(payload)
        return True

    @property
    def terminal(self) -> dict:
        return self.events[-1]


class _Step:
    def __init__(self, sid: str, op: str, src: Path, out_port: str) -> None:
        self.id, self.op, self.params, self.needs, self.optional = sid, op, {}, [], False
        self.inputs = [type("B", (), {"port": p, "url": None, "from_step": None, "path": str(src)})()
                       for p in _IN_PORTS]
        self.outputs = [type("B", (), {"port": out_port, "url": None, "urls": None})()]


class _Chain:
    def __init__(self, steps: list, job_id: str = "j-1") -> None:
        self.steps, self.job_id, self.pack = steps, job_id, None


@pytest.fixture
def op():
    """A real shipped op with one required output, so the runner's own checks stay live."""
    o = next(x for x in registry.all_ops().values() if any(not p.many for p in x.outputs))
    return o


_IN_PORTS: list[str] = []


@pytest.fixture
def wired(monkeypatch, tmp_path, op):
    """The real `run_chain` with the pack, the registry and the transport stubbed — everything else real."""
    global _IN_PORTS
    _IN_PORTS = [p.id for p in op.inputs]
    src = tmp_path / "in.bin"
    src.write_bytes(b"x" * 16)
    required = next(p.id for p in op.outputs if not p.optional and not p.many)

    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.registry, "assert_pod_safe", lambda *a, **k: None)
    monkeypatch.setattr(runner, "preflight_chain", lambda chain: None)
    monkeypatch.setattr(runner.pack, "activate", lambda ref: tmp_path)
    monkeypatch.setattr(runner, "log", lambda *a, **k: None)
    return src, required


def _handler_writing(required: str, payload: bytes = b"y" * 4096, hold_s: float = 0.0):
    def _fn(*, params, inputs, outputs):
        if hold_s:
            time.sleep(hold_s)
        outputs[required].write_bytes(payload)
    return _fn


# ── the terminal ─────────────────────────────────────────────────────────────────────────────────

def test_the_terminal_carries_one_timing_per_step(monkeypatch, wired, tmp_path, op):
    """THE DELIVERABLE. Without this the box's `steps` list names WHICH steps ran and nothing about what
    they cost, and `docs/gen/SEAM_ATLAS.md` records that Go drops even that.

    NEGATIVE: drop the `timings=` key from the terminal event and the box is back to one number per chain."""
    src, required = wired
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required))
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op.op, src, required)]), cp)

    t = cp.terminal
    assert t["status"] == "ok" and t["steps"] == ["s1"]
    assert "timings" in t, "the terminal carried no per-step seconds at all"
    assert t["timings"]["chain_s"] > 0, "the pod's own wall is what the box subtracts to get transport"
    got = t["timings"]["steps"]
    assert [s["id"] for s in got] == ["s1"] and got[0]["op"] == op.op, got
    assert got[0]["seconds"] >= 0.0 and set(got[0]["legs"]) <= {
        "slot_wait", "bind", "run", "put", "put_wait", "put_retry",
        "connect", "body", "seek_decode", "encode",
    }, got
    assert got[0]["legs"]["put_wait"] <= got[0]["legs"]["put"] + 1e-6, "put_wait is INSIDE put, not beside it"
    assert got[0]["slot_wait_s"] == got[0]["legs"]["slot_wait"]
    assert got[0]["outputs"], "the terminal names every output measured on the pod"


def test_live_events_keep_starts_plus_one_step_closure_while_terminal_keeps_detail(monkeypatch, wired, op):
    """Success keeps one boundary per phase that may hang plus one closure for the whole step; the terminal
    remains the full-chain receipt."""
    src, required = wired
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required))
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op.op, src, required)]), cp, corr_id="c", session_id="s")
    chain_phases = [e["phase"] for e in cp.events if e.get("op") == "ops" and e.get("phase")]
    assert chain_phases[:4] == [
        "preflight_started", "preflight_finished",
        "pack_activate_started", "pack_activate_finished",
    ]
    phases = [e for e in cp.events if e.get("step") == "s1" and e.get("phase")]
    assert [e["phase"] for e in phases] == [
        "bind_started", "slot_wait_started", "run_started", "upload_started", "step_finished",
    ]
    assert all(e["stage"] == "ops" and e["corr_id"] == "c" and e["session_id"] == "s"
               for e in phases)
    assert set(phases[1]["timings"]) == {"bind_s"}
    assert set(phases[2]["timings"]) == {"bind_s", "slot_wait_s"}
    assert set(phases[3]["timings"]) == {"run_s"}
    assert set(phases[4]["timings"]) == {
        "slot_wait_s", "bind_s", "run_s", "put_s", "seconds", "cache_hit"}
    assert phases[4]["timings"]["cache_hit"] == 0.0
    assert phases[4]["outcome"] == "ok" and phases[4]["outputs"]


def test_success_event_count_has_a_small_constant_per_step(monkeypatch, wired, op):
    """NEGATIVE/perf: restoring three phase-finished boundaries changes 5N+6 back to 8N+6 and makes the
    durable stream itself the critical path for wide b-roll chains."""
    src, required = wired
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required))
    cp = _CP()
    count = 8
    runner.run_chain(_Chain([_Step(f"s{i}", op.op, src, required) for i in range(count)]), cp)

    assert len(cp.events) == 5 * count + 6, [e.get("phase") for e in cp.events]
    terminal = cp.terminal["timings"]["steps"]
    assert len(terminal) == count
    assert all(set(row["legs"]) >= {"slot_wait", "bind", "run", "put"} for row in terminal)


def test_completed_sibling_is_durably_closed_while_another_sibling_hangs(monkeypatch, wired, op):
    """NEGATIVE: start-only events left every completed sibling looking stuck in upload until the chain
    terminal, but a sibling hung in the same chain prevents that terminal forever."""
    src, required = wired
    release = threading.Event()

    def selective(*, params, inputs, outputs):
        if outputs[required].parent.name == "hung":
            assert release.wait(timeout=2), "test release did not arrive"
        outputs[required].write_bytes(b"landed")

    monkeypatch.setattr(runner, "parallel_cap", lambda: (2, "test"))
    slots = threading.Semaphore(2)
    monkeypatch.setattr(runner, "step_slots", lambda: slots)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: selective)
    cp = _CP()
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            runner.run_chain(_Chain([
                _Step("done", op.op, src, required),
                _Step("hung", op.op, src, required),
            ]), cp)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=_run)
    thread.start()
    deadline = time.monotonic() + 1
    while not (any(e.get("step") == "done" and e.get("phase") == "step_finished" for e in cp.events)
               and any(e.get("step") == "hung" and e.get("phase") == "run_started" for e in cp.events)):
        assert time.monotonic() < deadline, cp.events
        time.sleep(0.005)

    closed = next(e for e in cp.events if e.get("step") == "done" and e.get("phase") == "step_finished")
    assert closed["outcome"] == "ok" and closed["outputs"][0]["present"] is True
    hung_phases = [e["phase"] for e in cp.events if e.get("step") == "hung" and e.get("phase")]
    assert hung_phases[-1] == "run_started", hung_phases
    assert not any(e.get("phase") == "work_finished" for e in cp.events)

    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and not errors


@pytest.mark.parametrize("boundary", [
    "bind_started", "slot_wait_started", "run_started", "upload_started",
])
def test_a_hung_step_names_its_exact_live_phase(monkeypatch, wired, op, boundary):
    """The terminal cannot exist while an op is hung. The last durable boundary must distinguish input bind,
    semaphore wait, handler execution and output upload without a completion event."""
    src, required = wired
    release = threading.Event()
    step = _Step("hung", op.op, src, required)
    real_bind = runner._bind_inputs

    if boundary == "bind_started":
        def _blocked_bind(*args, **kwargs):
            assert release.wait(timeout=2), "test release did not arrive"
            return real_bind(*args, **kwargs)

        monkeypatch.setattr(runner, "_bind_inputs", _blocked_bind)

    if boundary == "slot_wait_started":
        @contextmanager
        def _blocked_slot():
            assert release.wait(timeout=2), "test release did not arrive"
            yield

        monkeypatch.setattr(runner, "step_slots", _blocked_slot)

    def _hung(*, params, inputs, outputs):
        if boundary == "run_started":
            assert release.wait(timeout=2), "test release did not arrive"
        outputs[required].write_bytes(b"y")

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _hung)
    if boundary == "upload_started":
        step.outputs[0].url = "https://store.example/out"

        def _blocked_upload(*_args, **_kwargs):
            assert release.wait(timeout=2), "test release did not arrive"

        monkeypatch.setattr(runner, "upload", _blocked_upload)
    cp = _CP()
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            runner.run_chain(_Chain([step]), cp)
        except BaseException as exc:  # test thread must report into the assertion thread
            errors.append(exc)

    thread = threading.Thread(target=_run)
    thread.start()
    deadline = time.monotonic() + 1
    while not any(e.get("phase") == boundary for e in cp.events):
        assert time.monotonic() < deadline, cp.events
        time.sleep(0.005)
    live = next(e for e in cp.events if e.get("phase") == boundary)
    assert live["step"] == "hung" and live["op"] == op.op
    step_phases = [e["phase"] for e in cp.events if e.get("step") == "hung" and e.get("phase")]
    assert step_phases[-1] == boundary, step_phases
    assert not any(e.get("phase") == "work_finished" for e in cp.events)
    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and not errors


def test_slot_wait_is_not_mislabeled_as_handler_runtime(monkeypatch, wired, op):
    """NEGATIVE: starting the run clock before semaphore acquisition hides saturation inside `run`. The
    terminal must expose ready→acquired separately, both as a named leg and an explicit metric."""
    src, required = wired

    @contextmanager
    def delayed_slot():
        time.sleep(0.03)
        yield

    monkeypatch.setattr(runner, "step_slots", delayed_slot)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required))
    cp = _CP()
    runner.run_chain(_Chain([_Step("waited", op.op, src, required)]), cp)

    row = cp.terminal["timings"]["steps"][0]
    assert row["slot_wait_s"] >= 0.02, row
    assert row["legs"]["slot_wait"] == row["slot_wait_s"]
    assert row["legs"]["run"] < row["slot_wait_s"], row


def test_the_step_weighs_what_it_wrote(monkeypatch, wired, tmp_path, op):
    """Seconds without bytes name no defect: 33 s is a mood, 7 MB in 33 s is a diagnosis. This box is the
    only side that can weigh the file, so it does.

    NEGATIVE: stop weighing and the ledger can never print a rate for the leg the whole exercise is about."""
    src, required = wired
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required, b"z" * 8192))
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op.op, src, required)]), cp)
    row = cp.terminal["timings"]["steps"][0]
    assert row["bytes"] == 8192, cp.terminal["timings"]
    assert next(x for x in row["outputs"] if x["port"] == required) == {
        "port": required, "present": True, "bytes": 8192,
    }
    assert row["bytes"] == sum(x["bytes"] for x in row["outputs"])


def test_the_handler_legs_reach_the_terminal_when_the_pack_can_take_them(monkeypatch, wired, op):
    """A handler is handed typed params and local paths and returns nothing — deliberately. So for
    `media.fetch`, whose origin GET, redirect hops and GOP walk all happen INSIDE the one call, the runner's
    runner legs place the whole 33 s in `run` and answer nothing. The pack may carry a recorder; when it does,
    its legs must survive onto the wire.

    NEGATIVE: drop the `pack.legs()` hook in `_run_step_inner` and `connect`/`seek_decode` vanish, which is
    exactly the pair that tells a slow origin from a long GOP walk."""
    src, required = wired

    class _Recorder:
        def __init__(self) -> None:
            self.last: dict[str, float] = {}

        @contextmanager
        def recording(self):
            self.last = {}
            yield self.last

        def collect(self) -> dict[str, float]:
            return dict(self.last)

    rec = _Recorder()

    def _fn(*, params, inputs, outputs):
        rec.last["connect"] = 0.31
        rec.last["seek_decode"] = 3.55
        outputs[required].write_bytes(b"y")

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _fn)
    monkeypatch.setattr(runner.pack, "legs", lambda: rec)
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op.op, src, required)]), cp)
    legs = cp.terminal["timings"]["steps"][0]["legs"]
    assert legs["connect"] == 0.31 and legs["seek_decode"] == 3.55, legs
    assert "run" in legs, "the runner's own legs must not be lost to the handler's"


def test_an_older_pack_with_no_recorder_still_times_the_step(monkeypatch, wired, op):
    """The pack and the image ship separately, so a NEW agent WILL meet an OLD pack. It must degrade to the
    four legs it can always see itself, never crash and never lose the step.

    NEGATIVE: import the recorder at module scope instead of discovering it by name, and this agent refuses
    every pack built before it."""
    src, required = wired
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required))
    monkeypatch.setattr(runner.pack, "legs", lambda: None)     # what an older pack looks like
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op.op, src, required)]), cp)
    legs = cp.terminal["timings"]["steps"][0]["legs"]
    assert "run" in legs and not ({"connect", "seek_decode"} & set(legs)), legs


def test_a_recorder_that_raises_costs_its_legs_and_never_the_step(monkeypatch, wired, op):
    """A measurement that can fail the job it measures is strictly worse than no measurement."""
    src, required = wired

    class _Broken:
        @contextmanager
        def recording(self):
            yield {}

        def collect(self):
            raise RuntimeError("stopwatch exploded")

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required))
    monkeypatch.setattr(runner.pack, "legs", lambda: _Broken())
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op.op, src, required)]), cp)
    assert cp.terminal["status"] == "ok"
    assert cp.terminal["timings"]["steps"][0]["legs"], "the runner's own legs were lost with the handler's"


def test_a_step_that_failed_contributes_no_timing(monkeypatch, wired, op):
    """A duration booked for work that did not land makes a ledger worse than none: the box would price a
    fetch that delivered nothing as if it had. An OPTIONAL arm of a fan-out that dies is exactly this case.

    NEGATIVE: append the timing before the handler instead of after the uploads, and every failed arm shows
    up as work."""
    src, required = wired

    def _flaky(*, params, inputs, outputs):
        # keyed by the STEP (its output lives under <ws>/<step_id>/), never by call order — the two arms
        # run concurrently and a counter would make this test flap rather than assert
        if outputs[required].parent.name == "bad":
            raise RuntimeError("this arm's host said 403")
        outputs[required].write_bytes(b"y")

    monkeypatch.setattr(runner.pack, "resolve", lambda h: _flaky)
    bad = _Step("bad", op.op, src, required)
    bad.optional = True                       # one arm of a fan-out, not the chain
    cp = _CP()
    runner.run_chain(_Chain([bad, _Step("good", op.op, src, required)]), cp)
    ids = [s["id"] for s in cp.terminal["timings"]["steps"]]
    assert ids == ["good"], f"a step that delivered nothing was priced as work: {ids}"


def test_the_pods_own_wall_covers_every_step_it_reports(monkeypatch, wired, op):
    """`chain_s` is what the box subtracts from its own wall to get TRANSPORT. If it were smaller than the
    work it contains, that subtraction would invent transport out of nothing.

    NEGATIVE: start the chain clock after preflight/pack and a slow pack fetch lands in the transport row,
    where it is somebody else's bug."""
    src, required = wired
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required, hold_s=0.05))
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op.op, src, required)]), cp)
    t = cp.terminal["timings"]
    assert t["chain_s"] >= sum(s["seconds"] for s in t["steps"]), t


def test_the_timings_key_is_additive_and_nothing_that_already_crossed_moved(monkeypatch, wired, op):
    """`steps` is a list of ids both sides already know the shape of, and it is what an older control plane
    and an older box read. Changing its element type would be a BREAK on a field that already crosses; a new
    key is dropped with a 202 and ignored. So the old keys must be byte-identical to what they were."""
    src, required = wired
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler_writing(required))
    cp = _CP()
    runner.run_chain(_Chain([_Step("s1", op.op, src, required)]), cp)
    t = cp.terminal
    assert t["steps"] == ["s1"] and all(isinstance(x, str) for x in t["steps"])
    assert t["skipped"] == [] and t["stage"] == "ops" and t["status"] == "ok"


def test_the_leg_module_is_discovered_by_name_and_never_imported():
    """The agent must run a pack that predates the recorder. `pack.legs()` answers None rather than raising,
    and the name is DECLARED so the pack and the agent cannot drift into two spellings."""
    assert pack.LEGS_MODULE == "montyops.legs"
    assert pack.legs() is None, "no ops pack is active in this test process, so there is nothing to find"


# ── the store pool may not be narrower than the transfers that use it (cp.STORE_POOL_WHY) ────────────────
def test_the_store_pool_is_as_wide_as_the_transfers_the_box_assigned(monkeypatch):
    """MEASURED, and it is the churn we make ourselves: `pool_maxsize` was 4 while up to 64 binds and puts
    ran at once. urllib3 does not WAIT for a slot with `pool_block=False` — it mints a connection, uses it and
    throws it away, so 60 of every 64 puts paid a fresh TCP+TLS handshake and left a socket in TIME_WAIT.
    595 media steps a run, 391 of them putting: ~400 short-lived connections to one host, from a rented
    container whose SYNs then go unanswered for 45 s while the box gets a clean RST in 0.4 s."""
    import importlib

    from podagent import cp

    monkeypatch.setenv("OPS_MAX_TRANSFERS", "64")
    assert cp._store_pool() == 64
    monkeypatch.setenv("OPS_MAX_TRANSFERS", "")
    assert cp._store_pool() == 16, "an unset width must not fall back to the 4 that caused this"
    monkeypatch.setenv("OPS_MAX_TRANSFERS", "not-a-number")
    assert cp._store_pool() == 16
    monkeypatch.setenv("OPS_MAX_TRANSFERS", "1")
    assert cp._store_pool() >= 4, "a floor keeps a typo from serialising every transfer"
    importlib.reload  # noqa: B018 — the module-level session is built once; this test pins the sizer


def test_the_store_session_actually_mounts_that_pool():
    """A sizer nothing reads is a comment. The adapter must carry the number."""
    from podagent import cp

    ad = cp._store.get_adapter("https://x/")
    assert ad._pool_maxsize >= 16, f"the store session still mounts a narrow pool: {ad._pool_maxsize}"
