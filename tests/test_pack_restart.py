"""In-place ops-pack re-warm (rewarm_plan v2-v4): a chain bound to a NEWER pack than this process activated
must trigger a restart, never an error terminal — the chain is still durably claimed and a fresh incarnation
of this same process will see it replayed under the pack it actually names."""
from __future__ import annotations

import concurrent.futures as cf
import time

import pytest
from podagent import main as agent_main
from podagent.models import OpChain
from podagent.ops import pack, runner

_PACK_OLD = {"url": "https://x/old.tar", "sha256": "a" * 64, "size": 10}
_PACK_NEW = {"url": "https://x/new.tar", "sha256": "b" * 64, "size": 10}


def _chain(pack_ref: dict) -> OpChain:
    return OpChain(job_id="j-restart", pack=pack_ref, steps=[
        {"id": "a", "op": "media.scale", "needs": [],
         "params": {"height": 960, "encode_profile": "proxy"},
         "inputs": [{"port": "src", "url": "https://x/in.mp4"}],
         "outputs": [{"port": "dst", "url": "https://x/out.mp4"}]},
    ])


@pytest.fixture(autouse=True)
def _reset_pack():
    pack.reset_for_tests()
    yield
    pack.reset_for_tests()


@pytest.fixture(autouse=True)
def _isolated_restart_state(tmp_path, monkeypatch):
    """The real /tmp flip-history file is shared across the whole test session — every _drain_and_restart
    call here must count against a THROWAWAY history, or the 3rd unrelated test call trips the flip guard."""
    monkeypatch.setattr(agent_main, "_FLIP_HISTORY_MARK", tmp_path / "flip-history.json")
    monkeypatch.delenv(agent_main._PLANNED_RESTART_ENV, raising=False)


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.results: list[dict] = []
        self.closed = False

    def send_event(self, payload, *, wait=False):
        self.events.append(payload)
        return True

    def send_result(self, payload, *, wait=True):
        self.results.append(payload)
        return True

    def note(self, payload):
        self.events.append(payload)

    def close_stream(self):
        self.closed = True


# ── U1: runner-level sentinel ───────────────────────────────────────────────────────────────────

def test_a_pack_mismatch_returns_the_restart_sentinel_before_any_work(monkeypatch, tmp_path):
    monkeypatch.setattr(pack, "ensure", lambda ref: tmp_path)
    assert pack.activate_or_mismatch(_chain(_PACK_OLD).pack) == tmp_path
    monkeypatch.setattr(pack, "ensure", lambda ref: (_ for _ in ()).throw(
        AssertionError("a mismatch must never fetch the losing pack")))
    result = runner.run_chain(_chain(_PACK_NEW), cp=_CP(), corr_id="c", session_id="s")
    assert result is runner.RESTART_REQUIRED


# ── codex#1: check+activate is ONE lock acquisition, not a separate read then a separate activate ──

def test_the_first_activation_in_a_process_is_never_a_mismatch(monkeypatch, tmp_path):
    """`active_sha() is None` means nothing has activated yet — the first call always wins, whatever
    pack it names."""
    monkeypatch.setattr(pack, "ensure", lambda ref: tmp_path)
    assert pack.activate_or_mismatch(_chain(_PACK_NEW).pack) == tmp_path
    assert pack.active_sha() == _PACK_NEW["sha256"]


def test_reactivating_the_same_pack_stays_free_not_a_mismatch(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(pack, "ensure", lambda ref: (calls.append(ref.sha256), tmp_path)[1])
    assert pack.activate_or_mismatch(_chain(_PACK_OLD).pack) == tmp_path
    assert pack.activate_or_mismatch(_chain(_PACK_OLD).pack) == tmp_path
    assert pack.active_sha() == _PACK_OLD["sha256"]
    assert calls == [_PACK_OLD["sha256"], _PACK_OLD["sha256"]], "re-activation still hits ensure() (cache-checked)"


def test_a_losing_second_activation_never_raises_packerror(monkeypatch, tmp_path):
    """The OLD two-step design (separate active_sha() read, separate activate() call) let a losing second
    chain reach pack.activate() and raise PackError instead of signalling a restart. The atomic function
    must never raise here — codex#1/C2."""
    monkeypatch.setattr(pack, "ensure", lambda ref: tmp_path)
    assert pack.activate_or_mismatch(_chain(_PACK_OLD).pack) == tmp_path
    assert pack.activate_or_mismatch(_chain(_PACK_NEW).pack) is None
    assert pack.active_sha() == _PACK_OLD["sha256"], "the loser must not have disturbed the winner"


# ── U1: _run_ops builds NO terminal for the sentinel, and tells the coordinator ────────────────────

def test_run_ops_on_mismatch_requests_a_restart_and_sends_no_terminal(monkeypatch):
    monkeypatch.setattr(runner, "run_chain",
                        lambda *_a, **_kw: runner.RESTART_REQUIRED)
    cp = _CP()
    coordinator = agent_main.RestartCoordinator()
    chain = _chain(_PACK_NEW)

    agent_main._run_ops(chain, cp, corr_id="c", session_id="s", coordinator=coordinator)

    assert coordinator.restart_requested()
    assert "b" * 12 in coordinator.reason
    assert cp.results == [], "a non-terminal sentinel must never produce a result frame"
    assert not [e for e in cp.events if e.get("status") == "error"], (
        "a generation flip is not the chain's own failure")


def test_run_ops_on_mismatch_with_no_coordinator_fails_loudly(monkeypatch):
    """A production caller always supplies a coordinator; a bare RESTART_REQUIRED with none is a bug and
    must surface as a loud error terminal, not vanish the chain silently."""
    monkeypatch.setattr(runner, "run_chain", lambda *_a, **_kw: runner.RESTART_REQUIRED)
    cp = _CP()
    agent_main._run_ops(_chain(_PACK_NEW), cp, corr_id="c", session_id="s")
    assert cp.results and cp.results[0]["status"] == "error"
    assert "restart coordinator" in cp.results[0]["error"]


def test_a_matched_chain_still_completes_normally_alongside_a_mismatched_one(monkeypatch):
    """Other chains' results are unaffected by a sibling's generation flip."""
    calls = iter([runner.RESTART_REQUIRED, {"a": {"dst": "/x"}}])
    monkeypatch.setattr(runner, "run_chain", lambda *_a, **_kw: next(calls))
    cp = _CP()
    coordinator = agent_main.RestartCoordinator()

    agent_main._run_ops(_chain(_PACK_NEW), cp, corr_id="mismatch", session_id="s", coordinator=coordinator)
    agent_main._run_ops(_chain(_PACK_OLD), cp, corr_id="ok", session_id="s", coordinator=coordinator)

    assert cp.results == [], "run_chain here returns a dict, not RESTART_REQUIRED — _run_ops itself never " \
        "posts a result for a plain successful return; run_chain owns that terminal in production"
    assert coordinator.restart_requested()


# ── U2/U8: RestartCoordinator bookkeeping ───────────────────────────────────────────────────────

def test_coordinator_latch_is_idempotent_and_keeps_the_first_reason():
    c = agent_main.RestartCoordinator()
    assert not c.restart_requested()
    c.request_restart("first")
    c.request_restart("second")
    assert c.restart_requested() and c.reason == "first"


def test_coordinator_futures_untrack_themselves_on_completion():
    c = agent_main.RestartCoordinator()
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: 1)
        c.track("ops", "corr-1", fut)
        fut.result(timeout=5)
        deadline = time.monotonic() + 2
        while c.pending() and time.monotonic() < deadline:
            time.sleep(0.01)
    assert c.pending() == {}, "a finished future must not linger in the registry forever"


def test_dispatch_loop_stops_claiming_once_the_latch_is_set():
    coordinator = agent_main.RestartCoordinator()
    coordinator.request_restart("test")

    class _NeverPoll:
        def poll_job(self):
            raise AssertionError("poll_job must not be called once the restart latch is set")

    agent_main._dispatch_loop(_NeverPoll(), None, None, None, lambda _j: None, coordinator=coordinator)
    # returning at all (rather than hanging or raising) is the assertion


# ── U8/O1: close_stream() precedes the drain, deadline abandons and still restarts ──────────────

def test_drain_closes_the_stream_before_waiting_on_any_future(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(agent_main, "_mark_planned_restart", lambda *_a, **_kw: None)

    class _OrderedCP(_CP):
        def close_stream(self):
            order.append("close_stream")
            super().close_stream()

    cp = _OrderedCP()
    coordinator = agent_main.RestartCoordinator()
    coordinator.request_restart("pack flip")

    def _execv(_path, _argv):
        order.append("execv")

    agent_main._drain_and_restart(cp, coordinator, drain_s=5, sleep=lambda _s: None, execv=_execv)
    assert order == ["close_stream", "execv"]
    assert cp.closed


def test_drain_waits_for_a_future_that_finishes_in_time(monkeypatch):
    cp = _CP()
    coordinator = agent_main.RestartCoordinator()
    coordinator.request_restart("pack flip")
    calls = {"n": 0}
    marks: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(agent_main, "_mark_planned_restart", lambda reason, abandoned: marks.append(
        (reason, abandoned)))

    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: 1)
        fut.result(timeout=5)  # already finished before track(): pending() is empty on the first check
        coordinator.track("ops", "corr-1", fut)

        agent_main._drain_and_restart(
            cp, coordinator, drain_s=5, sleep=lambda _s: None,
            execv=lambda _path, _argv: calls.__setitem__("n", calls["n"] + 1))

    assert calls["n"] == 1
    assert marks == [("pack flip", [])], "a future that finished on its own must not be reported abandoned"


def test_drain_deadline_abandons_a_wedged_future_and_execs_anyway(monkeypatch):
    cp = _CP()
    coordinator = agent_main.RestartCoordinator()
    coordinator.request_restart("pack flip")

    never_done = cf.Future()  # deliberately never resolved
    coordinator.track("ops", "corr-stuck", never_done)

    clock = {"t": 0.0}
    monkeypatch.setattr(agent_main.time, "monotonic", lambda: clock["t"])

    def _sleep(_s):
        clock["t"] += 1.0

    marks: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(agent_main, "_mark_planned_restart", lambda reason, abandoned: marks.append(
        (reason, abandoned)))

    execs: list[tuple[str, list[str]]] = []
    agent_main._drain_and_restart(
        cp, coordinator, drain_s=3, sleep=_sleep,
        execv=lambda path, argv: execs.append((path, argv)))

    assert execs, "the deadline expiring must still restart — an unresulted chain replays correctly"
    assert marks and marks[0][1] == ["ops:corr-stuck"]
    assert cp.closed


# ── codex#3(b): a rate limit on the flip itself, durable across execv ──────────────────────────────

def _drained(cp=None, *, target="p" * 64, now=time.time):
    coordinator = agent_main.RestartCoordinator()
    coordinator.request_restart("pack flip", target_pack=target)
    agent_main._drain_and_restart(
        cp or _CP(), coordinator, drain_s=1, sleep=lambda _s: None, now=now,
        execv=lambda *_a, **_kw: None, kill_orphans=lambda: None)


def test_three_flips_in_an_hour_are_allowed_a_fourth_is_refused_loudly():
    for _ in range(3):
        _drained()
    with pytest.raises(agent_main.RestartRefused, match="3 ops-pack generation flip"):
        _drained()


def test_flips_outside_the_trailing_hour_do_not_count():
    clock = {"t": 1_000_000.0}
    for _ in range(3):
        _drained(now=lambda: clock["t"])
    clock["t"] += agent_main._FLIP_WINDOW_S + 1
    _drained(now=lambda: clock["t"])  # must NOT raise — the earlier three have aged out


def test_a_refused_flip_never_closes_the_stream_or_execs():
    cp = _CP()
    for _ in range(3):
        _drained(cp)
    cp2 = _CP()
    with pytest.raises(agent_main.RestartRefused):
        _drained(cp2)
    assert not cp2.closed


# ── codex#4: never exec with live child processes ───────────────────────────────────────────────

def test_drain_kills_orphan_children_before_execv():
    order: list[str] = []
    cp = _CP()
    coordinator = agent_main.RestartCoordinator()
    coordinator.request_restart("pack flip")
    agent_main._drain_and_restart(
        cp, coordinator, drain_s=1, sleep=lambda _s: None,
        kill_orphans=lambda: order.append("kill_orphans"),
        execv=lambda *_a, **_kw: order.append("execv"))
    assert order == ["kill_orphans", "execv"]


def test_kill_orphan_children_is_best_effort_on_an_unreadable_proc(monkeypatch):
    """No /proc/*/task/*/children (non-Linux, or a container without it) must never raise."""
    monkeypatch.setattr(agent_main.Path, "read_text", lambda self, *a, **k: (
        _ for _ in ()).throw(OSError("no such file")))
    agent_main._kill_orphan_children()  # must not raise


# ── codex#6/C3: planned-restart marker is coupled to exec success ──────────────────────────────────

def test_a_failed_execv_removes_the_planned_marker_and_exits_without_waiting(monkeypatch):
    cp = _CP()
    coordinator = agent_main.RestartCoordinator()
    coordinator.request_restart("pack flip")
    exits: list[int] = []
    monkeypatch.setattr(agent_main.os, "_exit", lambda code: exits.append(code))

    def _boom(_path, _argv):
        raise OSError("no such file or directory")

    agent_main._drain_and_restart(
        cp, coordinator, drain_s=1, sleep=lambda _s: None,
        kill_orphans=lambda: None, execv=_boom)

    assert exits == [1]
    assert agent_main._PLANNED_RESTART_ENV not in agent_main.os.environ


def test_a_successful_execv_leaves_the_marker_for_the_next_boot():
    cp = _CP()
    coordinator = agent_main.RestartCoordinator()
    coordinator.request_restart("pack flip")
    agent_main._drain_and_restart(
        cp, coordinator, drain_s=1, sleep=lambda _s: None,
        kill_orphans=lambda: None, execv=lambda *_a, **_kw: None)
    assert agent_main._PLANNED_RESTART_ENV in agent_main.os.environ
    assert agent_main._consume_planned_restart_mark() is not None
