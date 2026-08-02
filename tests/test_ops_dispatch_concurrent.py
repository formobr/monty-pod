"""The claim loop DISPATCHES; it must not execute.

The brain fans b-roll ranking out several rank chains wide and awaits them by corr_id — op_backend dropped
its per-jid lane lock on purpose so those chains overlap. If the agent runs each claimed envelope inline,
that fan-out is re-serialised HERE, on the one box the brain cannot see into, and the sourcing tail costs the
SUM of its chains instead of their max. Both tests below were watched fail against the inline loop.
"""
from __future__ import annotations

import concurrent.futures as cf
import threading

from podagent import main as agent_main

_CHAIN = {"chain_version": 1, "job_id": "j", "pack": {"url": "https://x/p.tar", "sha256": "a" * 64},
          "steps": [{"id": "s0", "op": "media.scale", "needs": [], "params": {}, "inputs": [], "outputs": []}]}
_SPEC = {
    "spec_version": 5, "job_id": "j", "slug": "s", "mode": "preview",
    "inputs": [{"id": "src", "kind": "video", "sha256": "0" * 64, "url": "u"}],
    "timeline": {"fps": 30, "width": 2, "height": 2,
                 "segments": [{"src": "src", "in": 0, "out": 1, "speed": 1}]},
    "encode": {"video": "libx264", "preset": "p4", "cq": 29, "pix_fmt": "yuv420p",
               "audio": "aac", "audio_bitrate": "192k"},
    "outputs": [{"id": "proxy", "kind": "proxy", "put_url": "p"}],
}


class _CP:
    """Hands out a fixed script of envelopes, then None forever."""

    def __init__(self, jobs):
        self.jobs = list(jobs)
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def poll_job(self):
        with self._lock:
            return self.jobs.pop(0) if self.jobs else None

    def post_event(self, payload):
        self.events.append(payload)

    def note(self, payload):
        self.events.append(payload)


def _pump(cp, ops_pool, heavy_pool, heavy, n, rank_pool=None):
    for _ in range(n):
        agent_main._dispatch_loop(cp, ops_pool, heavy_pool, rank_pool or heavy_pool, heavy, once=True)


def test_two_ops_chains_run_at_once(monkeypatch):
    """Two ops envelopes claimed back-to-back must be IN FLIGHT together, not one-after-the-other."""
    both_in = threading.Barrier(2)
    overlapped: list[str] = []

    def fake_run_ops(chain, cp, corr_id=None, session_id=None):
        # passes ONLY if a SECOND chain reaches here while this one is still running
        both_in.wait(timeout=3)
        overlapped.append(corr_id)

    monkeypatch.setattr(agent_main, "_run_ops", fake_run_ops)
    cp = _CP([{"type": "ops", "chain": _CHAIN, "corr_id": "a"},
              {"type": "ops", "chain": _CHAIN, "corr_id": "b"}])
    with cf.ThreadPoolExecutor(max_workers=4) as ops_pool, cf.ThreadPoolExecutor(max_workers=1) as heavy_pool:
        _pump(cp, ops_pool, heavy_pool, lambda job: None, 2)
    assert sorted(overlapped) == ["a", "b"], "the two ops chains never overlapped — the claim loop ran them"


def test_a_running_render_does_not_block_claiming_ops(monkeypatch):
    """A render holds the GPU for minutes. It must not also hold the CLAIM: the ops chain behind it in the
    queue is net-bound work with nothing to do with the GPU."""
    render_may_finish = threading.Event()
    ops_ran = threading.Event()
    render_done = threading.Event()

    def fake_heavy(job):
        render_may_finish.wait(timeout=3)
        render_done.set()

    def fake_run_ops(chain, cp, corr_id=None, session_id=None):
        ops_ran.set()

    monkeypatch.setattr(agent_main, "_run_ops", fake_run_ops)
    cp = _CP([{"type": "render", "spec": _SPEC}, {"type": "ops", "chain": _CHAIN, "corr_id": "a"}])
    with cf.ThreadPoolExecutor(max_workers=4) as ops_pool, cf.ThreadPoolExecutor(max_workers=1) as heavy_pool:
        _pump(cp, ops_pool, heavy_pool, fake_heavy, 2)
        assert ops_ran.wait(timeout=3), "the ops chain waited out the render"
        assert not render_done.is_set(), "the ops chain only ran because the render had already finished"
        render_may_finish.set()


def test_a_worker_that_raises_is_reported_not_fatal(monkeypatch):
    """Off the claim loop the old outer `except` no longer covers a worker — an unreported crash is a job
    the brain waits out in silence."""
    def boom(chain, cp, corr_id=None, session_id=None):
        raise RuntimeError("handler exploded")

    monkeypatch.setattr(agent_main, "_run_ops", boom)
    cp = _CP([{"type": "ops", "chain": _CHAIN, "corr_id": "a"}])
    with cf.ThreadPoolExecutor(max_workers=2) as ops_pool, cf.ThreadPoolExecutor(max_workers=1) as heavy_pool:
        _pump(cp, ops_pool, heavy_pool, lambda job: None, 1)
    assert any(e.get("status") == "error" and "handler exploded" in str(e.get("error")) for e in cp.events)


def test_ops_pool_size_is_env_tunable(monkeypatch):
    monkeypatch.setenv("OPS_MAX_CHAINS", "3")
    assert agent_main.ops_chain_pool_size() == 3
    monkeypatch.setenv("OPS_MAX_CHAINS", "nonsense")
    assert agent_main.ops_chain_pool_size() == agent_main._OPS_MAX_CHAINS_DEFAULT
