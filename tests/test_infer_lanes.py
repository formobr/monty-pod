"""Two weight-holding infer kinds on one card: parallel when the card holds both, in turns when it does not.
Each test names the reversion it catches."""
from __future__ import annotations

import concurrent.futures as cf
import threading

from podagent import infer_lanes as m
from podagent import main as agent_main

_W = {"url": "https://x/w.tar", "sha256": "b" * 64}
_RANK_REQ = {"infer_version": 6, "job_id": "j", "kind": "clip_rank", "model": "siglip",
             "put_url": "https://x/o/r.json", "weights": _W,
             "clip_rank": {"groups": [{"intent": "chart", "image_urls": ["u1"]}]}}
_ALIGN_REQ = {"infer_version": 6, "job_id": "j", "kind": "align", "model": "w2v",
              "put_url": "https://x/o/a.json", "weights": _W,
              "align": {"audio_url": "https://x/a.wav", "windows": [[0.0, 1.0]]}}


class _CP:
    def __init__(self, jobs):
        self.jobs = list(jobs)
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def poll_job(self):
        with self._lock:
            return self.jobs.pop(0) if self.jobs else None

    def send_event(self, payload, **_kw):
        self.events.append(payload)
        return True

    def note(self, payload):
        self.events.append(payload)


def test_the_dev_2060_cannot_hold_both_kinds_and_says_which_numbers_say_so():
    """The measured wall: 3254 MiB free against 1500 align + 2736 clip_rank + 512 reserve."""
    fits, why = m.kinds_fit_together(3254.0)
    assert fits is False
    assert "4748" in why and "3254" in why


def test_a_fleet_card_keeps_todays_parallel_lanes():
    """broker.base.Constraints.min_vram_gb = 14.5 GB is the WORST card the fleet admits — 14848 MiB."""
    assert m.kinds_fit_together(14848.0)[0] is True
    assert m.kinds_fit_together(m.coresident_mib())[0] is True


def test_a_card_that_reports_nothing_runs_one_kind_at_a_time_on_the_gpu():
    fits, why = m.kinds_fit_together(None)
    assert fits is False
    assert "never the CPU" in why


def test_a_probe_that_raises_is_a_narrow_reading_not_a_dead_boot():
    def boom():
        raise OSError("no nvidia-smi")

    fits, why = m.card_holds_both_kinds(probe=boom)
    assert fits is False and "OSError" in why


def test_the_probe_is_passed_through_to_the_derivation():
    assert m.card_holds_both_kinds(probe=lambda: 23000.0)[0] is True
    assert m.card_holds_both_kinds(probe=lambda: 3254.0)[0] is False


def test_only_the_other_kinds_weights_are_dropped():
    align, rank = {"sha-a": object()}, {"sha-r": object()}
    assert m.release_other_kinds("clip_rank", {"align": align, "clip_rank": rank}) == ["align"]
    assert align == {} and list(rank) == ["sha-r"]
    assert m.release_other_kinds("clip_rank", {"align": align, "clip_rank": rank}) == []


def _pump(cp, ops_pool, heavy_pool, rank_pool, heavy, n, **kw):
    for _ in range(n):
        agent_main._dispatch_loop(cp, ops_pool, heavy_pool, rank_pool, heavy, once=True, **kw)


def test_a_narrow_card_folds_clip_rank_onto_the_align_lane():
    """Watched fail without kinds_coexist: both kinds run at once and the second load OOMs the card."""
    lanes: list[str] = []

    def heavy(_job):
        lanes.append(threading.current_thread().name.split("_")[0])

    cp = _CP([{"type": "infer", "session_id": "s", "corr_id": "a", "request": _ALIGN_REQ},
              {"type": "infer", "session_id": "s", "corr_id": "r", "request": _RANK_REQ}])
    with cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ops") as ops, \
            cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu") as gpu, \
            cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="rank") as rank:
        _pump(cp, ops, gpu, rank, heavy, 2, kinds_coexist=False)
    assert lanes == ["gpu", "gpu"], f"the rank envelope left the one-wide lane: {lanes}"


def test_a_fleet_card_still_routes_clip_rank_to_its_own_lane():
    lanes: list[str] = []

    def heavy(_job):
        lanes.append(threading.current_thread().name.split("_")[0])

    cp = _CP([{"type": "infer", "session_id": "s", "corr_id": "a", "request": _ALIGN_REQ},
              {"type": "infer", "session_id": "s", "corr_id": "r", "request": _RANK_REQ}])
    with cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ops") as ops, \
            cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu") as gpu, \
            cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="rank") as rank:
        _pump(cp, ops, gpu, rank, heavy, 2, kinds_coexist=True)
    assert sorted(lanes) == ["gpu", "rank"], f"today's routing changed: {lanes}"
