"""Two weight-holding infer kinds on one card: parallel when the card holds both, in turns when it does not.
Each test names the reversion it catches."""
from __future__ import annotations

import concurrent.futures as cf
import threading

from podagent import infer_lanes as m
from podagent import main as agent_main

_W = {"url": "https://x/w.tar", "sha256": "b" * 64}
_RANK_REQ = {"infer_version": 7, "job_id": "j", "kind": "clip_rank", "model": "siglip",
             "put_url": "https://x/o/r.json", "weights": _W,
             "clip_rank": {"groups": [{"intent": "chart", "image_urls": ["u1"]}]}}
_ALIGN_REQ = {"infer_version": 7, "job_id": "j", "kind": "align", "model": "w2v",
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
    """The measured wall: 3254 MiB free against 2736 clip_rank + 2344 heavy lane + 512 reserve."""
    fits, why = m.kinds_fit_together(3254.0)
    assert fits is False
    assert "5592" in why and "3254" in why


def test_the_056_oom_card_serialises_now():
    """2026-10-22, 056 v13: 4835 MiB free passed the old 4748 floor and the lanes OOM'd. Measured (HEAVY_LANE_
    MEASURED_WHY): align 1244 resident + 518 forward, RVM +1100 autotune / 228 resident, SigLIP 2322 + 414/lane."""
    assert m.heavy_lane_mib() == 1244.0 + max(518.0 + 228.0, 1100.0) == 2344.0
    assert m.coresident_mib() == 2736.0 + 2344.0 + 512.0
    assert m.kinds_fit_together(4835.0)[0] is False
    # without person_matte the heavy lane is align alone: 1762
    assert m.heavy_lane_mib({"align", "clip_rank"}) == 1762.0
    assert m.kinds_fit_together(4835.0, {"align", "clip_rank"})[0] is False
    # person_matte alone beside clip_rank: its autotune peak
    assert m.heavy_lane_mib({"person_matte", "clip_rank"}) == 1100.0


def test_rank_lanes_are_sized_with_the_heavy_lane_beside_them():
    from podagent import infer_cliprank as ic
    # the OOM: 4835 free sized as if alone → 4 lanes; with the heavy lane held beside, the budget is gone → 1
    assert ic._width_from(4835.0, 12)[0] == 4
    assert ic._width_from(4835.0, 12, m.heavy_lane_mib())[0] == 1
    # kinds taking turns still leave one NVENC render session (RENDER_HEADROOM_MIB 960, the boot floor's figure):
    # (4835 − 512 − 2322 − 960) // 414 = 2 lanes; measured 2 lanes alone = 2322 + 726 MiB, + 960 = 4008 < 4323
    assert m.RENDER_HEADROOM_MIB == 960.0
    assert ic._width_from(4835.0, 12, m.RENDER_HEADROOM_MIB)[0] == 2
    # a fleet card (min 14848): (14848 − 512 − 2322 − 2344) // 414 = 23 → host-bound
    n, why = ic._width_from(14848.0, 16, m.heavy_lane_mib(), "heavy lane")
    assert n == 16 and "− 2344 heavy lane" in why


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


_PROBE_REQ = {"infer_version": 7, "job_id": "j", "kind": "face_probe", "model": "yunet",
              "put_url": "https://x/o/p.json",
              "face_probe": {"video_url": "https://x/v.mp4", "shots": [[0.0, 1.0]], "stride": 1,
                             "frame_diff": False}}


def test_face_probe_never_queues_behind_a_weight_holding_kind():
    """guards FACE_PROBE_LANE_WHY (056 v13 r8: face_probe claimed 38 s after person_matte, ran 0.044 s, on the
    preview's critical path): a CPU probe claimed while the one-wide heavy lane is busy runs on its own lane."""
    lanes: list[str] = []
    busy = threading.Event()

    def heavy(job):
        lanes.append(threading.current_thread().name.split("_")[0])
        if job.request.kind == "align":
            assert busy.wait(timeout=5), "the probe never ran beside the busy heavy lane"
        else:
            busy.set()

    cp = _CP([{"type": "infer", "session_id": "s", "corr_id": "a", "request": _ALIGN_REQ},
              {"type": "infer", "session_id": "s", "corr_id": "p", "request": _PROBE_REQ}])
    with cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ops") as ops, \
            cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu") as gpu, \
            cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="rank") as rank, \
            cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="probe") as probe:
        _pump(cp, ops, gpu, rank, heavy, 2, kinds_coexist=False, probe_pool=probe)
    assert sorted(lanes) == ["gpu", "probe"], f"face_probe shared the heavy lane: {lanes}"


def test_face_probe_drops_no_resident_weights_on_a_turn_taking_card(monkeypatch):
    """guards INFER_KIND_RESIDENCY_WHY's own scope: only a kind that LOADS onto the card may drop the idle one —
    a face_probe (0 MiB) on the 2060 used to throw SigLIP off the card for nothing, and the next rank reloaded it."""
    from pathlib import Path

    from podagent import infer_probe

    class _Probe:
        def __init__(self, *_a):
            pass

        def run(self, *_a):
            return 0.01

    class _ResultCP(_CP):
        def report_infer_result(self, result):
            self.events.append({"result": result})

    monkeypatch.setattr(infer_probe, "ProbeService", _Probe)
    rank = {"sha-r": object()}
    cp = _ResultCP([])
    agent_main._run_infer(dict(_PROBE_REQ), cp, {}, {}, rank, Path("/opt/models/yunet.onnx"), True,
                          corr_id="p", session_id="s", one_kind_resident=True)
    assert any("result" in e for e in cp.events), "the probe never reported — the test proves nothing"
    assert list(rank) == ["sha-r"], "a face_probe dropped SigLIP's weights"


def test_on_a_turn_taking_card_the_encode_takes_the_card_alone(monkeypatch):
    """guards CARD_TURNS_WHY (056 v13 r8: SigLIP resident through cut.apply, ledger grant 1768 MiB → 1 NVENC
    session, apply_cut 99 s): an admitted heavy op on a turn-taking card drops the idle infer weights and is
    granted the card as it is NOW; a weight-holding infer queues in the same ledger; a fleet card is untouched."""
    from podagent.ops import gpu_admission

    gpu_admission._reset_for_tests(1768.0)                # the budget the first heavy op of the pod's life saw
    monkeypatch.setattr(gpu_admission, "_free_vram_mb", lambda: 4100.0)   # the card with SigLIP gone
    rank, align = {"sha-r": object()}, {}
    try:
        m.arm_card_turns(False, {"align": align, "clip_rank": rank})      # a fleet card: kinds coexist
        assert not m.takes_card_turn("clip_rank")
        with gpu_admission.admission("cut.apply"):
            assert m.clear_card_for_render() == [] and list(rank) == ["sha-r"]
            assert gpu_admission.budget_mib() == 1768.0

        m.arm_card_turns(True, {"align": align, "clip_rank": rank})
        assert m.takes_card_turn("clip_rank") and m.takes_card_turn("person_matte")
        assert not m.takes_card_turn("face_probe")         # no card, no turn
        with gpu_admission.admission("cut.apply"):
            assert m.clear_card_for_render() == ["clip_rank"] and rank == {}
            assert gpu_admission.budget_mib() == 4100.0 - gpu_admission._VRAM_RESERVE_MB
            # a weight-holding infer that arrives now waits in the same FIFO for the whole card
            parked = threading.Event()
            t = threading.Thread(target=lambda: gpu_admission.admission(
                "infer.clip_rank", deadline_s=5.0, on_park=lambda **_k: parked.set()).__enter__())
            t.start()
            assert parked.wait(timeout=5), "the infer ran beside the encode"
        t.join(timeout=5)
        assert not t.is_alive()
    finally:
        m.arm_card_turns(False, {})
        gpu_admission._reset_for_tests()
