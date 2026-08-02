"""clip_rank gets its OWN card-sized lane, and its fetch is not its forward.
Everything model-shaped shared one worker, so a wave's ranking envelopes were serial pod-wide and any
render blocked all of them — while the envelope is ~80% network. Each test names the reversion it caught."""
from __future__ import annotations

import concurrent.futures as cf
import threading
import time
import types
from pathlib import Path

import pytest

from podagent import infer_cliprank as m
from podagent import main as agent_main
from podagent.infer_cliprank import ClipRankService
from podagent.models import ClipRankGroup

_W = {"url": "https://x/w.tar", "sha256": "b" * 64}
_RANK_REQ = {"infer_version": 5, "job_id": "j", "kind": "clip_rank", "model": "siglip",
             "put_url": "https://x/o/r.json", "weights": _W,
             "clip_rank": {"groups": [{"intent": "chart", "image_urls": ["u1"]}]}}
_ALIGN_REQ = {"infer_version": 5, "job_id": "j", "kind": "align", "model": "w2v",
              "put_url": "https://x/o/a.json", "weights": _W,
              "align": {"audio_url": "https://x/a.wav", "windows": [[0.0, 1.0]]}}
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


def _pump(cp, ops_pool, heavy_pool, rank_pool, heavy, n):
    for _ in range(n):
        agent_main._dispatch_loop(cp, ops_pool, heavy_pool, rank_pool, heavy, once=True)


# ── the width is derived from the card, and refuses to guess ─────────────────────────────────────

def test_a_big_card_scales_past_the_one_worker_it_used_to_get():
    """A rented 24 GB card must not inherit the single GPU worker: 23000 − 512 reserve − 2322 shared
    weights = 20166 MiB / 414 per lane = 48 by VRAM, so the host is the binding side."""
    n, why = m._width_from(free_mb=23000.0, threads=16, env_raw=None)
    assert n == 16
    assert "vram-bound=48" in why and "host-bound=16" in why


def test_the_dev_2060_gets_exactly_one_lane():
    """3254 MiB free under a desktop − 512 − 2322 = 420, which holds ONE 414 MiB forward — the formula must
    reproduce the card it was measured on, not flatter it."""
    assert m._width_from(free_mb=3254.0, threads=12, env_raw=None)[0] == 1


def test_a_card_that_reports_nothing_runs_narrow_on_the_gpu_and_says_so():
    """Placement law: no number => serialise ON THE CARD. 0 would raise in ThreadPoolExecutor and 'rank on
    CPU' would move the work across the placement axis."""
    n, why = m._width_from(free_mb=None, threads=32, env_raw=None)
    assert n == 1
    assert "never the CPU" in why


def test_a_card_with_no_room_left_still_gets_one_lane_never_zero():
    """Less free VRAM than the shared weights alone: the floor holds at 1 rather than going negative."""
    assert m._width_from(free_mb=800.0, threads=8, env_raw=None)[0] == 1
    assert m._width_from(free_mb=0.0, threads=8, env_raw=None)[0] == 1


def test_an_operators_number_wins_over_the_derivation():
    n, why = m._width_from(free_mb=3254.0, threads=12, env_raw="6")
    assert n == 6 and "env=6" in why


def test_a_garbage_override_is_loud_and_falls_back_to_the_card(capsys):
    """A typo may not kill a box that is already billed, nor silently become some other number. Watched
    fail with the except branch returning a default."""
    n, _ = m._width_from(free_mb=23000.0, threads=4, env_raw="wide-please")
    assert n == 4, "the derivation must still run"
    assert "not an integer" in capsys.readouterr().err


def test_lane_width_never_asks_the_card_when_an_operator_already_answered(monkeypatch):
    monkeypatch.setenv(m._LANES_ENV, "3")
    monkeypatch.setattr(m, "_free_vram_mb", lambda: pytest.fail("asked the card despite an explicit width"))
    assert m.lane_width() == 3


def test_an_unaskable_card_costs_one_lane_not_a_crashed_boot(monkeypatch):
    """A missing nvidia-smi raises out of subprocess.run — on a rented box that must be a narrow lane with a
    loud line, not an exception that ends main() before the first poll."""
    monkeypatch.delenv(m._LANES_ENV, raising=False)
    monkeypatch.setattr(m, "_free_vram_mb", lambda: (_ for _ in ()).throw(OSError("no nvidia-smi")))
    assert m.lane_width() == 1


# ── the lane is a DIFFERENT pool, and only clip_rank rides it ────────────────────────────────────

def test_a_running_render_does_not_block_a_clip_rank_envelope():
    """THE defect: a render holds the card for minutes and the wave's ranking queued behind it. Watched
    fail with _is_clip_rank forced False — both land on heavy_pool and the rank waits."""
    render_may_finish = threading.Event()
    ranked = threading.Event()
    render_done = threading.Event()

    def heavy(job):
        if job.type == "render":
            render_may_finish.wait(timeout=3)
            render_done.set()
        else:
            ranked.set()

    cp = _CP([{"type": "render", "spec": _SPEC}, {"type": "infer", "request": _RANK_REQ}])
    with cf.ThreadPoolExecutor(max_workers=2) as ops, cf.ThreadPoolExecutor(max_workers=1) as gpu, \
            cf.ThreadPoolExecutor(max_workers=2) as rank:
        _pump(cp, ops, gpu, rank, heavy, 2)
        assert ranked.wait(timeout=3), "the clip_rank envelope waited out the render"
        assert not render_done.is_set(), "it only ran because the render had already finished"
        render_may_finish.set()


def test_two_clip_rank_envelopes_are_in_flight_together():
    """A wave is ~18 of these; on one worker they cost their SUM. Watched fail with the rank routing
    removed — the second envelope never reaches the barrier."""
    both_in = threading.Barrier(2)
    seen: list[str] = []

    def heavy(job):
        both_in.wait(timeout=3)
        seen.append(job.request.kind)

    cp = _CP([{"type": "infer", "request": _RANK_REQ, "corr_id": "a"},
              {"type": "infer", "request": _RANK_REQ, "corr_id": "b"}])
    with cf.ThreadPoolExecutor(max_workers=2) as ops, cf.ThreadPoolExecutor(max_workers=1) as gpu, \
            cf.ThreadPoolExecutor(max_workers=4) as rank:
        _pump(cp, ops, gpu, rank, heavy, 2)
    assert seen == ["clip_rank", "clip_rank"], "the two ranking envelopes never overlapped"


@pytest.mark.parametrize("job,wide", [
    ({"type": "infer", "request": _ALIGN_REQ}, False),
    ({"type": "render", "spec": _SPEC}, False),
    ({"type": "infer", "request": _RANK_REQ}, True),
])
def test_only_clip_rank_leaves_the_narrow_lane(job, wide):
    """align holds a wav2vec2 checkpoint and render holds the encoder — real card tenants, so routing is by
    KIND and an align envelope may never reach the wide pool."""
    from podagent.models import PodJob

    assert agent_main._is_clip_rank(PodJob.model_validate(job)) is wide


def test_two_lanes_missing_the_same_weights_load_the_service_once(monkeypatch, tmp_path):
    """Two lanes racing a cold cache would load 4.6 GB of SigLIP twice — an OOM, not a cache miss. Watched
    fail with _SVC_LOAD_LOCK removed (loads == 2)."""
    loads: list[str] = []
    start = threading.Barrier(2, timeout=3)

    class _Svc:
        def run(self, params, put_url, progress=None):
            return 0.1

    def fake_service(model_id, wdir):
        loads.append(model_id)
        time.sleep(0.3)   # the second lane must reach its cache miss while this load is still running
        return _Svc()

    monkeypatch.setattr("podagent.infer_cliprank.ClipRankService", fake_service)
    monkeypatch.setattr("podagent.weights.ensure", lambda w, mid, note: tmp_path)

    cp = _CP([])
    cp.report_infer_result = lambda payload, wake=None: None
    cache: dict = {}

    def one():
        start.wait()
        agent_main._run_infer(dict(_RANK_REQ), cp, {}, {}, cache, Path("/opt/models/yunet.onnx"), True)

    threads = [threading.Thread(target=one) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert loads == ["siglip"], f"the weights were loaded {len(loads)} times"


# ── inside the envelope: the fetch is not the forward ────────────────────────────────────────────

def _svc() -> ClipRankService:
    svc = ClipRankService.__new__(ClipRankService)
    svc.model_id = "siglip"
    return svc


def test_a_groups_tiles_are_pulled_concurrently(monkeypatch, tmp_path):
    """14 serial tile GETs at ~211 ms each WAS the envelope. Watched fail with the per-url loop restored:
    the barrier times out because tile 2 only starts after tile 1 returned."""
    n = 6
    together = threading.Barrier(n, timeout=3)

    def fake_download(url, dest):
        together.wait()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"img")
        return dest

    monkeypatch.setattr(m, "download", fake_download)
    monkeypatch.setattr("PIL.Image.open", lambda p: types.SimpleNamespace(convert=lambda mode: "img"))
    monkeypatch.setenv(m._FETCH_WORKERS_ENV, str(n))
    m._FETCH_POOL = None

    images, ok = ClipRankService._fetch([f"u{i}" for i in range(n)], tmp_path / "g0")
    assert ok == list(range(n)) and len(images) == n


def test_the_next_groups_tiles_are_already_on_the_wire_while_this_one_forwards(monkeypatch, tmp_path):
    """The card must not idle on a download it could already have had. Watched fail with the fetch moved
    back inside each group's turn: group 1's tile is unrequested when group 0 forwards."""
    from podagent.models import ClipRankParams

    requested: list[str] = []
    seen_at_forward: list[list[str]] = []

    def fake_download(url, dest):
        requested.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"img")
        return dest

    monkeypatch.setattr(m, "download", fake_download)
    monkeypatch.setattr("PIL.Image.open", lambda p: types.SimpleNamespace(convert=lambda mode: "img"))
    monkeypatch.setattr(m, "upload", lambda src, url, ct=None: None)
    m._FETCH_POOL = None

    svc = _svc()
    svc._forward = lambda intent, images: (seen_at_forward.append(list(requested)),
                                           ([0.5] * len(images), [[0.5]] * len(images)))[1]
    params = ClipRankParams(groups=[ClipRankGroup(intent="a", image_urls=["g0t0"]),
                                    ClipRankGroup(intent="b", image_urls=["g1t0"])])
    svc.run(params, "https://storage.example/o/1.json?sig=PUT")

    assert "g1t0" in seen_at_forward[0], "group 1's tile was still unrequested when group 0 forwarded"


def test_the_fetch_pool_is_one_per_process_not_one_per_envelope(monkeypatch):
    """A per-envelope pool multiplies with the lanes — the box-wide budget lesson ops/runner already learned."""
    m._FETCH_POOL = None
    monkeypatch.setenv(m._FETCH_WORKERS_ENV, "2")
    assert m._fetch_pool() is m._fetch_pool()
    assert m._fetch_pool()._max_workers == 2


# ── the forward is chunked, so one lane's VRAM is a number ───────────────────────────────────────

class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class _T:
    float32 = "f32"

    class nn:
        class functional:
            @staticmethod
            def normalize(x, dim=-1):
                return x

    @staticmethod
    def cat(chunks, dim=0):
        return _Rows([r for c in chunks for r in c.rows])


def test_the_image_tower_runs_in_capped_chunks_and_keeps_request_order():
    """Uncapped, a lane's VRAM is a function of a beat's candidate count (2736 MiB at 12 tiles, 3782 at 48)
    and cannot be sized against. Watched fail with the single-forward body restored: one call of 26."""
    batches: list[int] = []

    class _Px:
        dtype = "u8"

        def __init__(self, n):
            self.n = n

    class _Model:
        def get_image_features(self, **kw):
            batches.append(kw["pixel_values"].n)
            return _Rows([[float(i)] for i in range(kw["pixel_values"].n)])

    svc = _svc()
    svc.torch = _T()
    svc.device = "cpu"
    svc.dtype = "f32"
    svc.model = _Model()
    svc.proc = lambda **kw: types.SimpleNamespace(
        to=lambda dev: {"pixel_values": _Px(len(kw["images"]))})

    out = svc._image_features(["img"] * 26)
    assert batches == [m.TILE_BATCH, m.TILE_BATCH, 2], f"the tower ran {batches}"
    assert out.rows == [[float(i)] for i in list(range(12)) + list(range(12)) + [0, 1]], \
        "the concatenation must keep the request's tile order"
