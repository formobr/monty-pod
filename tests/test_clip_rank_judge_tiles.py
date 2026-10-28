"""The pod answers a judge-tiles request by writing one small JPEG per tile to its PUT url, in parallel, from the
bytes its rank request already fetched (MISC-592).

  • a successful origin fetch keeps its bytes in a process-wide cache bounded by total bytes, LRU out first;
  • a judge tile reads the cache (`hit`) or refetches through the origin guard with the rank's caps (`miss`);
  • each tile is RGB, at most 512 px on its long side, JPEG quality 82, PUT as image/jpeg;
  • the tile PUTs run concurrently; a failed one is a named `put_failed` result, never a raise;
  • the result is ONE JSON PUT carrying the tile results and the PUT distribution in its timings;
  • main sends a judge-tiles request to run_judge_tiles with no weights fetch, load or kind release.

Hermetic: the network is a fake, PUTs are captured, and no verdict depends on a clock or a thread's timing —
the PUT walls come from injected put_trace rows, and concurrency is proven by a rendezvous serial code can't reach.
"""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import threading
from pathlib import Path

import pytest
import requests
from jsonschema import Draft202012Validator
from PIL import Image

import podagent.infer_cliprank as m
from podagent import cp, originrule
from podagent import main as agent_main
from podagent.models import ClipRankGroup, ClipRankJudgeTile, ClipRankParams
from test_infer_cliprank import _svc

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
RESULT_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "clip_rank.schema.json").read_text()))
RESULT_URL = "https://storage.example/o/r.json?sig=PUT"

A = "https://images.stock.example/photo/1/thumb.jpg"
B = "https://images.stock.example/photo/2/thumb.jpg"
C = "https://images.stock.example/photo/3/thumb.jpg"


def _put_url(i: int) -> str:
    return f"https://storage.example/o/judge/{i}.jpg?sig=PUT"


def _png(w: int, h: int) -> bytes:
    img = Image.new("RGB", (w, h))
    img.putdata([((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256) for y in range(h) for x in range(w)])
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class _InlinePool:
    """submit() runs on the caller's thread — the rank's fetch pool, made deterministic."""

    def submit(self, fn, *args, **kwargs):
        fut = cf.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 — mirror a real Future
            fut.set_exception(exc)
        return fut


@pytest.fixture
def rig(monkeypatch):
    """A fresh origin cache, a recording fake guard fetch, and captured PUTs (url → bytes, content type)."""
    monkeypatch.setattr(m, "_ORIGIN_CACHE", m._OriginCache(m._ORIGIN_CACHE_BYTES))
    monkeypatch.setattr(m, "_FETCH_POOL", _InlinePool())
    monkeypatch.setattr(m, "download", lambda url, dest: pytest.fail(f"cp.download fetched {url}"))
    bodies: dict = {}
    fetches: list[tuple[str, int, float]] = []
    puts: dict[str, tuple[bytes, str | None]] = {}
    lock = threading.Lock()

    def fetch(url, *, max_bytes, timeout_s, deadline=None):
        with lock:
            fetches.append((url, max_bytes, timeout_s))
        out = bodies[url]
        if isinstance(out, BaseException):
            raise out
        return out

    def put(src, url, content_type=None):
        data = Path(src).read_bytes()
        with lock:
            puts[url] = (data, content_type)

    monkeypatch.setattr(originrule, "fetch", fetch)
    monkeypatch.setattr(cp, "upload", put)
    monkeypatch.setattr(m, "upload", put)

    class Rig:
        pass

    r = Rig()
    r.bodies, r.fetches, r.puts = bodies, fetches, puts
    r.progress = []

    def judge(*origins: str) -> dict:
        params = ClipRankParams(groups=[], judge_tiles=[ClipRankJudgeTile(origin_url=u, put_url=_put_url(i))
                                                        for i, u in enumerate(origins)])
        r.run = m.run_judge_tiles(params, RESULT_URL, r.progress.append, model="siglip")
        body = json.loads(puts[RESULT_URL][0])
        assert next(RESULT_SCHEMA.iter_errors(body), None) is None
        return body

    r.judge = judge
    return r


def _rank(rig, *origins: str) -> None:
    svc = _svc()
    svc._prepare = lambda images, ok: ((None, [], ok) if not images
                                       else (list(images), [[0.6, 0.8] for _ in images], ok))
    svc._text_scores = lambda intent, ie: [0.5] * len(ie)
    svc.run(ClipRankParams(groups=[ClipRankGroup(intent="chart", origin_urls=list(origins))]), RESULT_URL)
    rig.puts.pop(RESULT_URL)


def test_a_judge_tile_reuses_the_rank_fetch(rig) -> None:
    rig.bodies.update({A: _png(40, 30), B: _png(20, 10)})
    _rank(rig, A, B)
    assert [u for u, *_ in rig.fetches] == [A, B]
    body = rig.judge(A, B)
    assert [u for u, *_ in rig.fetches] == [A, B], "a warm judge tile never pays the provider fetch twice"
    assert body["groups"] == []
    assert [t["cache"] for t in body["judge_tiles"]] == ["hit", "hit"]
    assert [t["status"] for t in body["judge_tiles"]] == ["ok", "ok"]
    for i in range(2):
        data, ctype = rig.puts[_put_url(i)]
        assert ctype == "image/jpeg" and Image.open(io.BytesIO(data)).format == "JPEG"
    assert body["timings"]["judge_cache_miss_n"] == 0
    # one JSON PUT for the request, one JPEG PUT per tile, nothing else
    assert sorted(rig.puts) == sorted([RESULT_URL, _put_url(0), _put_url(1)])


def test_a_cache_miss_refetches_through_the_guard_and_is_counted(rig) -> None:
    rig.bodies.update({A: _png(40, 30), B: originrule.OriginRefused("REFUSED: private address"),
                       C: requests.ConnectionError("reset")})
    body = rig.judge(A, B, C)
    assert rig.fetches == [(u, m._ORIGIN_MAX_BYTES, m._TILE_TIMEOUT_S) for u in (A, B, C)]
    ok, refused, failed = body["judge_tiles"]
    assert ok["status"] == "ok" and ok["cache"] == "miss"
    assert refused == {"status": "error", "error": {"kind": "origin_refused", "detail": refused["error"]["detail"]}}
    assert failed["status"] == "error" and failed["error"]["kind"] == "fetch_failed"
    assert body["timings"]["judge_cache_miss_n"] == 3
    assert sorted(rig.puts) == sorted([RESULT_URL, _put_url(0)])
    # the refetched bytes are kept: the next request hits
    assert rig.judge(A)["judge_tiles"][0]["cache"] == "hit"
    assert len(rig.fetches) == 3
    # one progress line per request naming result kinds and cache misses
    assert rig.progress[0] == "clip_rank judge tiles 3: fetch_failed=1, ok=1, origin_refused=1; cache_miss=3"
    assert rig.progress[1] == "clip_rank judge tiles 1: ok=1; cache_miss=0"
    assert len(rig.progress) == 2


def test_the_jpeg_is_512_long_side_quality_82(rig) -> None:
    src = _png(1200, 600)
    rig.bodies.update({A: src, B: _png(300, 200)})
    body = rig.judge(A, B)
    data, _ = rig.puts[_put_url(0)]
    ref = Image.open(io.BytesIO(src)).convert("RGB")
    ref.thumbnail((512, 512))
    want, other = io.BytesIO(), io.BytesIO()
    ref.save(want, "JPEG", quality=82)
    ref.save(other, "JPEG", quality=75)
    assert data == want.getvalue() and data != other.getvalue()
    assert Image.open(io.BytesIO(data)).size == (512, 256)
    assert body["judge_tiles"][0] == {"status": "ok", "bytes": len(data), "w": 512, "h": 256, "cache": "miss"}
    # a tile already under the cap is not upscaled
    assert Image.open(io.BytesIO(rig.puts[_put_url(1)][0])).size == (300, 200)
    assert (m._JUDGE_LONG_SIDE, m._JUDGE_QUALITY) == (512, 82)


def test_puts_run_in_parallel_not_serially(rig, monkeypatch) -> None:
    """Every tile PUT waits until ALL tile PUTs are in flight at once — serial code can never get there."""
    rig.bodies.update({A: _png(8, 8), B: _png(8, 8), C: _png(8, 8)})
    capture = cp.upload
    lock = threading.Lock()
    inflight = [0]
    all_in = threading.Event()

    def rendezvous(src, url, content_type=None):
        if url != RESULT_URL:
            with lock:
                inflight[0] += 1
                if inflight[0] == 3:
                    all_in.set()
            if not all_in.wait(timeout=60):
                raise requests.ConnectionError("tile PUTs ran one at a time")
        capture(src, url, content_type)

    monkeypatch.setattr(cp, "upload", rendezvous)
    body = rig.judge(A, B, C)
    assert [t["status"] for t in body["judge_tiles"]] == ["ok", "ok", "ok"]


def test_a_failed_put_is_a_named_tile_result(rig, monkeypatch) -> None:
    rig.bodies.update({A: _png(8, 8), B: _png(8, 8)})
    capture = cp.upload

    def put(src, url, content_type=None):
        if url == _put_url(1):
            raise cp.PutRejected("PUT rejected: HTTP 403")
        capture(src, url, content_type)

    monkeypatch.setattr(cp, "upload", put)
    body = rig.judge(A, B)
    ok, bad = body["judge_tiles"]
    assert ok["status"] == "ok"
    assert bad["status"] == "error" and bad["error"]["kind"] == "put_failed"
    assert "sig=" not in bad["error"]["detail"] and "storage.example" in bad["error"]["detail"]
    assert set(bad) == {"status", "error"}
    assert sorted(rig.puts) == sorted([RESULT_URL, _put_url(0)])
    assert "put_failed=1" in rig.progress[0]


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.results: list[dict] = []

    def send_event(self, ev: dict, *, wait: bool = False) -> bool:
        self.events.append(ev)
        return True

    def note(self, ev: dict) -> None:
        self.events.append(ev)

    def report_infer_result(self, payload: dict, wake=None) -> bool:
        self.results.append(payload)
        return True


@pytest.mark.parametrize("dry", [False, True], ids=["full", "dry"])
def test_a_judge_tiles_request_loads_no_weights_and_releases_nothing(monkeypatch, tmp_path, dry) -> None:
    monkeypatch.setattr(agent_main, "_LIVE_MARK", tmp_path / "podagent.alive")
    touched: list[str] = []
    calls: list[tuple] = []
    monkeypatch.setattr("podagent.ops.dry.armed", lambda: dry)
    monkeypatch.setattr("podagent.ops.dry.run_clip_rank", lambda *a, **k: touched.append("dry"))
    monkeypatch.setattr("podagent.weights.ensure", lambda *a, **k: touched.append("fetch") or tmp_path)
    monkeypatch.setattr("podagent.infer_cliprank.ClipRankService", lambda *a, **k: touched.append("load"))
    monkeypatch.setattr("podagent.infer_lanes.release_other_kinds", lambda *a, **k: touched.append("release"))

    def run_judge_tiles(params, put_url, progress=None, *, model="judge_tiles"):
        calls.append(([t.origin_url for t in params.judge_tiles], put_url, model))
        return m.ClipRankRun(infer_s=0.25, timings={"infer_s": 0.25, "judge_tiles_n": 2.0})

    def dry_judge_tiles(params, put_url, progress=None):
        calls.append(([t.origin_url for t in params.judge_tiles], put_url, "contour-dry"))
        return m.ClipRankRun(infer_s=0.25, timings={"infer_s": 0.25, "judge_tiles_n": 2.0})

    monkeypatch.setattr(m, "run_judge_tiles", run_judge_tiles)
    monkeypatch.setattr("podagent.ops.dry.run_judge_tiles", dry_judge_tiles)
    raw = json.loads((CONTRACTS / "examples" / "infer_request.clip_rank_judge_tiles.json").read_text())
    align_cache = {"a" * 64: object()}
    rank_cache = {"b" * 64: object()}
    cp_ = _CP()
    agent_main._run_infer(raw, cp_, align_cache, {}, rank_cache, Path("/opt/models/yunet.onnx"), True,
                          corr_id="c", session_id="s", one_kind_resident=True)
    assert touched == []
    assert list(align_cache) == ["a" * 64] and list(rank_cache) == ["b" * 64]
    # the dry tier answers with its own stub (no origin fetch); the full pod with the real one
    assert calls == [([t["origin_url"] for t in raw["clip_rank"]["judge_tiles"]], raw["put_url"],
                      "contour-dry" if dry else raw["model"])]
    (res,) = cp_.results
    assert res["status"] == "ok" and res["kind"] == "clip_rank" and res["corr_id"] == "c"


def test_the_cache_is_bounded_by_bytes() -> None:
    assert m._ORIGIN_CACHE_BYTES == 256 * 1024 * 1024
    cache = m._OriginCache(10)
    cache.put("a", b"1234")
    cache.put("b", b"1234")
    assert cache.get("a") == b"1234"          # a is now the most recently used
    cache.put("c", b"1234")                   # 12 > 10: the least recently used (b) goes
    assert cache.get("b") is None and cache.get("a") == b"1234" and cache.get("c") == b"1234"
    assert cache.size() == 8
    cache.put("a", b"1234567")                # a replaced in place: 11 > 10, so c (now oldest) goes
    assert cache.size() == 7 and cache.get("c") is None
    cache.put("huge", b"x" * 11)              # larger than the whole bound: never kept, nothing evicted
    assert cache.get("huge") is None and cache.get("a") == b"1234567"


def test_timings_carry_the_put_distribution(rig, monkeypatch) -> None:
    rig.bodies.update({A: _png(8, 8), B: _png(8, 8), C: _png(8, 8)})
    capture = cp.upload
    s = 1_000_000_000
    # per tile: its attempt rows as cp.upload records them — tile 2 retried once (3 s end to end)
    traces = {
        _put_url(0): [dict(attempt=1, start_mono_ns=0, end_mono_ns=1 * s, outcome="ok")],
        _put_url(1): [dict(attempt=1, start_mono_ns=0, end_mono_ns=1 * s, outcome="retry",
                           retry_end_mono_ns=2 * s),
                      dict(attempt=2, start_mono_ns=2 * s, end_mono_ns=3 * s, outcome="ok")],
        _put_url(2): [dict(attempt=1, start_mono_ns=5 * s, end_mono_ns=7 * s, outcome="ok")],
    }

    def put(src, url, content_type=None):
        for row in traces.get(url, []):
            cp.put_trace.add(**row)
        capture(src, url, content_type)

    monkeypatch.setattr(cp, "upload", put)
    body = rig.judge(A, B, C)
    t = body["timings"]
    assert t["judge_tiles_n"] == 3 and t["judge_cache_miss_n"] == 3
    assert t["judge_put_p50_s"] == 2.0 and t["judge_put_max_s"] == 3.0 and t["judge_put_retry_n"] == 1
    for key in ("judge_compose_s", "judge_put_wall_s", "infer_s"):
        assert t[key] >= 0
    for key, value in t.items():
        assert rig.run.timings[key] == value or key == "infer_s"
    assert {"payload_s", "upload_s", "work_s"} <= set(rig.run.timings)
