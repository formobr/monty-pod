"""ClipRankService — the SigLIP forward, hermetic (stub towers, no weights, no GPU, no network).

What the pod side must guarantee, and what each test pins:
  • scores AND embeds come out of ONE forward, 4dp, in the request's image order;
  • intent "" is embed-only: the TEXT tower is never touched and every score is null (named empty_intent);
  • an image the pod cannot fetch is score null/embed None + a NAMED error instead of failing a whole batch
    of beats — never a sentinel number (tests/test_failed_items_carry_errors.py pins the error shapes);
  • nothing in the result carries a ranking or a threshold — the payload is numbers only.
"""
from __future__ import annotations

import contextlib
import concurrent.futures as cf
import threading
import types
from pathlib import Path

import pytest

from podagent.infer_cliprank import ClipRankService
from podagent.models import ClipRankGroup


class _Vec:
    def __init__(self, rows):
        self.rows = rows

    def __matmul__(self, other):
        return _Vec([[sum(a * b for a, b in zip(r, c)) for c in other.rows] for r in self.rows])

    @property
    def T(self):
        return _Vec([list(c) for c in zip(*self.rows)])

    def squeeze(self, _d):
        return _Vec([r[0] for r in self.rows]) if self.rows and isinstance(self.rows[0], list) else self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.rows


class _In(dict):
    def to(self, _dev):
        return self


class _Torch:
    float32 = "f32"
    float16 = "f16"

    class nn:
        class functional:
            @staticmethod
            def normalize(x, dim=-1):
                return x

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()


def _svc(text_calls: list | None = None) -> ClipRankService:
    """A service with stub towers — bypasses __init__ so no weights load."""
    svc = ClipRankService.__new__(ClipRankService)
    svc.torch = _Torch()
    svc.device = "cpu"
    svc.dtype = _Torch.float32
    svc.model_id = "google/siglip2-so400m-patch14-384"
    svc.parallel = 1
    svc._slots = threading.BoundedSemaphore(1)
    svc.proc = lambda **kw: _In()

    class _Model:
        def get_image_features(self, **kw):
            return _Vec([[0.6, 0.8], [0.8, 0.6]])

        def get_text_features(self, **kw):
            if text_calls is not None:
                text_calls.append(kw)
            return _Vec([[1.0, 0.0]])

    svc.model = _Model()
    return svc


def test_one_forward_yields_scores_and_embeds() -> None:
    scores, embeds = _svc()._forward("a falling crypto chart", ["img", "img"])
    assert scores == [0.6, 0.8]
    assert embeds == [[0.6, 0.8], [0.8, 0.6]]


def test_empty_intent_is_embed_only_and_never_touches_the_text_tower() -> None:
    calls: list = []
    scores, embeds = _svc(calls)._forward("", ["img", "img"])
    assert scores == [None, None], "an embed-only group's scores have nothing to mean — null, never -1.0"
    assert embeds == [[0.6, 0.8], [0.8, 0.6]], "the image tower is text-independent — embeds still come back"
    assert calls == [], "no intent must mean no text forward"


def test_feat_unwraps_a_pooled_model_output_so_the_forward_never_crashes() -> None:
    """Regression: some transformers versions return a BaseModelOutputWithPooling (pooled embed in
    .pooler_output) from get_image/text_features instead of a bare tensor; F.normalize then calls .norm() on
    the OBJECT — the '...has no attribute norm' crash that lost every b-roll. _feat must unwrap .pooler_output
    so the forward runs. WITHOUT the guard this test raises AttributeError (the object has no .float())."""
    class _Pooled:
        def __init__(self, t):
            self.pooler_output = t

    svc = _svc()
    svc.model.get_image_features = lambda **kw: _Pooled(_Vec([[0.6, 0.8], [0.8, 0.6]]))
    svc.model.get_text_features = lambda **kw: _Pooled(_Vec([[1.0, 0.0]]))
    scores, embeds = svc._forward("some intent", ["img", "img"])
    assert scores == [0.6, 0.8]
    assert embeds == [[0.6, 0.8], [0.8, 0.6]]


def test_unfetchable_image_has_no_score_and_keeps_request_order(monkeypatch, tmp_path) -> None:
    import podagent.infer_cliprank as m

    def fake_download(url: str, dest: Path) -> Path:
        if url == "dead":
            raise OSError("404 on presigned GET")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"img")
        return dest

    monkeypatch.setattr(m, "download", fake_download)
    monkeypatch.setattr("PIL.Image.open", lambda p: types.SimpleNamespace(convert=lambda mode: "img"))

    group = ClipRankGroup(intent="chart", image_urls=["ok1", "dead", "ok2"])
    res = _svc()._run_group(group, tmp_path / "g0")

    assert res.status == "ok"
    assert res.scores == [0.6, None, 0.8], "a dead tile has no score; the live ones keep their slots"
    assert res.embeds == [[0.6, 0.8], None, [0.8, 0.6]]
    assert res.errors[0] is None and res.errors[2] is None
    assert res.errors[1].kind == "fetch_failed" and "404 on presigned GET" in res.errors[1].detail


def test_sheet_cells_are_exact_pixels_in_request_order() -> None:
    """Packed transport may change the object count, never a decoded pixel or the scorer's row order."""
    from PIL import Image
    from podagent.infer_cliprank import _gather

    left = Image.new("RGB", (2, 2), (11, 22, 33))
    right = Image.new("RGB", (2, 2), (101, 77, 9))
    sheet = Image.new("RGB", (4, 2))
    sheet.paste(left, (0, 0)); sheet.paste(right, (2, 0))
    future = cf.Future(); future.set_result(sheet)

    images, order, errors = _gather([future, future], [(0, 0, 2, 2, 4, 2), (2, 0, 2, 2, 4, 2)])

    assert order == [0, 1] and errors == [None, None]
    assert list(images[0].getdata()) == list(left.getdata())
    assert list(images[1].getdata()) == list(right.getdata())
    score = lambda im: sum(sum(px) for px in im.getdata())
    assert [score(im) for im in images] == [score(left), score(right)], "pixel-derived scores moved"


def test_sheet_cell_missing_or_wrong_shape_fails_loudly() -> None:
    from PIL import Image
    from podagent.infer_cliprank import _gather

    dead = cf.Future(); dead.set_result(None)
    with pytest.raises(ValueError, match="could not be fetched"):
        _gather([dead], [(0, 0, 2, 2, 2, 2)])
    wrong = cf.Future(); wrong.set_result(Image.new("RGB", (3, 2)))
    with pytest.raises(ValueError, match="expected sheet 2x2"):
        _gather([wrong], [(0, 0, 2, 2, 2, 2)])


def test_sheet_download_http_failure_names_status_and_redacts_capability(monkeypatch, tmp_path) -> None:
    import podagent.infer_cliprank as m

    class HttpError(Exception):
        response = types.SimpleNamespace(status_code=404)

    monkeypatch.setattr(m, "download",
                        lambda url, dest: (_ for _ in ()).throw(HttpError("signed GET failed")))
    future = cf.Future()
    future.set_result(m._fetch_tile("https://r2.test/sheets/a.png?X-Amz-Signature=secret", tmp_path / "a"))

    with pytest.raises(ValueError) as caught:
        m._gather([future], [(0, 0, 2, 2, 2, 2)])

    message = str(caught.value)
    assert "404" in message
    assert "https://r2.test/sheets/a.png" in message
    assert "X-Amz-Signature" not in message and "secret" not in message


def test_sheet_decode_failure_names_exception_class(monkeypatch, tmp_path) -> None:
    import podagent.infer_cliprank as m

    monkeypatch.setattr(m, "download", lambda url, dest: dest)

    class DecodeFailure(Exception):
        pass

    monkeypatch.setattr("PIL.Image.open",
                        lambda path: (_ for _ in ()).throw(DecodeFailure("bad pixels")))
    future = cf.Future()
    future.set_result(m._fetch_tile("https://r2.test/sheets/a.png?sig=secret", tmp_path / "a"))

    with pytest.raises(ValueError, match="DecodeFailure"):
        m._gather([future], [(0, 0, 2, 2, 2, 2)])


def test_no_cells_drops_failed_tile_and_keeps_other_images(tmp_path) -> None:
    import podagent.infer_cliprank as m
    from PIL import Image

    failed = cf.Future()
    failed.set_result((None, "OSError", "https://r2.test/a.png"))
    live = cf.Future()
    image = Image.new("RGB", (2, 2))
    live.set_result((image, None, "https://r2.test/b.png"))

    images, order, errors = m._gather([failed, live])

    assert images == [image] and order == [1]
    assert errors[1] is None and errors[0].kind == "fetch_failed" and "OSError" in errors[0].detail


def test_sheet_cell_contract_rejects_missing_parallel_entry_and_bad_geometry() -> None:
    with pytest.raises(ValueError, match="parallel"):
        ClipRankGroup(intent="x", image_urls=["sheet", "sheet"], image_cells=[(0, 0, 2, 2, 4, 2)])
    with pytest.raises(ValueError, match="positive sizes"):
        ClipRankGroup(intent="x", image_urls=["sheet"], image_cells=[(0, 0, 0, 2, 4, 2)])


def test_a_wholly_dead_group_is_an_error_without_a_forward(monkeypatch, tmp_path) -> None:
    import podagent.infer_cliprank as m

    monkeypatch.setattr(m, "download", lambda url, dest: (_ for _ in ()).throw(OSError("gone")))
    svc = _svc()
    svc.model = None  # any forward attempt would explode

    res = svc._run_group(ClipRankGroup(intent="chart", image_urls=["a", "b"]), tmp_path / "g0")
    assert res.status == "error" and res.error is not None and res.error.kind == "fetch_failed"
    assert res.scores == [None, None] and res.embeds == [None, None]


def test_payload_carries_numbers_only(monkeypatch, tmp_path) -> None:
    """The whole run: what lands at put_url is model + per-group scores/embeds, no ranking, no threshold."""
    import json

    import podagent.infer_cliprank as m
    from podagent.models import ClipRankParams

    monkeypatch.setattr(m, "download", lambda url, dest: (dest.parent.mkdir(parents=True, exist_ok=True),
                                                          dest.write_bytes(b"img"), dest)[-1])
    monkeypatch.setattr("PIL.Image.open", lambda p: types.SimpleNamespace(convert=lambda mode: "img"))
    put = tmp_path / "out" / "clip_rank.json"
    monkeypatch.setattr(m, "upload", lambda src, url, ct=None: (put.parent.mkdir(parents=True, exist_ok=True),
                                                               put.write_bytes(src.read_bytes())))

    params = ClipRankParams(groups=[ClipRankGroup(intent="chart", image_urls=["a", "b"]),
                                    ClipRankGroup(intent="", image_urls=["c", "d"])])
    result = _svc().run(params, "https://storage.example/o/1.json?sig=PUT")

    assert result.infer_s >= 0
    assert set(result.timings) == {
        "infer_s", "tile_gather_work_s", "forward_work_s", "payload_s", "upload_s", "work_s",
        "unique_image_sets_n", "reused_groups_n", "parallel_width_n"}
    body = json.loads(put.read_text())
    assert set(body) == {"model", "groups", "timings"}, "no ranking, no threshold, no rationale crosses back"
    assert [g["scores"] for g in body["groups"]] == [[0.6, 0.8], [None, None]]
    assert [[e and e["kind"] for e in g["errors"]] for g in body["groups"]] == [
        [None, None], ["empty_intent", "empty_intent"]]
    assert all(set(g) == {"status", "scores", "embeds", "errors"} for g in body["groups"])
    assert [g["status"] for g in body["groups"]] == ["ok", "ok"]


def test_identical_url_groups_reuse_one_image_forward_without_changing_order(monkeypatch, tmp_path) -> None:
    """Two phrasings are two text scores, not two downloads/decodes/image towers."""
    import json
    import podagent.infer_cliprank as m
    from podagent.models import ClipRankParams

    fetched: list[str] = []
    monkeypatch.setattr(m, "download", lambda url, dest: (fetched.append(url),
        dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"img"), dest)[-1])
    monkeypatch.setattr("PIL.Image.open", lambda p: types.SimpleNamespace(convert=lambda mode: "img"))
    put = tmp_path / "out.json"
    monkeypatch.setattr(m, "upload", lambda src, url, ct=None: put.write_bytes(src.read_bytes()))
    svc = _svc()
    image_calls = 0
    original = svc._image_features

    def counted(images):
        nonlocal image_calls
        image_calls += 1
        return original(images)

    svc._image_features = counted
    params = ClipRankParams(groups=[
        ClipRankGroup(intent="first", image_urls=["a", "b"]),
        ClipRankGroup(intent="second", image_urls=["a", "b"]),
    ])
    run = svc.run(params, "put")
    assert fetched == ["a", "b"] and image_calls == 1
    assert [g["scores"] for g in json.loads(put.read_text())["groups"]] == [[0.6, 0.8], [0.6, 0.8]]
    assert run.timings["unique_image_sets_n"] == 1 and run.timings["reused_groups_n"] == 1


def test_service_is_registered_for_the_clip_rank_kind() -> None:
    from podagent import main

    assert "clip_rank" in main.INFER_KINDS


@pytest.mark.parametrize("intent", ["", "some intent"])
def test_forward_rounds_to_four_places(intent: str) -> None:
    """4dp is the wire convention (cosine error ~1e-3) — it must hold on both branches."""
    _, embeds = _svc()._forward(intent, ["img", "img"])
    assert all(len(str(x).split(".")[-1]) <= 4 for row in embeds for x in row)


def _origin_error(monkeypatch, tmp_path, outcome, dest=None):
    """Run ONE origin tile through the real `_fetch_origin_tile` with the guard's fetch faked."""
    import podagent.infer_cliprank as m

    def fake(url, deadline=None):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(m, "_origin_fetch", fake)
    url = "https://images.pexels.com/photos/1/x.jpg?token=SIGNEDTOKEN"
    _, err, _, measure = m._fetch_origin_tile(url, dest or tmp_path / "w" / "0.img")
    assert measure is None and err is not None
    assert "SIGNEDTOKEN" not in err.detail and "?" not in err.detail and len(err.detail) <= 300
    return err


def test_a_cloudflare_403_reaches_the_tile_detail_with_status_and_code(monkeypatch, tmp_path) -> None:
    """PROD 2026-10-28 job 20168ee1: 538/538 origin tiles failed and no detail carried the HTTP status."""
    import http.client
    import urllib.error

    from podagent import originrule

    exc = urllib.error.HTTPError("https://images.pexels.com/photos/1/x.jpg", 403, "Forbidden",
                                 http.client.HTTPMessage(), None)
    exc.origin_failure = originrule.OriginFailure(
        phase="http", host="images.pexels.com", addr="104.18.1.1", family="v4", status=403,
        cf_mitigated="challenge", retry_after="30", cf_code="1020", exc="HTTPError")
    err = _origin_error(monkeypatch, tmp_path, exc)
    assert err.kind == "fetch_failed"
    assert err.detail.startswith("http HTTPError")
    for part in ("HTTP 403", "cf-1020", "mitigated=challenge", "retry-after=30", "via 104.18.1.1/v4"):
        assert part in err.detail
    assert err.detail.endswith("— https://images.pexels.com/photos/1/x.jpg")


def test_an_html_body_is_content_and_an_unwritable_dest_is_write(monkeypatch, tmp_path) -> None:
    from podagent import originrule

    html = originrule.OriginBody(b"<html>Just a moment...</html>")
    err = _origin_error(monkeypatch, tmp_path, html)
    assert err.kind == "decode_error" and err.detail.startswith("content ")
    # the same bytes without the flag are plain undecodable data
    err = _origin_error(monkeypatch, tmp_path, b"junk")
    assert err.kind == "decode_error" and err.detail.startswith("decode ")
    blocker = tmp_path / "file"
    blocker.write_bytes(b"")
    err = _origin_error(monkeypatch, tmp_path, b"junk", dest=blocker / "sub" / "0.img")
    assert err.kind == "decode_error" and err.detail.startswith("write ")


def test_an_exception_without_a_failure_record_keeps_the_old_detail(monkeypatch, tmp_path) -> None:
    err = _origin_error(monkeypatch, tmp_path, TimeoutError("slow"))
    assert err.kind == "fetch_timeout"
    assert err.detail == "TimeoutError: slow — https://images.pexels.com/photos/1/x.jpg"


def _phase_rig(monkeypatch, tmp_path):
    """An inline pool, an injected clock the origin fetch advances, stub towers scoring every image 0.5, and a
    captured PUT. Returns (run(urls) -> parsed result, clock cell)."""
    import io
    import json

    from PIL import Image

    import podagent.infer_cliprank as m
    from podagent import originrule
    from podagent.models import ClipRankParams

    class _Inline:
        def submit(self, fn, *args, **kwargs):
            fut = cf.Future()
            try:
                fut.set_result(fn(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001 — mirror a real Future
                fut.set_exception(exc)
            return fut

    class _Clock(list):
        png = b""

    clock = _Clock([0.0])
    monkeypatch.setattr(originrule, "_now", lambda: clock[0])
    monkeypatch.setattr(m, "_FETCH_POOL", _Inline())
    monkeypatch.setattr(m, "_ORIGIN_CACHE", m._OriginCache(m._ORIGIN_CACHE_BYTES))
    put = tmp_path / "out.json"
    monkeypatch.setattr(m, "upload", lambda src, url, ct=None: put.write_bytes(src.read_bytes()))
    svc = _svc()
    svc._prepare = lambda images, ok: ((None, [], ok) if not images
                                       else (list(images), [[0.6, 0.8] for _ in images], ok))
    svc._text_scores = lambda intent, ie: [0.5] * len(ie)
    buf = io.BytesIO()
    Image.new("RGB", (4, 3)).save(buf, "PNG")
    clock.png = buf.getvalue()

    def run(urls):
        svc.run(ClipRankParams(groups=[ClipRankGroup(intent="chart", origin_urls=urls)]), "put")
        return json.loads(put.read_text())["groups"][0]

    return run, clock


def test_a_fetch_phase_whose_host_answers_after_the_deadline_ends_at_120s_and_names_every_unfetched_tile(
        monkeypatch, tmp_path) -> None:
    """PROD 2026-10-28: a preview fetched 500-1657 origin tiles with nothing bounding the phase as a whole."""
    import podagent.infer_cliprank as m
    from podagent import originrule

    run, clock = _phase_rig(monkeypatch, tmp_path)
    urls = [f"https://images.pexels.com/p/{i}.jpg" for i in range(600)]
    fetched: list[str] = []
    deadlines: list[float] = []

    def fake(url, deadline=None):
        fetched.append(url)
        deadlines.append(deadline)
        if len(fetched) <= 2:
            clock[0] += 10.0                    # two tiles arrive
            return clock.png
        clock[0] = deadline                     # the third's host answers only after the phase is over:
        raise originrule.DeadlineExceeded("deadline")   # the bounded fetch gives up exactly at the deadline

    monkeypatch.setattr(m, "_origin_fetch", fake)
    g = run(urls)
    assert m._ORIGIN_FETCH_PHASE_S == 120 and set(deadlines) == {120.0}
    assert clock[0] == 120.0, "the phase ended at 120 s on the injected clock"
    assert len(fetched) == 3, "the 597 queued fetches were cancelled, not started"
    assert g["status"] == "ok" and g["scores"][:2] == [0.5, 0.5] and set(g["scores"][2:]) == {None}
    assert g["errors"][:2] == [None, None]
    assert {e["kind"] for e in g["errors"][2:]} == {"fetch_timeout"}
    assert all(e["detail"].startswith("deadline ") for e in g["errors"][2:]) and len(g["errors"][2:]) == 598


def test_a_healthy_fetch_phase_is_unchanged(monkeypatch, tmp_path) -> None:
    import podagent.infer_cliprank as m

    run, clock = _phase_rig(monkeypatch, tmp_path)
    urls = [f"https://images.pexels.com/p/{i}.jpg" for i in range(600)]

    def fake(url, deadline=None):
        clock[0] += 0.01
        return clock.png

    monkeypatch.setattr(m, "_origin_fetch", fake)
    g = run(urls)
    assert g["status"] == "ok" and g["errors"] == [None] * 600 and g["scores"] == [0.5] * 600
