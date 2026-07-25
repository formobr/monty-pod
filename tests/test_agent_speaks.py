"""A claimed job the pod cannot finish must still leave a terminal on the wire.
Observed live: a coalesced clip_rank batch was claimed and evaporated — no result, no error event, no
restart; the brain waited out INFER_TIMEOUT_S on a rented box. Each test is one route to that silence."""
from __future__ import annotations

from pathlib import Path

import pytest
import requests

from podagent import main as agent_main
from podagent.cp import ControlPlane


class _CP:
    """Records what reached the control plane. post_infer_result fails `fail_times` times first."""

    def __init__(self, fail_times: int = 0, event_fails: bool = False) -> None:
        self.results: list[dict] = []
        self.events: list[dict] = []
        self.fail_times = fail_times
        self.event_fails = event_fails
        self.attempts = 0

    def post_infer_result(self, payload: dict) -> None:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise requests.ConnectionError("Remote end closed connection without response")
        self.results.append(payload)

    def post_event(self, payload: dict) -> None:
        if self.event_fails:
            raise requests.ConnectionError("down")
        self.events.append(payload)

    report_infer_result = ControlPlane.report_infer_result
    note = ControlPlane.note


def _run(cp, raw: dict) -> None:
    agent_main._run_infer(raw, cp, {}, {}, {}, Path("/opt/models/yunet.onnx"), False, corr_id=None)


def _bad_request() -> dict:
    # clip_rank with no weights block fails InferRequest validation, i.e. raises before any service is
    # constructed — the failure path without a GPU.
    return {"infer_version": 5, "job_id": "j1", "kind": "clip_rank",
            "model": "siglip", "put_url": "https://r2.example/out/j1/clip_rank.json"}


def test_a_failed_job_posts_an_error_result():
    cp = _CP()
    _run(cp, _bad_request())
    assert len(cp.results) == 1, "a job that raised must leave a terminal, not silence"
    assert cp.results[0]["status"] == "error"
    assert cp.results[0]["kind"] == "clip_rank"


def test_the_agent_announces_the_claim_before_it_can_die():
    cp = _CP()
    _run(cp, _bad_request())
    steps = [e["step"] for e in cp.events if e.get("status") == "step"]
    assert any("claimed clip_rank" in s for s in steps), (
        "without a claim event a pod that dies mid-job looks like one that never got the job")


def test_a_dropped_terminal_is_retried_not_lost(monkeypatch):
    # The live signature: minutes of work, then the pooled socket is dead and the POST never reaches Caddy.
    monkeypatch.setattr("podagent.cp.time.sleep", lambda _s: None)
    cp = _CP(fail_times=3)
    _run(cp, _bad_request())
    assert cp.attempts == 4 and len(cp.results) == 1


def test_an_undeliverable_terminal_falls_back_to_an_event_carrying_the_wake_key(monkeypatch):
    monkeypatch.setattr("podagent.cp.time.sleep", lambda _s: None)
    cp = _CP(fail_times=99)
    _run(cp, _bad_request())
    assert not cp.results
    err = [e for e in cp.events if e.get("status") == "error"]
    assert len(err) == 1, "the last resort must still put the failure on the wire"
    assert err[0]["corr_id"] == "out/j1/clip_rank.json", (
        "the CP echoes corr_id onto its synthesized terminal — without it the awaiter starves anyway")


def test_a_progress_event_never_kills_the_job(monkeypatch):
    monkeypatch.setattr("podagent.cp.time.sleep", lambda _s: None)
    cp = _CP(event_fails=True)
    _run(cp, _bad_request())
    assert len(cp.results) == 1, "a pod must not die of failing to say what it is doing"


def test_a_base_exception_reports_before_it_unwinds(monkeypatch):
    # SystemExit/KeyboardInterrupt/a CUDA abort are BaseException: `except Exception` let them out with the
    # claimed job unreported, which is the silence itself.
    def _boom(*_a, **_kw):
        raise KeyboardInterrupt("torch aborted")

    monkeypatch.setattr(agent_main.InferRequest, "model_validate", staticmethod(_boom))
    cp = _CP()
    with pytest.raises(KeyboardInterrupt):
        _run(cp, _bad_request())
    assert cp.results and cp.results[0]["status"] == "error"


def test_weights_fetch_reports_progress(monkeypatch, tmp_path):
    # 4.3 GB over a datacenter link is ~10 min of a live pod being mute; the only honest reading of that
    # gap without these events is "the box is dead".
    import hashlib
    import io
    import tarfile

    from podagent import artifact

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"{}"
        info = tarfile.TarInfo("config.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()

    class _Ref:
        url = "https://r2.example/w.tar"
        sha256 = hashlib.sha256(blob).hexdigest()
        size = len(blob)

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, _n):
            yield blob

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(artifact.requests, "get", lambda *_a, **_kw: _Resp())
    seen: list[str] = []
    artifact.ensure_tree(_Ref(), tmp_path, "weights siglip", seen.append)
    assert any("cache MISS" in s for s in seen) and any("ready" in s for s in seen)
