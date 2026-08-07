"""A claimed job the pod cannot finish must still leave a terminal on the wire.
Observed live: a coalesced clip_rank batch was claimed and evaporated — no result, no error event, no
restart; the brain waited out INFER_TIMEOUT_S on a rented box. Each test is one route to that silence."""
from __future__ import annotations

from pathlib import Path

import pytest
from podagent import main as agent_main
from podagent.cp import ControlPlane
from podagent.event_stream import FrameRejected


class _CP:
    """Records which typed frame API the agent used."""

    def __init__(self, event_fails: bool = False) -> None:
        self.results: list[dict] = []
        self.events: list[dict] = []
        self.timeline: list[tuple[str, dict]] = []
        self.event_fails = event_fails

    def send_result(self, payload: dict, *, wait: bool = True) -> bool:
        self.results.append(payload)
        self.timeline.append(("result", payload))
        return True

    def send_event(self, payload: dict, *, wait: bool = False) -> bool:
        if self.event_fails:
            raise RuntimeError("down")
        self.events.append(payload)
        self.timeline.append(("event", payload))
        return True

    report_infer_result = ControlPlane.report_infer_result
    note = ControlPlane.note


def _run(cp, raw: dict) -> None:
    agent_main._run_infer(
        raw, cp, {}, {}, {}, Path("/opt/models/yunet.onnx"), False,
        corr_id="corr-1", session_id="session-1")


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
    work_finished = next(
        i for i, (kind, body) in enumerate(cp.timeline)
        if kind == "event" and body.get("phase") == "work_finished")
    result = next(i for i, (kind, _body) in enumerate(cp.timeline) if kind == "result")
    assert work_finished < result, "the pod must finish work before it can deliver and ACK the result"


def test_the_agent_announces_the_claim_before_it_can_die():
    cp = _CP()
    _run(cp, _bad_request())
    steps = [e["step"] for e in cp.events if e.get("status") == "step" and "step" in e]
    assert any("claimed clip_rank" in s for s in steps), (
        "without a claim event a pod that dies mid-job looks like one that never got the job")
    assert all(e.get("job_id") == "j1" for e in cp.events if e.get("status") == "step")
    assert all(e.get("corr_id") == "corr-1" and e.get("session_id") == "session-1"
               for e in cp.events if e.get("status") == "step")
    assert any("j1" in s for s in steps), "the request id still has to be readable in the step text"


def test_an_infer_terminal_uses_result_not_an_error_event_fallback():
    cp = _CP()
    _run(cp, _bad_request())
    assert len(cp.results) == 1 and cp.results[0]["corr_id"] == "corr-1"
    assert not [e for e in cp.events if e.get("status") == "error"], (
        "a result may not be downgraded into the event vocabulary")


def test_a_progress_persist_failure_stops_work_before_it_can_go_silent():
    cp = _CP(event_fails=True)
    with pytest.raises(RuntimeError, match="down"):
        _run(cp, _bad_request())
    assert cp.results == [], "work must not continue when its durable voice cannot append"


def test_a_rejected_result_is_not_rewritten_as_a_second_error_result():
    class _Rejected(_CP):
        def send_result(self, payload: dict, *, wait: bool = True) -> bool:
            self.results.append(payload)
            frame = {"type": "result", "stream_id": "s", "seq": 1, "result": payload}
            raise FrameRejected(frame, {
                "type": "ack", "stream_id": "s", "seq": 1, "status": 422, "error": "bad content"})

    cp = _Rejected()
    with pytest.raises(FrameRejected):
        _run(cp, _bad_request())
    assert len(cp.results) == 1, "a deterministic verdict must not be replaced by another terminal"


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


def test_exception_urls_and_bearers_are_scrubbed_from_events_and_results(monkeypatch):
    secret = "presigned-secret"

    def _boom(*_a, **_kw):
        raise RuntimeError(
            f"https://user:pass@store.example/object?X-Amz-Signature={secret} Bearer {secret}")

    monkeypatch.setattr(agent_main.InferRequest, "model_validate", staticmethod(_boom))
    cp = _CP()
    _run(cp, _bad_request())
    bodies = [*cp.events, *cp.results]
    rendered = repr(bodies)
    assert secret not in rendered and "user:pass" not in rendered
    assert "[redacted-url]" in rendered


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
