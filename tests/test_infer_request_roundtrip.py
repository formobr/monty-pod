"""The pod re-validates the infer request it dispatched (main.py: PodJob → request.model_dump → InferRequest.model_validate).
An optional block left unset must come back ABSENT, never as an explicit null — clip_rank's `judge_tiles` refuses a null
(«omit it instead»), so a plain dump turned every ordinary rank request into a ValidationError (056 local-pod run,
2026-10-24: pod_ready's clip_rank warm-up failed and the whole preview chain was cancelled)."""
from podagent.models import InferRequest, PodJob


def _job():
    req = {"infer_version": 7, "job_id": "j", "kind": "clip_rank", "model": "m", "put_url": "http://x/p",
           "weights": {"url": "http://x/w", "sha256": "0" * 64},
           "clip_rank": {"groups": [{"intent": "", "image_urls": ["http://x/a.jpg"]}]}}
    return PodJob.model_validate({"type": "infer", "request": req, "session_id": "s", "corr_id": "c"})


def test_an_ordinary_clip_rank_request_survives_the_pods_own_roundtrip():
    raw = _job().request.model_dump(by_alias=True, mode="json", exclude_none=True)
    assert "judge_tiles" not in raw["clip_rank"]
    assert InferRequest.model_validate(raw).clip_rank.judge_tiles is None


def test_the_dispatcher_dumps_without_nulls():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "podagent" / "main.py").read_text(encoding="utf-8")
    assert 'request_raw = pod_job.request.model_dump(by_alias=True, mode="json", exclude_none=True)' in src
