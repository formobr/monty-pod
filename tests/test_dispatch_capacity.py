from __future__ import annotations

from podagent import main as agent_main


class _CP:
    def __init__(self, job):
        self.job = job
        self.events: list[dict] = []
        self.results: list[dict] = []

    def poll_job(self):
        job, self.job = self.job, None
        return job

    def send_event(self, payload, *, wait=False):
        self.events.append(payload)
        return True

    def send_result(self, payload, *, wait=True):
        self.results.append(payload)
        return True

    def note(self, payload):
        self.events.append(payload)


def test_validation_failure_releases_credit_with_product_attribution():
    cp = _CP({
        "type": "infer",
        "session_id": "product-session",
        "corr_id": "corr-invalid",
        "request": {"job_id": "product-job", "kind": "clip_rank"},
    })
    agent_main._dispatch_loop(cp, None, None, None, lambda _job: None, once=True)

    assert len(cp.results) == 1
    terminal = cp.results[0]
    assert terminal["status"] == "error"
    assert terminal["corr_id"] == "corr-invalid"
    assert terminal["session_id"] == "product-session"
    assert terminal["job_id"] == "product-job"
    event = next(e for e in cp.events if e.get("phase") == "work_finished")
    assert event["corr_id"] == terminal["corr_id"]
    assert event["session_id"] == terminal["session_id"]


def test_capacity_advertises_worker_credit_limits(monkeypatch):
    monkeypatch.setenv("OPS_MAX_CHAINS", "5")
    capacity = agent_main.capacity_payload(rank_lanes=3, fetch_workers=7)
    assert capacity == {
        "rank_lanes": 3,
        "fetch_workers": 7,
        "max_inflight": 1,
        "max_parallel": 1,
    }
