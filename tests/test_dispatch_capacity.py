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
        "claim_capacity": {"ops": 4, "rank": 1, "heavy": 1},
    }


def test_capacity_never_advertises_more_ops_claims_than_executor(monkeypatch):
    monkeypatch.setenv("OPS_MAX_CHAINS", "2")
    assert agent_main.capacity_payload(rank_lanes=9, fetch_workers=9)["claim_capacity"] == {
        "ops": 2, "rank": 1, "heavy": 1,
    }


# ── VRAM: a real total when given one, never a fabricated peak (CAPACITY_VRAM_WHY) ─────────────────

def test_capacity_publishes_a_real_vram_total_when_given_one():
    capacity = agent_main.capacity_payload(rank_lanes=1, fetch_workers=1, vram_total_mb=6144.0)
    assert capacity["vram_total_mb"] == 6144.0


def test_capacity_never_fabricates_vram_for_a_box_with_no_reading():
    """NEGATIVE: a GPU-less box (or a card nvidia-smi could not read) must publish neither key — never 0."""
    capacity = agent_main.capacity_payload(rank_lanes=1, fetch_workers=1, vram_total_mb=None)
    assert "vram_total_mb" not in capacity and "vram_peak_used_mb" not in capacity


def test_capacity_has_no_way_to_spell_a_peak_reading_at_all():
    """nvidia-smi offers no high-water-mark query, so this function has no parameter that could even name
    vram_peak_used_mb — a mislabeled instant is refused by construction, not by a runtime check."""
    import inspect

    assert "vram_peak_used_mb" not in inspect.signature(agent_main.capacity_payload).parameters


def test_existing_callers_that_never_pass_vram_are_unaffected():
    """The default is additive: a caller written before this field existed gets the exact old shape."""
    capacity = agent_main.capacity_payload(rank_lanes=3, fetch_workers=7)
    assert "vram_total_mb" not in capacity and "vram_peak_used_mb" not in capacity
