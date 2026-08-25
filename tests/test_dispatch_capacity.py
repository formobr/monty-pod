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
        "boot_id": agent_main.BOOT_ID,
        "vulkan": False,
    }


def test_capacity_advertises_the_artifact_pool_separately_from_the_tile_pool(monkeypatch):
    """artifact.range_fetch_width and infer_cliprank.fetch_width are two different pools; the payload
    must name both or an operator reading `fetch_workers` alone under-reads the pod's real connections."""
    capacity = agent_main.capacity_payload(rank_lanes=3, fetch_workers=7, artifact_fetch_workers=6)
    assert capacity["fetch_workers"] == 7
    assert capacity["artifact_fetch_workers"] == 6


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


# ── boot_id: stable within a process, distinct across an execv-minted one (R3/U3) ──────────────────

def test_boot_id_is_stable_within_one_process():
    a = agent_main.capacity_payload(rank_lanes=1, fetch_workers=1)["boot_id"]
    b = agent_main.capacity_payload(rank_lanes=1, fetch_workers=1)["boot_id"]
    assert a == b == agent_main.BOOT_ID


def test_a_fresh_process_mints_a_different_boot_id():
    """execv() re-imports this module; the module-level constant is what makes that a NEW boot_id, not a
    thing anyone has to remember to reset."""
    import subprocess
    import sys

    got = subprocess.run(
        [sys.executable, "-c", "from podagent import main; print(main.BOOT_ID)"],
        capture_output=True, text=True, check=True, cwd=agent_main.__file__.rsplit("/podagent/", 1)[0],
    ).stdout.strip()
    assert got != agent_main.BOOT_ID and len(got) == 32


def test_capacity_has_no_way_to_spell_a_peak_reading_at_all():
    """nvidia-smi offers no high-water-mark query, so this function has no parameter that could even name
    vram_peak_used_mb — a mislabeled instant is refused by construction, not by a runtime check."""
    import inspect

    assert "vram_peak_used_mb" not in inspect.signature(agent_main.capacity_payload).parameters


def test_existing_callers_that_never_pass_vram_are_unaffected():
    """The default keeps the new capability fact explicit when no probe verdict was supplied yet."""
    capacity = agent_main.capacity_payload(rank_lanes=3, fetch_workers=7)
    assert "vram_total_mb" not in capacity and "vram_peak_used_mb" not in capacity
    assert capacity["vulkan"] is False


# ── usable_cores: the measured after-claim truth the box gates the lease on ────────────────────────

def test_capacity_declares_measured_usable_cores():
    """NEGATIVE (pre-change the payload had no CPU fact at all): the box-side min_cpu_cores gate reads it."""
    capacity = agent_main.capacity_payload(rank_lanes=1, fetch_workers=1, usable_cores=6)
    assert capacity["usable_cores"] == 6


def test_capacity_without_a_cores_reading_publishes_no_zero():
    capacity = agent_main.capacity_payload(rank_lanes=1, fetch_workers=1)
    assert "usable_cores" not in capacity


def test_usable_cores_is_the_affinity_mask_capped_by_the_cgroup_quota(monkeypatch):
    """A co-tenant box hands out the host's full mask while cpu.max caps the cycles — quota must win."""
    from podagent import infer_cliprank as icr

    monkeypatch.setattr(icr, "_host_threads", lambda: 16)
    monkeypatch.setattr(icr, "_cpu_quota_cores", lambda: 2.0)
    assert icr.usable_cores() == 2


def test_usable_cores_without_a_quota_is_the_affinity_mask(monkeypatch):
    from podagent import infer_cliprank as icr

    monkeypatch.setattr(icr, "_host_threads", lambda: 16)
    monkeypatch.setattr(icr, "_cpu_quota_cores", lambda: None)
    assert icr.usable_cores() == 16
