"""A parked heavy op is visible, bounded by a deadline priced once at entry, and never double-counted in
slot_wait (gpu_admission.GPU_ADMISSION_WHY, runner.HEAVY_PARK_WHY). Every test is NEGATIVE (docs/TESTING.md).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from podagent.ops import gpu_admission as ga
from podagent.ops import registry, runner

HANDLER_S = 0.25


@pytest.fixture(autouse=True)
def _reset():
    ga._reset_for_tests()
    yield
    ga._reset_for_tests()


def _spin_until(pred, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.005)


def _heavy_step(tmp_path: Path, sid: str):
    op = registry.get("media.normalize")
    assert op.op in ga.HEAVY_GPU_OPS and op.budget != "transport"
    src = tmp_path / f"{sid}.in"
    src.write_bytes(b"x")
    return type("S", (), {
        "id": sid, "op": op.op, "params": {}, "needs": [], "optional": False,
        "inputs": [type("B", (), {"port": p.id, "url": None, "from_step": None, "path": str(src)})()
                   for p in op.inputs],
        "outputs": [type("B", (), {"port": p.id, "url": None, "urls": None})()
                    for p in op.outputs if not p.optional],
    })()


def _handler(*, params, inputs, outputs):
    time.sleep(HANDLER_S)
    for path in outputs.values():
        for one in (path if isinstance(path, list) else [path]):
            Path(one).write_bytes(b"y")


def test_a_parked_heavy_op_is_visible_bounded_and_not_double_counted(monkeypatch, tmp_path: Path):
    """NEGATIVE: three preview-equivalent heavies on one card. With the old flat deadline the third timed out
    behind two legitimate runs; with slot_ready before admission its slot_wait CONTAINED the park, so adding a
    park leg beside it double-counted. Each parked op must announce its position and the deadline priced at
    entry, and its legs must sum to its own wall exactly once."""
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.registry, "assert_pod_safe", lambda *a, **k: None)
    monkeypatch.setattr(runner.pack, "resolve", lambda h: _handler)

    events: dict[str, list[dict]] = {}
    sinks: dict[str, list] = {}
    walls: dict[str, float] = {}
    errors: list[BaseException] = []

    def _run(sid: str) -> None:
        events[sid], sinks[sid] = [], []
        t0 = time.monotonic()
        try:
            runner._run_step(_heavy_step(tmp_path, sid), runner.Workspace(tmp_path / sid), {},
                             sink=sinks[sid], emit=lambda **kw: events[sid].append(kw))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        walls[sid] = time.monotonic() - t0

    threads = []
    for sid, ready in (("a", lambda: ga._live), ("b", lambda: len(ga._queue) == 1),
                       ("c", lambda: len(ga._queue) == 2)):
        t = threading.Thread(target=_run, args=(sid,))
        t.start()
        threads.append(t)
        _spin_until(ready)  # fixed arrival order: a admitted, b parked 1st, c parked 2nd
    for t in threads:
        t.join(timeout=10)
    assert not errors, errors

    parks = {sid: [e for e in evs if e["phase"] == "heavy_park"] for sid, evs in events.items()}
    assert parks["a"] == [], "an op admitted on arrival did not park and must not say it did"
    (b,), (c,) = parks["b"], parks["c"]
    budget = ga.budget_mib()
    assert b["timings"]["position"] == 1 and c["timings"]["position"] == 2, (b, c)
    assert b["timings"]["request_mib"] == c["timings"]["request_mib"] == budget
    assert b["timings"]["budget_mib"] == budget
    # priced ONCE at entry: (ahead + admitted) × the sanctioned heavy ceiling
    assert b["timings"]["deadline_s"] == 1 * ga.HEAVY_OP_CEILING_S
    assert c["timings"]["deadline_s"] == 2 * ga.HEAVY_OP_CEILING_S
    phases_c = [e["phase"] for e in events["c"]]
    assert phases_c.index("heavy_park") < phases_c.index("heavy_slot_wait_ended") < phases_c.index("run_started")

    for sid in "abc":
        (timing,) = sinks[sid]
        legs = timing.wire()["legs"]
        total = legs["slot_wait"] + legs["heavy_park"] + legs["bind"] + legs["run"] + legs["put"]
        assert total == pytest.approx(timing.seconds, abs=0.005), legs
        assert total == pytest.approx(walls[sid], abs=0.05), (sid, legs, walls[sid])
    # the park lives in heavy_park, and slot_wait no longer contains it
    c_legs = sinks["c"][0].wire()["legs"]
    assert c_legs["heavy_park"] >= 2 * HANDLER_S - 0.05, c_legs
    assert c_legs["slot_wait"] < HANDLER_S / 2, c_legs
    spans = sinks["c"][0].intervals
    assert spans["heavy_park"]["end_mono_ns"] <= spans["slot_wait"]["start_mono_ns"]


def test_a_request_that_cannot_fit_before_its_deadline_names_position_and_mib(monkeypatch):
    """NEGATIVE: a park must expire on the deadline priced at entry, and the refusal must name WHERE the op
    stood and WHAT it asked for — a bare 'timed out' cannot be told apart from a hung card."""
    monkeypatch.setattr(ga, "HEAVY_OP_CEILING_S", 0.05)
    ga._reset_for_tests(budget_mib=2000.0)
    holder = ga.reserve("cut.apply", 1500.0, deadline_s=1.0)
    parked: list[dict] = []
    try:
        with pytest.raises(ga.GpuAdmissionTimeout) as info:
            ga.reserve("camera.apply", 1000.0, on_park=lambda **kw: parked.append(kw))
    finally:
        ga.release(holder)
    msg = str(info.value)
    assert "position 1" in msg and "request 1000 MiB" in msg, msg
    assert parked and parked[0]["deadline_s"] == pytest.approx(0.05), parked
    assert not ga._queue, "an expired waiter must leave the queue"


def test_the_park_deadline_is_capped():
    """NEGATIVE: a deep queue may not price a park past HEAVY_PARK_CEILING_S."""
    assert ga.park_deadline_s(ga.MAX_PARKED - 1, 1) == ga.HEAVY_PARK_CEILING_S
    assert ga.park_deadline_s(0, 0) == ga.HEAVY_OP_CEILING_S
