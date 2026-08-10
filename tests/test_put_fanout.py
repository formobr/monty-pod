"""A step's output PUTs, and what the put phase can now say about itself (runner.PUT_FANOUT_WHY). Every
test is NEGATIVE in the sense docs/TESTING.md means: watched fail with the serial phase restored, or with
the sub-leg it asserts dropped from `StepTiming.wire`."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from podagent.models import OpChain, OpsPackRef
from podagent.ops import runner

_PACK = OpsPackRef(url="https://x/p.tgz", sha256="a" * 64)
_LATENCY = 0.25          # one mock store round trip; a handful still finish the test in under a second


def _sheet_step(sid="sheet"):
    """`media.sheet` as the rank wave submits it: two INDEPENDENT durable outputs, the contact sheet and
    the `.cells.json` sidecar that says which cells drew."""
    return {"id": sid, "op": "media.sheet", "needs": [],
            "params": {"cols": 1, "cell_w": 16, "cell_h": 16, "gap": 0, "head": 0,
                       "caption_h": 0, "plate": True, "bg": [18, 18, 18], "captions": [[]]},
            "inputs": [{"port": "tile0", "url": "https://x/t0.png"}],
            "outputs": [{"port": "sheet", "url": "https://x/s.png"},
                        {"port": "meta", "url": "https://x/s.cells.json"}]}


@pytest.fixture()
def wired(monkeypatch):
    """A step whose handler writes every declared output, with the input cache out of the picture so the
    test measures the transport and not the pod's disk."""
    monkeypatch.setattr(runner.registry, "validate_params", lambda *a, **k: None)
    monkeypatch.setattr(runner.registry, "assert_pod_safe", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_bind_inputs", lambda *a, **k: {})
    monkeypatch.setattr(runner.pack, "resolve", lambda _h: (
        lambda *, params, inputs, outputs: [p.write_bytes(b"o" * 64) for p in outputs.values()]))
    monkeypatch.setattr(runner.inputcache, "enabled", lambda: False)
    runner._reset_step_slots()
    yield
    runner._reset_step_slots()


def _slow_store(monkeypatch, seen=None, live=None):
    lock = threading.Lock()

    def _upload(src: Path, url: str, ct=None) -> None:
        if live is not None:
            with lock:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
        time.sleep(_LATENCY)
        if seen is not None:
            with lock:
                seen.append(url)
        if live is not None:
            with lock:
                live["now"] -= 1

    monkeypatch.setattr(runner, "upload", _upload)


def test_a_two_output_step_pays_one_store_latency_not_two(tmp_path, monkeypatch, wired):
    """THE DELIVERABLE: two independent objects went out one after another through ONE permit while the
    budget — four times the step cap — sat idle. Restore the serial loop and this costs 2x`_LATENCY`."""
    seen: list[str] = []
    _slow_store(monkeypatch, seen=seen)
    step = OpChain(job_id="j", pack=_PACK, steps=[_sheet_step()]).steps[0]

    t0 = time.monotonic()
    runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})
    wall = time.monotonic() - t0

    assert sorted(seen) == ["https://x/s.cells.json", "https://x/s.png"], "both objects must still cross"
    assert wall < 2 * _LATENCY, f"the two puts were serialised: {wall:.3f}s >= {2 * _LATENCY:.3f}s"


def test_a_permit_is_taken_per_object_so_the_fanout_cannot_outgrow_the_budget(tmp_path, monkeypatch, wired):
    """The budget counts transfers in flight, so the fan-out is bounded by IT and not by the port count.
    Fan out inside one permit and a two-port step runs two transfers on a budget of one."""
    live = {"now": 0, "peak": 0}
    _slow_store(monkeypatch, live=live)
    monkeypatch.setenv(runner.TRANSPORT_MAX_ENV, "1")
    runner._reset_step_slots()
    step = OpChain(job_id="j", pack=_PACK, steps=[_sheet_step()]).steps[0]

    runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})
    assert live["peak"] == 1, f"the put fan-out overran a transport budget of 1 (peak {live['peak']})"


def test_put_wait_names_the_half_of_put_that_was_queueing(tmp_path, monkeypatch, wired):
    """`put` was permit-wait plus wire under one name, so a 9.6-second put could be a saturated budget or
    a slow store and no reading told them apart."""
    _slow_store(monkeypatch)
    monkeypatch.setenv(runner.TRANSPORT_MAX_ENV, "1")
    runner._reset_step_slots()
    sink: list[runner.StepTiming] = []
    step = OpChain(job_id="j", pack=_PACK, steps=[_sheet_step()]).steps[0]

    runner._run_step(step, runner.Workspace(tmp_path / "ws"), {}, sink)
    legs = sink[0].wire()["legs"]
    # One permit, two objects: the second arm queues behind the first for a full round trip.
    assert legs["put_wait"] >= _LATENCY * 0.8, legs
    assert legs["put_wait"] <= legs["put"], "put_wait is INSIDE put, never beside it"
    assert sink[0].seconds == pytest.approx(round(
        sink[0].slot_wait_s + sink[0].bind_s + sink[0].run_s + sink[0].put_s, 3)), "a sub-leg is not a phase"


def test_put_retry_books_the_seconds_a_resend_from_byte_zero_cost(tmp_path, monkeypatch, wired):
    """A presigned PUT has no resume, so a failed attempt costs the whole object again — and that time
    landed inside `put` as if the wire were merely slow."""
    calls = {"n": 0}

    def _flaky(src: Path, url: str, ct=None) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            runner.retry.add(0.4)          # what cp.upload books when an attempt fails and it re-sends
        time.sleep(0.01)

    monkeypatch.setattr(runner, "upload", _flaky)
    sink: list[runner.StepTiming] = []
    step = OpChain(job_id="j", pack=_PACK, steps=[_sheet_step()]).steps[0]

    runner._run_step(step, runner.Workspace(tmp_path / "ws"), {}, sink)
    assert sink[0].wire()["legs"]["put_retry"] == pytest.approx(0.4, abs=1e-3)


def test_a_retry_clock_never_leaks_between_two_objects_on_one_pool_thread(monkeypatch):
    """Pool threads are REUSED, so a counter that is not reset per object charges the next object with the
    last one's re-send."""
    monkeypatch.setattr(runner.inputcache, "enabled", lambda: False)
    monkeypatch.setattr(runner, "upload", lambda src, url, ct=None: runner.retry.add(0.4))
    first = runner._upload_one("https://x/a.png", Path(__file__))
    monkeypatch.setattr(runner, "upload", lambda src, url, ct=None: None)
    second = runner._upload_one("https://x/b.png", Path(__file__))
    assert first[1] == pytest.approx(0.4) and second[1] == 0.0


def test_a_list_port_still_walks_its_addresses_in_order_and_stops_at_a_refusal(tmp_path, monkeypatch, wired):
    """The fan-out is per PORT, never per element: an address the store refused three times with backoff
    will refuse the next one too, and a rented box may not spend its lease proving it."""
    sent: list[str] = []

    def _upload(src: Path, url: str, ct=None) -> None:
        if url.endswith("g2.png"):
            raise RuntimeError("503 from the store")
        sent.append(url)

    monkeypatch.setattr(runner, "upload", _upload)
    monkeypatch.setattr(runner.pack, "resolve", lambda _h: (
        lambda *, params, inputs, outputs: [p.write_text("f") for p in outputs["frames"]]))
    step = OpChain(job_id="j", pack=_PACK, steps=[{
        "id": "g", "op": "media.frames", "needs": [],
        "params": {"positions": [0.0, 0.25, 0.5, 0.75], "width": 384, "height": 384},
        "inputs": [{"port": "src", "url": "https://x/in.mp4"}],
        "outputs": [{"port": "frames", "urls": [f"https://x/g{i}.png" for i in range(4)]}],
    }]).steps[0]

    with pytest.raises(runner.ChainError, match=r"frames'\[2\] of 4"):
        runner._run_step(step, runner.Workspace(tmp_path / "ws"), {})
    assert sent == ["https://x/g0.png", "https://x/g1.png"]
