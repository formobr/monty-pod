"""The perf wave's failure branches: prepare's parallel arms fail loud and first-fault-wins, the
download pool maps by spec order, and the gpu probe cache honours only a matching positive record.
Nothing here runs ffmpeg or node — every subprocess is stubbed."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

import test_render_onepass as t
from podagent import mograph, render, render_onepass as op


# --- prepare arms -------------------------------------------------------------

def test_a_failing_arm_fails_prepare_loud(monkeypatch, tmp_path) -> None:
    """First fault wins: a dead bed pre-render must raise out of prepare, never yield a half mix."""
    spec = t._spec()
    t._stub_prepare_passes(monkeypatch, tmp_path)

    def dead_bed(*_a, **_kw):
        raise RuntimeError("bed: ffmpeg exploded")
    monkeypatch.setattr(render, "_prerender_bed", dead_bed)
    with pytest.raises(RuntimeError, match="bed: ffmpeg exploded"):
        op.prepare(spec, t._paths(spec, tmp_path), tmp_path, False)


def test_parallel_prepare_hands_assemble_the_same_prepared(monkeypatch, tmp_path) -> None:
    """The arms change WHEN pre-passes run, never WHAT they produce: the mix, layers and flags must
    equal the serial build byte for byte (assemble purity then pins the filter-script)."""
    spec = t._spec()
    bed = t._stub_prepare_passes(monkeypatch, tmp_path)
    p = op.prepare(spec, t._paths(spec, tmp_path), tmp_path, False)
    assert p.bed == bed
    assert p.layers == tuple(t.LAYERS)
    assert p.audio is not None
    assert p.audio.clean == "highpass=f=80"
    assert p.audio.vln == "loudnorm=I=-20:TP=-1.5:LRA=11"
    assert p.audio.bed_idx == len(render.input_ids(spec))
    assert p.flares == ()


def test_a_fault_waits_for_running_siblings_before_raising(monkeypatch, tmp_path) -> None:
    """The raise must land BEHIND every running arm: the caller's TemporaryDirectory unwinds right
    after it, and a still-writing sibling under a dying tmp dir is the race this drains away."""
    spec = t._spec()
    t._stub_prepare_passes(monkeypatch, tmp_path)
    landed = threading.Event()

    def slow_layers(*_a, **_kw):
        time.sleep(0.3)
        landed.set()
        return list(t.LAYERS)

    def dead_bed(*_a, **_kw):
        raise RuntimeError("bed died")
    monkeypatch.setattr(mograph, "_render_layers", slow_layers)
    monkeypatch.setattr(render, "_prerender_bed", dead_bed)
    with pytest.raises(RuntimeError, match="bed died"):
        op.prepare(spec, t._paths(spec, tmp_path), tmp_path, False)
    assert landed.is_set(), "prepare raised while a sibling arm was still running"


def test_no_overlays_means_no_pool_at_all(monkeypatch, tmp_path) -> None:
    def boom(*_a, **_kw):
        raise AssertionError("an armless prepare must not build a pool")
    monkeypatch.setattr(op.cf, "ThreadPoolExecutor", boom)
    spec = t._spec(lambda d: (d["overlays"].__setitem__("music", None),
                              d["overlays"].__setitem__("sfx", []),
                              d["overlays"]["motion_plan"].__setitem__("sections", [])))
    p = op.prepare(spec, t._paths(spec, tmp_path), tmp_path, False)
    assert p.audio is None and p.layers == () and p.flares == ()


# --- the download pool --------------------------------------------------------

def test_downloads_map_by_spec_order_not_completion_order(monkeypatch, tmp_path) -> None:
    spec = t._spec()
    monkeypatch.setattr(render, "download", lambda _url, dest: dest)
    got = render._download_inputs(spec.inputs, tmp_path)
    assert list(got) == [i.id for i in spec.inputs]
    assert got["base"] == tmp_path / "base"
    assert got["music/bed.mp3"] == tmp_path / "music__bed.mp3"


def test_one_failed_download_fails_the_render_loud(monkeypatch, tmp_path) -> None:
    spec = t._spec()

    def dl(url, dest):
        if url.endswith("bed.mp3"):
            raise RuntimeError("transfer died")
        return dest
    monkeypatch.setattr(render, "download", dl)
    with pytest.raises(RuntimeError, match="transfer died"):
        render._download_inputs(spec.inputs, tmp_path)


def test_colliding_flattened_ids_refuse_before_any_transfer(monkeypatch, tmp_path) -> None:
    """ids are unique, but "/"→"__" flattening is not injective — two parallel writers on one path
    would race where the serial loop silently last-wrote. Refused by name, before any bytes move."""
    spec = t._spec(lambda d: d["inputs"].append(
        {"id": "music__bed.mp3", "kind": "audio", "sha256": t.SHA, "url": "https://x/bed2.mp3"}))
    monkeypatch.setattr(render, "download",
                        lambda *_a: pytest.fail("a transfer started despite the collision"))
    with pytest.raises(RuntimeError, match="collide after path flattening"):
        render._download_inputs(spec.inputs, tmp_path)


def test_tmpdir_is_leaked_when_an_arm_never_landed(tmp_path, monkeypatch) -> None:
    """An error marked `unlanded_arms` must LEAK the job tmp dir: an rmtree under a possibly-live
    writing child is the worse defect. An unmarked error still cleans up."""
    orig = render.tempfile.mkdtemp
    monkeypatch.setattr(render.tempfile, "mkdtemp",
                        lambda prefix: orig(prefix=prefix, dir=tmp_path))
    with pytest.raises(RuntimeError):
        with render._job_tmpdir() as d:
            kept = d
            e = RuntimeError("first fault")
            e.unlanded_arms = ["mograph"]
            raise e
    assert kept.exists(), "tmp dir was torn down under a possibly-live child"

    with pytest.raises(RuntimeError):
        with render._job_tmpdir() as d:
            gone = d
            raise RuntimeError("plain fault")
    assert not gone.exists(), "a landed failure must still clean up"


def test_the_unlanded_mark_survives_a_reporting_error(tmp_path, monkeypatch) -> None:
    """phase() may replace the marked error with an event-send failure; the mark must be found down
    the cause/context chain, or the leak guard is laundered away exactly when it matters."""
    orig = render.tempfile.mkdtemp
    monkeypatch.setattr(render.tempfile, "mkdtemp",
                        lambda prefix: orig(prefix=prefix, dir=tmp_path))
    with pytest.raises(ConnectionError):
        with render._job_tmpdir() as d:
            kept = d
            try:
                e = RuntimeError("first fault")
                e.unlanded_arms = ["mograph"]
                raise e
            except RuntimeError:
                raise ConnectionError("event send died")  # implicit __context__ carries the mark
    assert kept.exists(), "the mark was laundered by the reporting error"


def test_the_mark_survives_an_unmarked_explicit_cause_beside_it(tmp_path, monkeypatch) -> None:
    """raise X from unmarked_cause while the marked error sits in __context__: the walk must take
    BOTH branches, not `__cause__ or __context__`."""
    orig = render.tempfile.mkdtemp
    monkeypatch.setattr(render.tempfile, "mkdtemp",
                        lambda prefix: orig(prefix=prefix, dir=tmp_path))
    with pytest.raises(ConnectionError):
        with render._job_tmpdir() as d:
            kept = d
            try:
                e = RuntimeError("first fault")
                e.unlanded_arms = ["mograph"]
                raise e
            except RuntimeError:
                raise ConnectionError("send died") from OSError("transport")
    assert kept.exists(), "the walk followed __cause__ and dropped the marked __context__"


# --- the gpu probe cache ------------------------------------------------------

@pytest.fixture()
def gpu_cache(monkeypatch, tmp_path):
    cache = tmp_path / "probe.json"
    monkeypatch.setenv(render._GPU_CACHE_ENV, str(cache))
    monkeypatch.delenv("MONTY_GPU_MOTION", raising=False)
    monkeypatch.setattr(render, "_GPU", None)
    return cache


def _probe_runs(monkeypatch, rc: int | None):
    """rc=None forbids the PROBE subprocess entirely; otherwise it records probe calls and exits rc.
    The ffmpeg -version identity read (part of the cache key) is always answered, never counted."""
    calls: list = []

    def run(cmd, **_kw):
        if list(cmd[:2]) == ["ffmpeg", "-version"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="ffmpeg version test\n", stderr="")
        if rc is None:
            raise AssertionError("the probe ran when the cache should have answered")
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, rc)
    monkeypatch.setattr(render.subprocess, "run", run)
    return calls


def test_a_positive_probe_is_cached_and_reused_across_processes(monkeypatch, gpu_cache) -> None:
    calls = _probe_runs(monkeypatch, rc=0)
    assert render._gpu_available() is True
    assert len(calls) == 1
    assert json.loads(gpu_cache.read_text())["ok"] is True

    monkeypatch.setattr(render, "_GPU", None)  # a fresh worker process, same box
    _probe_runs(monkeypatch, rc=None)
    assert render._gpu_available() is True


def test_a_negative_probe_is_never_cached(monkeypatch, gpu_cache) -> None:
    calls = _probe_runs(monkeypatch, rc=1)
    assert render._gpu_available() is False
    assert not gpu_cache.exists()

    monkeypatch.setattr(render, "_GPU", None)
    assert render._gpu_available() is False
    assert len(calls) == 2  # the next process probes again — one flake must not demote the box


def test_a_corrupt_or_foreign_cache_record_is_a_miss(monkeypatch, gpu_cache) -> None:
    gpu_cache.write_text("{ not json")
    calls = _probe_runs(monkeypatch, rc=0)
    assert render._gpu_available() is True
    assert len(calls) == 1

    monkeypatch.setattr(render, "_GPU", None)
    gpu_cache.write_text(json.dumps({"key": "someone-elses-box", "ok": True}))
    assert render._gpu_available() is True
    assert len(calls) == 2
    assert json.loads(gpu_cache.read_text())["key"] == render._gpu_cache_key()


# --- the mograph batch wall ---------------------------------------------------

def test_a_wedged_render_batch_fails_loud_with_box_facts(monkeypatch, tmp_path) -> None:
    def wedged(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(mograph.subprocess, "run", wedged)
    monkeypatch.setattr(mograph, "_oom_count", lambda: None)
    with pytest.raises(RuntimeError, match="ran out its"):
        mograph._run_batch(tmp_path, [{"comp": "X", "seqdir": str(tmp_path)}],
                           tmp_path / "spec.json", None)
