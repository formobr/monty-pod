from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import pytest

from podagent.ops import pack, registry, resultcache, runner
from podagent.models import OpChain, OpsPackRef


@pytest.fixture(autouse=True)
def _wired(tmp_path, monkeypatch):
    monkeypatch.setenv(resultcache.CACHE_ENV, str(tmp_path / "results"))
    monkeypatch.delenv(resultcache.DISABLE_ENV, raising=False)
    monkeypatch.setenv("POD_IMAGE_TAG", "v-test")
    monkeypatch.setattr(pack, "active_sha", lambda: "a" * 64)
    monkeypatch.setattr(resultcache, "ffmpeg_identity", lambda: "ffmpeg n8.1.2-test")
    resultcache._locks.clear()


def _paths(tmp_path: Path, name: str = "out.mp4") -> tuple[dict[str, Path], dict[str, Path]]:
    src = tmp_path / "src.mov"
    if not src.exists():
        src.write_bytes(b"source bytes")
    return {"src": src}, {"dst": tmp_path / name}


def _params(**overrides):
    return {"height": 960, "encode_profile": "browser", **overrides}


def test_second_identical_scale_skips_handler_but_gets_a_run_owned_copy(tmp_path):
    op = registry.get("media.scale")
    calls = 0

    def handler(*, params, inputs, outputs):
        nonlocal calls
        calls += 1
        outputs["dst"].write_bytes(b"scaled bytes")

    inputs, out1 = _paths(tmp_path, "one.mp4")
    assert resultcache.execute(op, _params(), inputs, out1, handler, lambda _x: None) is False
    _, out2 = _paths(tmp_path, "two.mp4")
    assert resultcache.execute(op, _params(), inputs, out2, handler, lambda _x: None) is True
    assert calls == 1 and out2["dst"].read_bytes() == b"scaled bytes"
    payload = next(resultcache.root().glob("*/payload"))
    assert out2["dst"].stat().st_ino != payload.stat().st_ino, "hit hardlinked cache into workspace"
    assert payload.stat().st_mode & 0o222 == 0, "persistent payload remained mutable"


def test_canonical_param_order_is_one_identity(tmp_path):
    op = registry.get("media.scale")
    calls = 0

    def handler(*, params, inputs, outputs):
        nonlocal calls
        calls += 1
        outputs["dst"].write_bytes(b"scaled")

    inputs, first = _paths(tmp_path, "first.mp4")
    second = {"dst": tmp_path / "second.mp4"}
    resultcache.execute(op, {"height": 960, "encode_profile": "browser"},
                        inputs, first, handler, lambda _x: None)
    hit = resultcache.execute(op, {"encode_profile": "browser", "height": 960},
                              inputs, second, handler, lambda _x: None)
    assert hit and calls == 1


def test_runner_hit_still_uploads_and_adopts_run_owned_output(tmp_path, monkeypatch):
    src = tmp_path / "src.mov"
    src.write_bytes(b"source")
    calls = 0

    def handler(*, params, inputs, outputs):
        nonlocal calls
        calls += 1
        outputs["dst"].write_bytes(b"scaled")

    monkeypatch.setattr(runner.pack, "resolve", lambda _handler: handler)
    uploaded: list[tuple[Path, str]] = []
    adopted: list[tuple[str, Path]] = []
    monkeypatch.setattr(runner, "upload", lambda path, url: uploaded.append((path, url)))

    def _upload_and_adopt(url, path, upload, log=None):
        upload(path, url)
        adopted.append((url, path))

    monkeypatch.setattr(runner.inputcache, "upload_and_adopt", _upload_and_adopt)
    chain = OpChain(job_id="j", pack=OpsPackRef(url="https://x/p.tgz", sha256="a" * 64), steps=[{
        "id": "s", "op": "media.scale", "needs": [],
        "params": _params(), "inputs": [{"port": "src", "path": str(src)}],
        "outputs": [{"port": "dst", "url": "https://r2.example/o/result.mp4?sig=one"}],
    }])
    timings = []
    runner._run_step(chain.steps[0], runner.Workspace(tmp_path / "ws1"), {}, timings)
    runner._run_step(chain.steps[0], runner.Workspace(tmp_path / "ws2"), {}, timings)
    assert calls == 1 and [row.cache_hit for row in timings] == [False, True]
    assert len(uploaded) == len(adopted) == 2, "cache hit bypassed normal PUT/adoption transport"
    assert uploaded[0][0] != uploaded[1][0], "cache hit reused another run's workspace destination"


@pytest.mark.parametrize("axis", ["input", "params", "suffix", "op", "pack", "image", "ffmpeg"])
def test_every_execution_identity_axis_invalidates(axis, tmp_path, monkeypatch):
    op = registry.get("media.scale")
    calls = 0

    def handler(*, params, inputs, outputs):
        nonlocal calls
        calls += 1
        outputs["dst"].write_bytes(f"result-{calls}".encode())

    inputs, outputs = _paths(tmp_path, "a.mp4")
    resultcache.execute(op, _params(), inputs, outputs, handler, lambda _x: None)
    if axis == "input":
        inputs["src"].write_bytes(b"different source")
    elif axis == "params":
        pass
    elif axis == "suffix":
        outputs = {"dst": tmp_path / "a.mov"}
    elif axis == "op":
        op = dataclasses.replace(op, version=op.version + 1)
    elif axis == "pack":
        monkeypatch.setattr(pack, "active_sha", lambda: "b" * 64)
    elif axis == "image":
        monkeypatch.setenv("POD_IMAGE_TAG", "v-next")
    elif axis == "ffmpeg":
        monkeypatch.setattr(resultcache, "ffmpeg_identity", lambda: "ffmpeg next")
    params = _params(height=958) if axis == "params" else _params()
    resultcache.execute(op, params, inputs, outputs, handler, lambda _x: None)
    assert calls == 2, f"{axis} was absent from the cache identity"


@pytest.mark.parametrize("damage", ["payload", "meta", "digest", "size", "extra_meta"])
def test_corruption_is_a_miss_and_never_served(damage, tmp_path):
    op = registry.get("media.scale")
    calls = 0

    def handler(*, params, inputs, outputs):
        nonlocal calls
        calls += 1
        outputs["dst"].write_bytes(f"good-{calls}".encode())

    inputs, outputs = _paths(tmp_path)
    resultcache.execute(op, _params(), inputs, outputs, handler, lambda _x: None)
    slot = next(resultcache.root().iterdir())
    if damage == "payload":
        (slot / "payload").chmod(0o600)
        (slot / "payload").write_bytes(b"evil")
    else:
        import json
        meta = json.loads((slot / "meta.json").read_text())
        if damage == "meta":
            (slot / "meta.json").chmod(0o600)
            (slot / "meta.json").write_text("{")
        elif damage == "digest":
            meta["sha256"] = "0" * 64
        elif damage == "size":
            meta["bytes"] += 1
        else:
            meta["surprise"] = True
        if damage not in {"meta"}:
            (slot / "meta.json").chmod(0o600)
            (slot / "meta.json").write_text(json.dumps(meta))
    outputs = {"dst": tmp_path / "again.mp4"}
    assert resultcache.execute(op, _params(), inputs, outputs, handler, lambda _x: None) is False
    assert calls == 2 and outputs["dst"].read_bytes() == b"good-2"


def test_restore_exception_is_a_safe_loud_miss(tmp_path, monkeypatch):
    slot = tmp_path / "secret-slot-name"
    slot.mkdir()
    dst = tmp_path / "secret-output-name.mp4"
    monkeypatch.setattr(Path, "read_text", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("secret body")))
    lines: list[str] = []

    assert resultcache._restore(slot, "secret-cache-key", dst, lines.append) is False
    assert lines == ["[result-cache] restore miss (OSError); recomputing"]
    assert "secret" not in lines[0]


def test_identical_concurrent_calls_compute_once_and_copy_twice(tmp_path):
    op = registry.get("media.scale")
    inputs, first = _paths(tmp_path, "first.mp4")
    second = {"dst": tmp_path / "second.mp4"}
    calls = 0
    entered = threading.Event()

    def handler(*, params, inputs, outputs):
        nonlocal calls
        calls += 1
        entered.set()
        time.sleep(0.15)
        outputs["dst"].write_bytes(b"one computation")

    hits: list[bool] = []
    threads = [
        threading.Thread(target=lambda out=out: hits.append(
            resultcache.execute(op, _params(), inputs, out, handler, lambda _x: None)))
        for out in (first, second)
    ]
    for thread in threads:
        thread.start()
    assert entered.wait(2)
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert calls == 1 and sorted(hits) == [False, True]
    assert first["dst"].read_bytes() == second["dst"].read_bytes() == b"one computation"


def test_non_allowlisted_ops_and_kill_switch_always_run(tmp_path, monkeypatch):
    scale = registry.get("media.scale")
    other = registry.get("media.audio")
    calls = 0

    def handler(*, params, inputs, outputs):
        nonlocal calls
        calls += 1
        next(iter(outputs.values())).write_bytes(b"x")

    inputs, outputs = _paths(tmp_path)
    monkeypatch.setenv(resultcache.DISABLE_ENV, "1")
    for _ in range(2):
        resultcache.execute(scale, _params(), inputs, outputs, handler, lambda _x: None)
    monkeypatch.delenv(resultcache.DISABLE_ENV)
    audio_out = {"dst": tmp_path / "audio.m4a"}
    resultcache.execute(other, {}, inputs, audio_out, handler, lambda _x: None)
    assert calls == 3 and not list(resultcache.root().glob("*/meta.json"))


def test_unknown_image_or_pack_never_creates_an_ambiguous_entry(tmp_path, monkeypatch):
    op = registry.get("media.scale")
    inputs, outputs = _paths(tmp_path)
    calls = 0

    def handler(*, params, inputs, outputs):
        nonlocal calls
        calls += 1
        outputs["dst"].write_bytes(b"x")

    monkeypatch.delenv("POD_IMAGE_TAG")
    resultcache.execute(op, _params(), inputs, outputs, handler, lambda _x: None)
    monkeypatch.setenv("POD_IMAGE_TAG", "v-test")
    monkeypatch.setattr(pack, "active_sha", lambda: None)
    resultcache.execute(op, _params(), inputs, outputs, handler, lambda _x: None)
    assert calls == 2 and not list(resultcache.root().glob("*/meta.json"))


def test_cache_store_bug_never_changes_valid_handler_work(tmp_path, monkeypatch):
    op = registry.get("media.scale")
    inputs, outputs = _paths(tmp_path)
    monkeypatch.setattr(resultcache, "_store",
                        lambda *_a: (_ for _ in ()).throw(RuntimeError("cache implementation bug")))
    resultcache.execute(op, _params(), inputs, outputs,
                        lambda *, params, inputs, outputs: outputs["dst"].write_bytes(b"valid"),
                        lambda _x: None)
    assert outputs["dst"].read_bytes() == b"valid"


def test_cache_is_bounded_oldest_first(tmp_path):
    op = registry.get("media.scale")
    for i in range(3):
        inputs, outputs = _paths(tmp_path, f"{i}.mp4")
        inputs["src"].write_bytes(f"source-{i}".encode())
        resultcache.execute(op, _params(), inputs, outputs,
                            lambda *, params, inputs, outputs: outputs["dst"].write_bytes(b"x" * 20),
                            lambda _x: None)
    entries = sorted(resultcache._entries())
    keep = sum(size for _m, size, _s in entries[-2:])
    assert resultcache.prune(keep_bytes=keep) > 0
    assert len(resultcache._entries()) == 2
