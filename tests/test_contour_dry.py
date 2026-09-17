"""Contour-dry stand-in (podagent/ops/dry.py) and its pod-side boot lock (podagent.main)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from podagent import main as podagent_main
from podagent.ops import dry, pack, registry

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
FIXTURE = Path(__file__).resolve().parents[2] / "dev/localpod/fixtures/contour-smoke.mp4"

_ALL_OP_NAMES = sorted(p.stem for p in CONTRACTS.glob("ops/*.json"))
_STUB_OP_NAMES = sorted(n for n in _ALL_OP_NAMES if dry._CLASSIFICATION[n][0] == dry.STUB)


def test_armed_reads_only_its_own_env(monkeypatch):
    monkeypatch.delenv(dry.ARM_ENV, raising=False)
    assert dry.armed() is False
    monkeypatch.setenv(dry.ARM_ENV, "1")
    assert dry.armed() is True
    monkeypatch.setenv(dry.ARM_ENV, "0")
    assert dry.armed() is False


def test_every_declared_op_is_classified():
    missing = [n for n in _ALL_OP_NAMES if n not in dry._CLASSIFICATION]
    assert missing == [], f"contracts/ops/*.json op(s) with no dry-tier row in dry._CLASSIFICATION: {missing}"


@pytest.mark.parametrize("op_name", _STUB_OP_NAMES)
def test_every_stubbed_op_yields_its_declared_outputs_in_dry_mode(tmp_path, op_name, monkeypatch):
    op = registry.get(op_name)
    monkeypatch.setattr(pack, "resolve",
                         lambda h: pytest.fail(f"a STUB op must never resolve the real handler ({h})"))
    fn = dry.resolve(op)
    outputs = {}
    for port in op.outputs:
        ext = {"video": ".mp4", "audio": ".m4a", "image": ".png", "json": ".json"}[port.kind]
        if port.many:
            outputs[port.id] = [tmp_path / f"{port.id}_{i}{ext}" for i in range(2)]
        else:
            outputs[port.id] = tmp_path / f"{port.id}{ext}"
    fn(params={}, inputs={}, outputs=outputs)
    for port in op.outputs:
        paths = outputs[port.id] if port.many else [outputs[port.id]]
        for path in paths:
            assert path.exists() and path.stat().st_size > 0, f"{op_name}.{port.id}: dry stub wrote nothing"


def _fake_pack_tar(tmp_path: Path, module_name: str, body: str) -> object:
    root = tmp_path / "pack_src"
    pkg = root / "montyops"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / f"{module_name}.py").write_text(body)
    tar = tmp_path / "pack.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(pkg, arcname="montyops")
    sha = hashlib.sha256(tar.read_bytes()).hexdigest()

    class Ref:
        url = tar.as_uri()
        sha256 = sha
        size = tar.stat().st_size
    return Ref()


def test_real_op_resolves_through_the_activated_pack_not_the_synthetic_stub(tmp_path, monkeypatch):
    """Routing proof, not an astats re-implementation: pod-agent is public and never vendors the tuned
    handler (podagent/ops/pack.py WHY THIS EXISTS), so the fake pack's `run` only has to be REAL in the
    sense of reading the bound input, not identical to the private montyops.measure_audio algorithm."""
    pack.reset_for_tests()
    ref = _fake_pack_tar(
        tmp_path, "measure_audio",
        "import json\n"
        "def run(*, params, inputs, outputs):\n"
        "    src = inputs['src']\n"
        "    size = src.stat().st_size\n"
        "    outputs['measured'].write_text(json.dumps({'size': size, 'levels': {'mean_db': -12.3}}))\n",
    )
    monkeypatch.setenv(pack.PACK_CACHE_ENV, str(tmp_path / "cache"))
    sys.modules.pop("montyops", None)
    sys.modules.pop("montyops.measure_audio", None)
    pack.activate(ref)

    op = registry.get("measure.audio")
    fn = dry.resolve(op)
    dst = tmp_path / "measured.json"
    fn(params={}, inputs={"src": FIXTURE}, outputs={"measured": dst})

    data = json.loads(dst.read_text())
    assert data["size"] == FIXTURE.stat().st_size, "the real handler must see the real bound fixture"
    assert data["levels"]["mean_db"] != 0, "a non-silent level scalar, not a plan-shaped placeholder"

    pack.reset_for_tests()
    sys.modules.pop("montyops", None)
    sys.modules.pop("montyops.measure_audio", None)


def test_json_output_is_readable_json(tmp_path):
    dst = tmp_path / "out.json"
    dry._write_json(dst, seed="probe")
    json.loads(dst.read_text())


def test_unknown_output_kind_refuses_by_name(tmp_path):
    with pytest.raises(registry.OpError):
        dry._fill_one(tmp_path / "x", "browser", seed="s", op_name="fake.op")


@pytest.mark.parametrize("ext,codec", [(".mp3", "mp3"), (".wav", "pcm_s16le"), (".m4a", "aac"), (".aac", "aac")])
def test_stub_audio_placeholder_honours_declared_extension(tmp_path, ext, codec):
    dst = tmp_path / f"out{ext}"
    dry._write_audio(dst, op_name="media.pcm")
    probed = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "json", str(dst)],
        check=True, capture_output=True, text=True).stdout)
    assert probed["streams"][0]["codec_name"] == codec, f"{dst.name}: wrong codec for its own container"


@pytest.mark.parametrize("ext", [".mp4", ".mov"])
def test_stub_video_placeholder_honours_declared_extension(tmp_path, ext):
    dst = tmp_path / f"out{ext}"
    dry._write_video(dst, op_name="cut.apply")
    probed = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "json", str(dst)],
        check=True, capture_output=True, text=True).stdout)
    codecs = {s["codec_name"] for s in probed["streams"]}
    assert codecs == {"h264", "aac"}, f"{dst.name}: expected h264+aac streams, got {codecs}"


def test_stub_audio_placeholder_refuses_unsupported_extension_by_name(tmp_path):
    with pytest.raises(dry.DryStubUnsupportedOutput, match="media.pcm.*\\.ogg"):
        dry._write_audio(tmp_path / "out.ogg", op_name="media.pcm")


def test_stub_video_placeholder_refuses_unsupported_extension_by_name(tmp_path):
    with pytest.raises(dry.DryStubUnsupportedOutput, match="cut.apply.*\\.webm"):
        dry._write_video(tmp_path / "out.webm", op_name="cut.apply")


def test_cheap_cpu_audio_ops_are_real_not_stub():
    for op_name in ("cut.audio", "media.audio"):
        assert dry._CLASSIFICATION[op_name][0] == dry.REAL, f"{op_name} must stay REAL under contour-dry"


def test_boot_refuses_dry_arm_off_the_local_contour(monkeypatch):
    monkeypatch.setenv(dry.ARM_ENV, "1")
    monkeypatch.setenv("POD_IMAGE_TAG", "a" * 40)
    with pytest.raises(SystemExit):
        podagent_main._refuse_dry_off_local_contour()


def test_boot_accepts_dry_arm_on_the_local_contour(monkeypatch):
    monkeypatch.setenv(dry.ARM_ENV, "1")
    monkeypatch.delenv("POD_IMAGE_TAG", raising=False)
    podagent_main._refuse_dry_off_local_contour()


def test_boot_is_a_noop_when_the_switch_is_off(monkeypatch):
    monkeypatch.delenv(dry.ARM_ENV, raising=False)
    monkeypatch.setenv("POD_IMAGE_TAG", "a" * 40)
    podagent_main._refuse_dry_off_local_contour()
