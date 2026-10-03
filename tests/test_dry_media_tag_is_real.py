"""media.tag under contour-dry runs the pack's own handler, not the placeholder stub: the stub overwrote a
bt709-tagged dry master with an untagged 64x64 black file and the engine's check_master refused it."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from podagent.ops import dry, pack, registry

_NEEDS_FFMPEG = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not on this runner")

# pod-agent never vendors montyops (podagent/ops/pack.py), so the fake pack carries the same stream-copy
# argv as the engine's montyops.media_tag: `-c copy -map_metadata -1 -metadata k=v ... +faststart`.
_MEDIA_TAG_BODY = (
    "import subprocess\n"
    "def run(*, params, inputs, outputs):\n"
    "    src, dst = inputs['src'], outputs['dst']\n"
    "    cmd = ['ffmpeg', '-y', '-v', 'error', '-i', str(src), '-c', 'copy', '-map_metadata', '-1']\n"
    "    for k, v in params['tags'].items():\n"
    "        cmd += ['-metadata', f'{k}={v}']\n"
    "    dst.parent.mkdir(parents=True, exist_ok=True)\n"
    "    subprocess.run(cmd + ['-movflags', '+faststart', str(dst)], check=True, capture_output=True)\n"
)


def _pack_tar(tmp_path: Path) -> object:
    pkg = tmp_path / "pack_src" / "montyops"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "media_tag.py").write_text(_MEDIA_TAG_BODY)
    tar = tmp_path / "pack.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(pkg, arcname="montyops")

    class Ref:
        url = tar.as_uri()
        sha256 = hashlib.sha256(tar.read_bytes()).hexdigest()
        size = tar.stat().st_size
    return Ref()


def _bt709_master(dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=1",
         "-f", "lavfi", "-i", "sine=f=440:d=1",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
         "-c:a", "aac", "-shortest", str(dst)],
        check=True, capture_output=True, timeout=60)


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,color_primaries,color_transfer,color_space:format_tags",
         "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=20).stdout
    return json.loads(out)


def test_media_tag_is_classified_real_and_other_heavy_ops_stay_stubs():
    assert dry._CLASSIFICATION["media.tag"][0] == dry.REAL
    for op_name in ("cut.apply", "camera.apply", "media.normalize", "media.scale", "mograph.render"):
        assert dry._CLASSIFICATION[op_name][0] == dry.STUB, f"{op_name} must stay a dry stub"


@_NEEDS_FFMPEG
def test_dry_media_tag_keeps_the_masters_colour_tags(tmp_path, monkeypatch):
    pack.reset_for_tests()
    monkeypatch.setenv(pack.PACK_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(dry.ARM_ENV, "1")
    for mod in ("montyops", "montyops.media_tag"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    try:
        pack.activate(_pack_tar(tmp_path))
        src, dst = tmp_path / "master.mp4", tmp_path / "tagged" / "master.mp4"
        _bt709_master(src)
        tags = {"title": "Dry Master", "artist": "Tenant Brand", "comment": "https://example.com"}

        fn = dry.resolve(registry.get("media.tag"))
        fn(params={"tags": tags}, inputs={"src": src}, outputs={"dst": dst})

        doc = _probe(dst)
        stream = doc["streams"][0]
        assert (stream["width"], stream["height"]) == (320, 180), "the master, not a 64x64 placeholder"
        assert stream["color_primaries"] == "bt709"
        assert stream["color_transfer"] == "bt709"
        assert stream["color_space"] == "bt709"
        got = doc["format"]["tags"]
        for key, value in tags.items():
            assert got.get(key) == value, f"{key} not written: {got}"
    finally:
        pack.reset_for_tests()
