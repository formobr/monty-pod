"""podagent.mograph — the overlay filtergraph + public-staging (no node/chrome/ffmpeg execution)."""
from __future__ import annotations

from pathlib import Path

from podagent import mograph


def test_overlay_filtergraph_shifts_and_gates() -> None:
    layers = [{"start": 5.0, "dur": 4.0, "glass": False},
              {"start": 12.0, "dur": 3.0, "glass": True}]
    fc, last = mograph.overlay_filtergraph(layers)
    assert last == "v1"
    assert "[1:v]setpts=PTS-STARTPTS+5.0/TB[o0]" in fc            # layer shifted to its start
    assert "overlay=enable='between(t,5.0,9.0)':eof_action=pass" in fc  # gated to its window
    assert "[0:v]gblur" not in fc                                 # first (non-glass) layer: no blur
    assert "gblur=sigma=22:enable='between(t,12.0,15.0)'" in fc   # glass layer blurs the frame behind it


def test_stage_public_copies_by_prefix(tmp_path: Path) -> None:
    rd = tmp_path / "remotion"
    (rd / "public").mkdir(parents=True)
    src = tmp_path / "Inter.ttf"; src.write_bytes(b"ttf")
    media = tmp_path / "pic.png"; media.write_bytes(b"png")
    input_paths = {"mograph/public/Inter.ttf": src, "mograph/public/_photo/pic.png": media, "base": tmp_path / "x"}
    mograph._stage_public(input_paths, rd)
    assert (rd / "public" / "Inter.ttf").read_bytes() == b"ttf"
    assert (rd / "public" / "_photo" / "pic.png").read_bytes() == b"png"  # nested rel staged
    assert not (rd / "public" / "x").exists()                             # non-prefixed input NOT staged
