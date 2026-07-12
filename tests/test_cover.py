"""cover.compose — the Pillow still. Colour-token resolution + a real (small) draw over a base frame
using the system fallback font (no brand font needed). ffmpeg weld is a smoke concern, not here."""
from __future__ import annotations

from pathlib import Path

from podagent import cover


def test_color_resolution():
    colors = {"white": (242, 242, 240), "accent": (214, 255, 58), "black": (10, 10, 10)}
    assert cover._color("accent", colors) == (214, 255, 58)
    assert cover._color([1, 2, 3], colors) == (1, 2, 3)          # literal RGB passthrough
    assert cover._color("lime", colors) == (242, 242, 240)       # unknown token → default white
    assert cover._color("x", {}) == (255, 255, 255)              # empty map → hard white


def test_compose_writes_sized_png(tmp_path: Path):
    from PIL import Image
    base = tmp_path / "frame.jpg"
    Image.new("RGB", (400, 700), (30, 30, 30)).save(base)
    spec = {
        "frame_at": 5.0,
        "headline": {"lines": [[{"t": "НЕ", "c": "white"}], [{"t": "ЛЕНЬ", "c": "accent"}]],
                     "pos": "bottom", "y": 0.72, "size": 90.0, "box": True},
        "colors": {"white": [242, 242, 240], "accent": [214, 255, 58], "black": [10, 10, 10]},
        "elements": [{"type": "badge", "x": 0.5, "y": 0.2, "t": "LIVE", "bg": "accent", "fg": "black"}],
    }
    out = tmp_path / "cover.png"
    cover.compose(base, spec, {}, out, 216, 384)  # small canvas; fallback font
    assert out.is_file()
    with Image.open(out) as im:
        assert im.size == (216, 384)


def test_compose_no_headline_still_renders(tmp_path: Path):
    from PIL import Image
    base = tmp_path / "frame.jpg"
    Image.new("RGB", (216, 384), (0, 0, 0)).save(base)
    out = tmp_path / "c.png"
    cover.compose(base, {"frame_at": 0.0, "headline": {"lines": []}, "colors": {}}, {}, out, 216, 384)
    assert out.is_file()
