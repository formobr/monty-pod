"""Burned captions take an OPTIONAL horizontal centre (fraction of the frame width): absent keeps today's
output exactly, a number moves every event's x, clamped so the measured line stays in the safe box."""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from podagent import captions
from podagent.models import SpecCaptions

_W, _H = 1080, 1920
_FG, _ACCENT = "#123456", "#abcdef"
_WORDS = [
    {"text": "true", "start": 0.1, "end": 0.4, "hot": False},
    {"text": "story", "start": 0.45, "end": 0.9, "hot": True},
    {"text": "here", "start": 0.95, "end": 1.3, "hot": False},
]
_CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
_X = re.compile(r"\\(?:pos|move)\((-?\d+),")


def _font() -> Path:
    for pat in ("/usr/share/fonts/**/DejaVuSans.ttf", "/usr/share/fonts/**/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/**/*.ttf"):
        found = sorted(glob.glob(pat, recursive=True))
        if found:
            return Path(found[0])
    pytest.skip("no system TTF available to measure caption widths")


def _build(style: str, **kw) -> str:
    return captions.build_ass(_WORDS, font=_font(), w=_W, h=_H, fg=_FG, accent=_ACCENT, style=style, **kw)


def _xs(ass: str) -> list[int]:
    xs = []
    for ln in ass.splitlines():
        if ln.startswith("Dialogue:"):
            xs += [int(m) for m in _X.findall(ln)]
    return xs


_SAFE_C = round(captions._safe_center(_W))
_STYLES = ["oneword", "phrase", "phrase_jump", "bold"]


@pytest.mark.parametrize("style", _STYLES)
def test_absent_center_x_is_todays_output(style: str) -> None:
    base = _build(style)
    assert _build(style, center_x=None) == base
    if style != "phrase_jump":   # jump spreads words around the centre; the others sit exactly on it
        assert set(_xs(base)) == {_SAFE_C}


@pytest.mark.parametrize("style", _STYLES)
def test_center_x_moves_every_event_to_its_side(style: str) -> None:
    left, right = _xs(_build(style, center_x=0.3)), _xs(_build(style, center_x=0.7))
    assert left and right
    # jump lays words out around the block centre, so compare against the same layout at the safe centre
    base = _xs(_build(style))
    assert len(left) == len(right) == len(base)
    assert all(a < b for a, b in zip(left, base))
    assert all(a > b for a, b in zip(right, base))


@pytest.mark.parametrize("style", ["oneword", "phrase", "bold"])
@pytest.mark.parametrize("cx", [0.0, 1.0])
def test_extreme_center_x_stays_in_safe_box(style: str, cx: float) -> None:
    fnt = captions._font_at(str(_font()), captions._size_px(
        captions.TITLE if style == "oneword" else (captions.BOLD_SIZE if style == "bold" else captions.PHRASE_SIZE)))
    left_edge, right_edge = captions._SAFE_LEFT * _W / captions._REF_W, captions._SAFE_RIGHT * _W / captions._REF_W
    ass = _build(style, center_x=cx)
    n = 0
    for ln in ass.splitlines():
        if not ln.startswith("Dialogue:"):
            continue
        m = _X.search(ln)
        txt = re.sub(r"\{[^}]*\}", "", ln.split(",,0,0,0,,", 1)[1]).strip()
        half = fnt.getlength(txt) / 2
        x = int(m.group(1))
        assert x - half >= left_edge - 3 and x + half <= right_edge + 3
        n += 1
    assert n


def _spec_with(captions_block: dict) -> dict:
    spec = json.loads((_CONTRACTS / "examples" / "spec.final.json").read_text())

    def walk(node):
        if isinstance(node, dict):
            if "captions" in node and isinstance(node["captions"], dict):
                node["captions"] = captions_block
                return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    assert walk(spec)
    return spec


def _schema_errors(spec: dict) -> list:
    schema = json.loads((_CONTRACTS / "spec.schema.json").read_text())
    return list(Draft202012Validator(schema).iter_errors(spec))


def test_center_x_validates_and_unknown_key_is_refused() -> None:
    assert _schema_errors(_spec_with({"centerY": 0.78, "centerX": 0.4})) == []
    assert SpecCaptions.model_validate({"centerY": 0.78, "centerX": 0.4}).centerX == 0.4
    assert SpecCaptions.model_validate({}).centerX is None
    bad = _spec_with({"centerY": 0.78, "centerZ": 0.4})
    assert _schema_errors(bad)
    with pytest.raises(ValidationError):
        SpecCaptions.model_validate({"centerZ": 0.4})
