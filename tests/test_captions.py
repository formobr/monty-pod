"""podagent.captions — the libass ASS builder (no ffmpeg). Proves oneword emits Dialogue events,
the accent lights `hot` words, and the portrait safe-zone clamp. (Phrase parity: engine A/B.)

The colours are ARGUMENTS here, never fixtures-as-truth: this file used to feed one tenant's lime in and
assert it came back out, which made the pod's hardcoded palette look verified. The colours below are
deliberately NOT any brand's (see test_no_colour_literal_survives_in_the_module)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from podagent import captions

_WORDS = [
    {"text": "true", "start": 0.1, "end": 0.4, "hot": False},
    {"text": "story", "start": 0.45, "end": 0.9, "hot": True},
    {"text": "here", "start": 0.95, "end": 1.3, "hot": False},
]
_FONT = Path("/fonts/Inter-ExtraBold.ttf")  # never opened for oneword (no PIL measuring)
_FG = "#123456"       # arbitrary, on purpose: the builder must draw what it is HANDED
_ACCENT = "#abcdef"


def test_oneword_emits_events_and_lights_hot() -> None:
    ass = captions.build_ass(_WORDS, font=_FONT, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="oneword")
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert ass.count("Dialogue:") == 3
    assert "STORY" in ass and "TRUE" in ass          # words are upper-cased at draw time
    accent = captions._ac(_ACCENT)
    story = next(ln for ln in ass.splitlines() if "STORY" in ln)
    assert f"\\1c{accent}" in story                   # the hot word carries the accent colour
    true = next(ln for ln in ass.splitlines() if "TRUE" in ln)
    assert f"\\1c{accent}" not in true                # a non-hot word does not
    assert captions._ac(_FG) in ass                   # the body style is the fg it was handed


def test_the_body_colour_is_the_one_handed_in_not_a_house_white() -> None:
    """NEGATIVE — reddens the moment a module-level FG (or any other default) draws the body instead of the
    caller's brand value. The pod is the FINAL renderer: a literal here ships in every delivered video."""
    ass = captions.build_ass(_WORDS, font=_FONT, w=1080, h=1920, fg="#010203", accent=_ACCENT)
    assert captions._ac("#010203") in ass
    assert captions._ac("#f2f2f0") not in ass, "the retired hardcoded off-white is back in the ASS head"


def test_colours_are_required_arguments() -> None:
    """A default would be a second SSOT for a brand value; the seam must fail loudly, not silently recolour."""
    for missing in ({"accent": _ACCENT}, {"fg": _FG}):
        with pytest.raises(TypeError):
            captions.build_ass(_WORDS, font=_FONT, w=1080, h=1920, **missing)  # type: ignore[arg-type]


def test_no_colour_literal_survives_in_the_module() -> None:
    """NEGATIVE — the file itself must hold no `#rrggbb`. The engine gate (tests/test_brand_literals.py)
    proves it is not THIS brand's colour; this proves it is not ANY colour, defaults included."""
    src = Path(captions.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    found = [m for m in re.findall(r"#[0-9a-fA-F]{3,8}\b", code)]
    assert not found, f"colour literal(s) back in podagent/captions.py: {found}"


class _Caps:
    def __init__(self, accent=None):
        self.accent = accent


class _Brand:
    def __init__(self, tokens):
        self.tokens = tokens


class _MP:
    def __init__(self, brand=None):
        self.brand = brand


def _colours(caps, mp):
    from podagent.render import _caption_colours
    return _caption_colours(caps, mp)


def test_colours_come_from_the_data_that_crossed() -> None:
    mp = _MP(_Brand({"color": {"fg": "#111213", "accent": "#141516"}}))
    assert _colours(_Caps("#aabbcc"), mp) == ("#111213", "#aabbcc")   # spec field wins for accent
    assert _colours(_Caps(None), mp) == ("#111213", "#141516")        # else the crossed brand tokens


def test_a_burn_with_no_brand_at_all_fails_instead_of_guessing_a_palette() -> None:
    """NEGATIVE — reddens if `caps.accent or "#d6ff3a"` (or any other house colour) comes back: the pod is the
    final renderer, so a guessed accent is a DELIVERED video in another tenant's brand."""
    with pytest.raises(RuntimeError):
        _colours(_Caps(None), _MP(None))


def test_the_body_fallback_is_neutral_not_a_tenant_off_white() -> None:
    """A brandless job degrades to plain white — never #f2f2f0, which is one specific channel's warm white."""
    from podagent import render
    fg, _ = _colours(_Caps("#aabbcc"), _MP(None))
    assert fg == render._NEUTRAL_FG == "#ffffff"


def test_portrait_center_y_clamped_to_safe_zone() -> None:
    # a center_y past the bottom UI reserve is pulled up; landscape is left untouched
    assert captions._clamp_cy(0.99, 60, 1080, 1920) < 0.99
    assert captions._clamp_cy(0.99, 60, 1920, 1080) == 0.99


# ── bold (4th style): block layout like phrase, heavier weight ───────────────────────────────────────

def _real_font():
    """A real TTF: bold's block wrap measures glyph width via PIL, same as phrase/phrase_jump."""
    import glob
    for pat in ("/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/DejaVuSans*.ttf"):
        found = glob.glob(pat, recursive=True)
        if found:
            return Path(found[0])
    pytest.skip("no system TTF available to measure the bold block wrap")


def test_bold_style_head_carries_bold_flag_and_block_layout() -> None:
    font = _real_font()
    ass = captions.build_ass(_WORDS, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="bold")
    style_line = next(ln for ln in ass.splitlines() if ln.startswith("Style: Cap,"))
    fields = style_line.split(",")
    assert fields[2] == str(captions.BOLD_SIZE)                 # a dedicated, bigger Fontsize than phrase
    assert fields[7] == "1"                                     # Bold:1 (Format col index 7 after "Style:")
    assert "\\an8" in ass or "\\an5" in ass                     # block-anchored, not the oneword \move look


def test_bold_style_accents_the_spoken_word_like_phrase() -> None:
    font = _real_font()
    ass = captions.build_ass(_WORDS, font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, style="bold")
    accent = captions._inline_c(_ACCENT)
    assert f"\\1c{accent}" in ass
    assert "\\fscx" not in ass                                  # colour accent, not phrase_jump's scale bounce


def test_bold_size_sits_between_phrase_and_title() -> None:
    assert captions.PHRASE_SIZE < captions.BOLD_SIZE < captions.TITLE


def test_the_three_legacy_styles_keep_their_own_size_and_stay_not_bold() -> None:
    """NEGATIVE — proves the `bold` addition did not leak BOLD_SIZE/Bold:1 into phrase/phrase_jump (byte-exact
    pin for oneword lives in tests/test_captions_ass_golden.py; this covers the two _build_ass_phrase callers
    the golden test does not hash)."""
    font = _real_font()
    kw = dict(font=font, w=1080, h=1920, fg=_FG, accent=_ACCENT, center_y=0.76)
    for style in ("phrase", "phrase_jump"):
        ass = captions.build_ass(_WORDS, style=style, **kw)
        style_line = next(ln for ln in ass.splitlines() if ln.startswith("Style: Cap,"))
        fields = style_line.split(",")
        assert fields[2] == str(captions.PHRASE_SIZE), f"{style}: Fontsize drifted from PHRASE_SIZE"
        assert fields[7] == "0", f"{style}: Bold flag leaked on from the bold addition"
