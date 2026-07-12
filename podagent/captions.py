"""Subtitle track via libass (no browser): brand-neutral ASS builder for three looks
(oneword/phrase/phrase_jump). The brain bakes words+accent+centerY into the spec; the pod draws
them with a delivered TTF."""
from __future__ import annotations

from pathlib import Path

# ── look, tuned to the brand build_motion captions (A/B-verified) ────────────────────────────────────
TITLE = 92          # libass Fontsize == 64px on-screen (different metric than CSS px)
OUTLINE = 3
SHADOW = 4
BLUR = 4
FADE_IN_MS = 83
RISE_PX = 16
RISE_MS = 120
HOLD_AFTER = 0.8    # keep a word up this long past its end if no next word
FG = "#f2f2f0"

PHRASE_SIZE = 70
PHRASE_PX = round(PHRASE_SIZE * 64 / 92)   # libass Fontsize → on-screen px (measure/layout in THIS)
PHRASE_WINDOW_MS = 700
PHRASE_MAX_LINES = 2
PHRASE_MARGIN = 110
JUMP_OVERSHOOT = 120
JUMP_SCALE = 110
JUMP_GROW_MS = 90

# safe-zone bottom reserve (9:16 only), from the 1080×1920 reference; scales with height
_REF_H = 1920
_SAFE_BOTTOM = 1565


def _caption_max_y(below_px: int, h: int) -> int:
    return round(_SAFE_BOTTOM * h / _REF_H) - below_px


def _ac(hexc: str, aa: str = "00") -> str:
    """#rrggbb → ASS &HaaBBGGRR."""
    h = hexc.lstrip("#")
    return f"&H{aa}{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _inline_c(hexc: str) -> str:
    """#rrggbb → ASS inline \\1c form &HBBGGRR&."""
    h = hexc.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def _tc(t: float) -> str:
    cs = int(round(t * 100))
    hh = cs // 360000; cs %= 360000
    mm = cs // 6000; cs %= 6000
    ss = cs // 100; cs %= 100
    return f"{hh:d}:{mm:02d}:{ss:02d}.{cs:02d}"


def _clean(text: object) -> str:
    s = str(text).upper().replace("\n", " ").replace("{", "").replace("}", "")
    return s.strip(" .,!?;:…«»\"'()-—–").strip()


def _clamp_cy(center_y: float, below_px: int, w: int, h: int) -> float:
    """Portrait only: cap center_y so `below_px` under the anchor clears the bottom UI reserve."""
    if h <= w:
        return center_y
    return min(center_y, _caption_max_y(below_px, h) / h)


def _ass_head(white: str, w: int, h: int, size: int = TITLE) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Inter ExtraBold,{size},{white},&H000000FF,&H00000000,&H50000000,0,0,0,0,100,100,0,0,1,{OUTLINE},{SHADOW},5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass(words: list[dict], *, font: Path, w: int, h: int, accent: str = "#d6ff3a",
              center_y: float = 0.76, style: str = "oneword",
              window_ms: int = PHRASE_WINDOW_MS) -> str:
    """words = [{text,start,end,hot?}, …] → ASS text; `style` picks oneword/phrase/phrase_jump."""
    if style in ("phrase", "phrase_jump"):
        return _build_ass_phrase(words, font, w, h, accent, center_y, window_ms, kind=style)
    white = _ac(FG)
    lime = _ac(accent)
    center_y = _clamp_cy(center_y, TITLE // 2 + RISE_PX, w, h)
    ymid = round(h / 2 + (center_y - 0.5) * h)
    xc = w // 2
    head = _ass_head(white, w, h)
    n = len(words)
    lines = []
    for i, wd in enumerate(words):
        st = float(wd["start"])
        nxt = float(words[i + 1]["start"]) if i + 1 < n else 1e9
        su = min(nxt, float(wd["end"]) + HOLD_AFTER)
        su = su if su > st else st + 0.1
        tag = (f"{{\\move({xc},{ymid + RISE_PX},{xc},{ymid},0,{RISE_MS})"
               f"\\fad({FADE_IN_MS},0)\\blur{BLUR}")
        if wd.get("hot"):
            tag += f"\\1c{lime}"
        tag += "}"
        lines.append(f"Dialogue: 0,{_tc(st)},{_tc(su)},Cap,,0,0,0,,{tag}{_clean(wd['text'])}")
    return head + "\n".join(lines) + "\n"


def _phrase_font(font: Path):
    from PIL import ImageFont
    return ImageFont.truetype(str(font), PHRASE_PX)   # measure at on-screen px, not the libass Fontsize


def _wrap_lines(block, fnt, spc, w):
    """Greedily wrap words into ≤lines by pixel width. Returns [[(word, w_px), …], …]."""
    maxw = w - 2 * PHRASE_MARGIN
    lines, line, used = [], [], 0.0
    for wd in block:
        ww = fnt.getlength(_clean(wd["text"]))
        adv = ww if not line else spc + ww
        if line and used + adv > maxw:
            lines.append(line); line, used = [], 0.0; adv = ww
        line.append((wd, ww)); used += adv
    if line:
        lines.append(line)
    return lines


def _group_blocks(words, fnt, spc, window_ms, w):
    """Break a new block on a pause > window_ms, or when the next word would need a 3rd line."""
    gap_s = window_ms / 1000.0
    blocks, cur = [], []
    for wd in words:
        if cur:
            gap = float(wd["start"]) - float(cur[-1]["end"])
            if gap > gap_s or len(_wrap_lines(cur + [wd], fnt, spc, w)) > PHRASE_MAX_LINES:
                blocks.append(cur); cur = []
        cur.append(wd)
    if cur:
        blocks.append(cur)
    return blocks


def _build_ass_phrase(words, font, w, h, accent, center_y, window_ms, *, kind):
    """Stable, centred ≤2-line block pinned at a fixed y (never jumps); kind = phrase | phrase_jump."""
    white = _ac(FG)
    white_c = _inline_c(FG)
    accent_c = _inline_c(accent)
    fnt = _phrase_font(font)
    spc = fnt.getlength(" ")
    line_h = round(PHRASE_PX * 1.5)
    center_y = _clamp_cy(center_y, line_h, w, h)
    y_top = round(h * center_y) - line_h
    xc = w // 2
    blocks = _group_blocks(words, fnt, spc, window_ms, w)
    out = []
    for bi, block in enumerate(blocks):
        lines = _wrap_lines(block, fnt, spc, w)
        b_start = float(block[0]["start"])
        nxt = float(blocks[bi + 1][0]["start"]) if bi + 1 < len(blocks) else 1e9
        b_end = min(nxt, float(block[-1]["end"]) + HOLD_AFTER)
        if kind == "phrase_jump":
            out += _phrase_jump_block(lines, b_start, b_end, y_top, line_h, spc, xc, white_c, accent_c)
        else:
            out += _phrase_colour_block(lines, b_start, b_end, y_top, line_h, xc, white_c, accent_c)
    return _ass_head(white, w, h, PHRASE_SIZE) + "\n".join(out) + "\n"


def _phrase_colour_block(lines, b_start, b_end, y_top, line_h, xc, white_c, accent_c):
    """Colour-highlight: libass-native centred lines (static white) + a per-word accent overlay."""
    out = []
    for i, line in enumerate(lines):
        yc = y_top + i * line_h
        txt = " ".join((f"{{\\1c{accent_c}}}{_clean(wd['text'])}{{\\1c{white_c}}}" if wd.get("hot")
                        else _clean(wd["text"])) for wd, _ in line)
        out.append(f"Dialogue: 1,{_tc(b_start)},{_tc(b_end)},Cap,,0,0,0,,"
                   f"{{\\an8\\pos({xc},{yc})\\fad({FADE_IN_MS},0)\\blur{BLUR}}}{txt}")
    for i, line in enumerate(lines):
        yc = y_top + i * line_h
        m = len(line)
        for k, (wd, _) in enumerate(line):
            st = float(wd["start"])
            nxt = float(line[k + 1][0]["start"]) if k + 1 < m else b_end
            end = nxt if nxt > st else st + 0.1
            parts = [(f"{{\\1c{accent_c}}}{_clean(w2['text'])}{{\\1c{white_c}}}" if (jj == k or w2.get("hot"))
                      else _clean(w2["text"])) for jj, (w2, _) in enumerate(line)]
            out.append(f"Dialogue: 2,{_tc(st)},{_tc(end)},Cap,,0,0,0,,"
                       f"{{\\an8\\pos({xc},{yc})\\blur{BLUR}}}{' '.join(parts)}")
    return out


def _jump_bounce(d_ms):
    """Inline scale tags for one word's bounce: spring to overshoot, settle, then ease back to 100."""
    b = (f"\\fscx100\\fscy100\\t(0,{JUMP_GROW_MS},0.4,\\fscx{JUMP_OVERSHOOT}\\fscy{JUMP_OVERSHOOT})"
         f"\\t({JUMP_GROW_MS},{2 * JUMP_GROW_MS},\\fscx{JUMP_SCALE}\\fscy{JUMP_SCALE})")
    if d_ms > 2 * JUMP_GROW_MS + 80:
        b += f"\\t({d_ms - 80},{d_ms},\\fscx100\\fscy100)"
    return b


def _phrase_jump_block(lines, b_start, b_end, y_top, line_h, spc, xc, white_c, accent_c):
    """Every word at its own \\pos (fixed y); the spoken word bounces in scale, neighbours slide sideways."""
    out = []
    gap = spc + 8
    grow_f = JUMP_SCALE / 100.0 - 1.0
    for i, line in enumerate(lines):
        cy = round(y_top + i * line_h + PHRASE_PX * 0.5)
        ww = [w_px for _, w_px in line]
        toks = [_clean(wd["text"]) for wd, _ in line]
        starts = [float(wd["start"]) for wd, _ in line]
        m = len(line)
        total = sum(ww) + gap * (m - 1)
        x = xc - total / 2.0
        cx = []
        for k in range(m):
            cx.append(round(x + ww[k] / 2.0)); x += ww[k] + gap
        segs = []
        if starts[0] > b_start + 0.02:
            segs.append((b_start, starts[0], -1))
        for k in range(m):
            segs.append((starts[k], starts[k + 1] if k + 1 < m else b_end, k))
        for j in range(m):
            prev_x = None
            hot_col = f"\\1c{accent_c}" if line[j][0].get("hot") else ""
            for s, e, act in segs:
                e = e if e > s else s + 0.05
                fade = f"\\fad({FADE_IN_MS},0)" if abs(s - b_start) < 0.02 else ""
                if act == j or act < 0:
                    tx = cx[j]
                else:
                    push = round(grow_f * ww[act] / 2.0)
                    tx = cx[j] + (-push if j < act else push)
                if act == j:
                    tag = f"{{\\an5\\pos({cx[j]},{cy}){fade}{_jump_bounce(round((e - s) * 1000))}{hot_col}\\blur{BLUR}}}"
                elif prev_x is not None and prev_x != tx:
                    tag = f"{{\\an5\\move({prev_x},{cy},{tx},{cy},0,{JUMP_GROW_MS}){fade}{hot_col}\\blur{BLUR}}}"
                else:
                    tag = f"{{\\an5\\pos({tx},{cy}){fade}{hot_col}\\blur{BLUR}}}"
                out.append(f"Dialogue: 1,{_tc(s)},{_tc(e)},Cap,,0,0,0,,{tag}{toks[j]}")
                prev_x = tx
    return out
