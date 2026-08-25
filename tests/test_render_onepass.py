"""podagent.render_onepass — the COMPOSITION layer: one graph, one VIDEO encode, one sync
reference. Connectivity, goldens, reuse-by-spy and the negatives. Nothing here runs ffmpeg, and
nothing here asserts on a rendered frame (that harness is a later wave)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from podagent import accents, finalize, mograph, render, render_onepass as op
from podagent.models import RenderSpec

SHA = "0" * 64

_BASE_SPEC = {
    "spec_version": 6, "job_id": "job_w2a", "slug": "onepass", "mode": "final",
    "inputs": [
        {"id": "base", "kind": "video", "sha256": SHA, "url": "https://x/base.mp4"},
        {"id": "music/bed.mp3", "kind": "audio", "sha256": SHA, "url": "https://x/bed.mp3"},
        {"id": "caption_font", "kind": "font", "sha256": SHA, "url": "https://x/Inter.ttf"},
        {"id": "brand/logo.png", "kind": "image", "sha256": SHA, "url": "https://x/logo.png"},
        {"id": "brand/wm-sting.webm", "kind": "video", "sha256": SHA, "url": "https://x/sting.webm"},
        {"id": "brand/wm-idle.webm", "kind": "video", "sha256": SHA, "url": "https://x/idle.webm"},
    ],
    # multi-segment AND non-unit speed: a one-segment speed-1 fixture lets a wrong duration formula pass
    "timeline": {"fps": 30, "width": 1080, "height": 1920, "segments": [
        {"src": "base", "in": 0.0, "out": 6.0, "speed": 1.0},
        {"src": "base", "in": 10.0, "out": 15.0, "speed": 1.25},
    ]},
    "overlays": {
        "motion_plan": {
            "sections": [{"comp": "lower_third", "start": 2.0, "props": {"text": "hi"}, "glass": False}],
            "captions": {"centerY": 0.78, "accent": "#d6ff3a", "style": "oneword", "hot": [],
                         "words": [{"text": "HELLO", "start": 0.1, "end": 0.5, "hot": False}],
                         "font": "caption_font"},
            "brand": {"tokens": {"color": {"accent": "#d6ff3a", "fg": "#f2f2f0"}}, "fonts": {}},
            "bundle": {"url": "https://x/b.tar", "sha256": SHA},
        },
        "music": {"track": "music/bed.mp3", "start": 0.0, "gain": 0.35},
        "finalize": {
            "accents": [{"kind": "pixelate", "at": 3.0, "intensity": 0.5}],
            "logo": {"asset": "brand/logo.png", "corner": "tr", "width": 150, "opacity": 0.55,
                     "margin": 40, "cover_hold": 0.6},
            "watermark": {"sting": "brand/wm-sting.webm", "idle": "brand/wm-idle.webm", "width": 422,
                          "margin": 60, "position": "bottom-center", "delay": 2.5, "chime": True,
                          "chime_volume": 0.4},
            "loudnorm": {"i": -14.0, "tp": -1.0, "lra": 11.0, "attenuate_only": False},
        },
    },
    # cq23 on the spec: the deliverable must NOT be driven off it (§4.6)
    "encode": {"video": "h264_nvenc", "preset": "p7", "cq": 23, "pix_fmt": "yuv420p",
               "audio": "aac", "audio_bitrate": "192k"},
    "outputs": [
        {"id": "master", "kind": "master", "put_url": "https://x/master.mp4?PUT"},
        {"id": "presync", "kind": "presync", "put_url": "https://x/master.presync.mp4?PUT"},
    ],
}

DUR = 10.0                       # (6-0)/1.0 + (15-10)/1.25
MIX = render._AudioMix(voice_idx=0, bed_idx=1, clean="highpass=f=80",
                       vln="loudnorm=I=-20:TP=-1.5:LRA=11", dur=DUR, sfx=())
LAYERS = ({"mov": "/w/seq0.mov", "start": 2.0, "dur": 3.0, "glass": False},)
LN_JSON = ('{ "input_i" : "-19.4", "input_tp" : "-2.1", "input_lra" : "7.2", '
           '"input_thresh" : "-29.8", "target_offset" : "0.0" }')


def _spec(mutate=None) -> RenderSpec:
    data = json.loads(json.dumps(_BASE_SPEC))
    if mutate is not None:
        mutate(data)
    return RenderSpec.model_validate(data)


def _prepared(spec: RenderSpec, **kw) -> op.Prepared:
    """Fixed synthetic paths on purpose: a TemporaryDirectory would move the ASS/font text in the
    caption filter and the goldens would change every run."""
    base = dict(spec=spec, gpu=False, input_paths={i.id: Path("/w") / i.id for i in spec.inputs},
                duration=DUR, master_out=Path("/w/master.mp4"),
                presync_out=Path("/w/master.presync.mp4"), filter_script=Path("/w/body.filter"),
                bed=Path("/w/music_bed.flac"), audio=MIX, layers=LAYERS,
                ass=Path("/w/captions.ass"), font_dir=Path("/w/fonts"))
    base.update(kw)
    return op.Prepared(**base)


def _paths(spec: RenderSpec, root: Path) -> dict[str, Path]:
    return {i.id: root / i.id.replace("/", "__") for i in spec.inputs}


# --- graph reader (the connectivity assertion) --------------------------------

_EXT = re.compile(r"\d+:[av]")
_LEAD = re.compile(r"^(?:\[[^\[\]]+\])+")
_TRAIL = re.compile(r"(?:\[[^\[\]]+\])+$")


def _chains(script: str):
    out = []
    for raw in script.split(";"):
        lead, trail = _LEAD.match(raw), _TRAIL.search(raw)
        assert lead and trail, f"chain has no input or no output pad: {raw!r}"
        out.append((raw, re.findall(r"\[([^\[\]]+)\]", lead.group(0)),
                    re.findall(r"\[([^\[\]]+)\]", trail.group(0))))
    return out


def assert_connected(script: str, mapped: list[str]) -> None:
    """Every consumed pad produced exactly once, every produced pad used exactly once, one connected
    DAG from the declared inputs to the mapped outputs. Outdegree IS the fan-out rule: a filter with
    several DISTINCT outputs (concat a=1, split) is legal, one pad read twice is what ffmpeg refuses.
    Input STREAM pads are exempt — ffmpeg forks an -i stream itself, a link it does not."""
    chains = _chains(script)
    produced: dict[str, int] = {}
    consumed: dict[str, int] = {}
    for _text, ins, outs in chains:
        for pad in outs:
            produced[pad] = produced.get(pad, 0) + 1
        for pad in ins:
            consumed[pad] = consumed.get(pad, 0) + 1
    for pad, n in produced.items():
        assert n == 1, f"pad [{pad}] is produced {n} times — ffmpeg refuses the whole graph"
        used = consumed.get(pad, 0) + mapped.count(pad)
        assert used == 1, f"pad [{pad}] has outdegree {used}, not 1 (a link is single-use)"
    for pad in consumed:
        if not _EXT.fullmatch(pad):
            assert produced.get(pad) == 1, f"pad [{pad}] is consumed but never produced"
    avail = {p for p in consumed if _EXT.fullmatch(p)}
    pending = list(chains)
    while True:
        ready = [c for c in pending if all(i in avail for i in c[1])]
        if not ready:
            break
        for c in ready:
            avail |= set(c[2])
            pending.remove(c)
    assert not pending, f"unreachable (cycle or dangling producer): {[c[0] for c in pending]}"
    for pad in mapped:
        assert pad in avail, f"mapped pad [{pad}] is not produced by the graph"


def _ancestors(script: str, pad: str) -> set[str]:
    by_out = {o: ins for _t, ins, outs in _chains(script) for o in outs}
    seen, stack = set(), [pad]
    while stack:
        cur = stack.pop()
        for src in by_out.get(cur, []):
            if src not in seen:
                seen.add(src)
                stack.append(src)
    return seen


# --- argv reader (options belong to the destination they precede) -------------

_ZERO_ARG = {"-y", "-hide_banner", "-an", "-nostats"}


def _argv(cmd: list[str]):
    """(inputs, output clauses). inputs[0].flags carries the GLOBAL options — nothing precedes the
    first -i but them — and every later input's flags are its own."""
    inputs, outputs, cur, i = [], [], [], 1
    while i < len(cmd):
        tok = cmd[i]
        if tok == "-i":
            inputs.append((tuple(cur), cmd[i + 1])); cur = []; i += 2
        elif tok in _ZERO_ARG:
            cur.append(tok); i += 1
        elif tok == "-" or not tok.startswith("-"):
            outputs.append((tuple(cur), tok)); cur = []; i += 1
        else:
            cur += [tok, cmd[i + 1]]; i += 2
    return inputs, outputs


def _has_seq(hay, needle: list[str]) -> bool:
    h = list(hay)
    return any(h[i:i + len(needle)] == needle for i in range(len(h) - len(needle) + 1))


def _maps(opts) -> list[str]:
    return [opts[i + 1].strip("[]") for i, o in enumerate(opts) if o == "-map"]


def _video_encodes(runs: list[list[str]]) -> list[list[str]]:
    """Commands that ENCODE video — an output clause naming a -c:v that is not `copy`. The threshold
    of this wave is one video encode, not one subprocess: the loudnorm remux is a third pass and a
    legitimate one, and counting subprocesses would either forbid it or hide its absence."""
    hits = []
    for cmd in runs:
        if not cmd or cmd[0] != "ffmpeg":
            continue
        _ins, outs = _argv(cmd)
        for opts, _dst in outs:
            if any(o == "-c:v" and opts[i + 1] != "copy" for i, o in enumerate(opts[:-1])):
                hits.append(cmd)
                break
    return hits


# --- goldens (string-exact, fixed synthetic paths) ----------------------------

_COMPOSITE = [
    '[0:v]trim=start=0:end=6,setpts=(PTS-STARTPTS)/1,scale=1080:1920:flags=lanczos,setsar=1[v0__cmp]',
    '[0:v]trim=start=10:end=15,setpts=(PTS-STARTPTS)/1.25,scale=1080:1920:flags=lanczos,setsar=1[v1__cmp]',
    '[v0__cmp][v1__cmp]concat=n=2:v=1:a=0[vcomposite]',
    '[0:a]highpass=f=80,loudnorm=I=-20:TP=-1.5:LRA=11,apad=whole_dur=10,asplit=2[vc1__cmp][vc2__cmp]',
    '[1:a]volume=1.0[bg0__cmp]',
    '[bg0__cmp][vc2__cmp]sidechaincompress=threshold=0.06:ratio=3:attack=20:release=500[bg__cmp]',
    '[vc1__cmp][bg__cmp]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[premix__cmp]',
    '[premix__cmp]aresample=192000,alimiter=limit=0.63:attack=5:release=50:level=false,'
    'aresample=48000[amaster__cmp]',
    '[amaster__cmp]anull[acomposite]',
]
# no music, no sfx: segment audio rides the concat itself, which emits TWO distinct pads
_COMPOSITE_NO_MUSIC = [
    '[0:v]trim=start=0:end=6,setpts=(PTS-STARTPTS)/1,scale=1080:1920:flags=lanczos,setsar=1[v0__cmp]',
    '[0:a]atrim=start=0:end=6,asetpts=PTS-STARTPTS,atempo=1[a0__cmp]',
    '[0:v]trim=start=10:end=15,setpts=(PTS-STARTPTS)/1.25,scale=1080:1920:flags=lanczos,setsar=1[v1__cmp]',
    '[0:a]atrim=start=10:end=15,asetpts=PTS-STARTPTS,atempo=1.25[a1__cmp]',
    '[v0__cmp][a0__cmp][v1__cmp][a1__cmp]concat=n=2:v=1:a=1[vcomposite][acomposite]',
]
def _pixelate(base: str, out: str) -> list[str]:
    return [
        '[%s]split=2[base_a0__acc][px_a0__acc]' % base,
        '[px_a0__acc]trim=start=2.8333:end=3.1667,setpts=PTS-STARTPTS,scale=49:87:flags=neighbor,'
        'scale=1080:1920:flags=neighbor,setpts=PTS-STARTPTS+2.8333/TB[pxx_a0__acc]',
        "[base_a0__acc][pxx_a0__acc]overlay=enable='between(t,2.8333,3.1667)'[%s]" % out,
    ]


_ACCENT = _pixelate("vtail", "vaccents")

# add_offset_jump is seeded (seed=42): the slip expression below is a deterministic function of the
# two boundaries alone, so the literal pins it the same way every other golden line is pinned.
_JUMP_CROP = (
    "[jstk__acc]crop=1080:1920:0:y='mod(((between(t,3.080,3.095)*(0+(107)*(t-3.080)/0.01500)"
    "+between(t,3.095,3.162)*(107+(0)*(t-3.095)/0.06700)+between(t,3.162,3.169)*(107+(-248)*(t-3.162)/0.00700)"
    "+between(t,3.169,3.271)*(-141+(0)*(t-3.169)/0.10200)+between(t,3.271,3.284)*(-141+(244)*(t-3.271)/0.01300)"
    "+between(t,3.284,3.336)*(103+(0)*(t-3.284)/0.05200)+between(t,3.336,3.371)*(103+(-103)*(t-3.336)/0.03500)"
    "+between(t,3.371,3.500)*(0+(0)*(t-3.371)/0.12900))+(between(t,6.580,6.595)*(0+(131)*(t-6.580)/0.01500)"
    "+between(t,6.595,6.694)*(131+(0)*(t-6.595)/0.09900)+between(t,6.694,6.705)*(131+(-300)*(t-6.694)/0.01100)"
    "+between(t,6.705,6.775)*(-169+(0)*(t-6.705)/0.07000)+between(t,6.775,6.790)*(-169+(300)*(t-6.775)/0.01500)"
    "+between(t,6.790,6.851)*(131+(0)*(t-6.790)/0.06100)+between(t,6.851,6.886)*(131+(-131)*(t-6.851)/0.03500)"
    "+between(t,6.886,7.000)*(0+(0)*(t-6.886)/0.11400)))+192000,1920)'[vout__acc]"
)


def _burn_chains(idx: int, base: str) -> list[str]:
    return [
        '[%s]split=2[jc0__acc][jc1__acc]' % base,
        '[jc0__acc][jc1__acc]vstack=inputs=2[jstk__acc]',
        _JUMP_CROP,
        '[%d:v]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,'
        'crop=1080:1920,fps=30,format=gbrp,split=2[bc__acc][bl__acc]' % idx,
        "[bl__acc]format=gray,curves=all='0/0.05 0.12/0.5 1/0.8',colorchannelmixer=aa=0.6[bm__acc]",
        '[bc__acc][bm__acc]alphamerge,format=yuva420p,split=2[bn0__acc][bn1__acc]',
        '[bn0__acc]trim=start=0.06:duration=0.640,setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.140:alpha=1,'
        'fade=t=out:st=0.440:d=0.200:alpha=1,setpts=PTS-STARTPTS+3.080/TB[bs0__acc]',
        "[vout__acc][bs0__acc]overlay=enable='between(t,3.080,3.720)'[g0__acc]",
        '[bn1__acc]trim=start=0.06:duration=0.640,setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.140:alpha=1,'
        'fade=t=out:st=0.440:d=0.200:alpha=1,setpts=PTS-STARTPTS+6.580/TB[bs1__acc]',
        "[g0__acc][bs1__acc]overlay=enable='between(t,6.580,7.220)'[vaccents]",
    ]


def _mograph_chains(idx: int) -> list[str]:
    return ['[%d:v]setpts=PTS-STARTPTS+2.0/TB[o0__mog]' % idx,
            "[vcomposite][o0__mog]overlay=enable='between(t,2.0,5.0)':eof_action=pass[vmograph]"]


def _fork(src: str) -> list[str]:
    return ['[%s]split=2[vtail][vrefsrc]' % src,
            '[vrefsrc]scale=80:142,format=gray[vpresync]',
            '[acomposite]asplit=2[atail][apresync]']


def _logo(idx: int, base: str) -> list[str]:
    return ['[%d:v]format=rgba,colorchannelmixer=aa=0.55,scale=150:-1:flags=lanczos[lg__lgo]' % idx,
            "[%s][lg__lgo]overlay=W-w-40:40:enable='lt(t,10.000)'[vlogo]" % base]


def _watermark(sting: int, idle: int, base: str, base_a: str = "atail") -> list[str]:
    return [
        '[%d:v]pad=1200:600:0:(oh-ih)/2:color=black@0,fps=30,setpts=PTS-STARTPTS[i__wmk]' % sting,
        '[%d:v]pad=1200:600:0:(oh-ih)/2:color=black@0,fps=30,setpts=PTS-STARTPTS[d__wmk]' % idle,
        '[i__wmk][d__wmk]concat=n=2:v=1:a=0[wm0__wmk]',
        '[wm0__wmk]scale=422:-1[wm__wmk]',
        '[wm__wmk]setpts=PTS+2.5/TB[wmd__wmk]',
        "[%s][wmd__wmk]overlay=(W-w)/2:H-h-60:enable='gte(t,2.5)':shortest=1:format=auto[vwatermark]" % base,
        '[%d:a]aresample=48000,volume=0.4,adelay=2500|2500[chm__wmk]' % sting,
        '[%s][chm__wmk]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[awatermark]' % base_a,
    ]


_CAPTIONS = '[%s]subtitles=/w/captions.ass:fontsdir=/w/fonts[vcaptions]'
_MASTER_MAPS = ["vmaster", "awatermark"]
_REF_MAPS = ["vpresync", "apresync"]


def _bt709_stamp(base: str) -> list[str]:
    return [f'[{base}]{finalize._BT709_SET_PARAMS}[vmaster]']


def _drop_mograph(d):
    d["overlays"]["motion_plan"]["sections"] = []
    d["overlays"]["motion_plan"].pop("bundle")


def _drop_captions(d):
    d["overlays"]["motion_plan"]["captions"]["words"] = []


def _drop_accents_and_logo(d):
    d["overlays"]["finalize"]["accents"] = []
    d["overlays"]["finalize"].pop("logo")


def _drop_watermark(d):
    d["overlays"]["finalize"].pop("watermark")


def _drop_music(d):
    d["overlays"].pop("music")


def _drop_presync(d):
    d["outputs"] = [o for o in d["outputs"] if o["kind"] != "presync"]


def _no_chime(d):
    d["overlays"]["finalize"]["watermark"]["chime"] = False


def _add_film_burn(d):
    d["inputs"] += [{"id": "fx/burn.mp4", "kind": "video", "sha256": SHA, "url": "https://x/b"},
                    {"id": "fx/clicks.wav", "kind": "audio", "sha256": SHA, "url": "https://x/c"}]
    d["overlays"]["finalize"]["accents"] += [
        {"kind": "film_burn", "at": 3.5, "intensity": 0.6, "burn": "fx/burn.mp4",
         "clicks": "fx/clicks.wav"},
        {"kind": "film_burn", "at": 7.0, "intensity": 0.6, "burn": "fx/burn.mp4",
         "clicks": "fx/clicks.wav"},
    ]


def _only_film_burn(d):
    _add_film_burn(d)
    d["overlays"]["finalize"]["accents"] = [
        a for a in d["overlays"]["finalize"]["accents"] if a["kind"] == "film_burn"]


def test_golden_everything_present() -> None:
    graph, cmd = op.assemble(_prepared(_spec()))
    assert graph.split(";") == (
        _COMPOSITE + _mograph_chains(2) + [_CAPTIONS % "vmograph"] + _fork("vcaptions")
        + _ACCENT + _logo(3, "vaccents") + _watermark(4, 5, "vlogo") + _bt709_stamp("vwatermark"))
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)
    _ins, outs = _argv(cmd)
    assert [_maps(o[0]) for o in outs] == [_MASTER_MAPS, _REF_MAPS]


def test_golden_no_mograph() -> None:
    graph, _cmd = op.assemble(_prepared(_spec(_drop_mograph), layers=()))
    assert graph.split(";") == (
        _COMPOSITE + [_CAPTIONS % "vcomposite"] + _fork("vcaptions")
        + _ACCENT + _logo(2, "vaccents") + _watermark(3, 4, "vlogo") + _bt709_stamp("vwatermark"))
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_golden_no_captions() -> None:
    graph, _cmd = op.assemble(_prepared(_spec(_drop_captions), ass=None, font_dir=None))
    assert graph.split(";") == (
        _COMPOSITE + _mograph_chains(2) + _fork("vmograph")
        + _ACCENT + _logo(3, "vaccents") + _watermark(4, 5, "vlogo") + _bt709_stamp("vwatermark"))
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_golden_no_accents_no_logo() -> None:
    graph, _cmd = op.assemble(_prepared(_spec(_drop_accents_and_logo)))
    assert graph.split(";") == (
        _COMPOSITE + _mograph_chains(2) + [_CAPTIONS % "vmograph"] + _fork("vcaptions")
        + _watermark(3, 4, "vtail") + _bt709_stamp("vwatermark"))
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_golden_no_watermark() -> None:
    graph, cmd = op.assemble(_prepared(_spec(_drop_watermark)))
    assert graph.split(";") == (
        _COMPOSITE + _mograph_chains(2) + [_CAPTIONS % "vmograph"] + _fork("vcaptions")
        + _ACCENT + _logo(3, "vaccents") + _bt709_stamp("vlogo"))
    assert_connected(graph, ["vmaster", "atail"] + _REF_MAPS)
    _ins, outs = _argv(cmd)
    assert [_maps(o[0]) for o in outs] == [["vmaster", "atail"], _REF_MAPS]


def test_golden_no_music() -> None:
    """concat with a=1 emits TWO pads from one filter — legal, and the shape a naive fan-out check
    (rather than an outdegree check) calls a defect."""
    graph, _cmd = op.assemble(_prepared(_spec(_drop_music), audio=None, bed=None))
    assert graph.split(";") == (
        _COMPOSITE_NO_MUSIC + _mograph_chains(1) + [_CAPTIONS % "vmograph"] + _fork("vcaptions")
        + _ACCENT + _logo(2, "vaccents") + _watermark(3, 4, "vlogo") + _bt709_stamp("vwatermark"))
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_golden_no_presync_output() -> None:
    """No declared presync address = no reference branch. Building it anyway costs a second
    full-body encode that render_body would then have nowhere to PUT."""
    graph, cmd = op.assemble(_prepared(_spec(_drop_presync)))
    assert graph.split(";") == (
        _COMPOSITE + _mograph_chains(2) + [_CAPTIONS % "vmograph"]
        + ['[vcaptions]split=2[base_a0__acc][px_a0__acc]'] + _ACCENT[1:]
        + _logo(3, "vaccents") + _watermark(4, 5, "vlogo", base_a="acomposite") + _bt709_stamp("vwatermark"))
    assert "vpresync" not in graph and "asplit=2[atail]" not in graph
    assert_connected(graph, _MASTER_MAPS)
    _ins, outs = _argv(cmd)
    assert [_maps(o[0]) for o in outs] == [_MASTER_MAPS]
    assert [o[1] for o in outs] == ["/w/master.mp4"]


def test_golden_film_burn_with_singles() -> None:
    """The singles chain feeds [preburn], the slip precedes the burn overlay, the burn rides its own
    looped input — the exact multi-pass recipe (finalize._apply_film_burn_accents), merged."""
    graph, cmd = op.assemble(_prepared(_spec(_add_film_burn), flares=(0.3,)))
    assert graph.split(";") == (
        _COMPOSITE + _mograph_chains(2) + [_CAPTIONS % "vmograph"] + _fork("vcaptions")
        + _pixelate("vtail", "preburn__acc") + _burn_chains(3, "preburn__acc")
        + _logo(4, "vaccents") + _watermark(5, 6, "vlogo") + _bt709_stamp("vwatermark"))
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)
    ins, _outs = _argv(cmd)
    assert ins[3] == (("-stream_loop", "-1"), "/w/fx/burn.mp4")


def test_golden_film_burn_only() -> None:
    graph, _cmd = op.assemble(_prepared(_spec(_only_film_burn), flares=(0.3,)))
    assert graph.split(";") == (
        _COMPOSITE + _mograph_chains(2) + [_CAPTIONS % "vmograph"] + _fork("vcaptions")
        + _burn_chains(3, "vtail") + _logo(4, "vaccents") + _watermark(5, 6, "vlogo") + _bt709_stamp("vwatermark"))
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_branch_order_is_by_link_not_by_substring() -> None:
    """mograph -> captions -> fork -> accents -> logo -> watermark, read off the master's ancestry."""
    graph, _cmd = op.assemble(_prepared(_spec()))
    assert "vlogo" in _ancestors(graph, "vwatermark")
    assert "vaccents" in _ancestors(graph, "vlogo")
    assert "vtail" in _ancestors(graph, "vaccents")
    assert "vcaptions" in _ancestors(graph, "vtail")
    assert "vmograph" in _ancestors(graph, "vcaptions")
    assert "vcomposite" in _ancestors(graph, "vmograph")


# --- reuse proof by spy -------------------------------------------------------

class _Spy:
    def __init__(self, real, ret=None):
        self.real, self.ret, self.calls = real, ret, []

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        return self.ret if self.ret is not None else self.real(*a, **kw)


def _sub(graph: str, fragment: str) -> bool:
    hay, needle = graph.split(";"), fragment.split(";")
    return any(hay[i:i + len(needle)] == needle for i in range(len(hay) - len(needle) + 1))


def test_reuse_composite_builder(monkeypatch) -> None:
    spec = _spec()
    spy = _Spy(render.build_filtergraph)
    monkeypatch.setattr(render, "build_filtergraph", spy)
    graph, _cmd = op.assemble(_prepared(spec))
    assert spy.calls[0][0][0] is spec and spy.calls[0][0][1] is False and spy.calls[0][0][2] is MIX
    real = render.build_filtergraph.real(spec, False, MIX, terminal_bt709=False)
    ext = {f"{n}:{s}": f"{n}:{s}" for n in (0, 1, 2) for s in "va"}
    assert _sub(graph, op.rewire(real, "cmp", {**ext, "vout": "vcomposite", "aout": "acomposite"}))


def test_reuse_mograph_builder(monkeypatch) -> None:
    spy = _Spy(mograph.overlay_filtergraph)
    monkeypatch.setattr(mograph, "overlay_filtergraph", spy)
    graph, _cmd = op.assemble(_prepared(_spec()))
    assert spy.calls[0][0][0] == list(LAYERS)
    assert spy.calls[0][1] == {"base": "vcomposite", "layers_v": ["2:v"]}
    real, last = mograph.overlay_filtergraph.real(list(LAYERS), base="vcomposite", layers_v=["2:v"])
    assert _sub(graph, op.rewire(real, "mog", {"vcomposite": "vcomposite", "2:v": "2:v",
                                               last: "vmograph"}))


def test_reuse_accent_builder(monkeypatch) -> None:
    spec = _spec()
    spy = _Spy(accents.build_chain_filter)
    monkeypatch.setattr(accents, "build_chain_filter", spy)
    graph, _cmd = op.assemble(_prepared(spec))
    fin = spec.overlays.finalize
    assert spy.calls[0][0][0] == fin.accents
    assert spy.calls[0][1] == {"fps": 30.0, "w": 1080, "h": 1920, "gpu": False}
    real = accents.build_chain_filter.real(fin.accents, fps=30.0, w=1080, h=1920, gpu=False)
    assert _sub(graph, op.rewire(real, "acc", {"0:v": "vtail", "vout": "vaccents"}))


def test_reuse_logo_builder(monkeypatch) -> None:
    spy = _Spy(finalize.body_logo_filter)
    monkeypatch.setattr(finalize, "body_logo_filter", spy)
    graph, _cmd = op.assemble(_prepared(_spec()))
    assert spy.calls[0][0][:4] == ("tr", 150, 0.55, 40)
    assert spy.calls[0][0][4] == pytest.approx(DUR)
    assert spy.calls[0][1] == {"base_v": "vaccents", "logo_v": "3:v", "out_v": "vlogo"}
    real = finalize.body_logo_filter.real("tr", 150, 0.55, 40, DUR, base_v="vaccents",
                                          logo_v="3:v", out_v="vlogo")
    assert _sub(graph, op.rewire(real, "lgo", {"vaccents": "vaccents", "3:v": "3:v",
                                               "vlogo": "vlogo"}))


def test_reuse_watermark_builder(monkeypatch) -> None:
    spy = _Spy(finalize.watermark_filter)
    monkeypatch.setattr(finalize, "watermark_filter", spy)
    graph, _cmd = op.assemble(_prepared(_spec()))
    kw = spy.calls[0][1]
    assert kw == {"base_v": "vlogo", "sting_v": "4:v", "idle_v": "5:v", "width": 422,
                  "overlay_xy": "(W-w)/2:H-h-60", "base_a": "atail", "chime_a": "4:a",
                  "chime_vol": 0.4, "delay": 2.5, "grid": "30", "sample_rate": 48000,
                  "out_v": "vwatermark", "out_a": "awatermark"}
    real, _v, _a = finalize.watermark_filter.real(**kw)
    keep = {"vlogo": "vlogo", "4:v": "4:v", "5:v": "5:v", "4:a": "4:a", "atail": "atail",
            "vwatermark": "vwatermark", "awatermark": "awatermark"}
    assert _sub(graph, op.rewire(real, "wmk", keep))


# --- audio, encode, duration, lifetime ---------------------------------------

def test_audio_contract() -> None:
    """One audio map per output, both descending from the composite mix, chime amixed exactly once."""
    graph, cmd = op.assemble(_prepared(_spec()))
    _ins, outs = _argv(cmd)
    (master_opts, master_dst), (ref_opts, ref_dst) = outs
    assert _maps(master_opts)[1] == "awatermark" and _maps(ref_opts)[1] == "apresync"
    assert "acomposite" in _ancestors(graph, "awatermark")
    assert "acomposite" in _ancestors(graph, "apresync")
    assert _has_seq(master_opts, ["-c:a", "aac", "-b:a", "192k"])
    assert list(master_opts).count("-c:a") == 1
    assert len([c for c in graph.split(";") if "chm__wmk" in c and "amix" in c]) == 1
    assert master_dst == "/w/master.mp4" and ref_dst == "/w/master.presync.mp4"


def test_no_chime_still_maps_the_composite_audio() -> None:
    """chime=false must NOT silence the master: watermark_filter leaves the base audio alone (it
    returns no audio label), so the composite mix is mapped straight through."""
    graph, cmd = op.assemble(_prepared(_spec(_no_chime)))
    _ins, outs = _argv(cmd)
    (master_opts, _m), (ref_opts, _r) = outs
    assert _maps(master_opts) == ["vmaster", "atail"]
    assert _maps(ref_opts) == _REF_MAPS
    assert "-an" not in master_opts and "-an" not in ref_opts
    assert "chm__wmk" not in graph
    assert_connected(graph, ["vmaster", "atail"] + _REF_MAPS)


def test_encode_shape_per_output_clause() -> None:
    """The ref rung carries the SAME -r as the master (not the old contract's bare absence): the sync
    oracle diffs the two rungs by frame INDEX, so a different grid compares different instants."""
    _ins, outs = _argv(op.assemble(_prepared(_spec()))[1])
    (master_opts, _m), (ref_opts, _r) = outs
    assert _has_seq(master_opts, finalize._FINAL_CPU)
    assert _has_seq(master_opts, ["-movflags", "+faststart"])
    assert not _has_seq(master_opts, op._REF_VIDEO)
    assert _has_seq(ref_opts, op._REF_VIDEO)
    assert _has_seq(ref_opts, ["-c:a", "aac", "-b:a", "128k"])
    assert not _has_seq(ref_opts, finalize._FINAL_CPU)
    assert "-vf" not in ref_opts
    master_r = master_opts[master_opts.index("-r") + 1]
    ref_r = ref_opts[ref_opts.index("-r") + 1]
    assert master_r == ref_r == "30"


def test_master_clause_declares_the_measured_grid() -> None:
    """The grid is DECLARED from spec.timeline.fps, never inherited from whichever input branch ffmpeg
    negotiates — that inheritance is the root cause of a 25fps/96kHz master shipping from a
    30fps/48kHz source."""
    _ins, outs = _argv(op.assemble(_prepared(_spec()))[1])
    (master_opts, _m), _ref = outs
    assert _has_seq(master_opts, ["-r", "30", "-fps_mode", "cfr"])
    assert _has_seq(master_opts, ["-ar", "48000"])


def test_presync_clause_shares_the_masters_grid() -> None:
    """A ref on a different grid from the master makes check_sync.py's frame-index comparison
    meaningless — it must declare cfr and the master's own -ar too."""
    _ins, outs = _argv(op.assemble(_prepared(_spec()))[1])
    (master_opts, _m), (ref_opts, _r) = outs
    assert master_opts[master_opts.index("-r") + 1] == ref_opts[ref_opts.index("-r") + 1]
    assert _has_seq(ref_opts, ["-fps_mode", "cfr"])
    assert _has_seq(ref_opts, ["-ar", "48000"])


def test_non_integer_grid_emits_the_exact_rational_not_a_rounded_float() -> None:
    """29.97002997 must ship as 30000/1001, not ffmpeg's `-r 29.97` rounding, which drifts the
    delivered duration by whole frames over a long body."""
    spec = _spec(lambda d: d["timeline"].__setitem__("fps", 30000 / 1001))
    _ins, outs = _argv(op.assemble(_prepared(spec))[1])
    (master_opts, _m), _ref = outs
    assert master_opts[master_opts.index("-r") + 1] == "30000/1001"


def test_every_reencoding_output_clause_declares_the_grid() -> None:
    """NEGATIVE: a delivery clause that skips -r inherits whatever grid ffmpeg negotiates from its
    OWN inputs (60fps sting/idle assets, mismatched audio rates) — exactly the root cause of this wave."""
    _ins, outs = _argv(op.assemble(_prepared(_spec()))[1])
    assert outs, "no output clauses to check — the fixture stopped exercising the delivery tail"
    for opts, _dst in outs:
        if _has_seq(list(opts), ["-c:v", "copy"]):
            continue
        assert "-r" in opts and "-fps_mode" in opts


def test_the_spec_encode_rung_never_reaches_the_deliverable() -> None:
    """NEGATIVE: spec.encode carries cq23 and the model admits 0..51 — driving the master off it is
    a silent quality regression, so the delivery rung is hardcoded on both transports."""
    spec = _spec()
    assert spec.encode.cq == 23
    _ins, outs = _argv(op.assemble(_prepared(spec))[1])
    assert "23" not in outs[0][0]
    assert _has_seq(outs[0][0], ["-crf", "14"])
    assert _has_seq(outs[0][0], finalize._BT709)
    assert _has_seq(outs[1][0], finalize._BT709)
    gpu_outs = _argv(op.assemble(_prepared(spec, gpu=True))[1])[1]
    assert _has_seq(gpu_outs[0][0], finalize._FINAL_GPU)
    assert _has_seq(gpu_outs[1][0], op._REF_VIDEO)
    assert _has_seq(gpu_outs[0][0], finalize._BT709)
    assert _has_seq(gpu_outs[1][0], finalize._BT709)


def test_input_side_decoder_and_loop_flags() -> None:
    ins, _outs = _argv(op.assemble(_prepared(_spec()))[1])
    assert ins[0][0] == ("-y", "-hide_banner")
    flags = {path: fl for fl, path in ins}
    assert flags["/w/brand/wm-sting.webm"] == ("-c:v", "libvpx-vp9")
    assert flags["/w/brand/wm-idle.webm"] == ("-c:v", "libvpx-vp9", "-stream_loop", "-1")
    others = [fl for fl, path in ins if not path.endswith(".webm")]
    assert all("libvpx-vp9" not in fl and "-stream_loop" not in fl for fl in others)


def test_every_external_pad_names_an_allocated_input() -> None:
    """A builder's hardcoded [1:v] must never survive: in the merged argv input 1 is another timeline
    SOURCE, and blending a video clip in as the "logo" raises nothing anywhere."""
    graph, cmd = op.assemble(_prepared(_spec()))
    ins, _outs = _argv(cmd)
    used = {int(p.split(":")[0]) for _t, i, _o in _chains(graph) for p in i if _EXT.fullmatch(p)}
    assert used and max(used) < len(ins)
    logo_idx = ins.index(next(x for x in ins if x[1].endswith("logo.png")))
    assert f"[{logo_idx}:v]format=rgba" in graph


@pytest.mark.parametrize("segments,want", [
    ([{"src": "base", "in": 0.0, "out": 6.0, "speed": 1.0},
      {"src": "base", "in": 10.0, "out": 15.0, "speed": 1.25}], 10.0),
    ([{"src": "base", "in": 2.0, "out": 8.0, "speed": 2.0},
      {"src": "base", "in": 0.0, "out": 4.5, "speed": 0.75},
      {"src": "base", "in": 9.0, "out": 12.0, "speed": 1.0}], 12.0),
])
def test_one_duration_computed_once_reaches_all_three_places(segments, want, monkeypatch, tmp_path) -> None:
    """§4.3, proved through prepare so no fixture constant can stand in for the computation: the SAME
    number bounds each output, gates the logo and pads the voice."""
    def mutate(d):
        d["timeline"]["segments"] = segments
    spec = _spec(mutate)
    _stub_prepare_passes(monkeypatch, tmp_path)
    p = op.prepare(spec, _paths(spec, tmp_path), tmp_path, False)
    assert op.body_duration(spec) == pytest.approx(want)
    assert p.duration == pytest.approx(want) and p.audio.dur == pytest.approx(want)
    graph, cmd = op.assemble(p)
    _ins, outs = _argv(cmd)
    for opts, _dst in outs:
        assert opts.count("-t") == 1
        assert float(opts[opts.index("-t") + 1]) == pytest.approx(want)
    assert f"enable='lt(t,{want:.3f})'" in graph
    assert f"apad=whole_dur={render._num(want)}" in graph


def test_the_logo_runs_to_the_end_of_the_body() -> None:
    spec = _spec()
    assert spec.overlays.finalize.logo.cover_hold == 0.6
    graph, _cmd = op.assemble(_prepared(spec))
    assert "enable='lt(t,10.000)'" in graph
    assert "9.400" not in graph


def test_the_t_bound_survives_a_missing_watermark() -> None:
    """NEGATIVE: -t is the master's lifetime, not a watermark artefact — dropping the looped idle
    input must not drop the bound with it."""
    _ins, outs = _argv(op.assemble(_prepared(_spec(_drop_watermark)))[1])
    assert all(o[0].count("-t") == 1 for o in outs)
    assert all(float(o[0][o[0].index("-t") + 1]) == pytest.approx(DUR) for o in outs)


def test_gpu_opens_vulkan_and_a_software_boundary_before_the_bare_hwupload() -> None:
    """accents.py:81/145/300 upload with no format= in front; the intermediate encode that used to
    hand them yuv420p is exactly what this wave deletes."""
    graph, cmd = op.assemble(_prepared(_spec(), gpu=True))
    assert cmd[:5] == ["ffmpeg", "-y", "-hide_banner", "-init_hw_device", "vulkan"]
    assert "[vtail]format=yuv420p[vaccentin]" in graph.split(";")
    assert "[vaccentin]split=2[base_a0__acc][px_a0__acc]" in graph.split(";")
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


# --- the two mapped outputs ---------------------------------------------------

def test_the_reference_is_forked_before_the_accents() -> None:
    """The tail can genuinely desync (zoom_punch re-slices video only while the watermark delays and
    amixes audio), so merging must not remove the detector: the reference is PRE-accent."""
    graph, cmd = op.assemble(_prepared(_spec()))
    _ins, outs = _argv(cmd)
    master_v, ref_v = _maps(outs[0][0])[0], _maps(outs[1][0])[0]
    assert any(a.endswith("__acc") for a in _ancestors(graph, master_v))
    assert not any(a.endswith("__acc") for a in _ancestors(graph, ref_v))
    assert "vcomposite" in _ancestors(graph, ref_v) and "vcaptions" in _ancestors(graph, ref_v)
    assert "vaccents" not in _ancestors(graph, ref_v)


def test_the_reference_is_prescaled_to_the_guards_own_grid() -> None:
    """check_sync picks the matching frame by argmin over 80x142 GRAYSCALE, so a rung-dependent
    compression delta lands straight in that match — scale and grey it in the graph instead."""
    graph, _cmd = op.assemble(_prepared(_spec()))
    assert f"[vrefsrc]scale={op._REF_W}:{op._REF_H},format=gray[vpresync]" in graph.split(";")
    assert (op._REF_W, op._REF_H) == (80, 142)


# --- connectivity, proved to bite --------------------------------------------

def test_two_fragments_wanting_the_pad_v0_do_not_collide() -> None:
    """NEGATIVE: build_filtergraph names segment pads v0/v1 (render.py:392) and overlay_filtergraph
    names layer pads v0/v1 (mograph.py:252) — merged unrewritten, ffmpeg refuses the whole graph."""
    layers = LAYERS + ({"mov": "/w/seq1.mov", "start": 6.0, "dur": 2.0, "glass": False},)
    graph, _cmd = op.assemble(_prepared(_spec(), layers=layers))
    pads = [p for _t, _i, outs in _chains(graph) for p in outs]
    assert "v0__cmp" in pads and "v0__mog" in pads
    assert len(pads) == len(set(pads))
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_the_connectivity_assertion_rejects_a_reused_link() -> None:
    """NEGATIVE for the checker itself: a link consumed twice without a split is what ffmpeg refuses
    with 'already used elsewhere', so the assertion that guards the merge must catch it."""
    bad = "[0:v]anull[x];[x]anull[y];[x]anull[z];[y][z]hstack=inputs=2[out]"
    with pytest.raises(AssertionError, match=r"outdegree 2"):
        assert_connected(bad, ["out"])


def test_the_connectivity_assertion_rejects_a_duplicated_pad_name() -> None:
    bad = "[0:v]anull[v0];[1:v]anull[v0];[v0]anull[out]"
    with pytest.raises(AssertionError, match=r"produced 2 times"):
        assert_connected(bad, ["out"])


def test_the_connectivity_assertion_rejects_an_orphan_output() -> None:
    bad = "[0:v]anull[x];[1:v]anull[y];[x]anull[out]"
    with pytest.raises(AssertionError, match=r"outdegree 0"):
        assert_connected(bad, ["out"])


def test_the_accent_chainer_never_rewrites_its_own_substitution() -> None:
    """NEGATIVE: _namespace_labels ended in two sequential str.replace calls, so a src_in holding the
    downstream label — [vout], which is exactly what build_filtergraph calls the composite — came
    back rewritten, and the accent read its own output while the composite was dropped."""
    sub = "[0:v]split=2[base][sh];[base][sh]overlay[vout]"
    assert accents._namespace_labels(sub, "a0", "[vout]", "[fa0]") == (
        "[vout]split=2[base_a0][sh_a0];[base_a0][sh_a0]overlay[fa0]")


def test_the_merge_rewriter_never_rewrites_its_own_substitution() -> None:
    assert op.rewire("[0:v]anull[vout]", "acc", {"0:v": "vout", "vout": "vaccents"}) == \
        "[vout]anull[vaccents]"


def test_the_merge_rewriter_refuses_an_unallocated_external_pad() -> None:
    """NEGATIVE: a fragment reading an input pad the allocator never handed out would silently read
    another timeline's stream in the merged argv — it must raise, not namespace or pass through."""
    with pytest.raises(ValueError, match="never handed out"):
        op.rewire("[0:v][7:v]overlay[vout]", "lgo", {"0:v": "vbase", "vout": "vlogo"})


def test_a_pad_neither_side_declares_is_left_exactly_as_it_stands() -> None:
    """substitute_pads is a REWRITER, not a validator: resolve() returning None must leave the token
    byte-for-byte, which is what keeps a filter argument that merely looks like a pad safe."""
    assert accents.substitute_pads("[a][1:v]x[b]", lambda _n: None) == "[a][1:v]x[b]"


def test_the_connectivity_assertion_accepts_one_filter_with_two_distinct_pads() -> None:
    """NEGATIVE for an earlier version of the checker: concat a=1 emits [v][a] from one filter and
    that is not fan-out — reading it as one made the whole no-music shape unbuildable."""
    ok = "[0:v][0:a]concat=n=1:v=1:a=1[v][a];[v]anull[vo];[a]anull[ao]"
    assert_connected(ok, ["vo", "ao"])


# --- preflight: the refusals --------------------------------------------------

def _no_subprocess(monkeypatch) -> None:
    def boom(*_a, **_kw):
        raise AssertionError("a subprocess ran before the spec was refused")
    for mod in (op, render, mograph, finalize):
        monkeypatch.setattr(mod.subprocess, "run", boom)


@pytest.mark.parametrize("what,match", [
    ("opener", "opener"), ("trims", "trims"), ("cover", "cover"),
    ("cover-output", "cover"), ("cover-both", "cover"),
])
def test_the_non_goals_are_refused_before_any_subprocess(what, match, monkeypatch, tmp_path) -> None:
    def _cover_output(d):
        d["outputs"].append({"id": "cover", "kind": "cover", "put_url": "https://x/cover.png?PUT"})

    def mutate(d):
        if what == "opener":
            d["inputs"].append({"id": "cold.mp4", "kind": "video", "sha256": SHA, "url": "https://x/o"})
            d["overlays"]["opener"] = {"cold": "cold.mp4", "cold_trim": 0.1, "gain": 0.4}
        elif what == "trims":
            d["overlays"]["trims"] = [{"a": 1.0, "b": 2.0}]
        elif what == "cover":
            d["overlays"]["cover"] = {"frame_at": 4.0, "headline": {"lines": ["A"]}}
        elif what == "cover-output":
            _cover_output(d)
        else:  # cover-both: overlays.cover AND a declared cover output, both refuse
            d["overlays"]["cover"] = {"frame_at": 4.0, "headline": {"lines": ["A"]}}
            _cover_output(d)
    spec = _spec(mutate)
    _no_subprocess(monkeypatch)
    with pytest.raises(NotImplementedError, match=match):
        op.preflight(spec)
    with pytest.raises(NotImplementedError):
        op.render_body(spec, _paths(spec, tmp_path), tmp_path, False)


def test_a_clean_body_spec_passes_preflight() -> None:
    op.preflight(_spec())


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf"), None])
def test_preflight_refuses_an_unusable_grid_before_any_subprocess(fps, monkeypatch) -> None:
    """The ONLY refusal a lost render is worse than — it must fire before the encode ever starts."""
    spec = _spec()
    spec.timeline.fps = fps
    _no_subprocess(monkeypatch)
    with pytest.raises(ValueError):
        op.preflight(spec)


# --- prepare ------------------------------------------------------------------

def _stub_prepare_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(render, "_voice_is_dirty", lambda _p: False)
    monkeypatch.setattr(render, "_measure_loudnorm", lambda _p, _pre: "loudnorm=I=-20:TP=-1.5:LRA=11")
    bed = tmp_path / "music_bed.flac"
    monkeypatch.setattr(render, "_prerender_bed", lambda *_a, **_kw: bed)
    monkeypatch.setattr(mograph, "_render_layers", lambda *_a, **_kw: list(LAYERS))
    return bed


def test_prepare_hands_the_bed_the_same_duration(monkeypatch, tmp_path) -> None:
    spec = _spec()
    bed = _stub_prepare_passes(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(render, "_prerender_bed", lambda *a, **_kw: (calls.append(a), bed)[1])
    p = op.prepare(spec, _paths(spec, tmp_path), tmp_path, False)
    assert calls[0][2] == pytest.approx(DUR)
    assert p.layers == tuple(LAYERS) and p.ass is not None and p.ass.is_file()


def test_prepare_leaves_the_base_bare_when_no_mograph_layer_survived(monkeypatch, tmp_path) -> None:
    """mograph.py:143/275: a declared section whose bundle entry is missing prints SKIP and the
    render continues — zero surviving layers must mean no mograph branch, not a broken graph."""
    spec = _spec()
    _stub_prepare_passes(monkeypatch, tmp_path)
    monkeypatch.setattr(mograph, "_render_layers", lambda *_a, **_kw: [])
    p = op.prepare(spec, _paths(spec, tmp_path), tmp_path, False)
    assert p.layers == ()
    graph, _cmd = op.assemble(p)
    assert "__mog" not in graph
    assert_connected(graph, _MASTER_MAPS + _REF_MAPS)


def test_prepare_detects_flares_under_the_prepare_phase(monkeypatch, tmp_path) -> None:
    """The flare decode is the ONE film_burn I/O — it rides prepare's arm pool so assemble stays pure."""
    spec = _spec(_add_film_burn)
    _stub_prepare_passes(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(accents, "detect_flares",
                        lambda p: (calls.append(Path(p).name), [0.3, 1.7])[1])
    ops: list[str] = []

    @contextmanager
    def rec(name):
        ops.append(name)
        yield

    p = op.prepare(spec, _paths(spec, tmp_path), tmp_path, False, phase=rec)
    assert p.flares == (0.3, 1.7)
    assert calls == ["fx__burn.mp4"]
    assert "prepare" in ops


def test_prepare_refuses_a_malformed_burn_set_before_the_flare_decode(monkeypatch, tmp_path) -> None:
    def two_ids(d):
        _add_film_burn(d)
        d["inputs"].append({"id": "fx/burn2.mp4", "kind": "video", "sha256": SHA, "url": "https://x/b2"})
        d["overlays"]["finalize"]["accents"][-1]["burn"] = "fx/burn2.mp4"
    spec = _spec(two_ids)
    _stub_prepare_passes(monkeypatch, tmp_path)
    monkeypatch.setattr(accents, "detect_flares",
                        lambda _p: pytest.fail("the flare decode ran before the shape refusal"))
    with pytest.raises(RuntimeError, match="share one burn input id"):
        op.prepare(spec, _paths(spec, tmp_path), tmp_path, False)


def test_an_unresolved_burn_input_refuses_before_any_pass(monkeypatch, tmp_path) -> None:
    spec = _spec(_add_film_burn)
    paths = _paths(spec, tmp_path)
    del paths["fx/burn.mp4"]
    monkeypatch.setattr(accents, "detect_flares",
                        lambda _p: pytest.fail("the flare decode ran on an unresolved input"))
    with pytest.raises(RuntimeError, match="is not resolved"):
        op.prepare(spec, paths, tmp_path, False)


# --- the door: one VIDEO encode, the loudnorm, both outputs delivered ---------

class _FakeFFmpeg:
    """Records argv and creates whatever each output clause names, so apply_loudnorm's own
    'did the remux produce a file' check sees what it would see for real."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        self.kwargs = kw
        for _opts, dst in _argv(list(cmd))[1]:
            if dst != "-":
                Path(dst).write_bytes(b"v")
        return SimpleNamespace(returncode=0, stdout="", stderr=LN_JSON if "null" in cmd else "")


def _door(monkeypatch, tmp_path, spec, **kw):
    runs = _FakeFFmpeg()
    puts: list[tuple[str, str]] = []
    monkeypatch.setattr(op.subprocess, "run", runs)
    monkeypatch.setattr(op, "upload", lambda path, url, _mime: puts.append((Path(path).name, url)))
    _stub_prepare_passes(monkeypatch, tmp_path)
    # No forbid-loop: the builders it used to stub are gone (see test_the_multipass_builders_no_longer_exist).
    return runs, puts, op.render_body(spec, _paths(spec, tmp_path), tmp_path, False, **kw)


def test_the_multipass_builders_no_longer_exist() -> None:
    """Pins the deletion: the one-pass graph is the ONLY final encode core, so its predecessors
    (and the cover module the old tail welded through) cannot exist to be reached by anything."""
    for mod, name in ((finalize, "apply_accents"), (finalize, "apply_logo"),
                      (finalize, "apply_watermark"), (finalize, "finalize"),
                      (render, "_burn_captions"), (mograph, "_overlay"), (mograph, "composite")):
        assert not hasattr(mod, name), f"{mod.__name__}.{name} should have been deleted"
    assert not (Path(__file__).resolve().parents[1] / "podagent" / "cover.py").exists()


def test_one_video_encode_the_loudnorm_still_runs_and_both_outputs_ship(monkeypatch, tmp_path) -> None:
    """Without this an implementation can build the one-pass argv AND still run the old passes, or
    PUT the raw encode and ship every deliverable ~6 dB under the brand target."""
    spec = _spec(_drop_mograph)
    runs, puts, d = _door(monkeypatch, tmp_path, spec)

    encodes = _video_encodes(runs.calls)
    assert len(encodes) == 1 and "-filter_complex_script" in encodes[0]
    assert len(runs.calls) > 1, "the delivery loudnorm measures and remuxes — it is not one call"
    assert d.master == tmp_path / "fin_ln.mp4" and d.master.is_file()
    assert d.prepared.filter_script.read_text() == op.assemble(d.prepared)[0]
    _ins, outs = _argv(encodes[0])
    assert [o[1] for o in outs] == [str(d.prepared.master_out), str(d.prepared.presync_out)]
    assert puts == [("fin_ln.mp4", "https://x/master.mp4?PUT"),
                    ("render.presync.mp4", "https://x/master.presync.mp4?PUT")]
    assert d.outputs == ["master", "presync"]


def test_the_delivered_master_is_never_the_raw_encode(monkeypatch, tmp_path) -> None:
    """NEGATIVE for the P1: drop apply_loudnorm and the PUT silently becomes the un-levelled file."""
    spec = _spec(_drop_mograph)
    _runs, puts, d = _door(monkeypatch, tmp_path, spec)
    assert d.master != d.prepared.master_out
    assert puts[0][0] != d.prepared.master_out.name


def test_the_encode_wait_is_bounded(monkeypatch, tmp_path) -> None:
    """-stream_loop -1 + shortest=1 leaves -t as the only thing that ends the process; unbounded,
    a graph that cannot end wedges the render stage with nothing to read."""
    spec = _spec(_drop_mograph)
    runs, _puts, _d = _door(monkeypatch, tmp_path, spec)
    master = _video_encodes(runs.calls)[0]
    assert master[0] == "ffmpeg"
    seen = [c for c in runs.calls if "-filter_complex_script" in c]
    assert len(seen) == 1
    assert op.encode_budget_s(DUR) == pytest.approx(300.0 + 60.0 * DUR)
    assert op.encode_budget_s(0.0) >= 300.0


def test_a_wedged_encode_fails_loud_instead_of_hanging(monkeypatch, tmp_path) -> None:
    spec = _spec(_drop_mograph)
    seen: dict = {}

    def timeout(cmd, **kw):
        seen.update(kw)
        raise op.subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(op.subprocess, "run", timeout)
    _stub_prepare_passes(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="budget"):
        op.render_body(spec, _paths(spec, tmp_path), tmp_path, False)
    assert seen["timeout"] == pytest.approx(op.encode_budget_s(DUR))


def test_the_phase_hook_carries_the_render_stages_own_op_names(monkeypatch, tmp_path) -> None:
    """render.render_spec wraps every step in phase(op) -> cp.send_event; taking the same hook now
    means routing this door later does not delete the render stage's per-op timings."""
    ops: list[str] = []

    @contextmanager
    def rec(name):
        ops.append(name)
        yield

    _runs, _puts, _d = _door(monkeypatch, tmp_path, _spec(), phase=rec)
    assert ops == ["prepare", "ffmpeg", "finalize", "upload"]


def test_a_film_burn_door_run_never_reaches_the_multipass_burn(monkeypatch, tmp_path) -> None:
    """_door forbids finalize.apply_accents, so a burn spec passing through IS the proof the one-pass
    graph composited the burn itself; the flare decode rides prepare's arm pool before ffmpeg."""
    monkeypatch.setattr(accents, "detect_flares", lambda _p: [0.3])
    ops: list[str] = []

    @contextmanager
    def rec(name):
        ops.append(name)
        yield

    _runs, _puts, d = _door(monkeypatch, tmp_path, _spec(_add_film_burn), phase=rec)
    assert ops == ["prepare", "ffmpeg", "finalize", "upload"]
    assert d.prepared.flares == (0.3,)
    assert d.outputs == ["master", "presync"]


def test_the_door_runs_with_nobody_listening(monkeypatch, tmp_path) -> None:
    _runs, _puts, d = _door(monkeypatch, tmp_path, _spec(_drop_presync))
    assert d.presync is None
    assert d.outputs == ["master"]


# --- the post-render grid verdict: loud, never a reason to withhold the PUT ---

def test_render_body_ships_the_master_on_a_measured_grid_mismatch(monkeypatch, tmp_path) -> None:
    """NEGATIVE: the verdict is loud but must never withhold an already-paid-for master."""
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 25.0, 25, 1, DUR))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: False)
    _runs, puts, d = _door(monkeypatch, tmp_path, _spec(_drop_mograph))
    assert d.defect == {"video_rate": {"declared": "30", "measured": "25"}}
    assert puts, "the PUT must still happen despite the mismatch"


def test_render_body_reports_no_defect_on_a_matching_grid(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(finalize, "_probe", lambda _p: (1080, 1920, 30.0, 30, 1, DUR))
    monkeypatch.setattr(finalize, "_has_audio", lambda _p: True)
    monkeypatch.setattr(finalize, "_probe_audio", lambda _p: (48000, DUR))
    _runs, puts, d = _door(monkeypatch, tmp_path, _spec(_drop_mograph))
    assert d.defect is None
    assert puts


def test_render_body_survives_a_post_render_probe_failure(monkeypatch, tmp_path) -> None:
    """_FakeFFmpeg answers every ffprobe with garbage; grid_verdict must turn that into a defect
    entry rather than an exception that would cost the render its PUT."""
    _runs, puts, d = _door(monkeypatch, tmp_path, _spec(_drop_mograph))
    assert d.defect is not None and "probe_failed" in d.defect
    assert puts


# --- the encode pass's own throughput record (speed=/fps= off a swallowed stderr) -------------


def test_run_logs_the_last_speed_line_on_success(monkeypatch, capsys):
    stderr = (b"frame=  100 fps= 50 q=20.0 size=1024KiB time=00:00:04.00 speed=2.5x\r"
              b"frame=  200 fps= 48 q=20.0 size=2048KiB time=00:00:08.00 speed=1.9x\n"
              b"[out#0] video summary\n")
    monkeypatch.setattr(op.subprocess, "run",
                        lambda *_a, **_k: SimpleNamespace(stderr=stderr, returncode=0))
    op._run(["ffmpeg"], 10.0)
    err = capsys.readouterr().err
    assert "[onepass]" in err and "speed=1.9x" in err
    assert "speed=2.5x" not in err


def test_run_without_a_progress_line_logs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(op.subprocess, "run",
                        lambda *_a, **_k: SimpleNamespace(stderr=b"no progress here\n", returncode=0))
    op._run(["ffmpeg"], 10.0)
    assert "[onepass]" not in capsys.readouterr().err


def test_run_failure_still_raises_with_the_stderr_tail(monkeypatch):
    def boom(*_a, **_k):
        raise op.subprocess.CalledProcessError(1, "ffmpeg", stderr=b"speed=0.1x then died")
    monkeypatch.setattr(op.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="exited 1"):
        op._run(["ffmpeg"], 10.0)


@pytest.mark.integration
def test_real_onepass_master_survives_loudnorm_bt709_tagged(tmp_path: Path) -> None:
    """Argv assertions miss a remux that drops container/stream colour tags — run the REAL encode
    AND the real apply_loudnorm remux, then ffprobe the file actually delivered (check_master's own
    read), not the pre-remux one."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe unavailable")

    source = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
        "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(source),
    ], check=True)

    spec = RenderSpec.model_validate({
        "spec_version": 6, "job_id": "colour-onepass", "slug": "colour-onepass", "mode": "final",
        "inputs": [{"id": "source", "kind": "video", "sha256": SHA, "url": "unused"}],
        "timeline": {"fps": 10, "width": 64, "height": 64,
                     "segments": [{"src": "source", "in": 0.0, "out": 1.0, "speed": 1.0}]},
        "overlays": {"finalize": {"loudnorm": {"i": -14.0, "tp": -1.0, "lra": 11.0}}},
        "encode": {"video": "libx264", "preset": "medium", "cq": 14, "pix_fmt": "yuv420p",
                    "audio": "aac", "audio_bitrate": "128k"},
        "outputs": [{"id": "master", "kind": "master", "put_url": "unused"}],
    })
    p = op.prepare(spec, {"source": source}, tmp_path, gpu=False)
    op.run_encode(p)
    delivered = finalize.apply_loudnorm(spec.overlays.finalize, p.master_out, tmp_path / "fin_ln.mp4")
    assert delivered != p.master_out, "loudnorm must have actually remuxed, or this proves nothing"

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=color_space,color_primaries,color_transfer",
        "-of", "json", str(delivered),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["color_space"] == "bt709"
    assert stream["color_primaries"] == "bt709"
    assert stream["color_transfer"] == "bt709"
