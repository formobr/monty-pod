"""render.render_spec routing: an accepted final spec swaps in ONLY the one-pass encode core; every
refusal is a counted named event and falls back to the multi-pass chain; the tail stays render_spec's."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podagent import finalize, mograph, render, render_onepass as op
from podagent.models import RenderSpec
from podagent.stream_models import StreamEvent

SHA = "0" * 64
_ENCODE = {"video": "h264_nvenc", "preset": "p7", "cq": 23, "pix_fmt": "yuv420p",
           "audio": "aac", "audio_bitrate": "192k"}


def _spec(mutate=None) -> RenderSpec:
    data = {
        "spec_version": 6, "job_id": "route", "slug": "route", "mode": "final",
        "inputs": [{"id": "base", "kind": "video", "sha256": SHA, "url": "https://x/base.mp4"}],
        "timeline": {"fps": 30, "width": 1080, "height": 1920,
                     "segments": [{"src": "base", "in": 0.0, "out": 5.0, "speed": 1.0}]},
        "overlays": {
            "finalize": {
                "accents": [{"kind": "pixelate", "at": 3.0, "intensity": 0.5}],
                "loudnorm": {"i": -14.0, "tp": -1.0, "lra": 11.0, "attenuate_only": False},
            },
        },
        "encode": _ENCODE,
        "outputs": [{"id": "master", "kind": "master", "put_url": "https://x/master.mp4?PUT"},
                    {"id": "presync", "kind": "presync", "put_url": "https://x/presync.mp4?PUT"}],
    }
    if mutate is not None:
        mutate(data)
    return RenderSpec.model_validate(data)


def _add_film_burn(d) -> None:
    d["inputs"] += [{"id": "burn.mp4", "kind": "video", "sha256": SHA, "url": "https://x/b"},
                    {"id": "clicks.wav", "kind": "audio", "sha256": SHA, "url": "https://x/c"}]
    d["overlays"]["finalize"]["accents"].append(
        {"kind": "film_burn", "at": 4.0, "intensity": 0.6, "burn": "burn.mp4",
         "clicks": "clicks.wav"})


class _CP:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.results: list[dict] = []

    def send_event(self, payload, *, wait=False):
        self.events.append(payload)
        return True

    def send_result(self, payload, *, wait=True):
        self.results.append(payload)
        return True


def _argv_outputs(cmd: list[str]) -> list[str]:
    zero = {"-y", "-hide_banner", "-an", "-nostats"}
    outs, i = [], 1
    while i < len(cmd):
        tok = cmd[i]
        if tok in zero:
            i += 1
        elif tok == "-" or not tok.startswith("-"):
            outs.append(tok)
            i += 1
        else:
            i += 2
    return outs


class _FakeFFmpeg:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **_kw):
        self.calls.append(list(cmd))
        for dst in _argv_outputs(list(cmd)):
            if dst != "-":
                Path(dst).write_bytes(b"v")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _forbid(monkeypatch, mod, name):
    def boom(*_a, **_kw):
        raise AssertionError(f"{mod.__name__}.{name} ran on the wrong route")
    monkeypatch.setattr(mod, name, boom)


def _wire(monkeypatch, grid_defect=None):
    runs, puts = _FakeFFmpeg(), []
    monkeypatch.setattr(render, "download", lambda _u, dest: (dest.write_bytes(b"m"), dest)[1])
    monkeypatch.setattr(render, "upload", lambda src, url, _ct: puts.append((Path(src).name, url)))
    monkeypatch.setattr(render, "_gpu_available", lambda: False)
    monkeypatch.setattr(render.subprocess, "run", runs)
    monkeypatch.setattr(render._finalize, "grid_verdict", lambda *_a, **_kw: grid_defect)
    return runs, puts


def _decisions(cp: _CP) -> list[dict]:
    return [e for e in cp.events if e.get("op") == "onepass_preflight"]


def test_an_accepted_final_spec_runs_the_onepass_core_not_the_multipass_chain(monkeypatch):
    _runs, puts = _wire(monkeypatch)
    for mod, name in ((mograph, "composite"), (render, "_burn_captions"), (render, "build_command"),
                      (finalize, "finalize"), (finalize, "apply_accents"), (finalize, "apply_logo"),
                      (finalize, "apply_watermark")):
        _forbid(monkeypatch, mod, name)
    ln_calls = []
    monkeypatch.setattr(finalize, "apply_loudnorm",
                        lambda _fin, src, out: (ln_calls.append(src), Path(out).write_bytes(b"v"),
                                                out)[2])
    cp = _CP()
    render.render_spec(_spec(), cp)

    dec = _decisions(cp)
    assert [d["phase"] for d in dec] == ["onepass_accepted"]
    assert "timings" not in dec[0]
    # the delivered master is the loudnorm output over the ONE encode's master, never the raw encode
    assert ln_calls and ln_calls[0].name == "render.mp4"
    assert [(n, u) for n, u in puts] == [("fin_ln.mp4", "https://x/master.mp4?PUT"),
                                         ("render.presync.mp4", "https://x/presync.mp4?PUT")]
    assert cp.results and cp.results[0]["status"] == "ok"
    assert cp.results[0]["outputs"] == ["master", "presync"]
    assert "defects" not in cp.results[0]
    ops = [e.get("op") for e in cp.events if e.get("phase", "").endswith("_started")]
    assert "ffmpeg" in ops and "finalize" in ops and "upload" in ops


def test_a_film_burn_spec_now_runs_the_onepass_core(monkeypatch):
    """Wave B: the 1/1 production refusal reason is gone — a burn spec routes through the one-pass
    core, flare-decodes in prepare, and never touches the multi-pass chain."""
    _runs, puts = _wire(monkeypatch)
    from podagent import accents
    monkeypatch.setattr(accents, "detect_flares", lambda _p: [0.3])
    for mod, name in ((finalize, "finalize"), (finalize, "apply_accents")):
        _forbid(monkeypatch, mod, name)
    monkeypatch.setattr(finalize, "apply_loudnorm",
                        lambda _fin, _src, out: (Path(out).write_bytes(b"v"), out)[1])
    cp = _CP()
    render.render_spec(_spec(_add_film_burn), cp)

    dec = _decisions(cp)
    assert [d["phase"] for d in dec] == ["onepass_accepted"]
    assert "timings" not in dec[0]
    StreamEvent.model_validate({**dec[0]})
    assert [(n, u) for n, u in puts] == [("fin_ln.mp4", "https://x/master.mp4?PUT"),
                                         ("render.presync.mp4", "https://x/presync.mp4?PUT")]
    assert cp.results and cp.results[0]["status"] == "ok"
    ops = [e.get("op") for e in cp.events if e.get("phase", "").endswith("_started")]
    assert "flares" in ops and "ffmpeg" in ops


@pytest.mark.parametrize("what,mutate", [
    ("trims", lambda d: d["overlays"].__setitem__("trims", [{"a": 1.0, "b": 2.0}])),
    ("opener", lambda d: (d["inputs"].append(
        {"id": "cold.mp4", "kind": "video", "sha256": SHA, "url": "https://x/o"}),
        d["overlays"].__setitem__("opener", {"cold": "cold.mp4", "cold_trim": 0.1, "gain": 0.4}))),
])
def test_trims_and_opener_still_refuse_the_whole_stage_before_any_route(what, mutate, monkeypatch):
    _wire(monkeypatch)
    cp = _CP()
    with pytest.raises(NotImplementedError, match=what):
        render.render_spec(_spec(mutate), cp)
    assert _decisions(cp) == []


def test_a_preview_spec_takes_the_legacy_path_with_no_decision_event(monkeypatch):
    _runs, puts = _wire(monkeypatch)
    _forbid(monkeypatch, op, "prepare")
    cmds = []
    real = render.build_command
    monkeypatch.setattr(render, "build_command",
                        lambda *a, **kw: (cmds.append(1), real(*a, **kw))[1])
    cp = _CP()
    render.render_spec(_spec(lambda d: (d.__setitem__("mode", "preview"),
                                        d.__setitem__("overlays", None),
                                        d.__setitem__("outputs", [d["outputs"][0]]))), cp)
    assert _decisions(cp) == []
    assert cmds and puts and cp.results[0]["status"] == "ok"


def test_the_presync_put_keeps_the_legacy_fin_condition_on_the_onepass_route(monkeypatch):
    """guard_sync reads an absent presync object as "the tail never ran" — a finalize-less spec must
    not start PUTting one just because the graph built a reference."""
    _runs, puts = _wire(monkeypatch)
    cp = _CP()
    render.render_spec(_spec(lambda d: d["overlays"].__setitem__("finalize", None)), cp)
    assert [d["phase"] for d in _decisions(cp)] == ["onepass_accepted"]
    assert [(n, u) for n, u in puts] == [("render.mp4", "https://x/master.mp4?PUT")]
    assert cp.results[0]["outputs"] == ["master"]


def test_onepass_defects_reach_the_terminal_result_and_the_degraded_event(monkeypatch):
    defect = {"video_rate": {"declared": "30", "measured": "25"}}
    _runs, puts = _wire(monkeypatch, grid_defect=defect)
    monkeypatch.setattr(finalize, "apply_loudnorm", lambda _f, src, _o: src)
    cp = _CP()
    render.render_spec(_spec(), cp)
    assert [d["phase"] for d in _decisions(cp)] == ["onepass_accepted"]
    degraded = [e for e in cp.events if e.get("phase") == "grid_verify_degraded"]
    assert degraded and degraded[0]["outcome"] == "ok" and "error" not in degraded[0]
    assert cp.results[0]["defects"] == [defect]
    assert puts, "the PUT must still happen despite the mismatch"


def test_every_decision_event_is_a_valid_stream_event(monkeypatch):
    _wire(monkeypatch)
    monkeypatch.setattr(finalize, "apply_loudnorm", lambda _f, src, _o: src)
    from podagent import accents
    monkeypatch.setattr(accents, "detect_flares", lambda _p: [0.3])
    for mutate in (None, _add_film_burn):
        cp = _CP()
        render.render_spec(_spec(mutate), cp)
        for ev in _decisions(cp):
            StreamEvent.model_validate({**ev, "stage": ev["stage"]})


def test_a_cover_spec_falls_back_with_a_counted_named_reason(monkeypatch):
    """The refusal most likely to fire in production: the multi-pass tail WELDS a cover, the graph does not."""
    _runs, puts = _wire(monkeypatch)
    _forbid(monkeypatch, op, "prepare")
    _forbid(monkeypatch, op, "run_encode")
    fin_calls = []
    monkeypatch.setattr(finalize, "finalize",
                        lambda _f, master, *_a, **_kw: (fin_calls.append(master), master)[1])
    from podagent import cover as cover_mod
    monkeypatch.setattr(cover_mod, "render_cover",
                        lambda *_a, **_kw: _kw.get("out") or _a[3])

    def _add_cover(d):
        d["overlays"]["cover"] = {"frame_at": 1.0, "headline": {"lines": ["X"]}}

    cp = _CP()
    render.render_spec(_spec(_add_cover), cp)

    dec = _decisions(cp)
    assert [d["phase"] for d in dec] == ["onepass_refused"]
    assert dec[0]["timings"] == {"refused": ["cover"]}
    assert fin_calls or cp.results, "the legacy multi-pass tail must have run"
    assert cp.results and cp.results[0]["status"] == "ok"
