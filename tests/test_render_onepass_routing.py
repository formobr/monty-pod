"""render.render_spec routing: the one-pass graph is the ONLY final encode core; preview keeps its
single-pass composite via build_command; every non-goal (trims/opener/cover) is a hard refusal that
fires before any subprocess."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podagent import finalize, render, render_onepass as op
from podagent.models import RenderSpec

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


def _onepass_ops(cp: _CP) -> list[str]:
    return [e.get("op") for e in cp.events]


def test_an_accepted_final_spec_runs_the_onepass_core(monkeypatch):
    _runs, puts = _wire(monkeypatch)
    # build_command is preview's own composite; the multi-pass builders it also used to forbid are
    # gone (test_render_onepass.py::test_the_multipass_builders_no_longer_exist pins the deletion).
    _forbid(monkeypatch, render, "build_command")
    ln_calls = []
    monkeypatch.setattr(finalize, "apply_loudnorm",
                        lambda _fin, src, out: (ln_calls.append(src), Path(out).write_bytes(b"v"),
                                                out)[2])
    cp = _CP()
    render.render_spec(_spec(), cp)

    assert "onepass_preflight" not in _onepass_ops(cp)
    # the delivered master is the loudnorm output over the ONE encode's master
    assert ln_calls and ln_calls[0].name == "render.mp4"
    assert [(n, u) for n, u in puts] == [("fin_ln.mp4", "https://x/master.mp4?PUT"),
                                         ("render.presync.mp4", "https://x/presync.mp4?PUT")]
    assert cp.results and cp.results[0]["status"] == "ok"
    assert cp.results[0]["outputs"] == ["master", "presync"]
    assert "defects" not in cp.results[0]
    ops = [e.get("op") for e in cp.events if e.get("phase", "").endswith("_started")]
    assert "ffmpeg" in ops and "finalize" in ops and "upload" in ops


def test_a_film_burn_spec_runs_the_onepass_core(monkeypatch):
    """A burn spec routes through the one-pass core, flare-decodes in prepare, and never touches the
    deleted multi-pass finalize builders."""
    _runs, puts = _wire(monkeypatch)
    from podagent import accents
    monkeypatch.setattr(accents, "detect_flares", lambda _p: [0.3])
    monkeypatch.setattr(finalize, "apply_loudnorm",
                        lambda _fin, _src, out: (Path(out).write_bytes(b"v"), out)[1])
    cp = _CP()
    render.render_spec(_spec(_add_film_burn), cp)

    assert "onepass_preflight" not in _onepass_ops(cp)
    assert [(n, u) for n, u in puts] == [("fin_ln.mp4", "https://x/master.mp4?PUT"),
                                         ("render.presync.mp4", "https://x/presync.mp4?PUT")]
    assert cp.results and cp.results[0]["status"] == "ok"
    ops = [e.get("op") for e in cp.events if e.get("phase", "").endswith("_started")]
    assert "prepare" in ops and "ffmpeg" in ops


@pytest.mark.parametrize("what,mutate", [
    ("trims", lambda d: d["overlays"].__setitem__("trims", [{"a": 1.0, "b": 2.0}])),
    ("opener", lambda d: (d["inputs"].append(
        {"id": "cold.mp4", "kind": "video", "sha256": SHA, "url": "https://x/o"}),
        d["overlays"].__setitem__("opener", {"cold": "cold.mp4", "cold_trim": 0.1, "gain": 0.4}))),
    ("cover", lambda d: d["overlays"].__setitem__(
        "cover", {"frame_at": 1.0, "headline": {"lines": ["X"]}})),
    ("cover-output", lambda d: d["outputs"].append(
        {"id": "cover", "kind": "cover", "put_url": "https://x/cover.png?PUT"})),
    ("cover-both", lambda d: (
        d["overlays"].__setitem__("cover", {"frame_at": 1.0, "headline": {"lines": ["X"]}}),
        d["outputs"].append({"id": "cover", "kind": "cover", "put_url": "https://x/cover.png?PUT"}))),
])
def test_non_goals_refuse_before_any_route(what, mutate, monkeypatch):
    _wire(monkeypatch)
    cp = _CP()
    with pytest.raises(NotImplementedError, match="cover" if what.startswith("cover") else what):
        render.render_spec(_spec(mutate), cp)
    assert cp.events == []


def test_a_preview_spec_keeps_its_single_pass_composite(monkeypatch):
    _runs, puts = _wire(monkeypatch)
    _forbid(monkeypatch, op, "prepare")
    cmds = []
    real = render.build_command
    monkeypatch.setattr(render, "build_command",
                        lambda *a, **kw: (cmds.append(a), real(*a, **kw))[1])
    cp = _CP()
    render.render_spec(_spec(lambda d: (d.__setitem__("mode", "preview"),
                                        d.__setitem__("overlays", None),
                                        d.__setitem__("outputs", [d["outputs"][0]]))), cp)
    assert cmds and puts and cp.results[0]["status"] == "ok"
    # spec.encode (cq23) is honoured on preview — _FINAL_CPU's cq14 is the (deleted) delivery rung's
    # own number, never build_command's, so its presence would mean the wrong clause ran.
    argv = _runs.calls[0]
    assert argv[argv.index("-crf") + 1] == "23"
    assert not any(argv[i:i + len(finalize._FINAL_CPU)] == finalize._FINAL_CPU for i in range(len(argv)))


def test_a_preview_spec_declaring_a_cover_output_refuses_at_the_put(monkeypatch):
    """preflight only guards mode=final; without the upload-loop guard a preview spec declaring an
    outputs[kind=cover] would PUT the mp4 to the cover URL instead of refusing."""
    _runs, _puts = _wire(monkeypatch)
    cp = _CP()
    with pytest.raises(RuntimeError, match="kind=cover has no producer"):
        render.render_spec(_spec(lambda d: (d.__setitem__("mode", "preview"),
                                            d.__setitem__("overlays", None),
                                            d.__setitem__("outputs", [
                                                d["outputs"][0],
                                                {"id": "cover", "kind": "cover",
                                                 "put_url": "https://x/cover.png?PUT"}]))), cp)


def test_the_presync_put_keeps_the_fin_condition_on_the_onepass_route(monkeypatch):
    """guard_sync reads an absent presync object as "the tail never ran" — a finalize-less spec must
    not start PUTting one just because the graph built a reference."""
    _runs, puts = _wire(monkeypatch)
    cp = _CP()
    render.render_spec(_spec(lambda d: d["overlays"].__setitem__("finalize", None)), cp)
    assert [(n, u) for n, u in puts] == [("render.mp4", "https://x/master.mp4?PUT")]
    assert cp.results[0]["outputs"] == ["master"]


def test_onepass_defects_reach_the_terminal_result_and_the_degraded_event(monkeypatch):
    defect = {"video_rate": {"declared": "30", "measured": "25"}}
    _runs, puts = _wire(monkeypatch, grid_defect=defect)
    monkeypatch.setattr(finalize, "apply_loudnorm", lambda _f, src, _o: src)
    cp = _CP()
    render.render_spec(_spec(), cp)
    degraded = [e for e in cp.events if e.get("phase") == "grid_verify_degraded"]
    assert degraded and degraded[0]["outcome"] == "ok" and "error" not in degraded[0]
    assert cp.results[0]["defects"] == [defect]
    assert puts, "the PUT must still happen despite the mismatch"
