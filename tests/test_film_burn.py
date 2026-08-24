from __future__ import annotations

from types import SimpleNamespace

import pytest

from podagent import accents, render
from podagent.models import RenderSpec


def _burn(at=1.75, burn="burn.mp4", intensity=0.7):
    return SimpleNamespace(kind="film_burn", at=at, intensity=intensity,
                           burn=burn, clicks="clicks.wav")


def test_film_burn_plan_refuses_two_burn_ids():
    with pytest.raises(RuntimeError, match="share one burn input id"):
        accents.film_burn_plan([_burn(1.0, "a.mp4"), _burn(2.0, "b.mp4")])


def test_film_burn_plan_refuses_two_intensities():
    with pytest.raises(RuntimeError, match="share one intensity"):
        accents.film_burn_plan([_burn(1.0, intensity=0.4), _burn(2.0, intensity=0.8)])


def test_film_burn_plan_refuses_more_than_the_boundary_cap():
    with pytest.raises(RuntimeError, match=r"boundary count 9.*RARE accent, 2-3/video"):
        accents.film_burn_plan([_burn(float(i)) for i in range(9)])


def test_film_burn_boundary_cap_refuses_fanout():
    with pytest.raises(RuntimeError, match=r"boundary count 9.*RARE accent, 2-3/video"):
        accents.add_filmburn([], "[v]", 1, list(range(9)), [0.1])


@pytest.mark.parametrize("which", ["probe", "decode"])
def test_detect_flares_refuses_subprocess_failure(monkeypatch, which):
    bad = SimpleNamespace(returncode=1, stdout="", stderr=b"bad media")
    good_probe = SimpleNamespace(returncode=0, stdout="1.0", stderr=b"")
    def fake_run(command, **_kwargs):
        return bad if which == "probe" or command[0] == "ffmpeg" else good_probe

    monkeypatch.setattr(accents.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="film_burn"):
        accents.detect_flares("burn.mp4")


@pytest.mark.parametrize("which, timeout", [
    ("probe", accents.FILM_BURN_PROBE_TIMEOUT_S),
    ("decode", accents.FILM_BURN_DECODE_TIMEOUT_S),
])
def test_detect_flares_timeouts_refuse_loudly(monkeypatch, which, timeout):
    def fake_run(command, **kwargs):
        expected_timeout = accents.FILM_BURN_PROBE_TIMEOUT_S if command[0] == "ffprobe" else timeout
        assert kwargs["timeout"] == expected_timeout
        if which == "probe" or command[0] == "ffmpeg":
            raise accents.subprocess.TimeoutExpired(command, timeout)
        return SimpleNamespace(returncode=0, stdout="1.0", stderr=b"")

    monkeypatch.setattr(accents.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="film_burn .*timed out"):
        accents.detect_flares("burn.mp4")


def test_render_accepts_film_burn_spec_and_reaches_execution_gate(monkeypatch):
    spec = RenderSpec.model_validate({
        "spec_version": 6, "job_id": "j", "slug": "s", "mode": "final",
        "inputs": [
            {"id": "base", "kind": "video", "sha256": "0" * 64, "url": "u"},
            {"id": "burn.mp4", "kind": "video", "sha256": "1" * 64, "url": "u"},
            {"id": "clicks.wav", "kind": "audio", "sha256": "2" * 64, "url": "u"},
        ],
        "timeline": {"fps": 30, "width": 1080, "height": 1920,
                     "segments": [{"src": "base", "in": 0, "out": 1, "speed": 1}]},
        "encode": {"video": "libx264", "preset": "medium", "cq": 23, "pix_fmt": "yuv420p",
                   "audio": "aac", "audio_bitrate": "192k"},
        "outputs": [{"id": "master", "kind": "master", "put_url": "u"}],
        "overlays": {"finalize": {"accents": [{"kind": "film_burn", "at": 0.5,
                                                   "intensity": 1, "burn": "burn.mp4",
                                                   "clicks": "clicks.wav"}]}},
    })

    class CP:
        def send_event(self, *_a, **_k):
            return True

    monkeypatch.setattr(render, "_gpu_available", lambda: (_ for _ in ()).throw(
        AssertionError("execution gate reached")))
    with pytest.raises(AssertionError, match="execution gate reached"):
        render.render_spec(spec, CP())
