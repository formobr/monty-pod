"""render.render_spec — past a successful encode, event sends are best-effort (a broken stream must
never cost the paid master); before it they stay fatal (nothing spent yet, so fail fast is cheap)."""
from __future__ import annotations

import pytest

from podagent import render
from podagent.models import RenderSpec

_BASE_INPUT = {"id": "base", "kind": "video", "sha256": "0" * 64, "url": "u"}
_ENCODE = {"video": "libx264", "preset": "medium", "cq": 23, "pix_fmt": "yuv420p",
           "audio": "aac", "audio_bitrate": "192k"}
_TIMELINE = {"fps": 30, "width": 1080, "height": 1920,
             "segments": [{"src": "base", "in": 0.0, "out": 5.0, "speed": 1.0}]}


def _spec() -> RenderSpec:
    return RenderSpec.model_validate({
        "spec_version": 6, "job_id": "j", "slug": "s", "mode": "preview",
        "inputs": [_BASE_INPUT], "timeline": _TIMELINE, "encode": _ENCODE,
        "outputs": [{"id": "master", "kind": "master", "put_url": "p"}],
    })


class _Done:
    returncode = 0
    stderr = ""
    stdout = ""


def _wire_happy_path(monkeypatch, uploaded: list, grid_defect: dict | None = None):
    monkeypatch.setattr(render, "download", lambda _u, dest: (dest.write_bytes(b"m"), dest)[1])
    monkeypatch.setattr(render, "upload", lambda src, url, ctype: uploaded.append((src, url, ctype)))
    monkeypatch.setattr(render, "_gpu_available", lambda: False)
    monkeypatch.setattr(render.subprocess, "run", lambda *_a, **_kw: _Done())
    monkeypatch.setattr(render._finalize, "grid_verdict", lambda *_a, **_kw: grid_defect)


def test_a_control_plane_that_fails_from_the_encode_onward_still_delivers_and_reports(
    monkeypatch, tmp_path, capsys,
):
    """The property that matters: a control plane whose send_event raises for every event FROM THE
    SUCCESSFUL ENCODE ONWARD must not stop the PUT or the terminal result. ffmpeg_finished itself is
    the flag flip, not a raise site — it is the pre-encode path's own (still-fatal) exit event."""
    uploaded: list = []
    results: list = []

    class _CP:
        def __init__(self) -> None:
            self.encode_done = False

        def send_event(self, payload, *, wait=False):
            if payload.get("phase") == "ffmpeg_finished":
                self.encode_done = True
                return True
            if self.encode_done:
                raise RuntimeError(f"stream write failed ({payload.get('phase')})")
            return True

        def send_result(self, payload, *, wait=True):
            results.append(payload)
            return True

    _wire_happy_path(monkeypatch, uploaded)

    render.render_spec(_spec(), _CP())  # must NOT raise

    assert len(uploaded) == 1
    assert results and results[0]["status"] == "ok"
    err = capsys.readouterr().err
    assert "upload_started: event send failed, swallowed" in err
    assert "upload_finished: event send failed, swallowed" in err
    assert "work_finished: event send failed, swallowed" in err


def test_a_control_plane_that_fails_pre_encode_aborts_before_ffmpeg_runs(monkeypatch, tmp_path):
    """The other half of the boundary: a stream failure BEFORE the encode stays fatal and never reaches
    ffmpeg — spending ~20 GPU-min on a render nobody can be told about is the same loss, only pricier."""
    ffmpeg_calls: list = []

    class _CP:
        def send_event(self, payload, *, wait=False):
            raise RuntimeError("stream unreachable")

        def send_result(self, payload, *, wait=True):
            raise AssertionError("send_result must not be reached: encode never ran")

    monkeypatch.setattr(render, "download", lambda _u, dest: (dest.write_bytes(b"m"), dest)[1])
    monkeypatch.setattr(render, "_gpu_available", lambda: False)
    monkeypatch.setattr(render.subprocess, "run", lambda *_a, **_kw: (ffmpeg_calls.append(1), _Done())[1])

    with pytest.raises(RuntimeError):
        render.render_spec(_spec(), _CP())

    assert ffmpeg_calls == []


def test_a_failed_grid_verify_send_does_not_block_the_put(monkeypatch, tmp_path, capsys):
    uploaded: list = []
    results: list = []

    class _CP:
        def send_event(self, payload, *, wait=False):
            if payload.get("op") == "grid_verify":
                raise RuntimeError("stream write failed")
            return True

        def send_result(self, payload, *, wait=True):
            results.append(payload)
            return True

    _wire_happy_path(
        monkeypatch, uploaded,
        grid_defect={"video_rate": {"declared": "30", "measured": "25"}},
    )

    render.render_spec(_spec(), _CP())  # must NOT raise

    assert len(uploaded) == 1
    assert results and results[0]["status"] == "ok"
    assert "grid_verify_degraded: event send failed, swallowed" in capsys.readouterr().err
