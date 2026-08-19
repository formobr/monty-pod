"""render.render_spec — the grid_verify event fires after the master already exists; its send must
never be able to cost an already-paid-for encode."""
from __future__ import annotations

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


def test_a_failed_grid_verify_send_does_not_block_the_put(monkeypatch, tmp_path, capsys):
    uploaded = []
    results = []

    class _CP:
        def send_event(self, payload, *, wait=False):
            if payload.get("op") == "grid_verify":
                raise RuntimeError("stream write failed")
            return True

        def send_result(self, payload, *, wait=True):
            results.append(payload)
            return True

    monkeypatch.setattr(render, "download", lambda _u, dest: (dest.write_bytes(b"m"), dest)[1])
    monkeypatch.setattr(render, "upload", lambda src, url, ctype: uploaded.append((src, url, ctype)))
    monkeypatch.setattr(render, "_gpu_available", lambda: False)
    monkeypatch.setattr(render.subprocess, "run", lambda *_a, **_kw: _Done())
    monkeypatch.setattr(render._finalize, "grid_verdict",
                        lambda *_a, **_kw: {"video_rate": {"declared": "30", "measured": "25"}})

    render.render_spec(_spec(), _CP())  # must NOT raise

    assert len(uploaded) == 1
    assert results and results[0]["status"] == "ok"
    assert "grid_verify: event send failed, swallowed" in capsys.readouterr().err
