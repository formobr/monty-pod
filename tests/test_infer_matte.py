"""person_matte on the pod, offline: the real decode → forward → encode → PUT pipeline with a FAKE forward.

What the fake cannot prove (the RVM numbers, the GPU) is the pod smoke's job; what it can prove is the
contract: frame-for-frame outputs, a lossless alpha, an alpha-carrying WebM, the receipt shape, and every
refusal arriving as a named error instead of a short matte."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from podagent import infer_matte as matte
from podagent import main as agent_main
from podagent.models import PersonMatteParams, PersonMattePayload

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
_RESULT_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "infer_result.schema.json").read_text()))
_PAYLOAD_SCHEMA = Draft202012Validator(json.loads((CONTRACTS / "person_matte.schema.json").read_text()))

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")

W, H, N = 64, 96, 12


class FakeNet:
    """A deterministic 'matte': frame index ramps the alpha, so frame order is checkable after the encode."""

    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self.calls = 0

    def alpha(self, rgb: np.ndarray, downsample_ratio: float) -> np.ndarray:
        assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
        a = np.zeros(rgb.shape[:2], np.uint8)
        a[:, : W // 2] = (self.calls * 20) % 256
        a[:, W // 2:] = 255
        self.calls += 1
        return a


@pytest.fixture()
def clip(tmp_path: Path) -> Path:
    out = tmp_path / "plate.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=30",
                    "-frames:v", str(N), "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def _decode_gray(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W)


def _svc(net: Any = None) -> matte.MatteService:
    return matte.MatteService(matte.RVM_MODEL_ID, net=net or FakeNet())


def test_both_outputs_frame_for_frame(tmp_path: Path, clip: Path) -> None:
    alpha, cutout, receipt = tmp_path / "o" / "alpha.mkv", tmp_path / "o" / "cut.webm", tmp_path / "o" / "receipt"
    net = FakeNet()
    run = _svc(net).run(PersonMatteParams(video_url=clip.as_uri(), downsample_ratio=1.0,
                                          alpha_put_url=alpha.as_uri(), cutout_put_url=cutout.as_uri()),
                        receipt.as_uri())
    assert net.resets == 1, "the recurrent state starts empty for every video"
    payload = json.loads(receipt.read_text())
    assert not sorted(_PAYLOAD_SCHEMA.iter_errors(payload), key=str)
    got = PersonMattePayload.model_validate(payload)
    assert (got.width, got.height, got.frames, got.fps) == (W, H, N, "30/1")
    assert got.alpha is not None and got.cutout is not None
    assert got.alpha.bytes == alpha.stat().st_size and got.alpha.sha256 == matte.file_sha256(alpha)
    assert run.infer_s > 0 and run.timings["frames"] == N

    # the FINAL alpha is lossless and in frame order
    a = _decode_gray(alpha)
    assert a.shape == (N, H, W)
    for i in range(N):
        assert int(a[i, 0, 0]) == (i * 20) % 256 and int(a[i, 0, -1]) == 255

    # the PREVIEW cut-out carries an alpha plane (libvpx-vp9 decoder; the native one drops alpha)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name:stream_tags=alpha_mode",
                            "-of", "json", str(cutout)], capture_output=True, text=True, check=True).stdout
    st = json.loads(probe)["streams"][0]
    assert st["codec_name"] == "vp9" and st["tags"]["alpha_mode"] == "1"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-c:v", "libvpx-vp9", "-i", str(cutout), "-vf", "alphaextract",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE, check=True).stdout
    ca = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    assert ca.shape[0] == N
    assert abs(int(ca[-1, H // 2, W - 4]) - 255) <= 8 and abs(int(ca[5, H // 2, 4]) - 100) <= 12


def test_one_output_only(tmp_path: Path, clip: Path) -> None:
    alpha, receipt = tmp_path / "a.mkv", tmp_path / "receipt"
    _svc().run(PersonMatteParams(video_url=clip.as_uri(), downsample_ratio=0.5, alpha_put_url=alpha.as_uri()),
               receipt.as_uri())
    payload = json.loads(receipt.read_text())
    assert "cutout" not in payload and payload["alpha"]["bytes"] > 0 and payload["downsample_ratio"] == 0.5


def test_wrong_model_is_refused_by_name() -> None:
    with pytest.raises(matte.MatteRefused, match="rvm_mobilenetv3_fp16"):
        matte.MatteService("birefnet_lite_matting", net=FakeNet())


def test_baked_bytes_are_pinned(tmp_path: Path) -> None:
    fake = tmp_path / "rvm.torchscript"
    with pytest.raises(matte.MatteRefused, match="not on this pod"):
        matte.verify_model_file(fake)
    fake.write_bytes(b"not the model")
    with pytest.raises(matte.MatteRefused, match="pinned"):
        matte.verify_model_file(fake)


def test_a_wrong_shaped_forward_puts_nothing(tmp_path: Path, clip: Path) -> None:
    class Bad(FakeNet):
        def alpha(self, rgb: np.ndarray, downsample_ratio: float) -> np.ndarray:
            return np.zeros((H // 2, W), np.uint8)

    alpha, receipt = tmp_path / "a.mkv", tmp_path / "receipt"
    with pytest.raises(matte.MatteRefused, match="forward returned"):
        _svc(Bad()).run(PersonMatteParams(video_url=clip.as_uri(), downsample_ratio=1.0,
                                          alpha_put_url=alpha.as_uri()), receipt.as_uri())
    assert not alpha.exists() and not receipt.exists()


def test_undecodable_input_is_refused(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 1024)
    with pytest.raises(matte.MatteRefused, match="no readable video stream"):
        _svc().run(PersonMatteParams(video_url=junk.as_uri(), downsample_ratio=1.0,
                                     alpha_put_url=(tmp_path / "a.mkv").as_uri()), (tmp_path / "receipt").as_uri())


class _CP:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def note(self, _ev: dict[str, Any]) -> None:
        pass

    def report_infer_result(self, payload: dict[str, Any]) -> bool:
        self.results.append(payload)
        return True


def _request(tmp_path: Path, clip: Path) -> dict[str, Any]:
    req = json.loads((CONTRACTS / "examples" / "infer_request.person_matte.json").read_text())
    req["put_url"] = (tmp_path / "receipt").as_uri()
    req["person_matte"].update(video_url=clip.as_uri(), cutout_put_url=(tmp_path / "c.webm").as_uri())
    return req


def test_dispatch_reports_ok_with_the_cached_service(tmp_path: Path, clip: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent_main, "_lifecycle", lambda *a, **k: None)
    cache = {matte.RVM_MODEL_ID: _svc()}
    cp = _CP()
    agent_main._run_infer(_request(tmp_path, clip), cp, {}, {}, {}, Path("/opt/models/yunet.onnx"), True,
                          corr_id="work/s/r", session_id="s", matte_cache=cache)
    (result,) = cp.results
    assert result["status"] == "ok" and result["kind"] == "person_matte", result
    assert not sorted(_RESULT_SCHEMA.iter_errors({k: v for k, v in result.items() if k != "session_id"}), key=str)
    assert (tmp_path / "c.webm").stat().st_size > 0


def test_dispatch_refuses_on_a_dry_pod(tmp_path: Path, clip: Path, monkeypatch) -> None:
    from podagent.ops import dry

    monkeypatch.setattr(agent_main, "_lifecycle", lambda *a, **k: None)
    monkeypatch.setattr(dry, "armed", lambda: True)
    cp = _CP()
    agent_main._run_infer(_request(tmp_path, clip), cp, {}, {}, {}, Path("/opt/models/yunet.onnx"), True,
                          corr_id="work/s/r", session_id="s", matte_cache={matte.RVM_MODEL_ID: _svc()})
    (result,) = cp.results
    assert result["status"] == "error" and "contour-dry" in result["error"]
    assert not (tmp_path / "c.webm").exists()
