"""AlignService — the CPU fallback, and what the emissions npz actually costs on the wire."""
import io
import json
import types
import zipfile

import pytest

from podagent.infer_align import AlignService, pack_npz


def test_emit_falls_back_to_cpu_loudly_on_cuda_runtime_error(capsys):
    """A CUDA RuntimeError (e.g. an arch this torch build lacks kernels for) degrades to CPU ONCE, warns loudly,
    and still returns an emission. Reddens if the fallback silently swallows or crashes."""
    svc = AlignService.__new__(AlignService)          # bypass __init__ (no real model load)
    svc.device = "cuda"
    svc.torch = types.SimpleNamespace(log_softmax=lambda logits, dim=-1: logits)

    class _Model:
        calls = 0

        def to(self, d):
            return self

        def __call__(self, seg):
            _Model.calls += 1
            if _Model.calls == 1:
                raise RuntimeError("CUDA error: no kernel image is available for execution")
            return types.SimpleNamespace(logits=[[0.1, 0.9]])

    svc.model = _Model()
    seg = types.SimpleNamespace(to=lambda d: seg)      # .to(device) → itself
    out = svc._emit(seg)

    assert svc.device == "cpu"                          # degraded once
    assert out == [0.1, 0.9]                            # still produced an emission
    assert "CPU fallback" in capsys.readouterr().err    # LOUD, not silent


def test_emit_reraises_if_cpu_also_fails():
    """If we are ALREADY on CPU and the forward still errors, don't loop — re-raise (a real fault)."""
    svc = AlignService.__new__(AlignService)
    svc.device = "cpu"
    svc.torch = types.SimpleNamespace(log_softmax=lambda logits, dim=-1: logits)

    class _Model:
        def to(self, d):
            return self

        def __call__(self, seg):
            raise RuntimeError("genuinely broken")

    svc.model = _Model()
    seg = types.SimpleNamespace(to=lambda d: seg)
    try:
        svc._emit(seg)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


VOCAB_W = 9913          # voidful multi56's union vocab — the width that makes this payload 316.9 MB
KEEP = sorted({*range(1000, 1051), 9910})


def _arrays(width):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    return {f"emissions_{i}": rng.standard_normal((200, width)).astype("float32") - 8.0 for i in range(2)}


def test_align_npz_is_stored_not_deflated():
    """NEGATIVE: log-softmax float32 barely compresses (316.9 -> 274.8 MB measured) and zlib charges ~12 s of
    pod CPU for it. Reverting pack_npz to np.savez_compressed reddens this."""
    blob = pack_npz(_arrays(VOCAB_W), {"model": "m", "sr": 16000, "frame_stride_s": 0.02, "vocab": ["x"]})
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert {i.compress_type for i in z.infolist()} == {zipfile.ZIP_STORED}


def test_run_ships_only_the_requested_columns(tmp_path, monkeypatch):
    """NEGATIVE: the projection is the whole point of keep_ids — forced_align reads its targets and blank, so
    the other 99.5% of the union vocab is wire and CPU for nothing. Drop the index_select and this reddens."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    from podagent import infer_align
    from podagent.models import AlignParams

    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    monkeypatch.setattr(infer_align, "download", lambda url, dest: wav)
    put = {}
    monkeypatch.setattr(infer_align, "upload", lambda src, url, ct: put.update(blob=src.read_bytes()))

    svc = AlignService.__new__(AlignService)
    svc.torch, svc.device, svc.model_id = torch, "cpu", "m"
    svc._emit = lambda seg: torch.zeros(200, VOCAB_W)
    svc._vocab = lambda: ["x"] * VOCAB_W

    svc.run(AlignParams(audio_url="u", windows=[[0.0, 1.0]], keep_ids=KEEP), "https://r2/put/x")

    with np.load(io.BytesIO(put["blob"])) as z:
        assert z["emissions_0"].shape == (200, len(KEEP))
        assert json.loads(z["meta_json"].tobytes().decode())["keep_ids"] == KEEP


def test_run_without_keep_ids_still_ships_the_full_vocab(tmp_path, monkeypatch):
    """The absent field must mean exactly the old behaviour — an engine that never learned about the
    projection cannot be silently served a narrower matrix."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    from podagent import infer_align
    from podagent.models import AlignParams

    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    monkeypatch.setattr(infer_align, "download", lambda url, dest: wav)
    put = {}
    monkeypatch.setattr(infer_align, "upload", lambda src, url, ct: put.update(blob=src.read_bytes()))

    svc = AlignService.__new__(AlignService)
    svc.torch, svc.device, svc.model_id = torch, "cpu", "m"
    svc._emit = lambda seg: torch.zeros(200, 40)
    svc._vocab = lambda: ["x"] * 40

    svc.run(AlignParams(audio_url="u", windows=[[0.0, 1.0]]), "https://r2/put/x")

    with np.load(io.BytesIO(put["blob"])) as z:
        assert z["emissions_0"].shape == (200, 40)
        assert "keep_ids" not in json.loads(z["meta_json"].tobytes().decode())
