"""AlignService._emit — GPU-first with a LOUD one-time CPU fallback (no GPU needed for this test)."""
import types

from podagent.infer_align import AlignService


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
