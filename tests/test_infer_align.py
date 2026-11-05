"""AlignService — it refuses instead of degrading to CPU, and what the emissions npz actually costs on the wire."""
import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from podagent.infer_align import AlignGpuUnavailable, AlignService, pack_npz


def test_a_cuda_failure_in_align_refuses_and_never_moves_to_cpu():
    """MISC-998 (placement absolute, no silent CPU): a CUDA RuntimeError in the forward refuses with the typed
    error carrying the CUDA text; the model is never moved and the device stays 'cuda'."""
    svc = AlignService.__new__(AlignService)          # bypass __init__ (no real model load)
    svc.device = "cuda"
    svc.torch = types.SimpleNamespace(log_softmax=lambda logits, dim=-1: logits)
    moves = []

    class _Model:
        def to(self, d):
            moves.append(d)
            return self

        def __call__(self, seg):
            raise RuntimeError("CUDA error: the launch timed out and was terminated")

    svc.model = _Model()
    seg = types.SimpleNamespace(to=lambda d: seg)

    with pytest.raises(AlignGpuUnavailable, match="the launch timed out and was terminated"):
        svc._emit(seg)

    assert svc.device == "cuda"
    assert moves == []


def test_align_without_cuda_refuses_at_init(monkeypatch):
    """MISC-998: no CUDA at init → typed refusal naming the cause; no checkpoint is loaded onto any device."""
    loaded = []
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))

    class _Never:
        @staticmethod
        def from_pretrained(*a, **k):
            loaded.append(a)
            raise AssertionError("no model may be loaded without CUDA")

    fake_tf = types.SimpleNamespace(Wav2Vec2ForCTC=_Never, Wav2Vec2Processor=_Never)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    with pytest.raises(AlignGpuUnavailable, match=r"align needs CUDA; torch\.cuda\.is_available\(\) is False"):
        AlignService("m", Path("/nonexistent"))

    assert loaded == []


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


# ───────────────────────── words mode: the pod returns the alignment ─────────────────────────
import hashlib

from pydantic import ValidationError

from podagent import infer_align
from podagent.models import AlignParams

SPANS_SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "contracts" / "align_spans.schema.json").read_text())


def _log_softmax_emis(seed, frames, vocab):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(frames, vocab, generator=g) * 3.0, dim=-1)


def _reference_spans(emis, targets, blank):
    """Textbook CTC Viterbi over the blank-interleaved target sequence + token merge. Pure python; ties are
    impossible on seeded random floats, so the best path is unique."""
    e = emis.tolist()
    T = len(e)
    lab = [blank]
    for t in targets:
        lab += [t, blank]
    S = len(lab)
    NEG = float("-inf")
    score = [[NEG] * S for _ in range(T)]
    back = [[0] * S for _ in range(T)]
    score[0][0] = e[0][lab[0]]
    if S > 1:
        score[0][1] = e[0][lab[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(score[t - 1][s], s)]
            if s >= 1:
                cands.append((score[t - 1][s - 1], s - 1))
            if s >= 2 and lab[s] != blank and lab[s] != lab[s - 2]:
                cands.append((score[t - 1][s - 2], s - 2))
            best, prev = max(cands)
            if best > NEG:
                score[t][s] = best + e[t][lab[s]]
                back[t][s] = prev
    s = S - 1 if score[T - 1][S - 1] >= score[T - 1][S - 2] else S - 2
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t][s]
    toks = [lab[s] for s in path]
    ps = [e[t][toks[t]] for t in range(T)]
    spans, t = [], 0
    while t < T:
        u = t
        while u < T and toks[u] == toks[t]:
            u += 1
        if toks[t] != blank:
            spans.append([t, u, sum(ps[t:u]) / (u - t)])
        t = u
    return spans


def _targets(seed, n, vocab, repeats=0):
    import random

    rnd = random.Random(seed)
    out = [rnd.randrange(1, vocab) for _ in range(n)]
    for _ in range(repeats):
        i = rnd.randrange(0, len(out))
        out.insert(i + 1, out[i])
    return out


@pytest.mark.parametrize("case", [
    dict(seed=1, T=300, V=40, targets=_targets(11, 40, 40)),
    dict(seed=2, T=450, V=64, targets=_targets(12, 120, 64, repeats=15)),
    dict(seed=3, T=200, V=32, targets=[7]),
    dict(seed=4, T=600, V=50, targets=_targets(14, 200, 50, repeats=20)),
    dict(seed=5, T=7, V=32, targets=[1, 1, 2, 3, 3]),                    # fills capacity exactly: 5 + 2 == 7
    dict(seed=6, T=230, V=33, targets=[5, 5, 5, 6] * 20),               # 80 targets + 40 repeats = 120 of 230 frames
], ids=["plain", "repeats", "single", "long", "exact-capacity", "triple-repeat"])
def test_dp_spans_equal_the_textbook_reference(case):
    """The pod's DP function (torchaudio under this pin) must return the textbook Viterbi's spans: frames exact,
    score within 1e-5. Reddens if the blank, the target dtype, the merge or the end-exclusivity drifts."""
    pytest.importorskip("torchaudio")
    emis = _log_softmax_emis(case["seed"], case["T"], case["V"])
    overflow, spans = infer_align.align_spans(emis, case["targets"], blank=0)
    ref = _reference_spans(emis, case["targets"], 0)
    assert overflow is False
    assert len(spans) == len(case["targets"]) == len(ref)
    for got, want in zip(spans, ref):
        assert got[:2] == want[:2]
        assert got[2] == pytest.approx(want[2], abs=1e-5)


def test_capacity_overflow_never_runs_the_dp(monkeypatch):
    """len(targets) + equal-adjacent pairs > frames → overflow, spans null, no forced_align call."""
    torchaudio = pytest.importorskip("torchaudio")

    def boom(*a, **k):
        raise AssertionError("forced_align must not run on overflow")

    monkeypatch.setattr(torchaudio.functional, "forced_align", boom)
    emis = _log_softmax_emis(9, 6, 32)
    # 5 targets + 2 repeats = 7 > 6 frames, though 5 <= 6: the repeat term is what overflows
    assert infer_align.align_spans(emis, [1, 1, 2, 3, 3], blank=0) == (True, None)
    vocab = {"<pad>": 0, "a": 1, "b": 2, "c": 3}
    doc = json.loads(infer_align.pack_spans_json("m", ["a"], vocab, 0, emis, ["aa", "b", "cc"]))
    assert doc["overflow"] is True and doc["words"] is None and doc["frames"] == 6 and doc["chars"] == 5


# fake checkpoint: pad is NOT id 0 (blank must come from the tokenizer, never a literal 0)
FAKE_VOCAB = {"a": 0, "b": 1, "c": 2, "d": 3, "<pad>": 4, "e": 5, "f": 6}
PAD = FAKE_VOCAB["<pad>"]


def _reference_words(emis, words, vocab, blank):
    """Pure-python box semantics (cut_v3 :786-825): char→id per word, forced alignment, first-start/last-end."""
    targets, owner = [], []
    for wi, w in enumerate(words):
        for c in w:
            if c in vocab:
                targets.append(vocab[c])
                owner.append(wi)
    T = emis.shape[0]
    if len(targets) + sum(1 for a, b in zip(targets, targets[1:]) if a == b) > T:
        return None, len(targets)
    spans = _reference_spans(emis, targets, blank) if targets else []
    out = [None] * len(words)
    for (s, e, _), wi in zip(spans, owner):
        out[wi] = [s, e] if out[wi] is None else [out[wi][0], e]
    return out, len(targets)


def _fake_service(torch, per_window, vocab=FAKE_VOCAB, pad=PAD):
    svc = AlignService.__new__(AlignService)
    svc.torch, svc.device, svc.model_id = torch, "cpu", "m"
    it = iter(per_window)
    svc._emit = lambda seg: next(it)
    ordered = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    svc._vocab = lambda: ordered
    svc.processor = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(get_vocab=lambda: dict(vocab), pad_token_id=pad))
    return svc


WORD_CASES = [
    dict(seed=31, T=120, words=["abc", "dd", "ef"]),
    dict(seed=32, T=90, words=["ab", "zzz", "cd", "q"]),            # no-vocab-char words → null
    dict(seed=33, T=100, words=["abb", "bcc", "cd"]),               # b|b and c|c repeat across a word boundary
    dict(seed=34, T=60, words=["a", "a", "a", "b"]),                # one-char words, all-adjacent repeats
    dict(seed=35, T=200, words=["a<", "ed", "fe"]),                 # '<' is no vocab char; '<pad>' never typed
]


@pytest.mark.parametrize("case", WORD_CASES, ids=["plain", "no-vocab-char", "boundary-repeat", "single-chars", "partial"])
def test_pod_word_spans_equal_the_reference(case):
    """pack_spans_json on seeded emissions must give exactly the reference's per-word spans (frames exact)."""
    pytest.importorskip("torchaudio")
    emis = _log_softmax_emis(case["seed"], case["T"], len(FAKE_VOCAB))
    want, chars = _reference_words(emis, case["words"], FAKE_VOCAB, PAD)
    doc = json.loads(infer_align.pack_spans_json("m", ["x"], FAKE_VOCAB, PAD, emis, case["words"]))
    assert want is not None and doc["overflow"] is False
    assert doc["words"] == want and doc["chars"] == chars and doc["frames"] == case["T"]
    assert len(doc["words"]) == len(case["words"])


def test_word_with_no_vocab_char_is_null_and_others_survive():
    pytest.importorskip("torchaudio")
    emis = _log_softmax_emis(41, 50, len(FAKE_VOCAB))
    doc = json.loads(infer_align.pack_spans_json("m", ["x"], FAKE_VOCAB, PAD, emis, ["ab", "zz", "cd"]))
    assert doc["words"][1] is None and doc["words"][0] and doc["words"][2]


def test_all_words_without_vocab_chars_is_not_overflow():
    emis = _log_softmax_emis(42, 10, len(FAKE_VOCAB))
    doc = json.loads(infer_align.pack_spans_json("m", ["x"], FAKE_VOCAB, PAD, emis, ["zz", "yy"]))
    assert doc["overflow"] is False and doc["chars"] == 0 and doc["words"] == [None, None]


def _run_targets(tmp_path, monkeypatch, params_kw, windows=((0.0, 1.0), (1.0, 2.0)), vocab_w=40):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 16000)
    monkeypatch.setattr(infer_align, "download", lambda url, dest: wav)
    put = {}
    monkeypatch.setattr(infer_align, "upload",
                        lambda src, url, ct: put.update(blob=src.read_bytes(), ct=ct))
    per_window = iter([_log_softmax_emis(21, 150, vocab_w), _log_softmax_emis(22, 150, vocab_w)])
    svc = AlignService.__new__(AlignService)
    svc.torch, svc.device, svc.model_id = torch, "cpu", "m"
    svc._emit = lambda seg: next(per_window)
    vocab = [f"t{i}" for i in range(vocab_w)]
    svc._vocab = lambda: vocab
    svc.run(AlignParams(audio_url="u", windows=[list(w) for w in windows], **params_kw), "https://r2/put/x")
    return put, vocab


def _run_words(tmp_path, monkeypatch, words):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(32000, dtype="float32"), 16000)
    monkeypatch.setattr(infer_align, "download", lambda url, dest: wav)
    put = {}
    monkeypatch.setattr(infer_align, "upload",
                        lambda src, url, ct: put.update(blob=src.read_bytes(), ct=ct))
    n = len(FAKE_VOCAB)
    e1, e2 = _log_softmax_emis(21, 150, n), _log_softmax_emis(22, 150, n)
    svc = _fake_service(torch, [e1, e2])
    svc.run(AlignParams(audio_url="u", windows=[[0.0, 1.0], [1.0, 2.0]], words=words), "https://r2/put/x")
    return put, torch.cat([e1, e2]), svc


def test_run_with_words_puts_schema_valid_json(tmp_path, monkeypatch):
    """The words path PUTs application/json (never an npz) valid against align_spans.schema.json, one entry per
    request word, frames = the windows concatenated, vocab pinned by hash, spans equal to the reference."""
    jsonschema = pytest.importorskip("jsonschema")
    pytest.importorskip("torchaudio")
    words = ["abc", "zz", "cdd", "ef"]
    put, emis, svc = _run_words(tmp_path, monkeypatch, words)
    assert put["ct"] == "application/json"
    doc = json.loads(put["blob"])
    jsonschema.Draft202012Validator(SPANS_SCHEMA).validate(doc)
    assert doc["overflow"] is False and doc["frames"] == 300
    assert len(doc["words"]) == len(words) and doc["words"][1] is None
    assert doc["vocab_sha256"] == hashlib.sha256(json.dumps(svc._vocab()).encode()).hexdigest()
    assert doc["words"] == _reference_words(emis, words, FAKE_VOCAB, PAD)[0]


def test_run_with_words_overflow_json_is_schema_valid(tmp_path, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    put, _, _ = _run_words(tmp_path, monkeypatch, ["a" * 301])
    doc = json.loads(put["blob"])
    jsonschema.Draft202012Validator(SPANS_SCHEMA).validate(doc)
    assert doc["overflow"] is True and doc["words"] is None and doc["frames"] == 300 and doc["chars"] == 301


def test_run_without_words_still_puts_the_npz(tmp_path, monkeypatch):
    """Absent words = today's behaviour: octet-stream npz, keep_ids projection applied."""
    np = pytest.importorskip("numpy")
    put, _ = _run_targets(tmp_path, monkeypatch, dict(keep_ids=[2, 5, 7]))
    assert put["ct"] == "application/octet-stream"
    with np.load(io.BytesIO(put["blob"])) as z:
        assert z["emissions_0"].shape == (150, 3)


@pytest.mark.parametrize("kw", [
    dict(words=["a"], keep_ids=[1, 2]),
    dict(words=[]),
    dict(words=["a", ""]),
    dict(targets=[1, 2]),
    dict(blank=0),
], ids=["with-keep-ids", "empty-list", "empty-word", "targets-removed", "blank-removed"])
def test_align_params_refuses_bad_words(kw):
    with pytest.raises(ValidationError):
        AlignParams(audio_url="u", windows=[[0.0, 1.0]], **kw)
