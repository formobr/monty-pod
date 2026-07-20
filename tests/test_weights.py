"""Weights-as-an-input: the fetch, the content-addressed cache, and the guarantees around them.

Every test here is paired with the failure it exists to prevent. The cache ones matter most: the venv
tarball lane shipped a name-keyed cache once and silently served a stale environment for days, which is
exactly the class of bug a content hash makes unrepresentable.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest
from pydantic import ValidationError

from podagent import weights
from podagent.models import InferRequest, WeightsRef


def _tar_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, body in files.items():
            data = body.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk: int):
        for i in range(0, len(self._payload), chunk):
            yield self._payload[i:i + chunk]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def served(monkeypatch, tmp_path):
    """Serve a tar over a stubbed requests.get and count how many times it is fetched."""
    monkeypatch.setenv("WEIGHTS_CACHE", str(tmp_path / "cache"))
    calls: list[str] = []

    def _serve(payload: bytes):
        def _get(url, **kw):
            calls.append(url)
            return _FakeResp(payload)
        monkeypatch.setattr(weights.requests, "get", _get)
        return calls

    return _serve


def _ref(payload: bytes, **over) -> WeightsRef:
    d = {"url": "https://r2.example/models/x.tar",
         "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    d.update(over)
    return WeightsRef(**d)


# --- the fetch ---------------------------------------------------------------------------------

def test_a_hub_shaped_tar_resolves_to_the_snapshot_dir(served):
    """The seeded tars (scripts/render/upload_models.py) are HF-hub shaped, so the returned path must be the
    snapshot dir from_pretrained can actually read — not the tar root."""
    hub = "models--acme--m"
    payload = _tar_bytes({f"{hub}/refs/main": "abc123",
                          f"{hub}/snapshots/abc123/config.json": "{}",
                          f"{hub}/snapshots/abc123/model.safetensors": "W"})
    served(payload)

    got = weights.ensure(_ref(payload), "acme/m")

    assert got.name == "abc123"
    assert (got / "config.json").is_file()


def test_a_tar_with_no_config_json_fails_loud(served):
    """NEGATIVE: a mis-packed tar must fail at the fetch, not as an opaque from_pretrained error."""
    payload = _tar_bytes({"readme.txt": "nope"})
    served(payload)
    with pytest.raises(ValueError, match="no config.json"):
        weights.ensure(_ref(payload), "acme/m")


def test_ensure_fetches_extracts_and_returns_the_model_dir(served):
    payload = _tar_bytes({"config.json": json.dumps({"model_type": "siglip"}), "model.safetensors": "W"})
    served(payload)

    got = weights.ensure(_ref(payload), "acme/model")

    assert (got / "config.json").is_file()
    assert json.loads((got / "config.json").read_text())["model_type"] == "siglip"


def test_a_corrupt_download_raises_and_leaves_nothing_cached(served):
    """NEGATIVE: without the digest check, truncated bytes would extract and surface as mystery-bad
    inference hours later instead of as a failed job."""
    payload = _tar_bytes({"config.json": "{}"})
    served(payload)
    lying = _ref(payload, sha256="c" * 64)      # digest that does not describe these bytes

    with pytest.raises(ValueError, match="sha256 mismatch"):
        weights.ensure(lying, "acme/model")

    assert not (weights.cache_root() / lying.sha256).exists()


def test_a_size_mismatch_raises(served):
    payload = _tar_bytes({"config.json": "{}"})
    served(payload)
    with pytest.raises(ValueError, match="size mismatch"):
        weights.ensure(_ref(payload, size=len(payload) + 1), "acme/model")


def test_a_tar_that_escapes_its_directory_is_refused(served, tmp_path):
    """NEGATIVE: a plain extractall would let a member write outside the cache dir."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b"pwned"
        info = tarfile.TarInfo("../../escaped.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    payload = buf.getvalue()
    served(payload)

    with pytest.raises(ValueError, match="escapes its directory"):
        weights.ensure(_ref(payload), "acme/model")


# --- the cache ---------------------------------------------------------------------------------

def test_a_warm_pod_fetches_the_same_checkpoint_exactly_once(served):
    """The whole point of the cache: a second job of the same kind pays nothing."""
    payload = _tar_bytes({"config.json": "{}"})
    calls = served(payload)
    ref = _ref(payload)

    first = weights.ensure(ref, "acme/model")
    second = weights.ensure(ref, "acme/model")

    assert first == second
    assert len(calls) == 1, f"warm pod re-fetched the weights: {len(calls)} downloads"


def test_a_changed_checkpoint_is_a_cache_MISS_even_under_the_same_model_name(served, monkeypatch):
    """NEGATIVE: the bug this design exists to prevent — a name-keyed cache serving stale weights after
    the checkpoint is revised. Same model name, different bytes, must not reuse the first entry."""
    old = _tar_bytes({"config.json": "{}", "model.safetensors": "OLD"})
    new = _tar_bytes({"config.json": "{}", "model.safetensors": "NEW"})

    monkeypatch.setattr(weights.requests, "get", lambda url, **kw: _FakeResp(old))
    old_dir = weights.ensure(_ref(old), "acme/model")
    monkeypatch.setattr(weights.requests, "get", lambda url, **kw: _FakeResp(new))
    new_dir = weights.ensure(_ref(new), "acme/model")

    assert old_dir != new_dir, "a revised checkpoint reused the stale cache entry"
    assert (new_dir / "model.safetensors").read_text() == "NEW"


def test_an_interrupted_fetch_does_not_publish_a_usable_cache_entry(served, monkeypatch):
    """NEGATIVE: without the staging dir, a killed fetch would leave a half-extracted directory under the
    content hash that the next job would happily load as a model."""
    payload = _tar_bytes({"config.json": "{}"})
    ref = _ref(payload)

    class _Boom(_FakeResp):
        def iter_content(self, chunk: int):
            yield payload[: len(payload) // 2]
            raise OSError("connection reset")

    monkeypatch.setattr(weights.requests, "get", lambda url, **kw: _Boom(payload))
    with pytest.raises(OSError):
        weights.ensure(ref, "acme/model")

    assert not (weights.cache_root() / ref.sha256).exists()

    monkeypatch.setattr(weights.requests, "get", lambda url, **kw: _FakeResp(payload))
    assert (weights.ensure(ref, "acme/model") / "config.json").is_file()


def test_a_cache_dir_without_the_completion_sentinel_is_not_a_hit(served):
    """NEGATIVE: the sentinel is the second lock on the same door — a directory that exists but was never
    finished (an older layout, a manual copy, a half-restored backup) must be re-fetched, not loaded."""
    payload = _tar_bytes({"config.json": "{}", "model.safetensors": "REAL"})
    calls = served(payload)
    ref = _ref(payload)

    # a plausible-looking but unfinished entry sitting exactly where the cache would put one
    squatter = weights.cache_root() / ref.sha256
    squatter.mkdir(parents=True)
    (squatter / "config.json").write_text("{}")
    (squatter / "model.safetensors").write_text("TRUNCATED")

    got = weights.ensure(ref, "acme/model")

    assert len(calls) == 1, "an unfinished cache directory was served as a hit"
    assert (got / "model.safetensors").read_text() == "REAL"


# --- the contract ------------------------------------------------------------------------------

_BASE = {"infer_version": 3, "job_id": "j", "kind": "align", "model": "m", "put_url": "p",
         "align": {"audio_url": "u", "windows": [[0.0, 1.0]]}}
_W = {"url": "https://r2.example/x.tar", "sha256": "a" * 64}


def test_align_without_weights_is_rejected():
    """NEGATIVE: nothing heavy is baked any more, so a missing block is an origin bug that must fail at
    the seam — not 40 seconds later inside from_pretrained."""
    with pytest.raises(ValidationError, match="requires a weights block"):
        InferRequest.model_validate(_BASE)


def test_align_with_weights_is_accepted():
    """Positive twin: proves the test above measures the weights rule, not a broken fixture."""
    assert InferRequest.model_validate({**_BASE, "weights": _W}).weights is not None


def test_face_probe_must_not_carry_weights():
    req = {"infer_version": 3, "job_id": "j", "kind": "face_probe", "model": "m", "put_url": "p",
           "face_probe": {"video_url": "u", "shots": [[0.0, 1.0]], "stride": 5, "frame_diff": False},
           "weights": _W}
    with pytest.raises(ValidationError, match="must not carry weights"):
        InferRequest.model_validate(req)


def test_a_stale_v2_request_is_rejected_outright():
    """NEGATIVE: the version pin is what stops a v2 origin from half-working against a v3 pod. A v2
    request names no weights, so without this it would reach the loader and fail obscurely."""
    with pytest.raises(ValidationError, match="infer_version"):
        InferRequest.model_validate({**_BASE, "infer_version": 2, "weights": _W})


def test_a_non_sha256_cache_key_is_rejected():
    """NEGATIVE: a free-form key would let two different checkpoints collide on one cache directory."""
    with pytest.raises(ValidationError):
        InferRequest.model_validate({**_BASE, "weights": {**_W, "sha256": "not-a-digest"}})
