"""MISC-12: a manifest fetched by digest is trusted only when its body hashes to that digest."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_image_digest", ROOT / "scripts" / "release_image.py")
assert SPEC and SPEC.loader
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)

MANIFEST = json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "b" * 64}})


def _digest(body: str) -> str:
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


class _StubRegistry(release.Registry):
    def __init__(self, body: str):
        super().__init__(timeout=1)
        self.body = body

    def _read(self, url, *, headers=None):
        return self.body, {}


def test_a_manifest_body_that_does_not_hash_to_its_digest_is_refused():
    requested = _digest(MANIFEST)
    tampered = MANIFEST.replace("b" * 64, "c" * 64)
    with pytest.raises(release.ReleaseError, match="registry content digest mismatch"):
        release.verify_content_digest(tampered, requested)
    with pytest.raises(release.ReleaseError, match="registry content digest mismatch"):
        _StubRegistry(tampered)._by_digest("https://ghcr.io/x", requested, headers={})

    # The child-manifest fetch in `inspect` goes through the same check.
    index = {"manifests": [{"digest": requested, "platform": {"os": "linux", "architecture": "amd64"}}]}
    registry = _StubRegistry(tampered)
    with pytest.raises(release.ReleaseError, match="registry content digest mismatch"):
        release.select_amd64_manifest(index, {}, lambda ref: registry._by_digest(
            f"https://ghcr.io/v2/formobr/monty-pod/manifests/{ref}", ref, headers={}))


def test_a_manifest_body_that_hashes_to_its_digest_passes():
    requested = _digest(MANIFEST)
    assert release.verify_content_digest(MANIFEST, requested) == json.loads(MANIFEST)
    index = {"manifests": [{"digest": requested, "platform": {"os": "linux", "architecture": "amd64"}}]}
    registry = _StubRegistry(MANIFEST)
    digest, manifest = release.select_amd64_manifest(index, {}, lambda ref: registry._by_digest(
        f"https://ghcr.io/v2/formobr/monty-pod/manifests/{ref}", ref, headers={}))
    assert digest == requested and manifest == json.loads(MANIFEST)


def test_a_malformed_requested_digest_is_refused():
    with pytest.raises(release.ReleaseError, match="not a valid sha256 digest"):
        release.verify_content_digest(MANIFEST, "sha256:xyz")
