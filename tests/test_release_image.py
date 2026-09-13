from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_image.py"
SPEC = importlib.util.spec_from_file_location("release_image", SCRIPT)
assert SPEC and SPEC.loader
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)

OLD_SHA = "a" * 40
NEW_SHA = "b" * 40
DIGEST = "sha256:" + "d" * 64


def _receipt(sha: str = NEW_SHA, digest: str = DIGEST):
    return release.ImageReceipt(sha, sha, digest, sha, sha)


def _engine(tmp_path: Path, *, sha: str = OLD_SHA, digest: str | None = None) -> Path:
    engine = tmp_path / "engine"
    (engine / "scripts/broker").mkdir(parents=True)
    (engine / "docs/gen").mkdir(parents=True)
    (engine / "pod-agent").mkdir()
    (engine / ".venv/bin").mkdir(parents=True)
    (engine / ".venv/bin/python").touch()
    image = f"{release.IMAGE_REPO}:{sha}"
    pin_digest = digest or "sha256:" + "c" * 64
    (engine / "scripts/broker/pod_image.py").write_text(
        f'POD_AGENT_IMAGE = "{image}"\nPOD_AGENT_AMD64_DIGEST = "{pin_digest}"\n',
        encoding="utf-8",
    )
    (engine / "docs/gen/POD_IMAGE.md").write_text(
        f"{image}\n{pin_digest}\n", encoding="utf-8")
    return engine


class FakeCommands:
    def __init__(self, *, engine: Path, source_sha: str = NEW_SHA, reachable: bool = True):
        self.engine = engine
        self.source_sha = source_sha
        self.current_engine_sha = OLD_SHA
        self.reachable = reachable
        self.calls: list[tuple[str, ...]] = []

    def out(self, args, *, cwd=release.REPO, timeout=30):
        self.calls.append(tuple(args))
        if args[:2] == ["git", "status"]:
            return ""
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return self.current_engine_sha if cwd == self.engine / "pod-agent" else self.source_sha
        raise AssertionError((args, cwd, timeout))

    def run(self, args, *, cwd=release.REPO, timeout=30, check=True):
        self.calls.append(tuple(args))
        if args[:2] == ["git", "fetch"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return subprocess.CompletedProcess(args, 0 if self.reachable else 1, "", "")
        if args[:2] == ["git", "checkout"]:
            self.current_engine_sha = args[-1]
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError((args, cwd, timeout, check))


class GoodRegistry:
    def __init__(self, *, before=None):
        self.before = before

    def inspect(self, tag, commit):
        assert (tag, commit) == (NEW_SHA, NEW_SHA)
        if self.before is not None:
            self.before()
        return _receipt()


def test_image_identity_requires_a_full_lowercase_sha():
    assert release.require_full_sha(NEW_SHA, "image SHA") == NEW_SHA
    for bad in ("b" * 39, "B" * 40, "v0.18.6", "latest", "sha256:" + "b" * 64):
        with pytest.raises(release.ReleaseError, match="full lowercase"):
            release.require_full_sha(bad, "image SHA")


def test_index_requires_one_linux_amd64_manifest():
    digest = "sha256:" + "a" * 64
    manifest = {"config": {"digest": "sha256:" + "b" * 64}}
    root = {"manifests": [{"digest": digest,
                            "platform": {"os": "linux", "architecture": "amd64"}}]}
    assert release.select_amd64_manifest(
        root, {}, lambda ref: (manifest, {}) if ref == digest else ({}, {}),
    ) == (digest, manifest)

    for rows in ([], root["manifests"] * 2,
                 [{"digest": digest, "platform": {"os": "linux", "architecture": "arm64"}}]):
        with pytest.raises(release.ReleaseError, match="unique linux/amd64"):
            release.select_amd64_manifest({"manifests": rows}, {}, lambda _ref: ({}, {}))


def test_single_platform_manifest_requires_registry_digest_header():
    doc = {"config": {"digest": "sha256:" + "b" * 64}}
    with pytest.raises(release.ReleaseError, match="immutable digest"):
        release.select_amd64_manifest(doc, {}, lambda _ref: ({}, {}))
    digest, same = release.select_amd64_manifest(
        doc, {"docker-content-digest": "sha256:" + "a" * 64}, lambda _ref: ({}, {}))
    assert digest.endswith("a" * 64) and same is doc


def test_image_config_requires_sha_revision_tag_and_amd64():
    config = {"os": "linux", "architecture": "amd64", "config": {
        "Labels": {"org.opencontainers.image.revision": NEW_SHA},
        "Env": [f"POD_IMAGE_TAG={NEW_SHA}"],
    }}
    assert release.verify_config_identity(config, tag=NEW_SHA, commit=NEW_SHA) == (NEW_SHA, NEW_SHA)
    with pytest.raises(release.ReleaseError, match="OCI revision"):
        release.verify_config_identity(config, tag=NEW_SHA, commit="c" * 40)
    with pytest.raises(release.ReleaseError, match="POD_IMAGE_TAG"):
        release.verify_config_identity(config, tag="c" * 40, commit=NEW_SHA)
    config["architecture"] = "arm64"
    with pytest.raises(release.ReleaseError, match="linux/amd64"):
        release.verify_config_identity(config, tag=NEW_SHA, commit=NEW_SHA)


def test_source_sha_must_be_clean_head_reachable_from_main(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    commands = FakeCommands(engine=engine)
    monkeypatch.setattr(release, "REPO", tmp_path / "pod")
    assert release.verify_source(NEW_SHA, commands) == NEW_SHA
    assert ("git", "fetch", "--quiet", "origin", "main") in commands.calls
    assert ("git", "merge-base", "--is-ancestor", NEW_SHA, "origin/main") in commands.calls

    unreachable = FakeCommands(engine=engine, reachable=False)
    with pytest.raises(release.ReleaseError, match="not reachable"):
        release.verify_source(NEW_SHA, unreachable)


def test_engine_verifier_requires_sha_digest_gitlink_and_generated_doc(tmp_path):
    engine = _engine(tmp_path, sha=NEW_SHA, digest=DIGEST)
    commands = FakeCommands(engine=engine)
    commands.current_engine_sha = NEW_SHA
    release.verify_engine(engine, _receipt(), commands)
    (engine / "docs/gen/POD_IMAGE.md").write_text(
        f"{release.IMAGE_REPO}:{NEW_SHA}\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="generated"):
        release.verify_engine(engine, _receipt(), commands)


def test_verify_is_read_only_and_requires_existing_engine_equality(monkeypatch, tmp_path):
    engine = _engine(tmp_path, sha=NEW_SHA, digest=DIGEST)
    commands = FakeCommands(engine=engine)
    commands.current_engine_sha = NEW_SHA
    monkeypatch.setattr(release, "REPO", tmp_path / "pod")
    before = {path: path.read_bytes() for path in (
        engine / "scripts/broker/pod_image.py", engine / "docs/gen/POD_IMAGE.md")}
    assert release.verify(NEW_SHA, engine, commands, GoodRegistry()) == _receipt()
    assert {path: path.read_bytes() for path in before} == before
    assert not any(call[1:2] in (("push",), ("tag",)) or call[:2] == ("gh", "run")
                   for call in commands.calls)


def test_pin_proves_artifact_before_replacing_an_old_engine_pin(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    commands = FakeCommands(engine=engine)
    monkeypatch.setattr(release, "REPO", tmp_path / "pod")

    def still_old():
        image, digest = release.engine_pin_values(engine)
        assert image.endswith(OLD_SHA) and digest.endswith("c" * 64)
        assert commands.current_engine_sha == OLD_SHA

    def generate(argv, **_kwargs):
        assert argv[-2:] == ["--only", "doc:pod_image"]
        image, digest = release.engine_pin_values(engine)
        (engine / "docs/gen/POD_IMAGE.md").write_text(f"{image}\n{digest}\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(release.subprocess, "run", generate)
    assert release.pin(NEW_SHA, engine, commands, GoodRegistry(before=still_old)) == _receipt()
    assert release.engine_pin_values(engine) == (f"{release.IMAGE_REPO}:{NEW_SHA}", DIGEST)
    assert commands.current_engine_sha == NEW_SHA
    assert NEW_SHA in (engine / "docs/gen/POD_IMAGE.md").read_text(encoding="utf-8")
    assert not any(call[1:2] in (("push",), ("tag",)) or call[:2] == ("gh", "run")
                   for call in commands.calls)


@pytest.mark.parametrize("failure", ["missing", "wrong-platform", "ambiguous-platform"])
def test_registry_refusal_leaves_every_pin_path_unchanged(monkeypatch, tmp_path, failure):
    engine = _engine(tmp_path)
    commands = FakeCommands(engine=engine)
    monkeypatch.setattr(release, "REPO", tmp_path / "pod")
    paths = (engine / "scripts/broker/pod_image.py", engine / "docs/gen/POD_IMAGE.md")
    before = {path: path.read_bytes() for path in paths}

    class RefusingRegistry:
        def inspect(self, _tag, _commit):
            if failure == "missing":
                raise release.ReleaseError("GHCR image is not published")
            if failure == "wrong-platform":
                release.verify_config_identity(
                    {"os": "linux", "architecture": "arm64", "config": {}},
                    tag=NEW_SHA, commit=NEW_SHA)
            release.select_amd64_manifest({"manifests": []}, {}, lambda _ref: ({}, {}))
            raise AssertionError("unreachable")

    with pytest.raises(release.ReleaseError):
        release.pin(NEW_SHA, engine, commands, RefusingRegistry())
    assert {path: path.read_bytes() for path in paths} == before
    assert commands.current_engine_sha == OLD_SHA


def test_update_rolls_back_files_and_submodule_on_generator_failure(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    commands = FakeCommands(engine=engine)
    pin_file = engine / "scripts/broker/pod_image.py"
    doc_file = engine / "docs/gen/POD_IMAGE.md"
    before = (pin_file.read_bytes(), doc_file.read_bytes())
    monkeypatch.setattr(release.subprocess, "run", lambda argv, **_kwargs:
                        subprocess.CompletedProcess(argv, 1, "", ""))
    with pytest.raises(release.ReleaseError, match="generator failed"):
        release.update_engine(engine, _receipt(), commands)
    assert (pin_file.read_bytes(), doc_file.read_bytes()) == before
    assert commands.current_engine_sha == OLD_SHA


def test_workflow_publishes_only_the_successful_main_push_sha():
    repo = SCRIPT.parents[1]
    workflow = (repo / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    dockerfile = (repo / "Dockerfile").read_text(encoding="utf-8")
    assert "needs: test" in workflow
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main'" in workflow
    assert "IMAGE_TAG=${{ github.sha }}" in workflow
    assert "IMAGE_REVISION=${{ github.sha }}" in workflow
    assert "ghcr.io/${{ github.repository }}:${{ github.sha }}" in workflow
    assert "github.ref_name" not in workflow
    assert "ghcr.io/${{ github.repository }}:latest" not in workflow
    assert 'tags: ["v*"]' not in workflow
    assert "packages: write" in workflow and "password: ${{ secrets.GITHUB_TOKEN }}" in workflow
    assert "push: true" in workflow
    assert "ARG IMAGE_REVISION=unknown" in dockerfile
    assert "org.opencontainers.image.revision=${IMAGE_REVISION}" in dockerfile


def test_cli_and_module_have_no_release_build_or_ci_wait_path():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'for name in ("pin", "verify")' in source
    assert "wait_for_ci" not in source
    assert "git\", \"push" not in source
    assert "git\", \"tag" not in source
    assert "gh\", \"run" not in source
