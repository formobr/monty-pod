from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_image.py"
SPEC = importlib.util.spec_from_file_location("release_image", SCRIPT)
assert SPEC and SPEC.loader
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)


def test_tag_must_be_canonical_semver():
    assert release.semver("v0.18.6") == (0, 18, 6)
    for bad in ("0.18.6", "v0.18", "v01.2.3", "v0.18.6-rc1", "latest"):
        with pytest.raises(release.ReleaseError):
            release.semver(bad)


def test_index_requires_one_linux_amd64_manifest():
    digest = "sha256:" + "a" * 64
    manifest = {"config": {"digest": "sha256:" + "b" * 64}}
    root = {"manifests": [{"digest": digest, "platform": {"os": "linux", "architecture": "amd64"}}]}
    got_digest, got_manifest = release.select_amd64_manifest(
        root, {}, lambda ref: (manifest, {}) if ref == digest else ({}, {}))
    assert got_digest == digest and got_manifest == manifest

    with pytest.raises(release.ReleaseError):
        release.select_amd64_manifest({"manifests": []}, {}, lambda _ref: ({}, {}))
    with pytest.raises(release.ReleaseError):
        release.select_amd64_manifest(
            {"manifests": root["manifests"] * 2}, {}, lambda _ref: ({}, {}))


def test_single_platform_manifest_requires_registry_digest_header():
    doc = {"config": {"digest": "sha256:" + "b" * 64}}
    with pytest.raises(release.ReleaseError):
        release.select_amd64_manifest(doc, {}, lambda _ref: ({}, {}))
    digest, same = release.select_amd64_manifest(
        doc, {"docker-content-digest": "sha256:" + "a" * 64}, lambda _ref: ({}, {}))
    assert digest.endswith("a" * 64) and same is doc


def test_image_config_must_name_exact_tagged_commit():
    tag, commit = "v0.18.6", "c" * 40
    config = {"os": "linux", "architecture": "amd64", "config": {
        "Labels": {"org.opencontainers.image.revision": commit},
        "Env": [f"POD_IMAGE_TAG={tag}"],
    }}
    assert release.verify_config_identity(config, tag=tag, commit=commit) == (commit, tag)
    with pytest.raises(release.ReleaseError, match="OCI revision"):
        release.verify_config_identity(config, tag=tag, commit="d" * 40)
    with pytest.raises(release.ReleaseError, match="POD_IMAGE_TAG"):
        release.verify_config_identity(config, tag="v0.18.7", commit=commit)
    config["os"] = "windows"
    with pytest.raises(release.ReleaseError, match="linux/amd64"):
        release.verify_config_identity(config, tag=tag, commit=commit)


def test_local_tag_must_be_an_annotated_tag_object():
    commit = "c" * 40

    class Lightweight:
        def run(self, args, *, cwd=release.REPO, timeout=30, check=True):
            return type("Done", (), {"returncode": 0, "stdout": commit})()

        def out(self, args, *, cwd=release.REPO, timeout=30):
            return "commit"

    with pytest.raises(release.ReleaseError, match="lightweight"):
        release.local_tag_commit("v0.18.6", Lightweight(), allow_missing=False)


def test_new_tag_must_advance_remote_release_line():
    class Tags:
        def out(self, args, *, cwd=release.REPO, timeout=30):
            return "v0.18.5\nv0.18.7"

    with pytest.raises(release.ReleaseError, match="greater"):
        release.require_new_tag("v0.18.6", Tags())


def test_remote_tag_must_be_annotated_and_peel_to_reachable_commit():
    commit = "c" * 40

    class FakeCommands:
        def run(self, args, *, cwd=release.REPO, timeout=30, check=True):
            return type("Done", (), {"returncode": 0})()

        def out(self, args, *, cwd=release.REPO, timeout=30):
            return f"{'a' * 40}\trefs/tags/v0.18.6\n{commit}\trefs/tags/v0.18.6^{{}}"

    release.verify_remote("v0.18.6", commit, FakeCommands())

    class Lightweight(FakeCommands):
        def out(self, args, *, cwd=release.REPO, timeout=30):
            return f"{commit}\trefs/tags/v0.18.6"

    with pytest.raises(release.ReleaseError, match="lightweight"):
        release.verify_remote("v0.18.6", commit, Lightweight())

    class NotReachable(FakeCommands):
        def run(self, args, *, cwd=release.REPO, timeout=30, check=True):
            rc = 1 if args[:3] == ["git", "merge-base", "--is-ancestor"] else 0
            return type("Done", (), {"returncode": rc})()

    with pytest.raises(release.ReleaseError, match="not reachable"):
        release.verify_remote("v0.18.6", commit, NotReachable())


def test_replace_once_refuses_missing_or_duplicate_engine_pins():
    with pytest.raises(release.ReleaseError):
        release.replace_once("", r"^PIN=.*$", "PIN=new", "pin")
    with pytest.raises(release.ReleaseError):
        release.replace_once("PIN=old\nPIN=older\n", r"^PIN=.*$", "PIN=new", "pin")
    assert release.replace_once("PIN=old\n", r"^PIN=.*$", "PIN=new", "pin") == "PIN=new\n"


def test_engine_verifier_requires_tag_digest_gitlink_and_generated_doc(tmp_path):
    engine = tmp_path
    (engine / "scripts/broker").mkdir(parents=True)
    (engine / "docs/gen").mkdir(parents=True)
    (engine / "pod-agent").mkdir()
    digest = "sha256:" + "d" * 64
    commit = "c" * 40
    image = f"{release.IMAGE_REPO}:v0.18.6"
    (engine / "scripts/broker/pod_image.py").write_text(
        f'POD_AGENT_IMAGE = "{image}"\nPOD_AGENT_AMD64_DIGEST = "{digest}"\n', encoding="utf-8")
    (engine / "docs/gen/POD_IMAGE.md").write_text(f"{image}\n{digest}\n", encoding="utf-8")

    class FakeCommands:
        def out(self, args, *, cwd, timeout=30):
            assert cwd == engine / "pod-agent"
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return commit
            if args[:3] == ["git", "describe", "--tags"]:
                return "v0.18.6"
            raise AssertionError(args)

    receipt = release.ImageReceipt("v0.18.6", commit, digest, commit, "v0.18.6")
    release.verify_engine(engine, receipt, FakeCommands())
    (engine / "docs/gen/POD_IMAGE.md").write_text(image, encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="generated"):
        release.verify_engine(engine, receipt, FakeCommands())


def test_engine_update_rolls_back_all_files_and_submodule_on_generator_failure(
        monkeypatch, tmp_path):
    engine = tmp_path
    (engine / "scripts/broker").mkdir(parents=True)
    (engine / "docs/gen").mkdir(parents=True)
    (engine / "pod-agent").mkdir()
    python = engine / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    old_commit = "a" * 40
    new_commit = "b" * 40
    old_pin = ('POD_AGENT_IMAGE = "ghcr.io/formobr/monty-pod:v0.18.5"\n'
               'POD_AGENT_AMD64_DIGEST = "sha256:' + "c" * 64 + '"\n')
    old_doc = "old generated receipt\n"
    pin_file = engine / "scripts/broker/pod_image.py"
    doc_file = engine / "docs/gen/POD_IMAGE.md"
    pin_file.write_text(old_pin, encoding="utf-8")
    doc_file.write_text(old_doc, encoding="utf-8")

    class FakeCommands:
        current = old_commit

        def out(self, args, *, cwd=release.REPO, timeout=30):
            if args[:2] == ["git", "status"]:
                return ""
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return self.current
            raise AssertionError(args)

        def run(self, args, *, cwd=release.REPO, timeout=30, check=True):
            if args[:2] == ["git", "fetch"]:
                return type("Done", (), {"returncode": 0})()
            if args[:2] == ["git", "checkout"]:
                self.current = args[-1]
                return type("Done", (), {"returncode": 0})()
            raise AssertionError(args)

    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs:
                        type("Done", (), {"returncode": 1})())
    commands = FakeCommands()
    receipt = release.ImageReceipt("v0.18.6", new_commit, "sha256:" + "d" * 64,
                                   new_commit, "v0.18.6")
    with pytest.raises(release.ReleaseError, match="generator failed"):
        release.update_engine(engine, receipt, commands)
    assert pin_file.read_text(encoding="utf-8") == old_pin
    assert doc_file.read_text(encoding="utf-8") == old_doc
    assert commands.current == old_commit


def test_workflow_and_dockerfile_carry_tag_and_revision_build_identity():
    repo = SCRIPT.parents[1]
    workflow = (repo / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    dockerfile = (repo / "Dockerfile").read_text(encoding="utf-8")
    assert "IMAGE_TAG=${{ github.ref_name }}" in workflow
    assert "IMAGE_REVISION=${{ github.sha }}" in workflow
    assert "ARG IMAGE_REVISION=unknown" in dockerfile
    assert "org.opencontainers.image.revision=${IMAGE_REVISION}" in dockerfile


def test_dry_run_is_the_only_release_path_that_skips_mutations(monkeypatch, tmp_path):
    commit = "c" * 40

    class FakeCommands:
        def __init__(self):
            self.mutations = []

        def out(self, args, *, cwd=release.REPO, timeout=30):
            if args[:2] == ["git", "status"]:
                return ""
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return commit
            if args[:3] == ["git", "tag", "--list"]:
                return "v0.18.5"
            if args[:3] == ["git", "cat-file", "-t"]:
                return "tag"
            raise AssertionError(args)

        def run(self, args, *, cwd=release.REPO, timeout=30, check=True):
            if args[:4] == ["git", "rev-parse", "--verify", "refs/tags/v0.18.6^{commit}"]:
                return type("Done", (), {"returncode": 1, "stdout": ""})()
            self.mutations.append(args)
            raise AssertionError(args)

    commands = FakeCommands()
    monkeypatch.setattr(release, "REPO", tmp_path / "pod")
    (tmp_path / "engine").mkdir()
    assert release.release("v0.18.6", tmp_path / "engine", commands, object(),
                           dry_run=True, ci_timeout_s=1800) is None
    assert commands.mutations == []


def test_existing_exact_annotated_tag_resumes_after_interrupted_release(monkeypatch, tmp_path):
    commit = "c" * 40
    digest = "sha256:" + "d" * 64
    calls: list[list[str]] = []

    class FakeCommands:
        def out(self, args, *, cwd=release.REPO, timeout=30):
            if args[:2] == ["git", "status"]:
                return ""
            if args[:3] == ["git", "rev-parse", "HEAD"]:
                return commit
            if args[:3] == ["git", "cat-file", "-t"]:
                return "tag"
            if args[:3] == ["git", "tag", "--list"]:
                return "v0.18.5\nv0.18.6"
            raise AssertionError(args)

        def run(self, args, *, cwd=release.REPO, timeout=30, check=True):
            calls.append(args)
            if args[:3] == ["git", "rev-parse", "--verify"]:
                return type("Done", (), {"returncode": 0, "stdout": commit})()
            return type("Done", (), {"returncode": 0, "stdout": ""})()

    receipt = release.ImageReceipt("v0.18.6", commit, digest, commit, "v0.18.6")

    class FakeRegistry:
        def inspect(self, tag, tagged_commit):
            assert (tag, tagged_commit) == ("v0.18.6", commit)
            return receipt

    monkeypatch.setattr(release, "REPO", tmp_path / "pod")
    monkeypatch.setattr(release, "verify_remote", lambda *args: None)
    monkeypatch.setattr(release, "wait_for_ci", lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "update_engine", lambda *args: None)
    got = release.release("v0.18.6", tmp_path / "engine", FakeCommands(), FakeRegistry(),
                          dry_run=False, ci_timeout_s=1800)
    assert got == receipt
    assert not any(args[:3] == ["git", "tag", "-a"] for args in calls)
    assert ["git", "push", "origin", "HEAD:main"] in calls
