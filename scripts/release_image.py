#!/usr/bin/env python3
"""Fail-closed monty-pod tag → GHCR → engine-pin release transaction.

``release`` is the only mutating mode. It creates an annotated tag, pushes the
commit before the tag, waits for the tag CI under a bounded deadline, verifies
the published linux/amd64 config, then updates the clean engine checkout.
``verify`` performs the same source/origin/registry/engine proof without writes.
``release --dry-run`` validates local inputs and prints the transaction only.

No credential is accepted on the command line or printed. GitHub CLI and the
Actions workflow use their existing credential stores; GHCR verification is an
anonymous pull against the public repository.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
IMAGE_REPO = "ghcr.io/formobr/monty-pod"
GH_REPO = "formobr/monty-pod"
TAG_RE = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ACCEPT = ",".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


class ReleaseError(RuntimeError):
    pass


class _NoCrossHostAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the anonymous GHCR bearer token to a blob CDN."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (redirected is not None
                and urllib.parse.urlsplit(req.full_url).netloc
                != urllib.parse.urlsplit(newurl).netloc):
            redirected.remove_header("Authorization")
        return redirected


class Commands:
    def run(self, args: list[str], *, cwd: Path = REPO, timeout: float = 30,
            check: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(args, cwd=cwd, text=True, capture_output=True,
                                    timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ReleaseError(f"{args[0]} did not complete ({type(exc).__name__})") from exc
        if check and result.returncode:
            raise ReleaseError(f"{args[0]} command failed with rc={result.returncode}")
        return result

    def out(self, args: list[str], *, cwd: Path = REPO, timeout: float = 30) -> str:
        return self.run(args, cwd=cwd, timeout=timeout).stdout.strip()


@dataclass(frozen=True)
class ImageReceipt:
    tag: str
    commit: str
    amd64_digest: str
    config_revision: str
    config_tag: str


class Registry:
    def __init__(self, *, timeout: float = 15):
        self.timeout = timeout

    def _json(self, url: str, *, headers: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
        request = urllib.request.Request(url, headers=headers or {})
        try:
            opener = urllib.request.build_opener(_NoCrossHostAuthRedirect())
            with opener.open(request, timeout=self.timeout) as response:  # noqa: S310 — fixed GHCR
                body = json.loads(response.read().decode("utf-8"))
                return body, dict(response.headers.items())
        except Exception as exc:  # noqa: BLE001 — unknown registry state is a release refusal
            raise ReleaseError(f"GHCR read failed ({type(exc).__name__})") from exc

    def inspect(self, tag: str, commit: str) -> ImageReceipt:
        token_doc, _ = self._json(
            "https://ghcr.io/token?" + urllib.parse.urlencode({
                "scope": "repository:formobr/monty-pod:pull", "service": "ghcr.io",
            }))
        token = token_doc.get("token")
        if not isinstance(token, str) or not token:
            raise ReleaseError("GHCR returned no anonymous pull token")
        headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
        root, root_headers = self._json(
            f"https://ghcr.io/v2/formobr/monty-pod/manifests/{tag}", headers=headers)
        digest, manifest = select_amd64_manifest(root, root_headers, lambda ref: self._json(
            f"https://ghcr.io/v2/formobr/monty-pod/manifests/{ref}", headers=headers))
        config_ref = (manifest.get("config") or {}).get("digest")
        if not isinstance(config_ref, str) or not DIGEST_RE.fullmatch(config_ref):
            raise ReleaseError("linux/amd64 manifest has no valid config digest")
        config, _ = self._json(
            f"https://ghcr.io/v2/formobr/monty-pod/blobs/{config_ref}", headers=headers)
        revision, config_tag = verify_config_identity(config, tag=tag, commit=commit)
        return ImageReceipt(tag, commit, digest, revision, config_tag)


def select_amd64_manifest(root: dict[str, Any], headers: dict[str, str],
                          fetch: Callable[[str], tuple[dict[str, Any], dict[str, str]]]
                          ) -> tuple[str, dict[str, Any]]:
    manifests = root.get("manifests")
    if isinstance(manifests, list):
        matches = [row for row in manifests if isinstance(row, dict)
                   and (row.get("platform") or {}).get("os") == "linux"
                   and (row.get("platform") or {}).get("architecture") == "amd64"]
        if len(matches) != 1:
            raise ReleaseError("registry index has no unique linux/amd64 manifest")
        digest = matches[0].get("digest")
        if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
            raise ReleaseError("linux/amd64 descriptor has no valid digest")
        manifest, _ = fetch(digest)
        return digest, manifest
    digest = next((value for key, value in headers.items()
                   if key.lower() == "docker-content-digest"), "")
    if not DIGEST_RE.fullmatch(digest):
        raise ReleaseError("single-platform manifest returned no immutable digest")
    return digest, root


def verify_config_identity(config: dict[str, Any], *, tag: str, commit: str) -> tuple[str, str]:
    labels = (config.get("config") or {}).get("Labels") or {}
    revision = labels.get("org.opencontainers.image.revision")
    env = (config.get("config") or {}).get("Env") or []
    config_tag = next((str(row).split("=", 1)[1] for row in env
                       if str(row).startswith("POD_IMAGE_TAG=")), "")
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise ReleaseError("selected image config is not linux/amd64")
    if revision != commit:
        raise ReleaseError("image OCI revision does not equal the tagged commit")
    if config_tag != tag:
        raise ReleaseError("image POD_IMAGE_TAG does not equal the annotated tag")
    return str(revision), config_tag


def semver(tag: str) -> tuple[int, int, int]:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ReleaseError("tag must be canonical vX.Y.Z")
    return tuple(int(match.group(i)) for i in range(1, 4))


def require_full_sha(value: str, what: str) -> str:
    if not SHA_RE.fullmatch(value):
        raise ReleaseError(f"{what} is not a full lowercase git SHA")
    return value


def require_clean(repo: Path, commands: Commands, what: str) -> None:
    if commands.out(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo):
        raise ReleaseError(f"{what} checkout is not clean")


def local_tag_commit(tag: str, commands: Commands, *, allow_missing: bool) -> str | None:
    result = commands.run(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], check=False)
    if result.returncode:
        if allow_missing:
            return None
        raise ReleaseError(f"annotated tag {tag} is missing")
    if commands.out(["git", "cat-file", "-t", f"refs/tags/{tag}"]) != "tag":
        raise ReleaseError(f"{tag} is lightweight; an annotated tag is required")
    return require_full_sha(result.stdout.strip(), f"{tag} commit")


def require_new_tag(tag: str, commands: Commands) -> None:
    tags = commands.out(["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"]).splitlines()
    versions = [semver(row) for row in tags if row != tag and TAG_RE.fullmatch(row)]
    if versions and semver(tag) <= max(versions):
        raise ReleaseError(f"new tag {tag} must be greater than the existing release line")


def verify_remote(tag: str, commit: str, commands: Commands) -> None:
    commands.run(["git", "fetch", "--quiet", "origin", "main"], timeout=300)
    ancestor = commands.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], check=False)
    if ancestor.returncode:
        raise ReleaseError("tagged commit is not reachable from origin/main")
    raw = commands.out(["git", "ls-remote", "origin", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"])
    refs = {parts[1]: parts[0] for line in raw.splitlines() if len(parts := line.split()) == 2}
    tag_object = refs.get(f"refs/tags/{tag}")
    peeled = refs.get(f"refs/tags/{tag}^{{}}")
    if not tag_object or tag_object == commit or peeled != commit:
        raise ReleaseError("origin tag is missing, lightweight, or does not peel to the candidate commit")


def wait_for_ci(tag: str, commit: str, commands: Commands, *, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    run_id: str | None = None
    while time.monotonic() < deadline:
        raw = commands.out(["gh", "run", "list", "--repo", GH_REPO, "--workflow", "ci.yml",
                            "--commit", commit, "--event", "push", "--limit", "20",
                            "--json", "databaseId,headBranch,status,conclusion"], timeout=30)
        rows = json.loads(raw or "[]")
        hit = next((row for row in rows if row.get("headBranch") == tag), None)
        if hit:
            run_id = str(hit["databaseId"])
            if hit.get("status") == "completed":
                if hit.get("conclusion") != "success":
                    raise ReleaseError("tag CI completed unsuccessfully")
                return
            break
        time.sleep(5)
    if run_id is None:
        raise ReleaseError("tag CI did not appear before the release deadline")
    while time.monotonic() < deadline:
        row = json.loads(commands.out(["gh", "run", "view", run_id, "--repo", GH_REPO,
                                       "--json", "status,conclusion"], timeout=30))
        if row.get("status") == "completed":
            if row.get("conclusion") != "success":
                raise ReleaseError("tag CI completed unsuccessfully")
            return
        time.sleep(10)
    raise ReleaseError("tag CI exceeded the release deadline")


def engine_pin_values(engine: Path) -> tuple[str, str]:
    text = (engine / "scripts" / "broker" / "pod_image.py").read_text(encoding="utf-8")
    image = re.findall(r'^POD_AGENT_IMAGE\s*=\s*"([^"]+)"$', text, re.MULTILINE)
    digest = re.findall(r'^POD_AGENT_AMD64_DIGEST\s*=\s*"([^"]+)"$', text, re.MULTILINE)
    if len(image) != 1 or len(digest) != 1:
        raise ReleaseError("engine pod image pin declarations are missing or ambiguous")
    return image[0], digest[0]


def verify_engine(engine: Path, receipt: ImageReceipt, commands: Commands) -> None:
    image, digest = engine_pin_values(engine)
    if image != f"{IMAGE_REPO}:{receipt.tag}" or digest != receipt.amd64_digest:
        raise ReleaseError("engine image tag/digest do not equal the verified GHCR receipt")
    submodule_sha = require_full_sha(
        commands.out(["git", "rev-parse", "HEAD"], cwd=engine / "pod-agent"), "engine pod-agent gitlink")
    if submodule_sha != receipt.commit:
        raise ReleaseError("engine pod-agent gitlink does not equal the tagged commit")
    exact_tag = commands.out(
        ["git", "describe", "--tags", "--exact-match", "HEAD"], cwd=engine / "pod-agent")
    if exact_tag != receipt.tag:
        raise ReleaseError("engine pod-agent checkout has no exact matching release tag")
    doc = (engine / "docs" / "gen" / "POD_IMAGE.md").read_text(encoding="utf-8")
    if image not in doc or digest not in doc:
        raise ReleaseError("generated POD_IMAGE doc does not quote the exact tag and amd64 digest")


def replace_once(text: str, pattern: str, replacement: str, what: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ReleaseError(f"engine {what} declaration is missing or ambiguous")
    return updated


def update_engine(engine: Path, receipt: ImageReceipt, commands: Commands) -> None:
    require_clean(engine, commands, "engine")
    pin_file = engine / "scripts" / "broker" / "pod_image.py"
    doc_file = engine / "docs" / "gen" / "POD_IMAGE.md"
    old_pin = pin_file.read_text(encoding="utf-8")
    old_doc = doc_file.read_text(encoding="utf-8")
    old_submodule = commands.out(["git", "rev-parse", "HEAD"], cwd=engine / "pod-agent")
    try:
        commands.run(["git", "fetch", "--quiet", "origin",
                      f"refs/tags/{receipt.tag}:refs/tags/{receipt.tag}"],
                     cwd=engine / "pod-agent", timeout=300)
        commands.run(["git", "checkout", "--quiet", "--detach", receipt.commit],
                     cwd=engine / "pod-agent")
        updated = replace_once(old_pin, r'^POD_AGENT_IMAGE\s*=\s*"[^"]+"$',
                               f'POD_AGENT_IMAGE = "{IMAGE_REPO}:{receipt.tag}"', "image")
        updated = replace_once(updated, r'^POD_AGENT_AMD64_DIGEST\s*=\s*"[^"]+"$',
                               f'POD_AGENT_AMD64_DIGEST = "{receipt.amd64_digest}"', "digest")
        pin_file.write_text(updated, encoding="utf-8")
        python = engine / ".venv" / "bin" / "python"
        if not python.is_file():
            raise ReleaseError("engine .venv Python is missing; generated doc cannot be proven")
        env = dict(os.environ, PYTHONPATH=str(engine / "scripts"))
        result = subprocess.run([str(python), "-m", "gen", "--write", "--only", "doc:pod_image"],
                                cwd=engine, env=env, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise ReleaseError("engine POD_IMAGE generator failed")
        verify_engine(engine, receipt, commands)
    except Exception as exc:
        rollback_errors: list[str] = []
        try:
            pin_file.write_text(old_pin, encoding="utf-8")
        except OSError:
            rollback_errors.append("pin")
        try:
            doc_file.write_text(old_doc, encoding="utf-8")
        except OSError:
            rollback_errors.append("generated doc")
        restored = commands.run(["git", "checkout", "--quiet", "--detach", old_submodule],
                                cwd=engine / "pod-agent", check=False)
        if restored.returncode:
            rollback_errors.append("pod-agent gitlink")
        if rollback_errors:
            joined = ", ".join(rollback_errors)
            raise ReleaseError(f"engine rollback incomplete ({joined}); manual recovery required") from exc
        raise


def plan(tag: str, commit: str, engine: Path) -> None:
    print(f"[release] dry-run tag={tag} commit={commit} image={IMAGE_REPO}:{tag}")
    print("[release] would create annotated tag, push commit then tag, wait for bounded CI")
    print("[release] would verify linux/amd64 digest + OCI revision, then update engine pins")
    print(f"[release] engine={engine}")


def release(tag: str, engine: Path, commands: Commands, registry: Registry, *,
            dry_run: bool, ci_timeout_s: int) -> ImageReceipt | None:
    semver(tag)
    require_clean(REPO, commands, "pod-agent")
    require_clean(engine, commands, "engine")
    commit = require_full_sha(commands.out(["git", "rev-parse", "HEAD"]), "pod-agent HEAD")
    tagged = local_tag_commit(tag, commands, allow_missing=True)
    if tagged is not None and tagged != commit:
        raise ReleaseError(f"existing annotated tag {tag} does not point at HEAD")
    if dry_run:
        if tagged is None:
            require_new_tag(tag, commands)
        plan(tag, commit, engine)
        return None
    commands.run(["git", "fetch", "--quiet", "--tags", "origin"], timeout=300)
    tagged = local_tag_commit(tag, commands, allow_missing=True)
    if tagged is not None and tagged != commit:
        raise ReleaseError(f"existing annotated tag {tag} does not point at HEAD")
    require_new_tag(tag, commands)
    if tagged is None:
        commands.run(["git", "tag", "-a", tag, "-m", f"monty-pod {tag}"])
    commands.run(["git", "push", "origin", "HEAD:main"], timeout=300)
    commands.run(["git", "push", "origin", f"refs/tags/{tag}:refs/tags/{tag}"], timeout=300)
    verify_remote(tag, commit, commands)
    wait_for_ci(tag, commit, commands, timeout_s=ci_timeout_s)
    receipt = registry.inspect(tag, commit)
    update_engine(engine, receipt, commands)
    print(f"[release] verified {IMAGE_REPO}:{tag} amd64={receipt.amd64_digest} revision={commit}")
    print("[release] engine pins updated; review and commit the tag, digest, gitlink, and generated doc")
    return receipt


def verify(tag: str, engine: Path, commands: Commands, registry: Registry) -> ImageReceipt:
    semver(tag)
    require_clean(REPO, commands, "pod-agent")
    commit = require_full_sha(commands.out(["git", "rev-parse", "HEAD"]), "pod-agent HEAD")
    tagged = local_tag_commit(tag, commands, allow_missing=False)
    if tagged != commit:
        raise ReleaseError(f"annotated tag {tag} does not point at clean HEAD")
    verify_remote(tag, commit, commands)
    receipt = registry.inspect(tag, commit)
    verify_engine(engine, receipt, commands)
    print(f"[release] PASS tag={tag} commit={commit} amd64={receipt.amd64_digest}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fail-closed monty-pod image release")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("release", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("tag")
        cmd.add_argument("--engine-dir", type=Path, required=True)
        if name == "release":
            cmd.add_argument("--dry-run", action="store_true")
            cmd.add_argument("--ci-timeout-s", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.command == "release":
            if args.ci_timeout_s < 60:
                raise ReleaseError("--ci-timeout-s must be at least 60")
            release(args.tag, args.engine_dir.resolve(), Commands(), Registry(),
                    dry_run=args.dry_run, ci_timeout_s=args.ci_timeout_s)
        else:
            verify(args.tag, args.engine_dir.resolve(), Commands(), Registry())
        return 0
    except ReleaseError as exc:
        print(f"[release] REFUSE: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — unexpected state is still a bounded refusal
        print(f"[release] REFUSE: unexpected {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
