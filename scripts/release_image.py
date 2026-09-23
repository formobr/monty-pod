#!/usr/bin/env python3
"""Fail-closed monty-pod commit SHA -> GHCR -> engine-pin transaction.

Pod-agent CI publishes an image tagged with the full source commit SHA before
this tool runs. ``verify`` proves source/origin, the existing linux/amd64 image,
its embedded identity, and the engine pins without writes. ``pin`` performs the
same source/artifact proof independently of old engine pins, updates the clean
engine checkout, then proves the resulting pins.

Neither mode creates or pushes git tags or commits, dispatches CI, builds an
image, or waits for publication. GHCR verification is an anonymous bounded read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
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
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ACCEPT = ",".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))
REGISTRY_JSON_MAX_BYTES = 8 * 1024 * 1024
REGISTRY_WORKER_CLEANUP_S = 0.5
# Internal-only CLI verb, never in --help: `Registry._json` re-invokes THIS FILE as a fresh `python`
# process for the read (see the class docstring for why). Not a subparser, so a stray extra positional
# argument can never route here from a human's own `pin`/`verify` call.
REGISTRY_WORKER_ARG = "_registry-read"


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


def _registry_json_worker(url: str, headers: dict[str, str], socket_timeout: float) -> dict[str, Any]:
    """One bounded read. NEVER RAISES — every ordinary failure (HTTP error, decode error, a response over
    the bounded size) comes back as `{"ok": False, "error": <type name only, never response/token bytes>}`
    so the caller (in-process in a test, or the isolated `_registry-read` child in production) can tell an
    ANSWERED-but-red read from a read that never answered at all."""
    try:
        request = urllib.request.Request(url, headers=headers)
        opener = urllib.request.build_opener(_NoCrossHostAuthRedirect())
        with opener.open(request, timeout=socket_timeout) as response:  # noqa: S310 — fixed GHCR
            raw = response.read(REGISTRY_JSON_MAX_BYTES + 1)
            if len(raw) > REGISTRY_JSON_MAX_BYTES:
                raise ValueError("registry JSON exceeds the bounded response size")
            return {"ok": True, "body": raw.decode("utf-8"), "headers": dict(response.headers.items())}
    except BaseException as exc:  # noqa: BLE001 — see docstring
        return {"ok": False, "error": type(exc).__name__}


def _run_registry_worker() -> int:
    """The `_registry-read` child's own entry point: `{"url", "headers", "timeout"}` on stdin, the
    worker's own JSON payload on stdout, exit 0. Reading stdin (never argv) keeps the GHCR bearer header
    out of `ps`/`/proc/<pid>/cmdline` — argv is world-readable on this box, stdin is not. A non-zero or
    signal exit past this point is the INTERPRETER dying (OOM-kill, segfault) — every ORDINARY registry
    failure is `{"ok": false, ...}` on stdout with exit 0, never a process death."""
    request = json.loads(sys.stdin.read())
    payload = _registry_json_worker(str(request["url"]), dict(request.get("headers") or {}),
                                    float(request["timeout"]))
    sys.stdout.write(json.dumps(payload))
    return 0


def _exit_detail(returncode: int | None) -> str:
    """The child's own exit, named: `subprocess.CompletedProcess.returncode` is the process's own return
    code when it exited cleanly, and NEGATIVE the signal number that killed it on POSIX (Python's own
    convention, matching `os.wait`). TRK-79: this is the whole reason a dead-with-no-result child used to
    be reported as one bare sentence — the exit that would have named OOM-kill vs a plain non-zero return
    was read and then thrown away."""
    if returncode is None:
        return "returncode=None (never observed one)"
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"killed by {name} (exitcode={returncode})"
    return f"exitcode={returncode}"


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
    """Bounded GHCR reader; imported by the engine's local image boot proof."""

    def __init__(self, *, timeout: float = 15):
        self.timeout = timeout

    def _json(self, url: str, *, headers: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
        # TRK-79: `verify`/`pin` runs this from the engine's own image-boot proof, which the landing/ship
        # threaded proof pool (`proof_units.py`'s `run_units`, `ThreadPoolExecutor`) calls from a WORKER
        # THREAD. `multiprocessing`'s fork context only clones the calling thread, so a lock any sibling
        # thread held is frozen mid-state in the child, and the child can die with no result for reasons
        # that have nothing to do with the read itself (measured: the same read passed 71s earlier on
        # v0.20.70) — and `spawn`/`forkserver` need the target to be importable BY NAME from a fresh
        # interpreter, which this module is not when `_boot_probe_registry_module` (release_all.py) or
        # this test file load it via `importlib.util.spec_from_file_location`. A genuine `subprocess` —
        # re-invoking THIS FILE by its own resolved path, which is correct however this module was loaded
        # — has neither problem: no inherited threads/locks, and no by-name import at all.
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), REGISTRY_WORKER_ARG],
                input=json.dumps({"url": url, "headers": dict(headers or {}), "timeout": self.timeout}),
                capture_output=True, text=True, timeout=self.timeout + REGISTRY_WORKER_CLEANUP_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReleaseError(f"GHCR read exceeded {self.timeout:g}s wall-clock deadline") from exc
        except OSError as exc:
            raise ReleaseError(f"GHCR read worker could not start ({type(exc).__name__}: {exc})") from exc
        try:
            if result.returncode != 0:
                raise ReleaseError(
                    f"GHCR read worker exited without a result ({_exit_detail(result.returncode)})")
            try:
                payload = json.loads(result.stdout)
            except (ValueError, TypeError) as exc:
                raise ReleaseError(
                    f"GHCR read worker printed a malformed result ({type(exc).__name__})") from exc
            if not payload.get("ok"):
                raise ReleaseError(f"GHCR read failed ({payload.get('error', 'unknown')})")
            body = json.loads(payload["body"])
            response_headers = payload["headers"]
            if not isinstance(body, dict) or not isinstance(response_headers, dict):
                raise ReleaseError("GHCR returned a malformed JSON response")
            return body, {str(key): str(value) for key, value in response_headers.items()}
        except ReleaseError:
            raise
        except Exception as exc:  # noqa: BLE001 — unknown registry state is a release refusal
            raise ReleaseError(f"GHCR read failed ({type(exc).__name__})") from exc

    def inspect(self, tag: str, commit: str) -> ImageReceipt:
        """Return the exact linux/amd64 receipt after embedded SHA identity agrees."""
        image_sha = require_full_sha(tag, "image tag")
        commit = require_full_sha(commit, "image source revision")
        if image_sha != commit:
            raise ReleaseError("image tag does not equal the source commit SHA")
        token_doc, _ = self._json(
            "https://ghcr.io/token?" + urllib.parse.urlencode({
                "scope": "repository:formobr/monty-pod:pull", "service": "ghcr.io",
            }))
        token = token_doc.get("token")
        if not isinstance(token, str) or not token:
            raise ReleaseError("GHCR returned no anonymous pull token")
        headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
        root, root_headers = self._json(
            f"https://ghcr.io/v2/formobr/monty-pod/manifests/{image_sha}", headers=headers)
        digest, manifest = select_amd64_manifest(root, root_headers, lambda ref: self._json(
            f"https://ghcr.io/v2/formobr/monty-pod/manifests/{ref}", headers=headers))
        config_ref = (manifest.get("config") or {}).get("digest")
        if not isinstance(config_ref, str) or not DIGEST_RE.fullmatch(config_ref):
            raise ReleaseError("linux/amd64 manifest has no valid config digest")
        config, _ = self._json(
            f"https://ghcr.io/v2/formobr/monty-pod/blobs/{config_ref}", headers=headers)
        revision, config_tag = verify_config_identity(config, tag=image_sha, commit=commit)
        return ImageReceipt(image_sha, commit, digest, revision, config_tag)


def select_amd64_manifest(root: dict[str, Any], headers: dict[str, str],
                          fetch: Callable[[str], tuple[dict[str, Any], dict[str, str]]]
                          ) -> tuple[str, dict[str, Any]]:
    """Select exactly one linux/amd64 manifest; kept stable for engine boot-probe."""
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
        raise ReleaseError("image OCI revision does not equal the source commit SHA")
    if config_tag != tag:
        raise ReleaseError("image POD_IMAGE_TAG does not equal the commit SHA tag")
    return str(revision), config_tag


def require_full_sha(value: str, what: str) -> str:
    if not SHA_RE.fullmatch(value):
        raise ReleaseError(f"{what} is not a full lowercase git SHA")
    return value


def require_clean(repo: Path, commands: Commands, what: str) -> None:
    if commands.out(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo):
        raise ReleaseError(f"{what} checkout is not clean")


def verify_source(image_sha: str, commands: Commands) -> str:
    """Prove the requested source is this clean checkout and is on origin/main."""
    image_sha = require_full_sha(image_sha, "image SHA")
    require_clean(REPO, commands, "pod-agent")
    head = require_full_sha(commands.out(["git", "rev-parse", "HEAD"]), "pod-agent HEAD")
    if head != image_sha:
        raise ReleaseError("requested image SHA does not equal clean pod-agent HEAD")
    commands.run(["git", "fetch", "--quiet", "origin", "main"], timeout=300)
    ancestor = commands.run(
        ["git", "merge-base", "--is-ancestor", image_sha, "origin/main"], check=False)
    if ancestor.returncode:
        raise ReleaseError("image source commit is not reachable from origin/main")
    return image_sha


def inspect_source_artifact(image_sha: str, commands: Commands, registry: Registry) -> ImageReceipt:
    """Prove source and artifact identity without reading or changing engine pins."""
    commit = verify_source(image_sha, commands)
    return registry.inspect(commit, commit)


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
        raise ReleaseError("engine image SHA tag/digest do not equal the verified GHCR receipt")
    submodule_sha = require_full_sha(
        commands.out(["git", "rev-parse", "HEAD"], cwd=engine / "pod-agent"),
        "engine pod-agent gitlink")
    if submodule_sha != receipt.commit:
        raise ReleaseError("engine pod-agent gitlink does not equal the image source commit")
    doc = (engine / "docs" / "gen" / "POD_IMAGE.md").read_text(encoding="utf-8")
    if image not in doc or digest not in doc:
        raise ReleaseError("generated POD_IMAGE doc does not quote the exact SHA tag and amd64 digest")


def replace_once(text: str, pattern: str, replacement: str, what: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ReleaseError(f"engine {what} declaration is missing or ambiguous")
    return updated


def update_engine(engine: Path, receipt: ImageReceipt, commands: Commands) -> None:
    """Update exact local pin paths, rolling every one back on any refusal."""
    require_clean(engine, commands, "engine")
    pin_file = engine / "scripts" / "broker" / "pod_image.py"
    doc_file = engine / "docs" / "gen" / "POD_IMAGE.md"
    old_pin = pin_file.read_text(encoding="utf-8")
    old_doc = doc_file.read_text(encoding="utf-8")
    old_submodule = commands.out(["git", "rev-parse", "HEAD"], cwd=engine / "pod-agent")
    try:
        target = engine / "pod-agent"
        present = commands.run(["git", "cat-file", "-e", f"{receipt.commit}^{{commit}}"],
                               cwd=target, check=False)
        if present.returncode:
            commands.run(["git", "fetch", "--quiet", str(REPO), "HEAD"],
                         cwd=target, timeout=300)
        commands.run(["git", "checkout", "--quiet", "--detach", receipt.commit],
                     cwd=target)
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


def pin(image_sha: str, engine: Path, commands: Commands, registry: Registry) -> ImageReceipt:
    receipt = inspect_source_artifact(image_sha, commands, registry)
    update_engine(engine, receipt, commands)
    print(f"[image] PINNED sha={image_sha} amd64={receipt.amd64_digest}")
    return receipt


def verify(image_sha: str, engine: Path, commands: Commands, registry: Registry) -> ImageReceipt:
    receipt = inspect_source_artifact(image_sha, commands, registry)
    verify_engine(engine, receipt, commands)
    print(f"[image] PASS sha={image_sha} amd64={receipt.amd64_digest}")
    return receipt


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv[:1] == [REGISTRY_WORKER_ARG]:
        # `Registry._json`'s own isolated child, never a human-facing verb — no subparser, no --help entry.
        return _run_registry_worker()
    parser = argparse.ArgumentParser(description="verify or pin an already-published monty-pod SHA image")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pin", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("sha")
        cmd.add_argument("--engine-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pin":
            pin(args.sha, args.engine_dir.resolve(), Commands(), Registry())
        else:
            verify(args.sha, args.engine_dir.resolve(), Commands(), Registry())
        return 0
    except ReleaseError as exc:
        print(f"[image] REFUSE: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — unexpected state is still a bounded refusal
        print(f"[image] REFUSE: unexpected {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
