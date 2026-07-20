"""The Remotion bundle as a content-addressed job input (podagent/bundle.py + artifact.py).

Every test here is a NEGATIVE: it pins a way the cache could serve something wrong, and each one was
watched RED against the code with that guarantee removed. The two that matter most are the pair at the
bottom — an interrupted fetch must leave nothing usable AND must not wedge the slot forever. Written as
one test, the first half passes vacuously and hides the second.

Run: python3 -m pytest tests/test_bundle.py -q
"""
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from podagent import artifact, bundle
from podagent.models import BundleRef


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


_PROJECT = {
    "render_batch.mjs": b"// batch renderer\n",
    "src/index.ts": b"registerRoot(Root)\n",
    "src/tokens.ts": b"export const T = 1\n",
    "package.json": b'{"name":"remotion"}\n',
    "node_modules/remotion/index.js": b"module.exports = {}\n",
}


def _ref(payload: bytes) -> BundleRef:
    return BundleRef(url="https://example/bundle.tar", sha256=hashlib.sha256(payload).hexdigest(),
                     size=len(payload))


@pytest.fixture()
def serve(monkeypatch, tmp_path):
    """Point the cache at tmp and stub the network with a payload the test controls."""
    monkeypatch.setenv("REMOTION_BUNDLE_CACHE", str(tmp_path / "cache"))
    state: dict = {"calls": 0}

    def install(payload: bytes, *, truncate_at: int | None = None):
        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def raise_for_status(self): pass

            def iter_content(self, n):
                state["calls"] += 1
                body = payload if truncate_at is None else payload[:truncate_at]
                for i in range(0, len(body), n):
                    yield body[i:i + n]
                if truncate_at is not None:
                    raise ConnectionError("connection reset mid-body")

        monkeypatch.setattr(artifact.requests, "get", lambda *a, **k: _Resp())

    state["install"] = install
    return state


# ── integrity ───────────────────────────────────────────────────────────────
def test_digest_mismatch_never_extracts(serve):
    """Bytes that do not hash to the declared digest are refused BEFORE extraction. Without this a swapped
    or truncated object becomes a mystery-broken render instead of one clear error."""
    payload = _tar(_PROJECT)
    serve["install"](payload)
    bad = BundleRef(url="https://example/b.tar", sha256="a" * 64, size=len(payload))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        bundle.ensure(bad)


def test_non_remotion_tar_is_refused(serve):
    """A tar that verifies but is not a Remotion project fails at fetch with a named-missing-parts message,
    not later as an opaque Node resolution error inside a subprocess."""
    payload = _tar({"readme.txt": b"hi\n"})
    serve["install"](payload)
    with pytest.raises(RuntimeError, match="not a Remotion project"):
        bundle.ensure(_ref(payload))


def test_tar_escaping_its_directory_is_refused(tmp_path):
    """Path traversal in a member name is refused even though the tar is content-verified — a signed URL
    says who sent it, not that what they sent is safe to unpack anywhere."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"bad"))
    tar_path = tmp_path / "evil.tar"
    tar_path.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="escapes its directory"):
        artifact.safe_extract(tar_path, tmp_path / "dest")


# ── the cache key is the CONTENT, never a name ──────────────────────────────
def test_changed_bundle_is_never_served_from_the_old_entry(serve):
    """Two different bundles land in two different directories. A cache keyed by anything but content
    (a name, a version string) would serve the first one for the second — the exact stale-env bug the
    venv-tarball lane shipped."""
    first = _tar(_PROJECT)
    serve["install"](first)
    a = bundle.ensure(_ref(first))

    second = _tar({**_PROJECT, "src/tokens.ts": b"export const T = 2\n"})
    serve["install"](second)
    b = bundle.ensure(_ref(second))

    assert a != b
    assert (b / "src/tokens.ts").read_bytes() == b"export const T = 2\n"
    assert (a / "src/tokens.ts").read_bytes() == b"export const T = 1\n"


def test_warm_cache_does_not_refetch(serve):
    """A warm pod pays the transfer once per bundle. If this goes red the pod is re-downloading 500 MB
    per job and the whole per-job-input design is a cost regression instead of a saving."""
    payload = _tar(_PROJECT)
    serve["install"](payload)
    ref = _ref(payload)
    bundle.ensure(ref)
    calls_after_first = serve["calls"]
    bundle.ensure(ref)
    assert serve["calls"] == calls_after_first


# ── the cache is IMMUTABLE; jobs render out of their own workspace ──────────
def test_workspace_staging_cannot_touch_the_cache(serve, tmp_path):
    """Staging a job's assets into its workspace must leave the cached bundle byte-identical.

    This is the hardlink trap: a `cp -al`-style workspace would share inodes, and shutil.copy2 onto a
    shared inode opens O_TRUNC — one job's font would overwrite the cached bundle for every later job.
    """
    payload = _tar(_PROJECT)
    serve["install"](payload)
    root = bundle.ensure(_ref(payload))
    before = {p.relative_to(root): p.read_bytes()
              for p in root.rglob("*") if p.is_file() and not p.is_symlink()}

    ws = bundle.workspace(root, tmp_path / "job1")
    (ws / "public" / "brand.woff2").write_bytes(b"a job asset")
    (ws / "src" / "index.bespoke.Bespoke-dead.tsx").write_bytes(b"per-job entry")
    (ws / "src" / "tokens.ts").write_bytes(b"export const T = 999\n")   # overwrite an EXISTING file

    after = {p.relative_to(root): p.read_bytes()
             for p in root.rglob("*") if p.is_file() and not p.is_symlink()}
    assert after == before, "a job mutated the shared bundle cache"


def test_two_workspaces_do_not_leak_into_each_other(serve, tmp_path):
    """Job A's staged assets must be invisible to job B. A handed-out cache path would leak both ways."""
    payload = _tar(_PROJECT)
    serve["install"](payload)
    root = bundle.ensure(_ref(payload))

    a = bundle.workspace(root, tmp_path / "a")
    (a / "public" / "a-only.png").write_bytes(b"A")
    b = bundle.workspace(root, tmp_path / "b")

    assert not (b / "public" / "a-only.png").exists()


def test_node_modules_is_shared_by_symlink_not_copied(serve, tmp_path):
    """504 MB may not be copied per job. If this goes red every mograph job pays a half-gigabyte disk
    copy before it renders a single frame."""
    payload = _tar(_PROJECT)
    serve["install"](payload)
    root = bundle.ensure(_ref(payload))
    ws = bundle.workspace(root, tmp_path / "job")

    assert (ws / "node_modules").is_symlink()
    assert (ws / "node_modules").resolve() == (root / "node_modules").resolve()
    assert not (ws / "src").is_symlink(), "src is staged into, so it must be a real copy"


# ── the interrupted-fetch PAIR (Agent A's trap: one test hides the other) ───
def test_interrupted_fetch_leaves_no_usable_entry(serve):
    """A fetch killed mid-body must not publish a half-tree under the content hash — the next run must
    treat it as ABSENT, never as a usable bundle."""
    payload = _tar(_PROJECT)
    ref = _ref(payload)
    serve["install"](payload, truncate_at=len(payload) // 2)
    with pytest.raises(ConnectionError):
        bundle.ensure(ref)

    dest = bundle.cache_root() / ref.sha256
    assert not (dest / artifact.DONE).is_file(), "a partial fetch published a usable-looking cache entry"


def test_interrupted_fetch_does_not_wedge_the_cache(serve):
    """…and the slot must still be claimable. This is the half the combined test cannot see: a leftover
    sentinel-less directory on the target must be taken over, not treated as a permanent occupant, or one
    killed fetch disables mograph on that pod until someone deletes the directory by hand."""
    payload = _tar(_PROJECT)
    ref = _ref(payload)
    serve["install"](payload, truncate_at=len(payload) // 2)
    with pytest.raises(ConnectionError):
        bundle.ensure(ref)

    # simulate the leftover an older layout / half-restored backup leaves ON the target path
    dest = bundle.cache_root() / ref.sha256
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "stale-junk").write_text("no sentinel here")

    serve["install"](payload)
    root = bundle.ensure(ref)
    assert (root / artifact.DONE).is_file()
    assert (root / "render_batch.mjs").is_file()
    assert not (root / "stale-junk").exists(), "the stale directory was reused instead of replaced"


# ── local development is untouched ──────────────────────────────────────────
def test_local_remotion_dir_still_wins(monkeypatch, tmp_path):
    """A local run renders against a checkout's own remotion/ tree with no bundle at all — a binding
    constraint on this change. MONTY_REMOTION_DIR must short-circuit the whole fetch path."""
    from podagent import mograph

    local = tmp_path / "remotion"
    local.mkdir()
    (local / "render_batch.mjs").write_text("// local\n")
    monkeypatch.setenv("MONTY_REMOTION_DIR", str(local))

    assert mograph.remotion_dir(None, tmp_path / "tmp") == local


def test_bad_local_remotion_dir_fails_loud(monkeypatch, tmp_path):
    """A MONTY_REMOTION_DIR pointing at a tree with no render_batch.mjs is a broken dev setup and says so,
    rather than silently falling through to a fetch the local run cannot do."""
    from podagent import mograph

    monkeypatch.setenv("MONTY_REMOTION_DIR", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="no render_batch.mjs"):
        mograph.remotion_dir(None, tmp_path / "tmp")


# ── the CONTRACT guarantee: a bundle-less mograph job cannot even be represented ──
def _plan(**kw):
    from podagent.models import SpecMotionPlan
    return SpecMotionPlan(**kw)


def test_sections_without_a_bundle_are_rejected():
    """The defect this whole change exists to prevent: a pod with no bundle quietly skipping every section
    and publishing a green manifest for a video with no motion graphics. The contract refuses it instead."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="requires motion_plan.bundle"):
        _plan(sections=[{"comp": "TitleFX", "start": 0.0, "props": {}}])


def test_an_empty_motion_plan_needs_no_bundle():
    """…and the rule is scoped: nothing to render means nothing to fetch, so a captions-only plan stays
    legal. Without this the guarantee above would force a 500 MB input onto jobs that never use it."""
    assert _plan(sections=[]).bundle is None


def test_a_v3_spec_is_refused_by_a_v4_pod():
    """A pod must refuse a job from a control plane it does not share a contract with, loudly and at the
    edge. Silent acceptance is how a v3 pod would ignore motion_plan.bundle and skip mograph."""
    import json
    from pathlib import Path

    import pydantic

    from podagent.models import RenderSpec

    stale = json.loads((Path(__file__).resolve().parents[1] / "contracts" / "examples" / "invalid"
                        / "spec.stale-v3.json").read_text())
    with pytest.raises(pydantic.ValidationError):
        RenderSpec.model_validate(stale)
