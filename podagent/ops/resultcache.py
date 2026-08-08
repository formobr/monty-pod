"""Warm-pod, content-exact handler results. Deliberately allowlisted to ``media.scale`` only."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from . import pack, registry

CACHE_ENV = "OPS_RESULT_CACHE"
CACHE_DEFAULT = "/var/cache/monty/results"
DISABLE_ENV = "OPS_RESULT_CACHE_OFF"
MAX_GB_ENV = "OPS_RESULT_CACHE_MAX_GB"
MAX_GB_DEFAULT = 12.0
SCHEMA = 1
_ALLOW = frozenset({"media.scale"})
_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


def enabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in ("1", "true", "yes")


def root() -> Path:
    return Path(os.environ.get(CACHE_ENV) or CACHE_DEFAULT)


def _lock(key: str) -> threading.Lock:
    with _guard:
        return _locks.setdefault(key, threading.Lock())


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _copy_digest(src: Path, dst: Path) -> tuple[str, int]:
    """Copy once while proving the bytes copied; avoids rereading video-sized payloads."""
    h = hashlib.sha256()
    size = 0
    with src.open("rb") as source, dst.open("xb") as target:
        while block := source.read(8 * 1024 * 1024):
            target.write(block)
            h.update(block)
            size += len(block)
        target.flush()
        os.fsync(target.fileno())
    return h.hexdigest(), size


@lru_cache(maxsize=1)
def ffmpeg_identity() -> str:
    raw = subprocess.check_output(
        ["ffmpeg", "-hide_banner", "-version"], stderr=subprocess.STDOUT, timeout=10)
    if not raw.strip():
        raise RuntimeError("ffmpeg returned no build identity")
    # The first line names the git revision; the rest pins configure flags and linked library versions.
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _material(op: registry.Op, params: dict[str, Any], inputs: dict[str, Path],
              outputs: dict[str, Any]) -> dict[str, Any] | None:
    if op.op not in _ALLOW or set(inputs) != {"src"} or set(outputs) != {"dst"}:
        return None
    image = registry.image_tag()
    pack_sha = pack.active_sha()
    if image.startswith("unknown") or pack_sha is None:
        return None
    dst = outputs["dst"]
    if not isinstance(dst, Path):
        return None
    return {
        "schema": SCHEMA,
        "op": op.op,
        "op_version": op.version,
        "params": params,
        "inputs": [{"port": name, "sha256": _sha(inputs[name])} for name in sorted(inputs)],
        "suffix": dst.suffix,
        "ops_pack_sha256": pack_sha,
        "pod_image": image,
        "ffmpeg": ffmpeg_identity(),
    }


def _key(material: dict[str, Any]) -> str:
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _slot(key: str) -> Path:
    return root() / key


def _discard(slot: Path) -> None:
    shutil.rmtree(slot, ignore_errors=True)


def _restore(slot: Path, key: str, dst: Path, log: Callable[[str], None]) -> bool:
    payload, meta_path = slot / "payload", slot / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if set(meta) != {"schema", "key", "sha256", "bytes"}:
            return False
        if meta["schema"] != SCHEMA or meta["key"] != key or type(meta["bytes"]) is not int:
            return False
        if not payload.is_file() or payload.stat().st_size != meta["bytes"]:
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.cache-part")
        try:
            digest, size = _copy_digest(payload, tmp)
            if size != meta["bytes"] or digest != meta["sha256"]:
                raise OSError("restored result differs from verified cache payload")
            tmp.replace(dst)
        finally:
            tmp.unlink(missing_ok=True)
        os.utime(payload, None)
        return True
    except Exception as exc:  # cache corruption/IO is a miss; handler exceptions never enter this function
        # Type only: paths, cache keys and payload metadata are deliberately absent from the event stream.
        log(f"[result-cache] restore miss ({type(exc).__name__}); recomputing")
        return False


def _store(slot: Path, key: str, src: Path) -> None:
    parent = slot.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / f".{key}.{os.getpid()}.{threading.get_ident()}.part"
    _discard(tmp)
    tmp.mkdir()
    try:
        payload = tmp / "payload"
        digest, size = _copy_digest(src, payload)
        if size != src.stat().st_size:
            raise OSError("result cache copy differs from handler output")
        meta = {"schema": SCHEMA, "key": key, "sha256": digest, "bytes": size}
        (tmp / "meta.json").write_text(
            json.dumps(meta, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        payload.chmod(0o444)
        (tmp / "meta.json").chmod(0o444)
        _discard(slot)
        tmp.replace(slot)
    finally:
        _discard(tmp)


def _entries() -> list[tuple[float, int, Path]]:
    rows = []
    for slot in root().glob("[0-9a-f]*"):
        with _guard:
            active = _locks.get(slot.name)
        if active is not None and active.locked():
            continue
        payload = slot / "payload"
        meta = slot / "meta.json"
        try:
            if payload.is_file() and meta.is_file():
                rows.append((payload.stat().st_mtime, payload.stat().st_size + meta.stat().st_size, slot))
        except OSError:
            pass
    return rows


def prune(keep_bytes: int | None = None) -> int:
    cap = keep_bytes if keep_bytes is not None else int(
        float(os.environ.get(MAX_GB_ENV) or MAX_GB_DEFAULT) * 1e9)
    entries = sorted(_entries())
    total = sum(size for _mtime, size, _slot_path in entries)
    freed = 0
    for _mtime, size, slot in entries:
        if total <= cap:
            break
        _discard(slot)
        total -= size
        freed += size
    return freed


def execute(op: registry.Op, params: dict[str, Any], inputs: dict[str, Path], outputs: dict[str, Any],
            handler: Callable[..., Any], log: Callable[[str], None]) -> bool:
    """Run or restore. Returns whether the handler was skipped due to a verified cache hit."""
    if not enabled() or op.op not in _ALLOW:
        handler(params=params, inputs=inputs, outputs=outputs)
        return False
    try:
        material = _material(op, params, inputs, outputs)
        if material is None:
            handler(params=params, inputs=inputs, outputs=outputs)
            return False
        key = _key(material)
    except Exception as exc:  # identity/cache failure may not suppress the real handler
        log(f"[result-cache] identity unavailable ({type(exc).__name__}); running {op.op}")
        handler(params=params, inputs=inputs, outputs=outputs)
        return False

    with _lock(key):
        slot = _slot(key)
        if _restore(slot, key, outputs["dst"], log):
            log(f"[result-cache] hit {op.op} {key[:12]}")
            return True
        if slot.exists():
            _discard(slot)
            log(f"[result-cache] corrupt {op.op} {key[:12]}; recomputing")
        handler(params=params, inputs=inputs, outputs=outputs)
        stored = False
        if outputs["dst"].is_file():
            try:
                _store(slot, key, outputs["dst"])
                stored = True
            except Exception as exc:  # cache maintenance may not fail valid handler work
                _discard(slot)
                log(f"[result-cache] store failed ({type(exc).__name__}); result remains valid")
    if stored:
        try:
            prune()
        except Exception as exc:  # cache maintenance may not fail valid handler work
            log(f"[result-cache] prune failed ({type(exc).__name__}); result remains valid")
    return False
