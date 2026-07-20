"""Execute an OpChain: one box, one job, local-disk handoff, parallel where the DAG allows.

THE PERFORMANCE CONTRACT, because this file is where it is either kept or lost:

  * R2 is crossed ONCE IN and ONCE OUT of the chain, not per op. A binding that names `from_step` is a
    path lookup in the workspace — no upload, no download, no round trip. Only a binding that names a
    `url` costs transport, and the control plane only emits one where the artifact genuinely has to
    outlive the job (a deliverable, or something a later job reads after a human approve-gate).
  * Independent steps run CONCURRENTLY. The orchestrator already knows the dependency graph, so the
    transport must not be what serialises it.

`podagent.render.render_spec` is the existing proof this works: it downloads inputs into one tmpdir,
chains several ffmpeg passes rebinding a local Path each time, and uploads once at the end. This is that
pattern, generalised and made declarative.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from ..artifact import log
from ..cp import download, upload
from . import pack, registry

MAX_PARALLEL_ENV = "OPS_MAX_PARALLEL"


class ChainError(RuntimeError):
    pass


class Workspace:
    """The chain's shared local disk. One directory, removed when the chain ends.

    Step outputs live at `<root>/<step_id>/<port>`, which is what makes `from_step` a pure path lookup:
    the producer writes there and the consumer reads there, on the same filesystem, with no copy and no
    network. Downloads are memoised by URL so a fan-out of steps reading the same source input pays one
    transfer, not one per step.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._downloads: dict[str, Path] = {}
        self._lock = threading.Lock()

    def step_dir(self, step_id: str) -> Path:
        d = self.root / step_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fetch(self, url: str) -> Path:
        """Download once per distinct URL, however many steps bind it."""
        with self._lock:
            hit = self._downloads.get(url)
        if hit is not None:
            return hit
        dest = self.root / "_in" / f"{abs(hash(url)):016x}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Two steps racing the same URL both download; the loser's copy is byte-identical and discarded.
        # Holding the lock across a multi-hundred-MB transfer would serialise the whole chain, which is
        # the exact opposite of this module's job.
        download(url, dest)
        with self._lock:
            return self._downloads.setdefault(url, dest)


# A workspace path needs a real extension: ffmpeg picks its MUXER from the output suffix and fails with
# "Unable to choose an output format" on a bare name. Prefer the extension the binding's destination
# already implies (…/x.proxy.mp4 → .mp4, …/y.mov → .mov) so the intra-workspace file and the artifact that
# eventually lands are the same container; fall back to the port kind's default.
_KIND_EXT = {"video": ".mp4", "audio": ".m4a", "image": ".png", "font": ".ttf", "code": ".txt",
             "json": ".json", "dir": ""}


def _ext(binding: Any, port: registry.Port) -> str:
    for cand in (getattr(binding, "url", None), getattr(binding, "path", None)):
        if cand:
            suffix = PurePosixPath(urlparse(str(cand)).path).suffix
            if suffix:
                return suffix
    return _KIND_EXT.get(port.kind, "")


def _bind_inputs(step: Any, op: registry.Op, ws: Workspace, produced: dict[str, dict[str, Path]]) -> dict[str, Path]:
    declared = {p.id: p for p in op.inputs}
    bound: dict[str, Path] = {}
    for b in step.inputs:
        if b.port not in declared:
            raise ChainError(f"step {step.id!r}: op {op.op} declares no input port {b.port!r}")
        if b.from_step is not None:
            src = produced.get(b.from_step, {})
            if b.port not in src and len(src) == 1:
                # single-output producer: bind it positionally, the common chain shape
                bound[b.port] = next(iter(src.values()))
                continue
            if b.port not in src:
                raise ChainError(
                    f"step {step.id!r}: {b.from_step!r} produced {sorted(src)}, not {b.port!r}")
            bound[b.port] = src[b.port]
        elif b.path is not None:
            p = Path(b.path)
            if not p.exists():
                raise ChainError(f"step {step.id!r}: input {b.port!r} path {p} does not exist")
            bound[b.port] = p
        else:
            assert b.url is not None
            bound[b.port] = ws.fetch(b.url)
    missing = [p.id for p in op.inputs if not p.optional and p.id not in bound]
    if missing:
        raise ChainError(f"step {step.id!r}: op {op.op} requires inputs {missing}")
    return bound


def _run_step(step: Any, ws: Workspace, produced: dict[str, dict[str, Path]]) -> dict[str, Path]:
    op = registry.get(step.op)
    # Refuse a judgement op HERE, on the executing box, before anything is fetched or run. Redundant with
    # the control plane's placement check by design: a check that lives only where the routing decision is
    # made is a check the routing bug turns off.
    registry.assert_pod_safe(step.op)
    registry.validate_params(step.op, step.params)

    inputs = _bind_inputs(step, op, ws, produced)
    out_dir = ws.step_dir(step.id)
    declared_out = {p.id: p for p in op.outputs}
    outputs: dict[str, Path] = {}
    for b in step.outputs:
        if b.port not in declared_out:
            raise ChainError(f"step {step.id!r}: op {op.op} declares no output port {b.port!r}")
        outputs[b.port] = out_dir / (b.port + _ext(b, declared_out[b.port]))
    for p in op.outputs:
        outputs.setdefault(p.id, out_dir / (p.id + _ext(None, p)))

    fn = pack.resolve(op.handler)
    t0 = time.monotonic()
    # THE handler call. `LocalBackend` makes this exact call in-process on the origin machine; here the
    # pod makes it. ONE handler, two transports — parity is structural, not tested into existence. Note
    # what the handler is NOT given: no URL, no credential, no control-plane handle. It sees typed params
    # and local paths, so it cannot depend on where it is running.
    fn(params=step.params, inputs=inputs, outputs=outputs)
    dt = time.monotonic() - t0

    for port, path in outputs.items():
        if not path.exists() and not declared_out[port].optional:
            raise ChainError(f"step {step.id!r}: handler produced no {port!r} at {path}")
    log(f"op {step.op} [{step.id}] ok in {dt:.1f}s")

    # Only NOW does anything leave the box, and only for bindings that named a url.
    for b in step.outputs:
        if b.url is not None and outputs[b.port].exists():
            upload(outputs[b.port], b.url)
    return outputs


def run_chain(chain: Any, cp: Any) -> dict[str, Any]:
    """Execute the whole chain. Returns {step_id: {port: str(path)}} for the caller to inspect."""
    pack.activate(chain.pack)

    tmp = Path(tempfile.mkdtemp(prefix="opchain_"))
    ws = Workspace(tmp)
    produced: dict[str, dict[str, Path]] = {}
    by_id = {s.id: s for s in chain.steps}
    deps = {s.id: set(s.needs) | {b.from_step for b in s.inputs if b.from_step} for s in chain.steps}
    pending = set(by_id)
    lock = threading.Lock()

    cap = int(os.environ.get(MAX_PARALLEL_ENV) or 0) or min(8, (os.cpu_count() or 4))
    try:
        with cf.ThreadPoolExecutor(max_workers=cap) as ex:
            running: dict[cf.Future[Any], str] = {}
            while pending or running:
                with lock:
                    done_ids = set(produced)
                    ready = [sid for sid in pending if not (deps[sid] - done_ids)]
                for sid in ready:
                    pending.discard(sid)
                    running[ex.submit(_run_step, by_id[sid], ws, produced)] = sid
                if not running:
                    # pending non-empty with nothing runnable cannot happen (OpChain rejects cycles at
                    # validation) — but a deadlock on a rented box is expensive enough to name explicitly.
                    raise ChainError(f"chain stalled with {sorted(pending)} unrunnable")
                fut = next(cf.as_completed(list(running)))
                sid = running.pop(fut)
                try:
                    outs = fut.result()
                except Exception as e:
                    cp.post_event({"job_id": chain.job_id, "stage": "ops", "status": "error",
                                   "step": sid, "error": f"{by_id[sid].op}: {e}"[:500]})
                    raise
                with lock:
                    produced[sid] = outs
                cp.post_event({"job_id": chain.job_id, "stage": "ops", "status": "step",
                               "step": sid, "op": by_id[sid].op})
        cp.post_event({"job_id": chain.job_id, "stage": "ops", "status": "ok",
                       "steps": sorted(produced)})
        return {sid: {p: str(v) for p, v in outs.items()} for sid, outs in produced.items()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
