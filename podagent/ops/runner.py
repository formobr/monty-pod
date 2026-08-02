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

# ── how wide the chain runs ──────────────────────────────────────────────────────────────────────
#
# A fan-out stage is ~51 independent `media.scale` steps, so the cap IS the wall clock: a flat 8 left
# most of a rented 32-core box idle. But cores alone is the wrong bound — measured on a 12-thread /
# 31 GB dev box, five concurrent 4K ffmpeg encodes sat at ~1 GB RSS EACH and pushed the machine into
# 11 GB of swap, and swapping encodes finish later than fewer non-swapping ones. So the cap is the
# smaller of what the cores can schedule and what the RAM can hold.

# Bytes of resident memory to reserve per concurrent step. Gates how many steps may run at once when
# memory, not cores, is the scarce side. Anchored on the ~1 GB RSS a 4K ffmpeg encode was measured to
# hold on the dev box above, rounded up to 1.5 GiB so the arithmetic leaves room for the decode-side
# buffers and page cache that made that box swap rather than merely fill.
MEM_PER_STEP_BYTES = 1536 * 1024 * 1024

# Cores per concurrent step. Not 1: ffmpeg is internally multithreaded, so one process per core
# oversubscribes the scheduler and each encode gets slower without the box doing more work.
CORES_PER_STEP = 2

_CPU_FALLBACK = 4  # os.cpu_count() may return None in a container with no affinity mask


def _mem_available_bytes() -> int | None:
    """Memory the box can hand out RIGHT NOW, or None if this kernel will not say.

    MemAvailable is the honest number (it counts reclaimable page cache, which MemFree does not); the
    sysconf pair is the portable fallback. Unreadable is not fatal — the cap just falls back to cores.
    """
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def _cap_from(cores: int | None, mem_avail: int | None, env_raw: str | None) -> tuple[int, str]:
    """Pure cap arithmetic, so the decision is testable without a box to introspect.

    Returns (cap, reason) — the reason is logged, because a scheduling choice nobody can see is how a
    rented box ends up half idle for months.
    """
    env_cap = 0
    if env_raw:
        try:
            env_cap = int(env_raw)
        except ValueError:
            env_cap = 0

    n = cores if cores and cores > 0 else _CPU_FALLBACK
    core_bound = max(1, n // CORES_PER_STEP)
    mem_bound = max(1, mem_avail // MEM_PER_STEP_BYTES) if mem_avail and mem_avail > 0 else None

    derived = core_bound if mem_bound is None else min(core_bound, mem_bound)
    cap = env_cap if env_cap > 0 else derived
    cap = max(1, cap)
    reason = (f"cores={n}, core-bound={core_bound}, "
              f"mem-bound={'unknown' if mem_bound is None else mem_bound}, "
              f"env={env_cap if env_cap > 0 else 'unset'}")
    return cap, reason


def parallel_cap() -> tuple[int, str]:
    """The cap this box gets, read from this box."""
    return _cap_from(os.cpu_count(), _mem_available_bytes(), os.environ.get(MAX_PARALLEL_ENV))


_slots: threading.Semaphore | None = None
_slots_lock = threading.Lock()


def step_slots() -> threading.Semaphore:
    """ONE box-wide budget of concurrently RUNNING steps, shared by every chain in flight.

    parallel_cap() answers "how wide may this box go", and a per-chain executor sized by it is right for one
    chain and wrong for four: the agent now drains several ops envelopes at once, and four chains × cap 8 is
    how a 16-core box ends up 32 ffmpegs deep and swapping — the exact failure MEM_PER_STEP_BYTES exists to
    prevent. One chain alone still gets the whole cap, so nothing about single-chain behaviour moves."""
    global _slots
    with _slots_lock:
        if _slots is None:
            _slots = threading.Semaphore(parallel_cap()[0])
        return _slots


def _reset_step_slots() -> None:
    """Tests only: re-derive the budget after monkeypatching the cap."""
    global _slots
    with _slots_lock:
        _slots = None


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
            want = b.from_port or b.port
            if b.from_port is None and want not in src and len(src) == 1:
                # single-output producer: bind it positionally, the common chain shape
                bound[b.port] = next(iter(src.values()))
                continue
            if want not in src:
                raise ChainError(
                    f"step {step.id!r}: {b.from_step!r} produced {sorted(src)}, not {want!r}")
            bound[b.port] = src[want]
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
    # The box-wide slot is taken for the WHOLE step, binding included: a fetch is transport this box pays for
    # too, and letting N chains bind concurrently outside the budget is how the disk, not the CPU, becomes
    # the bound nobody sized.
    with step_slots():
        return _run_step_inner(step, ws, produced)


def _run_step_inner(step: Any, ws: Workspace, produced: dict[str, dict[str, Path]]) -> dict[str, Path]:
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


def preflight_chain(chain: Any) -> None:
    """Refuse a chain this IMAGE cannot run — before the first step, not when the doomed step is reached.

    The pod's registry is the ground truth about what this build carries: the control plane pins a tag, but
    only the box that booted it knows what is actually in it. Checking per-step (which is where these same
    three checks also live, deliberately) means the chain runs until it REACHES the unknown op — a real run
    spent 236 s transcoding and then died on `unknown op 'media.audio'`, having paid the rent, the pull and
    the decode to learn a fact that was true before the first byte moved. Every step's op, placement and
    params are knowable at claim, so they are decided at claim.
    """
    known = registry.all_ops()
    missing = sorted({s.op for s in chain.steps} - set(known))
    if missing:
        raise ChainError(
            f"image {registry.image_tag()} does not carry op(s) {missing} named by this chain — refusing "
            f"the WHOLE chain before any step runs (nothing was fetched, decoded or encoded).\n"
            f"  this image's registry holds: {sorted(known)}\n"
            f"  the control plane is ahead of the image: ship the op, tag+build the image, bump the pin, "
            f"and restart the provisioner so it rents the new tag.")
    op_of = {s.id: s.op for s in chain.steps}
    for step in chain.steps:
        registry.assert_pod_safe(step.op)
        registry.validate_params(step.op, step.params)
        for b in step.inputs:
            if b.from_port is None:
                continue
            # a hand-off that names a port its producer does not declare is knowable now, and knowing it
            # after the producing encode has run costs that encode
            ports = {p.id for p in registry.get(op_of[b.from_step]).outputs}
            if b.from_port not in ports:
                raise ChainError(
                    f"step {step.id!r}: input {b.port!r} reads {b.from_step!r}.{b.from_port!r}, but "
                    f"{op_of[b.from_step]} declares outputs {sorted(ports)}")


def run_chain(chain: Any, cp: Any, corr_id: str | None = None,
              session_id: str | None = None) -> dict[str, Any]:
    """Execute the whole chain. Returns {step_id: {port: str(path)}} for the caller to inspect.

    corr_id/session_id are echoed from the claimed envelope onto every event/terminal (pool demux)."""

    def _event(**payload: Any) -> None:
        if corr_id is not None:
            payload["corr_id"] = corr_id
        if session_id is not None:
            payload["session_id"] = session_id
        cp.post_event(payload)

    preflight_chain(chain)      # FIRST: an unrunnable chain must cost nothing but the claim
    pack.activate(chain.pack)

    tmp = Path(tempfile.mkdtemp(prefix="opchain_"))
    ws = Workspace(tmp)
    produced: dict[str, dict[str, Path]] = {}
    by_id = {s.id: s for s in chain.steps}
    deps = {s.id: set(s.needs) | {b.from_step for b in s.inputs if b.from_step} for s in chain.steps}
    pending = set(by_id)
    failed: set[str] = set()
    lock = threading.Lock()

    def _drop_dependents_of_failed() -> None:
        """A step whose producer failed can never run. If it is OPTIONAL it is dropped with a diagnostic; if
        it is not, the chain has lost work it was required to deliver and says so — the alternative is the
        runner sitting on an unrunnable step until the stall check calls it a deadlock."""
        while True:
            doomed = [sid for sid in sorted(pending) if deps[sid] & failed]
            if not doomed:
                return
            for sid in doomed:
                if not by_id[sid].optional:
                    raise ChainError(
                        f"step {sid!r} ({by_id[sid].op}) needs failed step(s) "
                        f"{sorted(deps[sid] & failed)} and is not optional")
                pending.discard(sid)
                failed.add(sid)
                _event(job_id=chain.job_id, stage="ops", status="step", step=sid, optional=True,
                       error=f"{by_id[sid].op}: skipped, needs failed step(s) {sorted(deps[sid] & failed)}")

    cap, why = parallel_cap()
    log(f"ops parallel cap={cap} ({why})")
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
                    if by_id[sid].optional:
                        # ONE arm of a fan-out, not the chain: a 403 from one candidate's host must not
                        # discard the twelve siblings that were fetching fine. Status stays inside the pod
                        # wire vocabulary (done|ok|step|error) and NON-terminal — an "error" here would be
                        # read as the chain's own, and a word outside it would 422 and abort the chain.
                        _event(job_id=chain.job_id, stage="ops", status="step", step=sid, optional=True,
                               error=f"{by_id[sid].op}: {e}"[:500])
                        failed.add(sid)
                        _drop_dependents_of_failed()
                        continue
                    _event(job_id=chain.job_id, stage="ops", status="error",
                           step=sid, error=f"{by_id[sid].op}: {e}"[:500])
                    raise
                with lock:
                    produced[sid] = outs
                _event(job_id=chain.job_id, stage="ops", status="step",
                       step=sid, op=by_id[sid].op)
        _event(job_id=chain.job_id, stage="ops", status="ok", steps=sorted(produced),
               skipped=sorted(failed))
        return {sid: {p: str(v) for p, v in outs.items()} for sid, outs in produced.items()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
