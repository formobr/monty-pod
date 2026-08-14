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
import contextvars
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final
from urllib.parse import urlparse

from ..artifact import log
from ..cp import download, put_trace, retry, upload
from ..identity import worker_identity
from ..sanitize import safe_error
from . import inputcache, pack, registry, resultcache

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

TRANSPORT_BUDGET_WHY = """
ONE BUDGET WAS SIZED BY CORES AND SPENT ON THE NETWORK.

The step slot used to cover the WHOLE step — bind, run and put — and the reason given for that was the DISK:
"letting N chains bind concurrently outside the budget is how the disk, not the CPU, becomes the bound nobody
sized". That reason is real and it is not the CPU's. Two different physical limits were being rationed by one
counter, and the counter was derived from `cores // 2`.

WHAT IT COST, MEASURED. On the b-roll fan-out of a warm prod run: `put 208.1 s` of `375.7 s` of total step
time across `media.fetch` + `media.filmstrip` + `media.sheet`. More than half the budget of a 32-core pod was
held by objects being uploaded to R2 — work that burns no core and finishes no sooner for having one. The
visible consequence was a ONE-step `media.fetch` (the opener's photo) submitted at run-offset 64.7 s that
never got a slot before its caller's 120 s subprocess timeout killed it, while fifteen-step chains held the
budget for 192 and 199 seconds.

SO THERE ARE TWO COUNTERS, EACH DERIVED FROM THE THING IT ACTUALLY PROTECTS. The CPU budget wraps the HANDLER
CALL only — that is where ffmpeg runs and where `MEM_PER_STEP_BYTES` and `CORES_PER_STEP` mean something.
Transport gets its own, wider counter: bind and put are bounded so a fan-out cannot open unlimited sockets
and unlimited partial files, but they no longer consume a slot sized for an encode.

WHY THE TRANSPORT WIDTH IS A MULTIPLE AND NOT A CONSTANT. It has to scale with the box like the CPU one does,
and it is bounded by sockets and disk head, not by cores — a transfer spends its time waiting. The multiple
is deliberately modest: the failure it still has to prevent is the one the original comment names, N chains
binding at once and filling the disk.
"""

# Concurrent transfers per CPU slot (TRANSPORT_BUDGET_WHY). 4x: a transfer measured at 0.3 s of latency for
# ~170 KB is ~90% wait, so four of them overlap inside one slot's worth of wall without contending for it.
TRANSFERS_PER_STEP = 4
TRANSPORT_MAX_ENV = "OPS_MAX_TRANSFERS"


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
_xfer: threading.Semaphore | None = None


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


def transport_cap() -> tuple[int, str]:
    """How many BINDS and PUTS may be in flight at once (TRANSPORT_BUDGET_WHY)."""
    raw = (os.environ.get(TRANSPORT_MAX_ENV) or "").strip()
    if raw:
        try:
            if (n := int(raw)) > 0:
                return n, f"{TRANSPORT_MAX_ENV}={n}"
        except ValueError:
            pass                       # an unreadable knob is not a bound; the derivation stands and says so
    cap, why = parallel_cap()
    return max(1, cap * TRANSFERS_PER_STEP), f"{TRANSFERS_PER_STEP}x the step cap ({why})"


def transport_slots() -> threading.Semaphore:
    """The transfer budget, separate from the CPU one because it protects a different thing."""
    global _xfer
    with _slots_lock:
        if _xfer is None:
            _xfer = threading.Semaphore(transport_cap()[0])
        return _xfer


def handler_slots(op: Any) -> threading.Semaphore:
    """The permit for the handler's actual work.

    Most handlers hold the CPU/RAM permit because they decode or encode pixels. A declared transport
    handler is different: its bounded network/remux work may be wider, but it still takes the same
    transport semaphore that protects sockets and disk. The declaration is explicit so a future op
    cannot accidentally become wide merely because it happens to mention ``net``.
    """
    return transport_slots() if op.budget == "transport" else step_slots()


def executor_workers(step_cap: int, transfer_cap: int) -> int:
    """Threads needed to expose both budgets; semaphores remain the actual global bounds."""
    return max(1, int(step_cap), int(transfer_cap))


def _reset_step_slots() -> None:
    """Tests only: re-derive both budgets after monkeypatching the cap."""
    global _slots, _xfer
    with _slots_lock:
        _slots = None
        _xfer = None


class ChainError(RuntimeError):
    pass


RETAIN_WHY: Final[str] = """
THE PREVIEW MASTER WAS PUT SO THAT THE SAME BOX COULD GET IT BACK.

`cut.apply` writes <slug>.cut.mp4 and five later chains bind it by R2 address. On the preview tier every one
of those chains lands on the worker that produced it, so the PUT buys nothing but latency — and the pod's own
input cache proves it: `upload_and_adopt` immediately turns the uploaded snapshot into the local entry those
same chains then hit. The upload is the round trip; the adoption is the part that was load-bearing.

`retain: "local"` keeps the adoption and drops the upload. What makes it safe is that the loss is NAMED. The
object exists on exactly ONE box, so a chain that binds it declares `retained_on`, and this runner answers
before it fetches anything:

  · same worker, entry present  — a cache hit, which is what the chain would have got after the PUT anyway.
  · DIFFERENT worker            — refused by name, at PREFLIGHT, before a byte moves or a frame decodes.
  · same worker, entry gone     — refused by name too. Retained slots are exempt from pruning, so this means
                                  the pod restarted or the cache was wiped; either way there is no remote
                                  copy, and a silent re-render would be paid work nobody asked for.

THE TWO FAILURES THIS REPLACES ARE BOTH SILENT. A plain bind of a never-uploaded key is a 404 several minutes
and one rented box away from its cause; a "helpful" fallback that re-derives the master is the same money
spent twice with nothing in the log to say so. Neither is a state this transport is allowed to reach.

IT IS THE CALLER'S CLAIM, NOT AN INFERENCE. Nothing here decides which tier may retain — the flag arrives on
the binding, and a final-tier step that does not carry it uploads exactly as it always did.
"""


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
        """Download once per distinct OBJECT, however many steps — or chains — bind it."""
        with self._lock:
            hit = self._downloads.get(url)
        if hit is not None:
            return hit
        dest = self.root / "_in" / f"{abs(hash(url)):016x}"
        # keyed by the object, so it survives the chain and a re-minted presign. The returned path is an
        # independent workspace lease: cache eviction cannot unlink a file while ffmpeg reads it.
        shared = inputcache.get(url, download, lease=dest, log=log)
        if shared is not None:
            with self._lock:
                return self._downloads.setdefault(url, shared)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Two steps racing the same URL both download; the loser's copy is byte-identical and discarded.
        # Holding the lock across a multi-hundred-MB transfer would serialise the whole chain, which is
        # the exact opposite of this module's job.
        download(url, dest)
        with self._lock:
            return self._downloads.setdefault(url, dest)

    def fetch_retained(self, url: str, holder: str) -> Path:
        """An object that was never uploaded, read back on the box that kept it (RETAIN_WHY).

        There is no remote fallback by construction, so every way this can fail is raised with the worker
        names in it. A lease is still taken, exactly as `fetch` does: the persistent payload is never handed
        to a handler that might outlive its slot.
        """
        assert_retained_here(url, holder)
        with self._lock:
            hit = self._downloads.get(url)
        if hit is not None:
            return hit
        dest = self.root / "_in" / f"{abs(hash(url)):016x}"
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _never(_url: str, _dest: Path) -> None:      # a retained object has nowhere to be fetched from
            raise ChainError(f"retained object {inputcache.object_key(_url)} was about to be DOWNLOADED; "
                             f"nothing ever uploaded it, so that request could only 404")

        lease = inputcache.get(url, _never, lease=dest, log=log)
        if lease is None:
            raise ChainError(
                f"input binds an object retained on worker {holder}, but this pod's input cache is disabled "
                f"({inputcache.DISABLE_ENV}) — retention has no other store. Re-run without retain.")
        with self._lock:
            return self._downloads.setdefault(url, lease)


def assert_retained_here(url: str, holder: str, *, check_entry: bool = True) -> None:
    """Refuse a retained bind this box cannot serve, by NAME (RETAIN_WHY). Pure identity check when
    `check_entry` is false, which is what lets preflight refuse a mis-routed chain before it costs a claim."""
    me = worker_identity()
    if holder != me:
        raise ChainError(
            f"retained on worker {holder}, claimed on worker {me} — re-run without retain or route to the "
            f"same worker. This object was deliberately NOT uploaded, so there is no address to fall back "
            f"to; nothing was re-rendered and nothing 404'd silently.")
    if check_entry and inputcache.retained_holder(url) is None:
        raise ChainError(
            f"retained on worker {holder}, which IS this worker, but its input cache no longer holds "
            f"{inputcache.object_key(url)} — the agent restarted or the cache was wiped. Nothing uploaded "
            f"this object, so it exists nowhere: re-run the producing chain without retain.")


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


ARITY_WHY: Final[str] = """
ONE STEP MAY PRODUCE N ADDRESSABLE FILES. A port the registry declares `many` is bound with `urls` — one
address per element — and the handler is handed a LIST of destination paths under the step's directory
instead of a single one. Everything else about that port is unchanged: the same existence check, the same
uploads, the same first-failure-raises polarity a step with two ports already had.

WHY IT IS THE PORT, NOT THE STEP, THAT GREW. The step is the unit of scheduling and of failure, and neither
of those wants to be N — a fan-out of N steps is already expressible and is the right shape when the N pieces
of work are independent. This is the other case: N results that one pass over one input produces together,
where splitting them into N steps means N passes. Reading the same input N times to write N files is a cost
the transport was imposing on the work.

A `many` PORT IS WRITE-ONLY WITHIN THE CHAIN: `from_step` may not read one, because "which of the N" is a
question a binding cannot ask today, and guessing is how a later step silently composites the wrong file.

THE FAILURE POLARITY IS THE ONE A TWO-PORT STEP ALREADY HAD. `upload` retries three times per object and
then raises, and the raise ends the STEP — exactly as a step whose second port will not move loses its
first. Elements past the failure are not attempted: an address the store has refused three times with
backoff will refuse the next one too, and a rented box may not spend its lease proving that. What already
landed STAYS landed, so a caller that addresses its files by content re-cuts only the holes. An element the
handler chose not to write is not a failure at all — the port carries its own `optional`, and a missing
element simply has nothing to move.
"""


def _many_dir(out_dir: Path, port: str) -> Path:
    return out_dir / f"_{port}"


def _bind_inputs(step: Any, op: registry.Op, ws: Workspace,
                 produced: dict[str, dict[str, Any]]) -> dict[str, Path]:
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
                only = next(iter(src.values()))
                if isinstance(only, list):
                    raise ChainError(f"step {step.id!r}: {b.from_step!r} produced a LIST port; a binding "
                                     f"cannot say which of its {len(only)} files to read (ARITY_WHY)")
                bound[b.port] = only
                continue
            if want not in src:
                raise ChainError(
                    f"step {step.id!r}: {b.from_step!r} produced {sorted(src)}, not {want!r}")
            if isinstance(src[want], list):
                raise ChainError(f"step {step.id!r}: {b.from_step!r}.{want!r} is a LIST port; a binding "
                                 f"cannot say which of its {len(src[want])} files to read (ARITY_WHY)")
            bound[b.port] = src[want]
        elif b.path is not None:
            p = Path(b.path)
            if not p.exists():
                raise ChainError(f"step {step.id!r}: input {b.port!r} path {p} does not exist")
            bound[b.port] = p
        else:
            assert b.url is not None
            held = getattr(b, "retained_on", None)
            bound[b.port] = ws.fetch_retained(b.url, str(held)) if held else ws.fetch(b.url)
    missing = [p.id for p in op.inputs if not p.optional and p.id not in bound]
    if missing:
        raise ChainError(f"step {step.id!r}: op {op.op} requires inputs {missing}")
    return bound


def assert_output_arity(step: Any, op: registry.Op) -> None:
    """Do the step's output bindings agree with the arity the op declares (ARITY_WHY)? Pure — no disk."""
    declared = {p.id: p for p in op.outputs}
    bound = {b.port for b in step.outputs}
    for b in step.outputs:
        if b.port not in declared:
            raise ChainError(f"step {step.id!r}: op {op.op} declares no output port {b.port!r}")
        if declared[b.port].many and b.urls is None:
            raise ChainError(f"step {step.id!r}: output {b.port!r} is a LIST port and must bind `urls`; "
                             f"a single `url` cannot address its files")
        if b.urls is not None and not declared[b.port].many:
            raise ChainError(f"step {step.id!r}: output {b.port!r} binds `urls`, but {op.op} declares it as "
                             f"ONE file — the addresses would have nothing to name")
    for p in op.outputs:
        # Unbound, so nobody said how many. A LIST port nothing addresses is a decode nothing reads.
        if p.many and p.id not in bound:
            raise ChainError(f"step {step.id!r}: op {op.op} declares LIST output {p.id!r}, which must be "
                             f"bound with `urls` — its arity is the binding's, not the handler's")


def _bind_outputs(step: Any, op: registry.Op, out_dir: Path) -> dict[str, Any]:
    """Destination paths per declared output port: one Path, or a LIST of them for a `many` port."""
    assert_output_arity(step, op)
    declared = {p.id: p for p in op.outputs}
    outputs: dict[str, Any] = {}
    for b in step.outputs:
        port = declared[b.port]
        if port.many:
            d = _many_dir(out_dir, b.port)
            d.mkdir(parents=True, exist_ok=True)
            outputs[b.port] = [d / f"{i:04d}{_ext(b, port)}" for i in range(len(b.urls))]
        else:
            outputs[b.port] = out_dir / (b.port + _ext(b, port))
    for p in op.outputs:
        if not p.many:
            outputs.setdefault(p.id, out_dir / (p.id + _ext(None, p)))
    return outputs


STEP_TIMING_WHY: Final[str] = """
THE BOX BOOKED ONE NUMBER PER CHAIN AND COULD NOT SAY WHAT IT BOUGHT.

Measured 2026-08-04/05: a b-roll fetch leg cost 4.13 s at the median and 33.61 s at the slowest one that
DELIVERED, for the same ~5-7 MB interior window of a 4K master. 7 MB in 33 s is 0.2 MB/s, which no CDN does.
The box could not tell a slow origin from a long GOP walk from its own transport losing the terminal over
finished work, because one span covered the enqueue, the claim, every step and the trip home.

THIS BOX IS THE ONLY PLACE THE SPLIT EXISTS, so this is where it is measured and it rides home on the
terminal. Four legs the runner itself can always see, for EVERY op:

  · `slot_wait` — ready-to-run time spent behind the box-wide handler semaphore. This is not handler work:
                  booking it as `run` made a saturated box look like a slow ffmpeg/CDN.

  · `bind`  — pulling this step's inputs into the workspace. Real transport, paid before any frame decodes,
              and free on a `from_step` hand-off (which is the whole point of the local-disk chain).
  · `run`   — the handler call. What the op actually costs on this hardware.
  · `put`   — the outputs leaving for their durable address. The other half of the transport.

...plus whatever the HANDLER can see inside `run`, through the pack's optional recorder (pack.LEGS_MODULE).
`media.fetch` is the case that motivated all of this: its origin GET happens INSIDE the handler, so the
runner's own legs would place the whole 33 s in `run` and answer nothing.

ADDITIVE IN BOTH DIRECTIONS. An older pack has no recorder and the handler legs are simply absent; an older
control plane drops the whole `timings` key and answers 202. Nothing here may ever be load-bearing for the
work — a measurement that can fail the job it measures is worse than no measurement.
"""


@dataclass
class StepTiming:
    """What one step cost, split the four ways this process can always see it (STEP_TIMING_WHY)."""
    step_id: str
    op: str
    slot_wait_s: float = 0.0
    bind_s: float = 0.0
    run_s: float = 0.0
    put_s: float = 0.0
    # Inside put_s, never added to it: the phase's wall is still the four (PUT_FANOUT_WHY).
    put_wait_s: float = 0.0
    put_retry_s: float = 0.0
    nbytes: int = 0
    outputs: list[dict[str, Any]] = field(default_factory=list)
    legs: dict[str, float] = field(default_factory=dict)
    cache_hit: bool = False
    intervals: dict[str, dict[str, int]] = field(default_factory=dict)
    handler_intervals: list[dict[str, Any]] = field(default_factory=list)
    puts: list[dict[str, Any]] = field(default_factory=list)
    incomplete_reasons: list[str] = field(default_factory=list)
    # ports this step kept on THIS worker instead of uploading (RETAIN_WHY) — the terminal names the
    # holder off these, because a reader on another worker can only be warned by a name it can bind
    retained_ports: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return round(self.slot_wait_s + self.bind_s + self.run_s + self.put_s, 3)

    def wire(self) -> dict[str, Any]:
        """The shape that crosses the seam. `legs` merges the runner's four with the handler's, and the
        runner's win a name collision: a handler cannot know what its own binding cost.

        The four are ALWAYS present, zero included. A `bind` of 0.0 is a measurement, not an absence — it
        says this step's input was a `from_step` hand-off on local disk and crossed no network, which is the
        performance contract at the top of this file being kept. Dropping it would make "free" and "never
        measured" the same reading, which is the class of error this whole change exists to remove."""
        legs = {**{k: round(v, 3) for k, v in self.legs.items()},
                "slot_wait": round(self.slot_wait_s, 3), "bind": round(self.bind_s, 3),
                "run": round(self.run_s, 3), "put": round(self.put_s, 3),
                "put_wait": round(self.put_wait_s, 3), "put_retry": round(self.put_retry_s, 3)}
        out: dict[str, Any] = {
            "id": self.step_id,
            "op": self.op,
            "seconds": self.seconds,
            "slot_wait_s": round(self.slot_wait_s, 3),
            "legs": legs,
            "outputs": list(self.outputs),
            "bytes": self.nbytes,
            "cache_hit": self.cache_hit,
        }
        return out

    def timeline_wire(self, *, job_id: str, corr_id: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for put in self.puts:
            port, index = str(put["port"]), int(put["index"])
            semantic = json.dumps(
                [job_id, corr_id, self.step_id, "output", port, index],
                ensure_ascii=True, separators=(",", ":"),
            ).encode()
            rows.append({
                **put,
                "object_id": hashlib.sha256(semantic).hexdigest(),
            })
        return {
            "id": self.step_id,
            "op": self.op,
            "intervals": dict(self.intervals),
            "handler_legs": [dict(row) for row in self.handler_intervals],
            "puts": rows,
            "complete": not self.incomplete_reasons,
            "incomplete_reasons": sorted(set(self.incomplete_reasons)),
        }


def _collect_legs(recorder: Any) -> tuple[dict[str, float], list[dict[str, Any]], bool]:
    """Whatever the pack's recorder gathered for the call that just returned. Best-effort by contract: a
    recorder that raises loses its legs and never the step."""
    if recorder is None:
        return {}, [], False
    try:
        got = recorder.collect()
    except Exception as e:  # noqa: BLE001 — announced; a stopwatch may not break the work it times
        log(f"ops: leg recorder failed ({type(e).__name__}: {e}) — step timed without handler legs")
        return {}, [], False
    totals = {str(k): float(v) for k, v in dict(got or {}).items()}
    try:
        intervals = recorder.collect_intervals()
    except Exception as e:  # noqa: BLE001 - old/broken recorder only loses placement
        if not isinstance(e, AttributeError):
            log(f"ops: leg interval recorder failed ({type(e).__name__}: {e})")
        return totals, [], not totals
    checked: list[dict[str, Any]] = []
    for row in list(intervals or []):
        checked.append({
            "name": str(row["name"]),
            "start_mono_ns": int(row["start_mono_ns"]),
            "end_mono_ns": int(row["end_mono_ns"]),
        })
    observed = {str(row["name"]) for row in checked}
    return totals, checked, set(totals) <= observed


def _moved_outputs(outputs: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """What this step actually wrote. Seconds without bytes name no defect — 33 s is a mood, 7 MB in 33 s is
    a diagnosis — and this box is the only side that can weigh the files."""
    total = 0
    rows: list[dict[str, Any]] = []
    for port, path in outputs.items():
        many = isinstance(path, list)
        for index, one in enumerate(path if many else [path]):
            row: dict[str, Any] = {"port": str(port)}
            if many:
                row["index"] = index
            try:
                nbytes = one.stat().st_size
            except OSError:
                nbytes = 0     # optional absent output remains visible, but weighs nothing
                row["present"] = False
            else:
                row["present"] = True
                total += nbytes
            row["bytes"] = nbytes
            rows.append(row)
    return total, rows


def _upload_with_affinity(url: str, path: Path) -> None:
    """One immutable snapshot is both PUT body and, after success, the warm input-cache entry."""
    inputcache.upload_and_adopt(url, path, upload, log=log)


PUT_FANOUT_WHY = """
A STEP'S OUTPUTS ARE INDEPENDENT OBJECTS AND WERE SENT ONE AFTER ANOTHER THROUGH ONE PERMIT.

`media.sheet` declares two durable outputs (the contact sheet and the `.cells.json` sidecar that says which
cells drew); `media.range_filmstrip` declares two (the strip and its transport receipt). The put phase held
ONE `transport_slots()` permit and walked them serially, so a two-output step paid two full store latencies
end to end while the transport budget — `TRANSFERS_PER_STEP` times the step cap, twenty on a five-step rent
— sat idle. Measured on the b-roll rank wave of run img-8010: `media.sheet@put` 277.9 s over 29 calls for
0.58 MB of object each, `media.range_filmstrip@put` 132.5 s over 48. Neither is bandwidth: the same pod put
an 87 MB `cut.apply` master in 6.4 s.

SO THE PERMIT IS TAKEN PER OBJECT, which is what the budget already says it counts ("how many BINDS and PUTS
may be in flight at once"), and a step's objects go out together. The arms cannot exceed the budget, because
each takes a permit of its own from it. Nothing about WHICH objects cross, or their bytes, changes.

AND THE PHASE NOW SAYS WHICH HALF IT SPENT. `put` was permit-wait plus wire under one name, so a 9.6-second
put could be a saturated transport budget or a slow store and no reading could tell them apart. `put_wait`
(worst permit wait among the arms) and `put_retry` (seconds burned on attempts that failed and were re-sent
from byte zero) are sub-legs of `put`, the way `connect`/`body` are sub-legs of `run`.
"""


def _upload_one(url: str, path: Path) -> tuple[float, float, dict[str, Any]]:
    """ONE object under ONE permit → (permit wait, seconds lost to failed attempts). The acquire is the
    same unbounded one the serial phase had; the transfer inside carries `cp._XFER_DEADLINE_S` per attempt."""
    t0 = time.monotonic()
    wait_start_ns = time.monotonic_ns()
    with transport_slots():
        wait_end_ns = time.monotonic_ns()
        waited = time.monotonic() - t0
        retry.reset()
        put_trace.reset()
        try:
            _upload_with_affinity(url, path)
        finally:
            spent = retry.seconds()
            attempts = put_trace.collect()
    return waited, spent, {
        "wait": {"start_mono_ns": wait_start_ns, "end_mono_ns": wait_end_ns},
        "attempts": attempts,
    }


def _output_arms(step: Any, outputs: dict[str, Any]) -> list[list[tuple[str, Path, str, str, int]]]:
    """One arm per PORT — never per element (PUT_FANOUT_WHY). A binding with no url crosses nothing and an
    absent optional output weighs nothing; a LIST port's `what` names its index (ARITY_WHY), a single one
    stays unnamed so its refusal reaches the caller exactly as the serial phase raised it."""
    arms: list[list[tuple[str, Path, str, str, int]]] = []
    for b in step.outputs:
        if getattr(b, "retain", None) is not None:
            continue                      # kept on the box under this url's key instead (RETAIN_WHY)
        if b.url is not None:
            if outputs[b.port].exists():
                arms.append([(b.url, outputs[b.port], "", str(b.port), 0)])
        elif b.urls is not None:
            arms.append([(url, path, f"{str(b.port)!r}[{i}] of {len(b.urls)}", str(b.port), i)
                         for i, (path, url) in enumerate(zip(outputs[b.port], b.urls)) if path.exists()])
    return [arm for arm in arms if arm]


def _upload_arm(
        step_id: str, arm: list[tuple[str, Path, str, str, int]],
) -> tuple[float, float, list[dict[str, Any]]]:
    """A port's objects, in order, stopping at the first refusal: an address the store refused three times
    with backoff will refuse the next one too, and a rented box may not spend its lease proving it."""
    waited = retried = 0.0
    traces: list[dict[str, Any]] = []
    for i, (url, path, what, port, index) in enumerate(arm):
        try:
            one_wait, one_retry, trace = _upload_one(url, path)
        except Exception as exc:
            if not what:
                raise
            raise ChainError(f"step {step_id!r}: output {what} would not upload ({safe_error(exc)}); "
                             f"{i} element(s) before it did land") from exc
        waited, retried = max(waited, one_wait), retried + one_retry
        traces.append({"direction": "output", "port": port, "index": index, **trace})
    return waited, retried, traces


def terminal_retained_on(timings) -> str | None:
    """The holder the terminal must name when ANY step kept an output on this worker (RETAIN_WHY) — the
    engine's require_retained_worker refuses a preview cut whose terminal cannot say WHERE the bytes are."""
    return worker_identity() if any(getattr(t, "retained_ports", None) for t in timings) else None


def _retain_outputs(step: Any, outputs: dict[str, Any]) -> list[str]:
    """Adopt every `retain: local` output under its own url's key instead of PUTting it (RETAIN_WHY).

    Runs BEFORE the uploads: it is a local copy that fails in milliseconds, and a retention that cannot be
    made readable must not be discovered after a sibling port has already crossed the wire.
    """
    kept: list[str] = []
    for b in step.outputs:
        if getattr(b, "retain", None) is None:
            continue
        path = outputs[b.port]
        if isinstance(path, list):
            raise ChainError(f"step {step.id!r}: output {b.port!r} is a LIST port and cannot be retained — "
                             f"retention keys on ONE object address (ARITY_WHY)")
        if not path.exists():
            # An optional output the handler chose not to write has nothing to keep; the port's own
            # `optional` flag already ruled on whether that is legal.
            continue
        try:
            inputcache.adopt_local(str(b.url), path, holder=worker_identity(), log=log)
        except inputcache.RetentionUnavailable as exc:
            raise ChainError(f"step {step.id!r}: output {b.port!r} was not uploaded because it declares "
                             f"retain=local, and it could not be kept either ({safe_error(exc)})") from exc
        kept.append(str(b.port))
    return kept


def _upload_outputs(step: Any, outputs: dict[str, Any]) -> tuple[float, float, list[dict[str, Any]]]:
    """A step's PORTS out concurrently → (worst permit wait, retry seconds) (PUT_FANOUT_WHY). Every arm is
    joined, failure included: unwinding while a transfer still reads the workspace deletes it underneath."""
    arms = _output_arms(step, outputs)
    if not arms:
        return 0.0, 0.0, []
    if len(arms) == 1:
        return _upload_arm(str(step.id), arms[0])
    # One thread per PORT, and the permit is the only bound: sizing the pool by the budget too would
    # serialise arms without ever booking the queueing as `put_wait`.
    waits, retries, traces, failed = [0.0], [0.0], [], []
    with cf.ThreadPoolExecutor(max_workers=len(arms), thread_name_prefix="ops-put") as ex:
        futures = [ex.submit(_upload_arm, str(step.id), arm) for arm in arms]
        for fut in futures:
            try:
                waited, spent, rows = fut.result()
            except Exception as exc:
                failed.append(exc)
            else:
                waits.append(waited)
                retries.append(spent)
                traces.extend(rows)
    if failed:
        raise failed[0]
    return max(waits), sum(retries), traces


def _run_step(step: Any, ws: Workspace, produced: dict[str, dict[str, Any]],
              sink: list[StepTiming] | None = None, emit: Any = None) -> dict[str, Any]:
    # The CPU slot is taken around the HANDLER ONLY (TRANSPORT_BUDGET_WHY) — see `_run_step_inner`. The disk
    # and socket bound the original comment was really protecting is now its own counter, which is what lets
    # a one-step fetch through while fifteen-step chains are encoding.
    return _run_step_inner(step, ws, produced, sink, emit)


def _run_step_inner(step: Any, ws: Workspace, produced: dict[str, dict[str, Any]],
                    sink: list[StepTiming] | None = None, emit: Any = None) -> dict[str, Any]:
    op = registry.get(step.op)
    # Refuse a judgement op HERE, on the executing box, before anything is fetched or run. Redundant with
    # the control plane's placement check by design: a check that lives only where the routing decision is
    # made is a check the routing bug turns off.
    registry.assert_pod_safe(step.op)
    registry.validate_params(step.op, step.params)

    timing = StepTiming(step_id=str(step.id), op=str(step.op))

    def live(phase: str, *, started: float | None = None,
             exc: BaseException | None = None, timings: dict[str, float] | None = None,
             outputs: list[dict[str, Any]] | None = None, outcome: str | None = None) -> None:
        if emit is None:
            return
        payload: dict[str, Any] = {
            "status": "step",
            "step": str(step.id),
            "op": str(step.op),
            "phase": phase,
        }
        if timings:
            payload["timings"] = {k: round(v, 3) for k, v in timings.items()}
        if outputs is not None:
            payload["outputs"] = list(outputs)
        if outcome is not None:
            payload["outcome"] = outcome
        if started is not None:
            payload.setdefault("timings", {})["phase_s"] = round(time.monotonic() - started, 3)
        if exc is not None:
            payload["outcome"] = "error"
            payload["error_type"] = type(exc).__name__
            payload["error"] = safe_error(exc)
        emit(**payload)

    # LIVE is deliberately one START boundary per waitable phase plus ONE success closure for the whole step.
    # Each event is a durable outbox rewrite+fsync, so restoring every phase-finished event would make wide
    # chains pay for noise. The single closure is not optional: without it, completed siblings and siblings
    # hung in upload are indistinguishable until a terminal that a hung chain can never produce.
    live("bind_started")
    t_bind = time.monotonic()
    bind_start_ns = time.monotonic_ns()
    try:
        with transport_slots():
            inputs = _bind_inputs(step, op, ws, produced)
    except BaseException as exc:
        live("bind_error", started=t_bind, exc=exc)
        raise
    timing.bind_s = time.monotonic() - t_bind
    timing.intervals["bind"] = {
        "start_mono_ns": bind_start_ns, "end_mono_ns": time.monotonic_ns()}
    out_dir = ws.step_dir(step.id)
    declared_out = {p.id: p for p in op.outputs}
    outputs = _bind_outputs(step, op, out_dir)

    fn = pack.resolve(op.handler)
    recorder = pack.legs()
    live("slot_wait_started", timings={"bind_s": timing.bind_s})
    slot_ready = time.monotonic()
    slot_start_ns = time.monotonic_ns()
    # THE handler call. `LocalBackend` makes this exact call in-process on the origin machine; here the
    # pod makes it. ONE handler, two transports — parity is structural, not tested into existence. Note
    # what the handler is NOT given: no URL, no credential, no control-plane handle. It sees typed params
    # and local paths, so it cannot depend on where it is running.
    # The op declaration selects the permit here. Pixel handlers take the CPU budget; the one transport
    # class (`media.fetch`) takes the wider socket/disk budget because its stream-copy/remux wait is not an
    # encode. Both are still bounded globally (TRANSPORT_BUDGET_WHY).
    run_announced = False
    try:
        with handler_slots(op):
            timing.slot_wait_s = time.monotonic() - slot_ready
            timing.intervals["slot_wait"] = {
                "start_mono_ns": slot_start_ns, "end_mono_ns": time.monotonic_ns()}
            live("run_started", timings={"bind_s": timing.bind_s, "slot_wait_s": timing.slot_wait_s})
            run_announced = True
            t0 = time.monotonic()
            run_start_ns = time.monotonic_ns()
            if recorder is not None:
                with recorder.recording():
                    timing.cache_hit = resultcache.execute(op, step.params, inputs, outputs, fn, log)
                timing.legs, timing.handler_intervals, placed = _collect_legs(recorder)
                if not placed:
                    timing.incomplete_reasons.append("handler_leg_intervals_missing")
            else:
                timing.cache_hit = resultcache.execute(op, step.params, inputs, outputs, fn, log)
                timing.incomplete_reasons.append("handler_recorder_missing")
    except BaseException as exc:
        # If the durable `step_started` append itself failed, do not try another append and mask the transport
        # refusal; work has not started, and main must stop before paid work can go silent.
        if run_announced:
            live("run_error", started=t0, exc=exc,
                 timings={"bind_s": timing.bind_s, "slot_wait_s": timing.slot_wait_s})
        raise
    dt = time.monotonic() - t0
    timing.run_s = dt
    timing.intervals["run"] = {
        "start_mono_ns": run_start_ns, "end_mono_ns": time.monotonic_ns()}

    for port, path in outputs.items():
        for one in (path if isinstance(path, list) else [path]):
            if not one.exists() and not declared_out[port].optional:
                raise ChainError(f"step {step.id!r}: handler produced no {port!r} at {one}")
    log(f"op {step.op} [{step.id}] ok in {dt:.1f}s")
    timing.nbytes, timing.outputs = _moved_outputs(outputs)

    # Only NOW does anything leave the box, and only for bindings that named a url.
    live("upload_started", timings={"run_s": timing.run_s})
    t_put = time.monotonic()
    put_start_ns = time.monotonic_ns()
    try:
        if kept := _retain_outputs(step, outputs):
            timing.retained_ports = kept
            log(f"op {step.op} [{step.id}] kept {kept} on this worker — no PUT (RETAIN_WHY)")
        timing.put_wait_s, timing.put_retry_s, timing.puts = _upload_outputs(step, outputs)
    except BaseException as exc:
        live("upload_error", started=t_put, exc=exc)
        raise
    timing.put_s = time.monotonic() - t_put
    timing.intervals["put"] = {
        "start_mono_ns": put_start_ns, "end_mono_ns": time.monotonic_ns()}
    if any(not row.get("attempts") for row in timing.puts):
        timing.incomplete_reasons.append("put_attempt_intervals_missing")
    # LAST, and only on the success road: a step that raised delivered nothing, and a duration booked for
    # work that did not land is the kind of number that makes a ledger worse than none.
    if sink is not None:
        sink.append(timing)
    live("step_finished", timings={
        "slot_wait_s": timing.slot_wait_s,
        "bind_s": timing.bind_s,
        "run_s": timing.run_s,
        "put_s": timing.put_s,
        "seconds": timing.seconds,
        "cache_hit": float(timing.cache_hit),
    }, outputs=timing.outputs, outcome="ok")
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
        # arity is knowable now too: N addresses against a port that yields one file is a chain that cannot
        # be run, and learning it after the decode costs the decode
        assert_output_arity(step, registry.get(step.op))
        for b in step.inputs:
            # WHICH BOX holds a retained object is knowable at claim, and learning it after the chain has
            # rented, pulled and decoded costs all three for a fact that was true before the first byte
            # moved. Identity only — whether the ENTRY is still there is a disk question, asked at bind.
            if (held := getattr(b, "retained_on", None)) is not None:
                assert_retained_here(str(b.url), str(held), check_entry=False)
            if b.from_port is None:
                continue
            # a hand-off that names a port its producer does not declare is knowable now, and knowing it
            # after the producing encode has run costs that encode
            out_ports = {p.id: p for p in registry.get(op_of[b.from_step]).outputs}
            if b.from_port not in out_ports:
                raise ChainError(
                    f"step {step.id!r}: input {b.port!r} reads {b.from_step!r}.{b.from_port!r}, but "
                    f"{op_of[b.from_step]} declares outputs {sorted(out_ports)}")
            if out_ports[b.from_port].many:
                raise ChainError(
                    f"step {step.id!r}: input {b.port!r} reads {b.from_step!r}.{b.from_port!r}, which is a "
                    f"LIST port — a binding cannot say which of its files to read (ARITY_WHY)")


def _chain_digest(chain: Any) -> str:
    """Digest only DAG shape and semantic port names; never bindings, params, URLs, paths or query data."""
    shape = [{
        "id": str(step.id),
        "op": str(step.op),
        "needs": sorted(str(value) for value in step.needs),
        "inputs": [{
            "port": str(binding.port),
            "from_step": str(binding.from_step) if getattr(binding, "from_step", None) is not None else None,
            "from_port": str(binding.from_port) if getattr(binding, "from_port", None) is not None else None,
            "external": getattr(binding, "url", None) is not None or getattr(binding, "path", None) is not None,
        } for binding in step.inputs],
        "outputs": [{
            "port": str(binding.port),
            "many": binding.urls is not None,
            "durable": binding.url is not None or binding.urls is not None,
        } for binding in step.outputs],
    } for step in chain.steps]
    encoded = json.dumps(shape, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _timeline_spans(
        *, pod_clock_id: str | None, chain_start_ns: int, chain_end_ns: int,
        steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten nested measurement into independently placeable, explicitly parented intervals."""
    clock = str(pod_clock_id or "")
    spans: list[dict[str, Any]] = [{
        "span_id": "chain",
        "scope": "chain",
        "lane": "pod",
        "op": "ops",
        "leg": "chain",
        "pod_clock_id": clock,
        "start_mono_ns": chain_start_ns,
        "end_mono_ns": chain_end_ns,
    }]
    handler_lanes = {
        "connect": "transport", "body": "transport",
        "seek_decode": "compute", "encode": "compute",
    }
    phase_lanes = {
        "bind": "transport", "slot_wait": "scheduler", "run": "compute", "put": "transport"}
    for step in steps:
        sid, op = str(step["id"]), str(step["op"])
        for leg, interval in step["intervals"].items():
            spans.append({
                "span_id": f"step:{sid}:{leg}",
                "scope": "phase",
                "lane": phase_lanes[leg],
                "parent": "chain",
                "step": sid,
                "op": op,
                "leg": leg,
                "pod_clock_id": clock,
                **interval,
            })
        occurrences: dict[str, int] = {}
        for interval in step["handler_legs"]:
            leg = str(interval["name"])
            ordinal = occurrences.get(leg, 0)
            occurrences[leg] = ordinal + 1
            spans.append({
                "span_id": f"step:{sid}:handler:{leg}:{ordinal}",
                "scope": "handler_leg",
                "lane": handler_lanes.get(leg, "compute"),
                "parent": f"step:{sid}:run",
                "step": sid,
                "op": op,
                "leg": leg,
                "pod_clock_id": clock,
                "start_mono_ns": int(interval["start_mono_ns"]),
                "end_mono_ns": int(interval["end_mono_ns"]),
            })
        for put in step["puts"]:
            object_id = str(put["object_id"])
            object_fields = {
                "object_id": object_id,
                "direction": str(put["direction"]),
                "port": str(put["port"]),
                "index": int(put["index"]),
            }
            wait = put["wait"]
            spans.append({
                "span_id": f"step:{sid}:put:{object_id}:wait",
                "scope": "put_wait",
                "lane": "transport_wait",
                "parent": f"step:{sid}:put",
                "step": sid,
                "op": op,
                "leg": "wait",
                "pod_clock_id": clock,
                **object_fields,
                **wait,
            })
            for attempt in put["attempts"]:
                attempt_n = int(attempt["attempt"])
                wire = attempt["wire"]
                spans.append({
                    "span_id": f"step:{sid}:put:{object_id}:wire:{attempt_n}",
                    "scope": "put_wire",
                    "lane": "transport",
                    "parent": f"step:{sid}:put",
                    "step": sid,
                    "op": op,
                    "leg": "wire",
                    "pod_clock_id": clock,
                    **object_fields,
                    "attempt": attempt_n,
                    "outcome": str(attempt["outcome"]),
                    **wire,
                })
                if (retry_interval := attempt.get("retry")) is not None:
                    spans.append({
                        "span_id": f"step:{sid}:put:{object_id}:retry:{attempt_n}",
                        "scope": "put_retry",
                        "lane": "transport_retry",
                        "parent": f"step:{sid}:put",
                        "step": sid,
                        "op": op,
                        "leg": "retry",
                        "pod_clock_id": clock,
                        **object_fields,
                        "attempt": attempt_n,
                        **retry_interval,
                    })
    return spans


def _terminal_timeline_context(
        cp: Any, *, corr_id: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Take one terminal-near clock sample without making measurement part of the product result."""
    sync_event: dict[str, Any] = {
        "stage": "ops",
        "status": "step",
        "phase": "timeline_sync",
    }
    if corr_id is not None:
        sync_event["corr_id"] = corr_id

    reasons: list[str] = []
    try:
        if not cp.send_event(sync_event, wait=True):
            reasons.append("terminal_clock_sync_unavailable")
    except Exception as exc:  # noqa: BLE001 - measurement may not change a successful work result
        reasons.append("terminal_clock_sync_unavailable")
        log(f"ops: terminal clock sync unavailable ({type(exc).__name__})")
    try:
        context = cp.timeline_context(str(corr_id or ""))
    except Exception as exc:  # noqa: BLE001 - measurement may not change a successful work result
        context = {"complete": False, "incomplete_reasons": ["delivery_context_unavailable"]}
        log(f"ops: terminal timeline context unavailable ({type(exc).__name__})")
    return context, reasons


# A checked, non-exception signal: pack.activate() would raise on a second pack, turning a still-claimed
# chain into a lost result the box cannot afford — the caller must build no terminal for this sentinel.
RESTART_REQUIRED: Final[object] = object()


def run_chain(chain: Any, cp: Any, corr_id: str | None = None,
              session_id: str | None = None) -> dict[str, Any]:
    """Execute the whole chain. Returns {step_id: {port: str(path)}}, or RESTART_REQUIRED when this process
    activated a different ops-pack than the chain names. corr_id/session_id echo onto every event/terminal."""

    def _event(**payload: Any) -> None:
        payload.setdefault("job_id", chain.job_id)
        payload.setdefault("stage", "ops")
        if corr_id is not None:
            payload["corr_id"] = corr_id
        if session_id is not None:
            payload["session_id"] = session_id
        cp.send_event(payload)

    @contextmanager
    def chain_phase(name: str):
        started = time.monotonic()
        _event(status="step", op="ops", phase=f"{name}_started")
        try:
            yield
        except BaseException as exc:
            _event(
                status="step",
                op="ops",
                phase=f"{name}_error",
                outcome="error",
                error_type=type(exc).__name__,
                error=safe_error(exc),
                timings={"phase_s": round(time.monotonic() - started, 3)},
            )
            raise
        else:
            _event(
                status="step",
                op="ops",
                phase=f"{name}_finished",
                outcome="ok",
                timings={"phase_s": round(time.monotonic() - started, 3)},
            )

    # The clock the box cannot read. Started BEFORE preflight and the pack fetch on purpose: everything from
    # here to the terminal is the pod's own, and what the box's wall holds beyond it is transport and queue
    # (STEP_TIMING_WHY). `timings` on every list.append below is safe unlocked — `append` is atomic under the
    # GIL, and a lock here would be a wait with no deadline anyone could state.
    t_chain = time.monotonic()
    chain_start_ns = time.monotonic_ns()
    timings: list[StepTiming] = []

    with chain_phase("preflight"):
        preflight_chain(chain)      # FIRST: an unrunnable chain must cost nothing but the claim
    with chain_phase("pack_activate"):
        activated = pack.activate_or_mismatch(chain.pack)
    if activated is None:
        # Still no terminal: the chain stays durably claimed, and a fresh incarnation of this same
        # process will see it replayed and run it under the pack it actually names.
        return RESTART_REQUIRED

    tmp = Path(tempfile.mkdtemp(prefix="opchain_"))
    ws = Workspace(tmp)
    produced: dict[str, dict[str, Any]] = {}
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
    xcap, xwhy = transport_cap()
    log(f"ops parallel cap={cap} ({why}) · transport cap={xcap} ({xwhy})")
    _event(job_id=chain.job_id, stage="ops", status="step", phase="capacity", op="ops",
           capacity={"step_workers": cap, "transfer_workers": xcap})
    try:
        with cf.ThreadPoolExecutor(max_workers=executor_workers(cap, xcap)) as ex:
            running: dict[cf.Future[Any], str] = {}
            while pending or running:
                with lock:
                    done_ids = set(produced)
                    ready = [sid for sid in pending if not (deps[sid] - done_ids)]
                for sid in ready:
                    pending.discard(sid)
                    # ContextVars carry the claimed corr into an op's LLM call without a process-wide env
                    # race.  Each future gets its own snapshot because sibling steps may run concurrently.
                    ctx = contextvars.copy_context()
                    running[ex.submit(
                        ctx.run, _run_step, by_id[sid], ws, produced, timings, _event)] = sid
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
                               error=f"{by_id[sid].op}: {safe_error(e)}"[:500])
                        failed.add(sid)
                        _drop_dependents_of_failed()
                        continue
                    _event(job_id=chain.job_id, stage="ops", status="error",
                           step=sid, error=f"{by_id[sid].op}: {safe_error(e)}"[:500])
                    raise
                with lock:
                    produced[sid] = outs
        # PER-STEP SECONDS RIDE THE TERMINAL, in one additive key. Not folded into `steps` (a list of ids
        # that both sides already know the shape of) — a new key is dropped by an older control plane with a
        # 202 and ignored by an older box, while a changed element type would be a break on a field that
        # already crosses. `chain_s` is the pod's OWN wall: the box subtracts it from its own to get the
        # transport it has never been able to see (STEP_TIMING_WHY).
        chain_end_ns = time.monotonic_ns()
        terminal_timings = {
            "chain_s": round(time.monotonic() - t_chain, 3),
            "steps": [t.wire() for t in list(timings)],
        }
        context, timeline_reasons = _terminal_timeline_context(cp, corr_id=corr_id)
        timeline_reasons.extend(str(reason) for reason in context.get("incomplete_reasons", []))
        step_rows = [
            timing.timeline_wire(job_id=str(chain.job_id), corr_id=str(corr_id or ""))
            for timing in list(timings)
        ]
        for row in step_rows:
            timeline_reasons.extend(str(reason) for reason in row["incomplete_reasons"])
        pod_clock_id = context.get("pod_clock_id")
        terminal_timeline = {
            "schema_version": 1,
            "pod": {
                "complete": bool(context.get("complete")) and not timeline_reasons,
                "incomplete_reasons": sorted(set(timeline_reasons)),
                "pod_clock_id": pod_clock_id,
                "attempt_id": context.get("attempt_id"),
                "chain_digest": _chain_digest(chain),
                "delivery": context.get("delivery"),
                "sync": list(context.get("clock_sync", [])),
                "spans": _timeline_spans(
                    pod_clock_id=pod_clock_id,
                    chain_start_ns=chain_start_ns,
                    chain_end_ns=chain_end_ns,
                    steps=step_rows,
                ),
            },
        }
        _event(job_id=chain.job_id, stage="ops", status="ok", phase="work_finished",
               outcome="ok", steps=sorted(produced), skipped=sorted(failed), timings=terminal_timings,
               timeline=terminal_timeline)
        cp.send_result({
            "job_id": chain.job_id,
            "stage": "ops",
            "status": "ok",
            "corr_id": corr_id,
            "timings": terminal_timings,
            "timeline": terminal_timeline,
            # additive, like chain_s: an older control plane drops the key with a 202. Without it the box
            # KNOWS a retain happened but not WHERE — and the engine refuses the cut rather than letting
            # five later readers 404 (op_chains.require_retained_worker).
            **({"retained_on": held} if (held := terminal_retained_on(timings)) else {}),
            **({"session_id": session_id} if session_id is not None else {}),
        })
        return {sid: {p: ([str(x) for x in v] if isinstance(v, list) else str(v))
                      for p, v in outs.items()} for sid, outs in produced.items()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
