# monty-pod — dumb render & inference executor

A worker for rented GPU boxes. It dials out to a control plane, receives
fully-resolved render specs and batched inference tasks as **data**, and
returns artifacts via presigned URLs.

The image is deliberately **thin**: runtime only (CUDA base + torch + ffmpeg),
no model weights. Rented boxes wipe the image between rents, so every gigabyte
baked in is a gigabyte re-pulled before any work starts. **Model weights are
just another input** — a job that needs a checkpoint carries a presigned tar
for it (`InferRequest.weights`), and the pod caches it on local disk by content
hash, so a warm pod pays for each checkpoint exactly once and never pays for
one it does not use.

It makes zero editing decisions. Every number in every job was decided
upstream by the planner; the pod just executes it. See `contracts/README.md`
for the exact seam.

## Run

```
docker run --gpus all \
  -e CP_URL=https://control-plane.example \
  -e JOB_TOKEN=... \
  ghcr.io/formobr/monty-pod:latest
```

`CP_URL` and `JOB_TOKEN` are the **entire** runtime configuration. The box
holds no other credentials — auth to everything else (media storage, model
weights) rides in as presigned URLs from the control plane. `WEIGHTS_CACHE`
(default `/var/cache/monty/weights`) is where fetched checkpoints land; point
it at the roomiest local disk.

Warm pods also keep two bounded, local-only caches. An immutable reflink/copy snapshot is both the successful
object PUT body and the input-cache entry (`OPS_INPUT_CACHE`, off with `OPS_INPUT_CACHE_OFF=1`), so later
workspace mutation cannot make R2 and the cache disagree. Same-object PUT+adoption is serialised; a failed
PUT preserves the prior value, while a successful PUT whose adoption fails invalidates it so no stale bytes
can be served. Readers get independent workspace leases, never an evictable cache path. `media.scale` is
the sole result-cache allowlist entry (`OPS_RESULT_CACHE`, default 12 GB, `OPS_RESULT_CACHE_MAX_GB`, off with
`OPS_RESULT_CACHE_OFF=1`): its key includes input bytes, canonical params, suffix, op and pack versions, pod
image, and ffmpeg build. Hits are checksum-verified, copied to the run-owned output, uploaded normally, and
reported as `cache_hit`; no cache object is shared across pods or stored in R2.

Jobs, events, and results share one typed WebSocket. Before a client frame is
sent it is fsynced to `POD_STREAM_OUTBOX` (default
`/var/cache/monty/pod-stream/outbox.json`). Unacknowledged frames replay after
reconnect or process restart with their original `stream_id` and `seq`; the
agent refuses to start if this durable state is unreadable or unwritable. One
ordered sender pipelines at most `POD_STREAM_ACK_WINDOW` frames (default 32,
hard cap 256), settles them from the durable head only, and reconnects with
only the suffix that has not received a terminal ACK. Contract v12 stamps every
wire attempt with `client_send_mono_ns`; the ACK returns server receive/send Unix
nanoseconds, so the pod stores offset bounds rather than inventing one shared
clock. Each pushed job is durably answered by a `job_ack` carrying its exact
claim `attempt_id` and pod receive boundary. A successful N-step ops chain emits
`5N+6` event frames; its `job_ack`, result and durable `result_acked` boundary
make `5N+9` total client frames on the transport.

An ops terminal also carries an additive `timeline`: chain and phase intervals
on one pod monotonic clock, repeated handler legs, and every PUT permit wait,
wire attempt and retry. Existing second totals are unchanged. Timeline rows use
semantic object IDs derived only from run/correlation/step/direction/port/index;
presigned URLs, query strings and workspace paths never enter the receipt. If a
recorder or clock sample is unavailable, the work still succeeds but the
timeline is explicitly incomplete and cannot be used as a performance baseline.

The WebSocket structure and its v12 version come from the deterministic
`vendor/monty-contracts/wire_bundle.json`; `podagent/wire_generated.py` is regenerated from that bundle.
Only cross-field identity/clock rules remain handwritten in `podagent/stream_models.py`. This shared-wire
version is independent of `contracts/VERSION=5`, which still owns only render, inference and op payloads.

At boot, the agent reports its diagnostic beacon, proves that `h264_nvenc`
actually opens on the rented host, then synchronously sends and waits for ACK of a typed
`boot/ready` event. Capacity advertising and job admission happen only after
that barrier. A failed capability probe reports a bounded, secret-safe stderr
head and tail synchronously and exits as a deliberate refusal, not an unclean
agent death. The boot diagnostic also records the injected `nvidia-smi` device
name and driver independently of Torch, so provider metadata and container reality
can be compared without treating the diagnostic itself as an admission verdict.

## Layout

| path | what |
|---|---|
| `contracts/` | the render/inference seam — SSOT (JSON Schema + goldens), consumed by both sides |
| `vendor/monty-contracts/` | pinned deterministic Go↔pod wire bundle used to generate EventStream fields |
| `podagent/` | the agent: control-plane client, align/face-probe/clip-rank inference, spec renderer, weight fetch+cache |
| `Dockerfile` | the thin runtime image (no weights, no browser) |
| `tests/` | contract-mirror goldens + a secret-scan gate |

## Design rules

- **Data, not code.** Every job the pod receives is a fully-resolved
  `RenderSpec` or `InferRequest` — plain JSON, no thresholds, no rationale, no
  prompts. `additionalProperties: false` makes a leaked planning knob a hard
  schema error on both sides (declared opaque render props — `sections[].props`,
  `cover.logo/elements/headline` extras — are the exceptions, by design).
- **Keyless pod.** No API keys, no long-lived credentials live on the box.
  The job token is short-lived and scoped to one job's storage URLs.
- **One batched inference call per kind.** `align`, `face_probe` and `clip_rank`
  each run once over a whole request's windows/shots/groups, never per-segment.
- **Every number is a number.** No `"auto"`, no sentinels — the planner
  resolves every decision before it crosses the seam.

## Status

**v2**: single-pass timeline+motion render (preview tier), plus `align`,
`face_probe` and `clip_rank` batched inference. Final-tier compositing raises
`NotImplementedError` for now.
