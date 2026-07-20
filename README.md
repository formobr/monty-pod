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

## Layout

| path | what |
|---|---|
| `contracts/` | the render/inference seam — SSOT (JSON Schema + goldens), consumed by both sides |
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
