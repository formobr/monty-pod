# monty-pod — dumb render & inference executor

A worker for rented GPU boxes. It boots ready — everything is baked into the
image, no setup step, no cold-start download — dials out to a control plane,
receives fully-resolved render specs and batched inference tasks as **data**,
and returns artifacts via presigned URLs.

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
downloads) rides through the control plane or was baked in at build time.

## Layout

| path | what |
|---|---|
| `contracts/` | the render/inference seam — SSOT (JSON Schema + goldens), consumed by both sides |
| `podagent/` | the agent: control-plane client, align/face-probe inference, spec renderer |
| `Dockerfile` | one image, everything baked in |
| `tests/` | contract-mirror goldens + a secret-scan gate |

## Design rules

- **Data, not code.** Every job the pod receives is a fully-resolved
  `RenderSpec` or `InferRequest` — plain JSON, no thresholds, no rationale, no
  prompts. `additionalProperties: false` makes a leaked planning knob a hard
  schema error on both sides (declared opaque render props — `sections[].props`,
  `cover.logo/elements/headline` extras — are the exceptions, by design).
- **Keyless pod.** No API keys, no long-lived credentials live on the box.
  The job token is short-lived and scoped to one job's storage URLs.
- **One batched inference call per kind.** `align` and `face_probe` each run
  once over a whole request's windows/shots, never per-segment.
- **Every number is a number.** No `"auto"`, no sentinels — the planner
  resolves every decision before it crosses the seam.

## Status

**v1**: single-pass timeline+motion render (preview tier), plus `align` and
`face_probe` batched inference. Final-tier compositing raises
`NotImplementedError` for now.
