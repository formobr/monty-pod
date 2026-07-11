# contracts — the render seam

The pod is a dumb executor on rented GPUs. Everything that crosses this seam is **data, never
code**: the planning side decides, writes a `spec.json`, the pod applies it. This directory is the
SSOT for that seam — JSON Schema (draft 2020-12) + golden examples + a `validate.py` tripwire.
Consumers on both sides mirror these schemas and re-run the same goldens against their mirrors.

## The four schemas

| schema | direction | what |
|---|---|---|
| `spec.schema.json` | planner→pod (object storage) | the render instruction: inputs, timeline (EDL+speed), motion keyframes, overlays (final only), encode, outputs |
| `infer_request.schema.json` | planner→pod (CP job queue) | one BATCHED inference task: `align` or `face_probe` |
| `infer_result.schema.json` | pod→CP (`POST /pod/infer-result`) | completion envelope; payload already PUT to storage |
| `face_probe.schema.json` | payload (object storage) | raw face boxes + frame_diff per shot, pixel space |

Transport: the planner and the pod NEVER talk directly. Requests ride the control-plane job queue
(the pod polls `GET /pod/job`), payloads ride presigned URLs, completion is reported to the CP.

## Invariants (enforced by schema + goldens)

- **Every number is a number.** No `"auto"`, no sentinels. The planner resolves every decision
  before writing the spec.
- **No threshold carries meaning.** Editing knobs never appear — they were already applied
  upstream. The pod is told *what*, never *why*. `additionalProperties: false` makes a leaked
  knob a hard error.
- **No prompts, no scores, no rationale, no planning metadata.** Only resolved render fields cross.
- **Every media reference resolves.** `timeline.segments[].src`, `overlays.broll_final.broll[].clip`,
  `overlays.music.track` must each equal an `inputs[].id` (mirror-model validation — JSON Schema
  cannot express it).
- **Inference stays dumb.** `align` = pure wav2vec2 forward, emissions come back. `face_probe` =
  raw boxes back. One batched call per kind, never per-segment.

## align payload (binary, not JSON-schema'd)

The pod PUTs a single `.npz`:

- `emissions_<i>`: float32 `[frames, vocab]` — log-softmax CTC emissions for `windows[i]`.
- `meta.json` (stored as an npz string entry): `{"model": "<hf id>", "sr": 16000,
  "frame_stride_s": 0.02, "vocab": ["<pad>", ...]}` — `vocab` pins the checkpoint's token order so
  alignment targets can never silently shift against a re-baked image.

## Versioning

`VERSION` (plain integer) == the `spec_version` / `infer_version` consts in the schemas. Bump ALL
together on ANY change; there is no back-compat — a mismatch is a loud fail on both sides.
Goldens: every `examples/*.json` must validate, every `examples/invalid/*.json` must be rejected
(`python validate.py`).

## Clock conventions

- `timeline.segments[].in/out` — SOURCE seconds, frame-snapped at `timeline.fps`.
- `segments[].speed` — atempo/setpts factor applied to that segment.
- `motion.segments[].keyframes[].t` — seconds from that rendered segment's start, OUTPUT clock
  (post-speed). `rect` = `[x, y, w, h]` normalized to the source frame.
- `overlays.*` times (`broll.start`, `motion_plan.sections[].start`, `trims`, `cover.frame_at`) —
  FINAL output clock.
