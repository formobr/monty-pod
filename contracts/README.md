# contracts — the render seam

The pod is a dumb executor on rented GPUs. Everything that crosses this seam is **data, never
code**: the planning side decides, writes a `spec.json`, the pod applies it. This directory is the
SSOT for that seam — JSON Schema (draft 2020-12) + golden examples + a `validate.py` tripwire.
Consumers on both sides mirror these schemas and re-run the same goldens against their mirrors.

## The six schemas

| schema | direction | what |
|---|---|---|
| `pod_job.schema.json` | CP→pod (`GET /pod/job`) | the job envelope: `type` dispatches to `request` (infer) or `spec` (render) |
| `spec.schema.json` | planner→pod (object storage) | the render instruction: inputs, timeline (EDL+speed), motion keyframes, overlays (final only), encode, outputs |
| `infer_request.schema.json` | planner→pod (CP job queue) | one BATCHED inference task: `align`, `face_probe` or `clip_rank` |
| `infer_result.schema.json` | pod→CP (`POST /pod/infer-result`) | completion envelope; payload already PUT to storage |
| `face_probe.schema.json` | payload (object storage) | raw face boxes + frame_diff per shot, pixel space |
| `clip_rank.schema.json` | payload (object storage) | per-group SigLIP cosines + L2-normalized image embeddings |

Transport: the planner and the pod NEVER talk directly. Requests ride the control-plane job queue
(the pod polls `GET /pod/job`), payloads ride presigned URLs, completion is reported to the CP.

## The job envelope (frozen, v1)

`pod_job.schema.json` is the exact shape the pod's poll loop dispatches on — frozen, not
transitional. `{"type": "infer", "request": {...}}` or `{"type": "render", "spec": {...}}`;
`additionalProperties: false` and the `type`-conditional `allOf` make the other block a hard
error. It has no version const of its own — `request`/`spec` each pin their own
(`infer_version`/`spec_version`); `contracts/VERSION` stays the single shared pin (see
Versioning below) since the envelope is additive, not a new seam.

Transport conventions (frozen alongside the envelope):

- `GET /pod/job` — long-poll; `204` = no work, poll again.
- Auth — `Authorization: Bearer <JOB_TOKEN>` on every request. The pod's entire runtime config
  is `CP_URL` + `JOB_TOKEN` (env); the pod dials out only, nothing dials in.
- `POST /pod/event` — free-form progress/error events (stage, status, ...).
- `POST /pod/infer-result` — completion envelope for `kind=infer` jobs (`infer_result.schema.json`).
- `result_key` — the presigned `put_url`'s path with the leading slash stripped. OPAQUE and
  informational only: under path-style presigning the URL path starts with the bucket name, so
  the value may be bucket-prefixed. Consumers must locate the payload by the key they presigned
  `put_url` for — never by parsing `result_key`.

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
  raw boxes back. `clip_rank` = both SigLIP towers plus the cosine, numbers back — the reorder, the
  relevance floor and the MMR dedup are the planner's. One batched call per kind, never per-segment.

## align payload (binary, not JSON-schema'd)

The pod PUTs a single `.npz`:

- `emissions_<i>`: float32 `[frames, vocab]` — log-softmax CTC emissions for `windows[i]`.
- `meta.json` (stored as an npz string entry): `{"model": "<hf id>", "sr": 16000,
  "frame_stride_s": 0.02, "vocab": ["<pad>", ...]}` — `vocab` pins the checkpoint's token order so
  alignment targets can never silently shift against a re-baked image.

## clip_rank payload (JSON, `clip_rank.schema.json`)

`groups[i]` answers `clip_rank.groups[i]` of the request, and within a group `scores[j]`/`embeds[j]`
answer `image_urls[j]` — position IS the join, nothing is reordered.

- `scores[j]` — cosine(image, intent). `-1.0` means "no score to give": an image the pod could not
  fetch or decode, or an embed-only group.
- `embeds[j]` — the L2-normalized image embedding (image↔image cosine is then a plain dot, which is
  what the planner's MMR anti-repeat runs on); `null` for an image that never decoded.
- `intent: ""` — **embed-only**. The image tower is text-independent, so the group still yields
  embeddings and every score comes back `-1.0`. Bailing out instead would blind a caller that asked
  ONLY for embeddings.
- Both towers and the cosine run inside one fp16/no_grad block, so a group is a single forward.
- A dead image is data, not a fault: it is scored `-1.0`/`null` and the batch completes. Only a
  failure that invalidates the whole call raises.

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
