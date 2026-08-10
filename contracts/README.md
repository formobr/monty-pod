# contracts — the render seam

The pod is a dumb executor on rented GPUs. Everything that crosses this seam is **data, never
code**: the planning side decides, writes a `spec.json`, the pod applies it. This directory is the
SSOT for that seam — JSON Schema (draft 2020-12) + golden examples + a `validate.py` tripwire.
Consumers on both sides mirror these schemas and re-run the same goldens against their mirrors.

## The six schemas

| schema | direction | what |
|---|---|---|
| `pod_job.schema.json` | CP→pod (`job` frame) | the job envelope: `type` dispatches to `request`, `spec`, or `chain` |
| `spec.schema.json` | planner→pod (object storage) | the render instruction: inputs, timeline (EDL+speed), motion keyframes, overlays (final only), encode, outputs |
| `infer_request.schema.json` | planner→pod (CP job queue) | one BATCHED inference task: `align`, `face_probe` or `clip_rank` |
| `infer_result.schema.json` | pod→CP (`result` frame) | completion envelope; payload already PUT to storage |
| `face_probe.schema.json` | payload (object storage) | raw face boxes + frame_diff per shot, pixel space |
| `clip_rank.schema.json` | payload (object storage) | per-group SigLIP cosines + L2-normalized image embeddings |

Transport: the planner and the pod NEVER talk directly. Jobs, lifecycle events, and results use one
typed EventStream WebSocket; media payloads ride presigned URLs.

## The job envelope

`pod_job.schema.json` is the exact payload inside a typed `job` frame.
`{"type": "infer", "session_id": "...", "corr_id": "...", "request": {...}}`,
`{"type": "render", "session_id": "...", "corr_id": "...", "spec": {...}}`, or
`{"type": "ops", "session_id": "...", "corr_id": "...", "chain": {...}}`;
`additionalProperties: false` and the `type`-conditional `allOf` make the other block a hard
error. It has no version const of its own — `request`/`spec` each pin their own
(`infer_version`/`spec_version`).

Transport conventions:

- `/pod/stream` — the only pod↔control-plane channel. Server frames are typed v12 `job` or `ack`;
  client frames are typed `event`, `result`, or `job_ack`, each identified by `stream_id` + monotonic `seq`.
- Every client wire attempt carries `client_send_mono_ns`; every ACK carries control-plane
  `server_recv_unix_ns`/`server_send_unix_ns`. A pushed job names its Redis `attempt_id`, replay flag and
  enqueue/claim/pre-write bounds; `job_ack` echoes that exact attempt with the pod receive boundary.
  These are bounded clock evidence, never permission to treat pod monotonic time as Unix time.
- Auth — `Authorization: Bearer <JOB_TOKEN>` on the WebSocket handshake. The pod's entire runtime config
  is `CP_URL` + `JOB_TOKEN` (env); the pod dials out only, nothing dials in.
- Client frames are fsynced before send and retired only by a matching 2xx ACK. A 4xx is moved to
  durable dead-letter; 5xx, timeout, and socket ambiguity remain replayable.
- Every job and result carries mandatory `session_id` and `corr_id`; `corr_id` is the sole routing,
  delivery, and dedupe identity. There is no FIFO-positional or event-as-result fallback.
- `result_key` is optional transport metadata. When present it must equal `corr_id`; it is never derived
  from a presigned URL. Consumers locate payload bytes by the object address they issued, not by parsing
  `result_key`.

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
- **The delivery tail is data too.** `overlays.finalize` carries the accents, logo, watermark and delivery
  loudness that turn a composite into a deliverable. Every value is a number or an enum: an accent is
  `{kind, at, intensity}`, a placement is pixels (never an ffmpeg expression), a level is a number. Brand
  assets ride as `inputs[]` — the pod has no brand profile and no asset tree, so anything not delivered as
  an input is simply not available. A disabled step is an ABSENT block, never a flag the pod interprets.
- **Inference stays dumb.** `align` = pure wav2vec2 forward, emissions come back. `face_probe` =
  raw boxes back. `clip_rank` = both SigLIP towers plus the cosine, numbers back — the reorder, the
  relevance floor and the MMR dedup are the planner's. One batched call per kind, never per-segment.

## outputs

`outputs[].kind` says what a PUT is for: `master` (the deliverable), `cover` (the standalone cover.png),
`proxy`, `cache`, and `presync` — the composite as it stood BEFORE the delivery tail. `presync` exists so
the origin can measure the finished master against a video-identical reference and attribute any A/V drift
to the tail; it is uploaded only when a `finalize` block actually ran.

## align payload (binary, not JSON-schema'd)

The pod PUTs a single `.npz`, **STORED (uncompressed)**: log-softmax float32 is near-incompressible
(measured 316.9 MB → 274.8 MB) and deflating it costs ~12 s of pod CPU per call for 13%.

- `emissions_<i>`: float32 `[frames, vocab]` — log-softmax CTC emissions for `windows[i]`.
- `meta.json` (stored as an npz string entry): `{"model": "<hf id>", "sr": 16000,
  "frame_stride_s": 0.02, "vocab": ["<pad>", ...]}` — `vocab` pins the checkpoint's token order so
  alignment targets can never silently shift against a re-delivered checkpoint.

**Column projection (`align.keep_ids`).** A CTC forced alignment reads only the target ids and blank, so
at multi56's 9913-token union vocab a 159 s source ships 316.9 MB where 1.7 MB carries the same answer.
When the request sets `keep_ids`, `emissions_<i>` has exactly those columns **in that order** and `meta`
echoes `keep_ids`; the caller scatters them back (or re-indexes its targets) and the alignment is
unchanged — the values at the columns it reads are the same floats. Absent → full vocab width, exactly as
before. `keep_ids` is a sorted, de-duplicated **set**: an alphabet, never text.

## clip_rank payload (JSON, `clip_rank.schema.json`)

`groups[i]` answers `clip_rank.groups[i]` of the request, and within a group `scores[j]`/`embeds[j]`
answer `image_urls[j]` — position IS the join, nothing is reordered. `image_cells`, when present, is an
optional parallel array of `[x,y,width,height,sheet_width,sheet_height]` crops. The agent downloads each
distinct sheet once, validates its exact declared shape, and slices cells locally; a missing/malformed cell
fails the request instead of scoring the whole sheet.

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

## weights — the model is an INPUT

Nothing heavy is baked into the image. `InferRequest.weights = {url, sha256, size?}` is a presigned GET
for a **tar of the model directory** plus that tar's digest. **Required for `align` and `clip_rank`;
forbidden on `face_probe`** (its 227 KB YuNet stays in the image) — enforced by both the schema and the
model mirrors, so a request naming no checkpoint is rejected at the seam.

The pod verifies the digest before extracting, caches under `WEIGHTS_CACHE/<sha256>/`, and points
`from_pretrained` at the extracted directory — **never at a hub id**. Two consequences worth stating:
the **cache key is the content hash**, so a revised checkpoint can never be served from a stale entry;
and the only weights a pod can load are weights the origin hashed, since it holds no hub credential.

Tar layout is not fixed: the pod locates the directory containing `config.json` (a flat model dir and an
HF-hub-shaped `<repo>/snapshots/<rev>/` tar both work), so re-exporting weights cannot silently break it.

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
