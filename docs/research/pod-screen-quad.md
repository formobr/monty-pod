---
covers:
  - "podagent/main.py"
  - "podagent/infer_screen_quad.py"
  - "podagent/models.py"
  - "podagent/infer_lanes.py"
  - "Dockerfile"
  - "constraints.txt"
  - "contracts/infer_request.schema.json"
  - "contracts/infer_result.schema.json"
  - "contracts/screen_quad.schema.json"
  - "the screen_quad infer kind: Florence-2 base phrase grounding -> SAM 2.1 hiera base-plus box mask -> opencv 4-corner fit of a device screen in a still picture"
  - "screen_quad"
  - "Florence-2"
  - "Florence2"
  - "SAM 2.1"
  - "Sam2"
sources:
  - "https://huggingface.co/api/models/florence-community/Florence-2-base?blobs=true — revision 00921df66db728a9ceb750f5eca43e5c203a2051, card license: mit, config model_type florence2 / Florence2ForConditionalGeneration with NO auto_map, transformers_version 4.56.1, dtype float16; model.safetensors 463 178 864 bytes sha256 62f3e696da74f8869a68ddb529a9b3e14eb25b21c592cb3dea6179bf944df6a0"
  - "https://huggingface.co/api/models/microsoft/Florence-2-base?blobs=true — revision 5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac, LICENSE file MIT (Microsoft); its config carries auto_map to remote code and does NOT load natively in transformers 4.57.6 (measured: weights unused / newly initialised, processor AttributeError image_token) — not used"
  - "https://huggingface.co/api/models/facebook/sam2.1-hiera-base-plus?blobs=true — revision b7320756a13354e7530a63935656d35b2f91a290, card license: apache-2.0 (upstream https://github.com/facebookresearch/sam2/blob/main/LICENSE Apache-2.0); model.safetensors 323 476 296 bytes sha256 2012733a0de5d03efd1bba550a2847c4551be9ef2e0d497c83074df66189f780; config model_type sam2_video (Sam2Model loads it with empty missing / unexpected / mismatched keys, measured)"
  - "https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/florence2 — task <CAPTION_TO_PHRASE_GROUNDING> + phrase; processor.post_process_generation(text, task, image_size=(W, H)) -> {task: {bboxes: [[x0,y0,x1,y1] px], labels}}; no score is returned (the transformers 4.57.6 Florence-2 processor source)"
  - "https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/sam2 — input_boxes=[[[x0,y0,x1,y1]]] in source pixels, optional input_points / input_labels; processor.post_process_masks(outputs.pred_masks.cpu(), inputs['original_sizes']) (mask_threshold 0.0, binarize True); multimask_output=True gives 3 masks + iou_scores"
  - "https://github.com/huggingface/transformers/releases/tag/v4.56.0 — Florence-2 (#38188) and SAM2 (#32317) added; the image pins transformers==4.57.6"
  - "https://github.com/opencv/opencv/blob/5.0.0/modules/geometry/include/opencv2/geometry/2d.hpp — approxPolyN(curve, nsides, epsilon_percentage=-1, ensure_convex=true) :353, convexHull :538, contourArea :822; findContours / RETR_EXTERNAL / CHAIN_APPROX_SIMPLE in modules/imgproc/include/opencv2/imgproc.hpp :3780 / :409 / :432; the image pins opencv-python-headless==5.0.0.93"
verified: >-
  2026-11-16 on a desktop RTX 2060 6 GB, driver 580.178.04, python 3.11.15 with the image's exact pins (torch 2.11.0+cu128,
  torchvision 0.26.0+cu128, transformers 4.57.6, opencv-python-headless 5.0.0.93, numpy 2.4.6, pillow 12.3.0, tokenizers 0.22.2),
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 from local snapshots, no trust_remote_code; both models load with empty missing / unexpected /
  mismatched keys (output_loading_info). (1) Load: Florence fp16 1.06 s cold / 0.49-0.52 s warm, SAM fp32 0.43 s cold / 0.17-0.18 s
  warm; unload 0.09-0.10 s. (2) VRAM: allocated resident 452 + 285 = 737 MiB, allocated peak 1166 MiB, process ≈ 1.65 GB (nvidia-smi
  sampled every ~50 ms: a lower bound). (3) The algorithm below (bench v3), one SAM image embedding per picture: 0.46-0.67 s per
  picture warm (2.5 s for the first call of a process). (4) Quality on 15 stock pictures from past runs, the device named per
  picture: 9 device faces right (phones in hand / flat / one of two side by side, tablet, iMac, monitor, laptop), 5 `none` (a phone on a
  tripod, two laptops — one box > 80 % of the frame, one mask that does not fit a quad; an occluded monitor; a monitor seen at a steep
  angle whose quad had a wrong corner), the microphone with no screen `none`; ZERO wrong devices. Earlier variants for the record:
  without the device name and without the containment rule 3 wrong devices of 15 (v2). Repeat run on the same picture: identical boxes.
  The quad covers the device FACE incl. its bezel (SAM does not separate the glass; a second box + centre-point pass gave the same region).
  Florence returns no score and a derived loc-token probability does not separate hits from misses — no score gate.
  NOT verified: the rented GPU's latency and fp16 beam-search determinism on another GPU architecture; the thresholds (0.90, 0.3, 0.8,
  1.02, 0.9) are calibrated on 15 pictures, not on an annotated set.
critics:
  - {family: orchestrator, verdict: PASS-WITH-CONDITIONS, receipt: "reviewer 2026-11-16 ~11:20 EST — 11 conditions; the re-measure it asked for (condition 1) is in `verified`, all conditions folded into «Conditions» below"}
  - {family: codex, verdict: UNAVAILABLE, receipt: "codex-review door exit 3 — quota until 12:55, 2026-11-16"}
---

# A device screen's 4 corners on the pod: the `screen_quad` infer kind

Decision record: a stock picture of a phone / laptop / monitor / tablet whose screen
shows the wrong thing gets OUR content pinned onto its screen (a perspective warp in the caller's renderer); the pin needs the screen's 4
corners as fractions of the picture. This kind finds them on the pod GPU.

**Shape: an inference kind like `cutout`** (weights delivery by WeightsRef tar + per-kind VRAM booking; an op has no weights).

**Runtime settings (the external truth this receipt pins):**
- Weights: ONE tar holding two HF snapshots at the pinned revisions — florence-community/Florence-2-base@00921df66db7 (all repo files,
  model.safetensors sha above) and facebook/sam2.1-hiera-base-plus@b7320756a133 (the model config, the preprocessor config,
  the processor config, model.safetensors; NOT the duplicate .pt) — plus a manifest config.json {florence: {hf_repo, hf_revision},
  sam: {hf_repo, hf_revision}, transformers, torch}; ≈ 790 MB. Loaded with from_pretrained(<dir>, local_files_only=True); no
  trust_remote_code, no HF access at run time (the image is HF-offline). No new pip dependency: transformers 4.57.6, opencv 5.0.0.93,
  torch 2.11 are already pinned.
- Intake: ImageOps.exif_transpose; any mode — real transparency composited onto white, then RGB; long side capped at 2048 px (PIL
  LANCZOS); the capped, upright picture's (W, H) is what Florence's post-process, the quad fractions and the answer's w / h use. A file PIL
  cannot open → `unreadable`.
- Florence: Florence2ForConditionalGeneration fp16 on CUDA + Florence2Processor.from_pretrained(dir, local_files_only=True) (explicit class,
  never AutoProcessor — the snapshot's preprocessor config carries a remote-code auto_map; NOT use_fast=False: measured, it switches the
  tokenizer too and every box disappears); its image processor is CLIPImageProcessor (768 × 768, BICUBIC, aspect not kept), tokenizer
  BartTokenizerFast. Prompt `<CAPTION_TO_PHRASE_GROUNDING>` + `<device> screen`, device ∈ {phone, tablet, laptop, monitor, tv} (an enum in
  the schema, English; no free text, no «screen» fallback). generate(max_new_tokens=256, num_beams=3, do_sample=False) on top of the
  snapshot's generation_config (no_repeat_ngram_size 3, early_stopping true, forced bos / eos) — the receipt of the kind logs the effective
  config. post_process_generation(..., image_size=(W, H)) → boxes in px.
- Box selection: drop a box > 80 % of the picture; drop a box that contains ≥ 2 other returned boxes (each ≥ 90 % inside it); Florence's
  order is not a ranking — the rest go largest first, at most 4.
- SAM: Sam2Model fp32 + Sam2Processor (local_files_only); ONE image embedding per picture (get_image_embeddings) reused by every pass;
  per box: input_boxes=[[box]], multimask_output=True → 3 masks + iou_scores; a box whose best iou_score < 0.9 is skipped; then a second
  pass with the best mask's bbox + one positive point at its median pixel.
- Quad fit per mask: largest external contour → convexHull; a hull of < 4 points, a cv2.error, or a result that is not 4 points → no quad
  (never a job error); approxPolyN(hull, 4, epsilon_percentage=-1, ensure_convex=True).reshape(-1, 2) (5.0.0 returns (1, 4, 2)); order
  TL, TR, BR, BL (angle around the centroid, roll to min(x+y)); reject a quad whose shortest edge < 5 % of its longest.
- Choice: a candidate mask has a quad with polygon-IoU ≥ 0.90 against the mask, area ≥ 0.3 × the first-pass best mask and < 80 % of the
  picture (second-pass masks also ≤ 1.02 × the best mask); the smallest candidate of the first box (largest-first) that has one wins.
  Corners are NOT clamped: fractions in [-0.25, 1.25] (the pin's warp is clipped by the picture box); a corner beyond that → that box
  gives no answer, reason `screen_cut_by_frame`. No box answers → `none` with the per-box reasons.
- VRAM: process peak ≈ 1.65 GB (placeholder; re-measure in the pod image). The cache rule of every heavy kind (podagent/main.py, cutout
  Condition 2): cached by weights sha256, dropped only by card turns (infer_lanes.release_other_kinds); screen_quad joins INFER_KINDS, the
  caches passed to release_other_kinds, KIND_VRAM_MIB, heavy_lane_mib and coresident_mib (resident 737 MiB, peak 1650 MiB placeholders) and
  the caller's VRAM mirror; the slice states the new coresident_mib number.
- Answers per item: `found` {quad, w, h, device, surface: "device_face"} | `none` {reasons} (a normal answer: the lane tries its next pick)
  | `unreadable`. Kind-level failure (CUDA missing / OOM / weights sha mismatch / manifest-vs-runtime mismatch) is a named refusal; never
  a CPU run, never a fallback. Bounds: ≤ 16 items per request; worst case ≈ 16 × (0.5 s + 4 boxes × 2 SAM decoder passes) — the job deadline
  is set from the measured per-item time on the pod.

## Conditions (first reviewer 2026-11-16, all adopted)
1. The device phrase is the reason for zero wrong picks only together with the containment rule and no «screen» fallback — re-measured
   (bench v3, `verified`); thresholds are named above.
2. No clamping; [-0.25, 1.25] bound, `screen_cut_by_frame`; the schema's quad point type is its own bounded type, not RectT (podagent/models.py).
3. The heavy-kind cache rule (sha-keyed cache, card turns), not «loaded per job, released».
4. A degenerate mask is a per-item no-quad, never a job error (hull < 4 points, cv2.error, shortest edge < 5 % of longest).
5. Intake: exif_transpose, any mode with alpha onto white, 2048 LANCZOS cap, the same (W, H) everywhere, `unreadable`.
6. Explicit Florence2Processor / Sam2Processor, local_files_only; both models load with output_loading_info and the kind refuses on any
   missing / unexpected / mismatched key (the «sam2_video to instantiate sam2» warning is expected; this check is its gate).
7. Manifest {florence: {hf_repo, hf_revision}, sam: {hf_repo, hf_revision}, transformers, torch, torchvision, dtypes (float16 /
   float32)}; the pod refuses any mismatch with the runtime; torchvision is now also needed by Sam2ImageProcessorFast (batched_nms,
   resize) — the Dockerfile comment saying «only for deform_conv2d» is updated. The weights build checks every packed file against the HF
   LFS sha256 and packs the licence texts: Microsoft's MIT LICENSE (microsoft/Florence-2-base@5ca5edf5bd01; florence-community has none)
   and the Apache-2.0 text for SAM 2.1.
8. The effective generation config is logged in the kind's receipt; determinism measured on the dev card only.
9. The phrase is a closed device enum {phone, tablet, laptop, monitor, tv} → «<device> screen» (English).
10. One SAM image embedding per picture; ≤ 4 boxes per picture, ≤ 16 items per request; the job deadline from the measured per-item time.
11. The answer says `surface: "device_face"`: the quad is the device face incl. its bezel. Decision (orchestrator, owner 2026-11-16 «решай
    сам»): the pin covers the face; any inset is the caller's look decision made on rendered frames, not a detector change.

## Amendment 2026-11-16 — the glass inside the device face
The grounded box + SAM mask is the device FACE (bezel, notch body, case). The glass is found inside it with the same weights: the picture is
warped upright by the face quad, one SAM image embedding of that face, and a box prompt shrunk from 0 to 15 % of the face in 1.25 % steps
(multimask); a candidate covers at least 30 % of the face with iou_score at least 0.7; the kept candidate has the strongest mean Sobel
gradient (3x3, grey) on its 1-px morphological boundary. Measured on 11 device pictures (phones, tablet, iMac, monitors, laptops) on a
6 GB card: the glass on 8, the whole face on 2 (a dark screen on a dark bezel; a monitor whose face quad was already wrong), a sub-window
on 1 (a laptop cut by the frame); about 590 ms per picture. Eight other approaches were measured and rejected (Florence grounding on the
rectified face returns the whole crop; Florence referring segmentation returns slivers; Hough / LSD rectangles lose rounded corners and
notches; SAM automatic mask generation finds no candidate on 4 of 11 and is 10x slower). The mask is answered as a PNG so a consumer can
composite inside the glass only; small interior holes are filled, notches at the border are kept.
