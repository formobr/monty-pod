---
covers: [podagent/finalize.py, podagent/render_onepass.py, registry/encode.yaml, "-g"]
sources:
  - https://ffmpeg.org/ffmpeg-codecs.html#Codec-Options
  - https://support.google.com/youtube/answer/1722171
  - https://developers.tiktok.com/doc/content-posting-api-media-transfer-guide
verified: "2026-11-17 local ffmpeg (capped), testsrc2 1080x1920@30 10 s with the exact master argv: h264_nvenc p7/hq/cq19/vbr/b:v0/maxrate12M/bufsize24M without -g -> keyframes at 0, 8.333 s (encoder default 250); same with -g 60 -> 0,2,4,6,8 s; libx264 crf21/medium/maxrate12M/bufsize24M -g 60 -> 0,2,4,6,8 s. Prod master of job b8d555e7 (2026-11-15) probed: 10 keyframes in 52.6 s, gaps up to 8.333 s. Representative content (the creators own 52 s clip upscaled to 1080x1920, NVENC master argv, no audio): no -g 69,538,324 B / 10 IDR frames; -g 60 70,180,280 B (+0.9%) / 28 IDR frames. Toolchain: ffmpeg 6.1.1-3ubuntu5, NVIDIA GeForce RTX 2060, driver 580.178.04."
critics: [{family: codex, verdict: PASS-WITH-CONDITIONS, receipt: codex-review-2026-11-17-gop}, {family: claude, verdict: PASS-WITH-CONDITIONS, receipt: reviewer-2026-11-17-gop}]
---

# Master keyframe interval (`-g`) for the terminal encode

## Question
The terminal master encode (`podagent/finalize.py` `_FINAL_GPU` = h264_nvenc, `_FINAL_CPU` = libx264) passes no `-g`.
Should it bound the keyframe interval, and to what?

## Facts
- ffmpeg generic codec option `-g` = "set the group of picture (GOP) size" (ffmpeg-codecs, Codec Options); it is the
  AVCodecContext `gop_size` both libx264 and h264_nvenc read. When unset, h264_nvenc measured at 250 frames here (8.333 s at
  30 fps; ffmpeg's NVENC wrapper default) and libx264's documented default keyint is 250 - observed/documented defaults, not a promise. -g alone does not force a
  closed GOP; the measured NVENC output puts an IDR at every interval. Scene-cut I-frames may still be added between them.
- YouTube's recommended upload encoding settings ask for "Closed GOP. GOP of half the frame rate" (i.e. ~0.5 s); a
  shorter GOP than 8.3 s is at least what the largest destination recommends.
- TikTok's Content Posting API media transfer guide lists container/codec/fps/resolution/size limits and says nothing
  about GOP; that documents no requirement, it does not prove none exists.
- Prod: the master sent to TikTok drafts for job b8d555e7 has keyframe gaps up to 8.333 s; the creator reports the TikTok
  draft preview stalls after ~1/3 and the cover picker fails. Seeking (cover picker / scrubbing) must decode from the
  previous keyframe, up to 249 frames here. Link between the stall and the GOP is an inference, not documented.

## Decision
Add `-g 60` to both master argv lists (2 s at 30 fps, 1 s at 60). Not YouTube's 0.5 s: at cq 19 with a 12M maxrate a
0.5 s GOP spends more bits on I-frames under the same VBV caps (2 s measured at +0.9% size), and 2 s already bounds a
seek to at most 59 presentation frames after an access point. The TikTok stall / cover-picker fix stays an unproven
hypothesis until an affected clip passes an upload comparison. Nothing else in the argv changes. Preview / working-source encodes are out of scope.

## Critic conditions (both PASS-WITH-CONDITIONS) and how they are met
- codex: record toolchain + commands, measure size on representative content (done, `verified`); do not overclaim the
  TikTok fix (stated as hypothesis above); 250 is a default, not universal (reworded).
- claude: the one-pass master (`podagent/render_onepass.py:678`) splices the same lists, so it gets `-g 60` too (named in
  `covers`); the engine's `registry/encode.yaml` `master` block is the hand-mirrored source of these numbers and must carry
  the GOP in its own engine change; a 60 fps source gets a 1 s GOP (accepted); commit/card must not claim the stall is fixed
  until an affected clip passes an upload comparison.
