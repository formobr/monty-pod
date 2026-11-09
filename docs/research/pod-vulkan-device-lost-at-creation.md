---
covers: [podagent/ops/runner.py, the engine defect-class registry, vulkan, VK_ERROR_DEVICE_LOST]
sources:
  - https://raw.githubusercontent.com/FFmpeg/FFmpeg/n6.1.1/libavutil/hwcontext_vulkan.c
  - https://raw.githubusercontent.com/FFmpeg/FFmpeg/n6.1.1/libavutil/vulkan.c
  - https://github.com/KhronosGroup/Vulkan-Docs/blob/main/chapters/devsandqueues.adoc
  - https://github.com/KhronosGroup/Vulkan-Docs/blob/main/chapters/fundamentals.adoc
  - https://github.com/KhronosGroup/Vulkan-Docs/blob/main/xml/vk.xml
verified: "PROD 2026-11-08 (MISC-1035): one warm pod (RTX 4060 Ti, boot Vulkan preflight green via the synthesized EGL ICD) failed 8 final renders of one user over 12 h and was never retired. Two per-op camera.apply probe refusals carried «probe: stderr=[AVHWDeviceContext …] Device creation failure: VK_ERROR_DEVICE_LOST | Device creation failed: -542398533.»; the probe argv is the FIXED VULKAN_PROBE (lavfi testsrc 256x256 + libplacebo 256x256, the engine camera step:115-120, _gpu_available :530) — no job input. One assemble_master single-pass render failed with «Device creation failure: VK_ERROR_DEVICE_LOST» (ffmpeg rc 187). The same pod also timed out 4 camera.apply bakes at their 140 s budget. Neither the pool classifier nor the pod latch matched (only «Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER» is a vulkan_false pattern)."
critics:
  - {family: orchestrator, verdict: PASS-WITH-CONDITIONS, receipt: "reviewer 2026-11-09: same pattern on both sides (registry vulkan_false + pod runner _INFRA_CLASSES); positive test built with the real camera_apply refusal code with DEVICE_LOST in stderr, bare VK_ERROR_DEVICE_LOST negative added; the assemble_master positive is text-only — the render path is NOT classified (pod _run_render sends work_finished stage=render, pool pod_frame_failure_text accepts only run_error / infer work_finished) → separate paired slice + board card; state the self-inflicted-hang caveat and how a repeat across distinct offers is noticed"}
  - {family: codex, verdict: PASS-WITH-CONDITIONS, receipt: "codex-review 2026-11-09: valid for probe refusals; «never job-caused» overstates — earlier/concurrent workloads or a transient reset could retire usable hardware; effect statement only for matched ops probe refusals; explicitly exclude assemble recovery from this slice and record the transport gap; refresh line refs against pinned revisions"}
---

# A lost Vulkan device at device creation is a pod capability fault

Amends vulkan-run-time-device-loss.md (which kept only the INCOMPATIBLE_DRIVER instance pattern). Board MISC-1035.

## Findings
1. FFmpeg n6.1.1 hwcontext_vulkan.c: `vk->CreateDevice(...)` (l.1354); on `ret != VK_SUCCESS` it logs
   `"Device creation failure: %s\n", ff_vk_ret2str(ret)` (l.1362-1363). Instance creation is a separate call site
   (`CreateInstance` l.790, `"Instance creation failure: %s"` l.794-795). `ff_vk_ret2str` stringifies the enum
   (libavutil/vulkan.c:34-36; `CASE(VK_ERROR_DEVICE_LOST)` l.47). So the exact text names vkCreateDevice → DEVICE_LOST.
2. Khronos: VK_ERROR_DEVICE_LOST — «The logical or physical device has been lost» (fundamentals.adoc:1335); vk.xml lists
   it among vkCreateDevice error codes (l.14852); devsandqueues.adoc «Lost Device»: «In some cases, the physical device may
   also be lost, and attempting to create a new logical device will fail, returning VK_ERROR_DEVICE_LOST. This is usually
   indicative of a problem with the underlying implementation, or its connection to the host» (l.3456-3459). Typical
   causes listed: execution timing out, power management, platform resource management, implementation errors (l.3437-3443).
3. Our argv: every site passes the literal pair `-init_hw_device vulkan` with no options, before any job input
   (the engine camera step:459 bake, :115-120 probe; pod render.py:38/725; render_onepass.py:671; main.py:60). The probe that
   failed has no job input at all, so a DEVICE_LOST there is strong evidence that this pod cannot provide Vulkan — not
   proof of hardware damage: an earlier or concurrent workload (ours) or a transient reset could also leave the device lost.
   Caveat: if this class repeats across DISTINCT offers, suspect our own workload; there is no systemic breaker for this
   class yet (the egress class has one) — a repeat shows as several quarantined offers with this reason in the pool log.
4. Pattern kept narrow: `Device creation failure: VK_ERROR_DEVICE_LOST\b`. Not added: bare `VK_ERROR_DEVICE_LOST` (a
   queue-submit device loss mid-filter could in principle be provoked by invalid usage — spec: «not guaranteed»),
   `Device creation failed: -542398533` (ffmpeg_hw.c prints it for every backend), and timeouts / `GPU … unavailable`
   (any probe failure). Class stays `vulkan_false` (capability unavailable), no claim of hardware damage.
5. Effect, ONLY for ops-chain failures that arrive as run_error (the camera.apply probe refusal): the pod latch
   (podagent/ops/runner.py `_INFRA_CLASSES`) stops the pod claiming and emits infra_fault; the pool classifier
   (`classify_infra_error` over the engine defect-class registry) condemns the pod and the job is re-run once on another pod —
   the existing MISC-998 path, unchanged.
6. NOT covered (transport gap, separate slice MISC-1036): the single-pass assemble_master render path. The pod's
   `_run_render` reports `phase=work_finished, stage=render, outcome=error`, never run_error, and never calls the latch; the
   pool's `pod_frame_failure_text` accepts only run_error or an infer work_finished. A DEVICE_LOST there is matched as TEXT
   only and still neither retires the pod nor re-runs the job.
