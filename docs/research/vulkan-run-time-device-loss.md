---
covers: [podagent/ops/runner.py, vulkan, VK_ERROR_INCOMPATIBLE_DRIVER]
sources:
  - https://registry.khronos.org/vulkan/specs/latest/man/html/VkResult.html
  - https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavutil/error.h
  - https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md
  - https://raw.githubusercontent.com/FFmpeg/FFmpeg/n6.1.1/libavutil/hwcontext_vulkan.c
  - https://raw.githubusercontent.com/FFmpeg/FFmpeg/n6.1.1/fftools/ffmpeg_hw.c
verified: "2026-11-04 on the dev card (RTX 2060, driver 580.178.04, ffmpeg 6.1.1): pointing VK_ICD_FILENAMES at a missing manifest makes the libplacebo/vulkan probe fail with «Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER» then «Device creation failed: -542398533», ffmpeg rc 187 — byte-identical to the prod pod failure of 2026-11-02/03."
critics:
  - {family: codex, verdict: PASS-WITH-CONDITIONS, receipt: "codex-review 2026-11-05 round 2 (round 1 BLOCK on 4 patterns): narrow findings 1-2, cost is expected isolated-case not upper bound; no re-probe required"}
  - {family: orchestrator, verdict: PASS-WITH-CONDITIONS, receipt: "reviewer 2026-11-05: same pattern both sides; tests positive via safe_error + negatives (Device creation failed alone, NVENC exited 187 …, exited 234, GPU unavailable kind=timeout); stderr evidence first in the probe refusal (500-char sanitizer); claim probe refusal only"}
---

# A Vulkan instance the driver refuses at run time is a pod capability fault

## Revision after the cross-family critic BLOCK (2026-11-05)
The first draft proposed four patterns. The critic showed three of them are not card/driver evidence:
- `Device creation failed: -542398533` is printed by fftools/ffmpeg_hw.c (not by the Vulkan hwcontext) for EVERY hw backend (e.g. a bad
  `-init_hw_device cuda:999` ordinal gives the same text) — dropped.
- bare `VK_ERROR_INITIALIZATION_FAILED` is also returned for caller/config mistakes (loader direct-driver path) — dropped.
- `GPU (Vulkan+libplacebo+NVENC) unavailable` collapses ANY probe failure (argv, missing filter, timeout) — dropped.
Only one pattern is kept: `Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER`.

## Findings
1. hwcontext_vulkan.c (n6.1.1) prints «Instance creation failure: %s» with the VkResult name when vkCreateInstance
   fails (`.apiVersion = VK_API_VERSION_1_3` fixed by the build). Our probe and render argv deliberately reach this
   call through `-init_hw_device vulkan`, but job media and filter/encoder settings do not cause this
   instance-creation result.
2. The Vulkan loader driver-interface doc: the loader returns VK_ERROR_INCOMPATIBLE_DRIVER when it fails to find and load a
   valid Vulkan driver on the system. The requested apiVersion is fixed by our ffmpeg build, identical to the one
   the boot preflight ladder ran green on the same pod; a run-time occurrence means this process could not
   initialize the required Vulkan instance; the earlier successful preflight makes runtime availability/configuration
   differences plausible (a deleted ICD manifest is one sufficient cause, reproduced — now moved out of /tmp; not
   proven the only cause).
3. The class name stays `vulkan_false` (pod capability unavailable — the boot class); no claim of hardware damage.
4. Encoder exit codes (234, 187) alone match nothing. What this catches is the per-op Vulkan probe REFUSAL, whose
   retained stderr evidence carries the Instance-creation line (the refusal puts that evidence first so the pod's
   500-char error sanitizer cannot cut it). A bake-time ffmpeg failure is NOT caught: its error-line filter drops the
   Instance line — a miss never retires a healthy pod; widening it is a separate change.
5. Codex condition not adopted here: a fresh job-independent re-probe before retirement. The latch already ends the
   pod (drain, exit 5) and the replacement boots through the full preflight ladder; the expected isolated-case cost of a false
   retire is one pod replacement and one stage re-run (not an upper bound: the drain deadline may cut other in-flight
   chains, and a shared configuration fault can recur on replacements), the cost of a miss is client work failing on a broken card (prod MISC-998).

## Pattern the edit may write (regex, case-sensitive)
- `Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER\b`
