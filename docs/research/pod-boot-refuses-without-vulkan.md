---
covers:
  - "podagent/main.py"
  - "vulkan_false"
sources:
  - "https://registry.khronos.org/vulkan/specs/latest/man/html/VkResult.html — VK_ERROR_INCOMPATIBLE_DRIVER: «The requested version of Vulkan is not supported by the driver or is otherwise incompatible for implementation-specific reasons» — an instance-creation failure of the host's driver/ICD, not of any argv"
  - "https://github.com/FFmpeg/FFmpeg/blob/n6.1.3/libavutil/error.h — AVERROR_EXTERNAL = FFERRTAG('E','X','T',' ') = -542398533 «Generic error in an external library»: the `Device creation failed: -542398533` line is ffmpeg wrapping the Vulkan failure, rc 187 is that value mod 256"
  - "prod 2026-10-27 monty-api-error.log 10:04:18 CET, worker fleet-worker-08a93feee9d (rented pod 2359813 at the GPU provider, RTX 5060 Ti, driver 595.112.07): boot frame seq=2 `WARNING VULKAN UNAVAILABLE: exit 187: … Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER …`, seq=3 ready, seq=4..7 job_accepted for a real user in the same second; both finals then failed in camera.apply"
  - "docs/research/pod-image-headless-vulkan-libegl.md (superproject) — the 2026-09-30 occurrence on the same GPU class (RTX 5060 Ti, driver 595.84), cured then by libegl1 in the image; the symptom returned on driver 595.112.07"
verified: "prod logs only (the pod was retired 2026-10-27 12:21 +03; no live repro). The change adds no ffmpeg/driver/ICD setting: it keeps the existing VULKAN_PROBE argv and 3-rank ICD ladder and only turns the ladder's exhausted outcome from a report into a refusal before ready."
critics:
  - {family: orchestrator, verdict: PASS, receipt: "orchestrator + Plan design agent 2026-10-27: the boot probe already caught this pod; no production job can run without Vulkan (camera.apply refuses unless MONTY_ALLOW_STATIC_CAMERA=1); nothing routes on capacity vulkan=false"}
  - {family: codex, verdict: PASS-WITH-CONDITIONS, receipt: "codex-review door 2026-10-27: terminal-path-only change; a failed delivery of the refusal event must still refuse (exit 3); a green fallback rank keeps its ICD env; cover no-ready, both EGL fallback successes, and the contour-dry path"}
---
# Pod boot refuses work when no Vulkan device opens

The pod-agent's boot ladder (default ICD discovery, then a synthesized EGL ICD at 1.3.0, then 1.2.0 on
INCOMPATIBLE_DRIVER) is unchanged. When every eligible rank fails, the pod now refuses before ready, with the same
shape as the NVENC and NVDEC refusals (boot error event, `_mark_stopped()`, exit 3), instead of reporting
`vulkan=false` and serving. A transient failure at boot can refuse a pod that might later recover; that is the
accepted fail-closed admission for a capability every final needs, as for the codec gates.
