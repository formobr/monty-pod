---
covers: [Dockerfile, NVIDIA_DRIVER_CAPABILITIES, podagent/main.py, vulkan]
sources:
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
  - https://github.com/NVIDIA/libnvidia-container/blob/main/src/nvc_info.c
  - https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/internal/config/image/capabilities.go
  - https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/CHANGELOG.md
  - https://github.com/moby/moby/blob/master/daemon/devices_nvidia_linux.go
verified: "2026-11-05 on a dev card (RTX 2060, driver 580.178.04, docker 29.1.3, toolkit 1.20.1) with this image: `--gpus all` and no -e → libEGL_nvidia count 0, caps compute,utility (inherited from the CUDA base image ENV); `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all` → libEGL_nvidia, libGLX_nvidia, glcore, glvkspirv present; `--runtime nvidia` / CDI device → libraries and the NVIDIA Vulkan ICD json present regardless of caps."
critics:
  - {family: orchestrator, verdict: PASS-WITH-CONDITIONS, receipt: "reviewer 2026-11-05: safe no-regret default, not the root fix; explicit list, not all; base stage; pin with a test (set equality, no display, not re-set later); record full host evidence"}
  - {family: codex, verdict: PASS-WITH-CONDITIONS, receipt: "codex-review 2026-11-05: graphics alone is in the old-toolkit start-failure class (fixed v1.12.0); JSON injection depends on the runtime path; validate on a failing host with CUDA, NVML, NVENC/NVDEC and Vulkan; one host validates one host"}
---

# The image declares the driver capabilities it needs

## Findings
1. The CUDA base image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility`; the value REPLACES the default rather than
   adding to it (NVIDIA docs). With plain `--gpus all`, docker appends no capabilities (moby), so the image value or a
   run-time `-e` decides.
2. libnvidia-container gates libEGL_nvidia, libGLX_nvidia, libnvidia-glcore, libnvidia-glvkspirv behind `graphics`
   and libnvidia-encode / libnvcuvid behind `video` (nvc_info.c). A host library it cannot find is logged
   «missing library» and skipped; start still succeeds but usability is not guaranteed.
3. Vulkan ICD/EGL JSON injection depends on the runtime path (graphics modifier or CDI) and host files, not on this
   ENV alone. Under CDI the graphics mounts are discovered without the image-capability gate.
4. A run-time `-e` overwrites the image ENV. Renters that already pass `all` at create time are unaffected; the line
   governs only injection paths that do not consume a create-time value. It is a no-regret default and a hypothesis
   test, not a proven fix for the failing hosts (which report `caps=all` in-container yet lack libEGL_nvidia).
5. `graphics` is in the old-toolkit start-failure class («includes graphics or display», fixed in v1.12.0); renters
   already request `all` ⊇ graphics, so the line adds no new exposure.

## Exact setting the edit may write
`ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics` in the base stage after NVIDIA_REQUIRE_CUDA. No
NVIDIA_VISIBLE_DEVICES line (the base sets all). No later stage re-sets it.

## Evidence the boot refusal must carry (so the next failing host settles the cause)
Presence (yes/no) of libGLX_nvidia.so.0, libnvidia-glcore, libnvidia-glvkspirv, libnvidia-eglcore,
libnvidia-gpucomp, libnvidia-encode.so.1, libnvcuvid.so.1 in ldconfig, and the `library_path` of every ICD json
under /etc/vulkan/icd.d and /usr/share/vulkan/icd.d — placed before the truncatable library list.

  - PUBLIC repo: no engine-internal names (provider names, engine paths) in code, tests or docs.
  - monty-pod has no pytest-xdist: no -n flag.
