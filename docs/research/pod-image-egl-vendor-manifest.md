---
covers:
  - "Dockerfile"
  - "docker/10_nvidia.json"
  - "egl_vendor.d"
  - "10_nvidia.json"
sources:
  - "https://github.com/NVIDIA/nvidia-container-toolkit/issues/2128 — opened 2026-10-07, toolkit 1.20.1, driver 595.104.02: «Toolkit omits glvnd EGL vendor manifest (10_nvidia.json)» in legacy mode; workaround bind-mounts the ICD and EGL vendor manifests from the host; the reporter says libegl1 alone did not fix it"
  - "https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/internal/discover/graphics.go — the toolkit's graphics discovery targets glvnd/egl_vendor.d/10_nvidia.json (conflicts with #2128: listed on main, reported missing in the field)"
  - "https://download.nvidia.com/XFree86/Linux-x86_64/595.112.07/README/installedcomponents.html — the Vulkan ICD references libGLX_nvidia.so.0; where X11 client libraries are not available, libEGL_nvidia.so.0 should be used"
  - "https://github.com/NVIDIA/libglvnd/blob/v1.4.0/src/EGL/libeglvendor.c — GLVND scans /etc and /usr/share egl_vendor.d and dedups loaded vendor libraries by dlopen handle, so a baked file and an injected one coexist"
  - "https://github.com/nv-tlabs/GET3D/blob/master/docker/10_nvidia.json — the exact manifest content NVIDIA's own projects bake: {\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"libEGL_nvidia.so.0\"}}"
  - "prod 2026-10-27 rented pod 2359813 at the GPU provider RTX 5060 Ti driver 595.112.07: `Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER` at boot while CUDA/NVENC/NVDEC worked; image has libegl1 + libvulkan1 and no egl_vendor.d manifest"
verified: "image contents only (Dockerfile: no egl_vendor.d file today). NOT yet verified on an affected host — the condition below requires one deliberate boot of the new image on an RTX 5060 Ti / driver 595.x host before this is called fixed."
critics:
  - {family: orchestrator, verdict: PASS, receipt: "orchestrator + researcher 2026-10-27: no primary source names a Vulkan ICD change in 590/595; the documented container cause on 595.x is the missing EGL vendor manifest (#2128); the file names the library by soname, so it is driver-version independent"}
  - {family: codex, verdict: PASS-WITH-CONDITIONS, receipt: "codex-review door 2026-10-27: content/path match NVIDIA conventions; a same-path bind mount hides the baked file, /etc and /usr/share coexist (glvnd dedups by handle); keep the runtime Vulkan ICD fallback; add a final-image JSON check; confirm ffmpeg Vulkan init on an affected host"}
---
# Pod image: ship the NVIDIA EGL vendor manifest

The NVIDIA Vulkan ICD loads through libEGL_nvidia on a headless pod. libglvnd's libEGL finds the vendor library
only through an EGL vendor manifest in egl_vendor.d. nvidia-container-toolkit is supposed to inject
`10_nvidia.json`, but on the 595 driver branch it has been reported missing (#2128), and then Vulkan instance
creation fails with VK_ERROR_INCOMPATIBLE_DRIVER while CUDA works — our 2026-10-27 symptom. The image now bakes
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` with the soname-only content NVIDIA ships. It carries no driver
version, so a new driver needs no new image; an injected copy at the same path simply hides it, and one under
/etc coexists. The pod-agent's runtime Vulkan ICD fallback (the boot ladder) stays.
