---
covers:
  - "Dockerfile"
  - "pod image apt packages for headless NVIDIA Vulkan: libegl1 (pulls libglvnd0) next to the existing libvulkan1"
sources:
  - "https://gitlab.com/nvidia/container-images/vulkan/-/raw/master/docker/Dockerfile.ubuntu — NVIDIA's own Vulkan container image installs «libglvnd0 libgl1 libglx0 libegl1 libgles2 libxcb1-dev wget vulkan-utils» and sets NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility; it ships no nvidia_icd.json (the toolkit injects it)"
  - "https://github.com/containers/ramalama/pull/2932 — «The Vulkan ICD only comes in under the 'graphics' capability»; libGLX_nvidia.so.0 needs «libXext.so.6 (a DT_NEEDED of the ICD, which fails loudly)» and «libEGL.so.1 (resolved internally, which fails silently by returning NULL from vk_icdGetInstanceProcAddr)»"
  - "https://github.com/NVIDIA/nvidia-container-toolkit/issues/1952 — same symptom on toolkit 1.19.1 / driver 580.126.09: «Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0»"
  - "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html — `graphics` is «required for rendering OpenGL, EGL, and Vulkan applications»; the rented pods and the local run already pass NVIDIA_DRIVER_CAPABILITIES=all"
  - "measured 2026-09-30 on the laptop (RTX 2060, driver 580.173.02, toolkit 1.20.0) in a throwaway container of ghcr.io/formobr/monty-pod:1346c99d… with --runtime nvidia and capabilities all: the toolkit injected /etc/vulkan/icd.d/nvidia_icd.json and all .580.173.02 driver libs; VK_LOADER_DEBUG=all vulkaninfo → «Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0» → ERROR_INCOMPATIBLE_DRIVER; strace: the driver's failed open of /usr/lib/x86_64-linux-gnu/libEGL.so.1 (ENOENT); after `apt-get install --no-install-recommends libegl1` (pulls libglvnd0 1.4.0-1): vulkaninfo lists the RTX 2060 (apiVersion 1.4.312) and ffmpeg libplacebo processes 25 frames without error"
  - "prod 2026-09-30 pod clore/2214844 (RTX 5060 Ti, driver 595.84): ffmpeg libplacebo «Failed initializing vulkan device / Failed creating Vulkan device» with the same image; pod-agent/Dockerfile:20 base nvidia/cuda:12.8.1-base-ubuntu22.04, :38 installs libvulkan1 vulkan-tools and no libegl1/libglvnd0"
critics:
  - {family: claude, verdict: GO, receipt: "orchestrator session 2026-09-30, reviewer role: re-ran in a throwaway container - before ERROR_INCOMPATIBLE_DRIVER, after libegl1 vulkaninfo lists the RTX 2060 and ffmpeg -init_hw_device vulkan + libplacebo (the camera_apply.py:108,451 form) exits 0"}
  - {family: codex, verdict: GO, receipt: "codex exec gpt-5.6-terra high 2026-09-30: libegl1 is the minimal owning package on 22.04; libxext6 already present via libgtk-3-0; graphics satisfied by capabilities all"}
---
# Pod image: headless NVIDIA Vulkan needs libEGL in the image

The toolkit injects the NVIDIA Vulkan ICD (libGLX_nvidia + nvidia_icd.json) under the `graphics` capability, but
that ICD resolves libEGL.so.1 internally and silently fails vkCreateInstance when the image has none - which is
our image (nvidia/cuda base, libvulkan1 only). Result: ERROR_INCOMPATIBLE_DRIVER, libplacebo cannot create a
Vulkan device, and every libplacebo pass (camera.apply) fails on a pod. The fix is one apt package in the image,
`libegl1` (with --no-install-recommends it brings its hard dependencies: libglvnd0, libegl-mesa0, libglapi-mesa and five libxcb-* libraries - still one apt name), installed with --no-install-recommends next to libvulkan1; nothing changes in the
run flags. Verified in a throwaway container of the current image on the laptop. The driver-version difference
(580 laptop / 595 pods) does not matter here: the missing file is the image's, not the host's.
