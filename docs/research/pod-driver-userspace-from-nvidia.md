---
covers: [podagent/main.py, vulkan, NVIDIA-Linux-x86_64, VK_ICD_FILENAMES, LD_LIBRARY_PATH]
sources: [https://download.nvidia.com/XFree86/Linux-x86_64/580.178.04/, https://download.nvidia.com/XFree86/Linux-x86_64/595.71.05/, https://download.nvidia.com/XFree86/Linux-x86_64/580.178.04/NVIDIA-Linux-x86_64-580.178.04.run.sha256sum, https://download.nvidia.com/XFree86/Linux-x86_64/580.178.04/README/installedcomponents.html, https://download.nvidia.com/XFree86/Linux-x86_64/latest.txt, https://raw.githubusercontent.com/NVIDIA/libnvidia-container/main/src/nvc_info.c, https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html, https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md]
verified: "2026-11-05 laptop RTX 2060, host kernel driver 580.178.04, image ghcr.io/formobr/monty-pod:b004d542..., --gpus all, NVIDIA_DRIVER_CAPABILITIES=compute,utility,video (no graphics). Baseline: ffmpeg -init_hw_device vulkan -> 'Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER', vulkaninfo -> ERROR_INCOMPATIBLE_DRIVER. With 4 libs extracted from NVIDIA-Linux-x86_64-580.178.04.run (libEGL_nvidia, libnvidia-eglcore, libnvidia-glsi, libnvidia-glvkspirv; 45 MB) bind-mounted ro at /nvx, LD_LIBRARY_PATH=/nvx, VK_ICD_FILENAMES=a manifest under the mounted lib dir (library_path=/nvx/libEGL_nvidia.so.0): ffmpeg rc=0 'Device 0 selected: NVIDIA GeForce RTX 2060', libplacebo hwupload/scale/hwdownload 3 frames rc=0, vulkaninfo driverInfo=580.178.04 apiVersion 1.4.312. GLX manifest (8-lib set) also rc=0. Same set WITHOUT LD_LIBRARY_PATH -> INCOMPATIBLE_DRIVER. Same set from 580.173.02 against 580.178.04 kernel -> INCOMPATIBLE_DRIVER + kernel 'NVRM: API mismatch: the client vulkaninfo has the version 580.173.02, but this kernel module has the version 580.178.04'."
critics:
  - {family: orchestrator, verdict: PASS-WITH-CONDITIONS, receipt: "reviewer 2026-11-05: deadlines (total wall + stall floor, timeouts on sha fetch and extract, cap under the boot budget); restore VK_ICD_FILENAMES/VK_DRIVER_FILES/LD_LIBRARY_PATH on failure; lib dir = explicit allowlist only; validate one driver version; sha before running the .run; fresh empty target, finally-cleanup; disk check; api_version from the shipped json; named reasons in refusal; public-repo gate"}
  - {family: codex, verdict: PASS-WITH-CONDITIONS, receipt: "codex-review 2026-11-05: env must reach fresh subprocesses (not in-process dlopen); same manifest in both Vulkan vars; keep versioned names, alias only libEGL_nvidia.so.0; prefer -no-compat32 with its own checksum, never latest; named failure classes; probe real libplacebo work; Blackwell needs open kernel modules (scope); one host validates one host"}
---

# NVIDIA graphics userspace extracted from the .run per host driver version: research receipt (draft)

## 1. URL layout, existence, sizes, checksums
- Pattern: `https://download.nvidia.com/XFree86/Linux-x86_64/<v>/NVIDIA-Linux-x86_64-<v>.run`
  Each version directory also contains (directory index, 580.178.04 and 595.71.05 both checked):
  `NVIDIA-Linux-x86_64-<v>.run.md5sum`, `...run.sha256sum`, `NVIDIA-Linux-x86_64-<v>-no-compat32.run`
  (+ `.md5sum`, `.sha256sum`), `README/`, `license.txt`.
- HEAD 200 + Content-Length (2026-11-05):
  | version | .run | -no-compat32.run |
  |---|---|---|
  | 580.126.18 | 396862844 | - |
  | 580.173.02 | 398312209 | 326239589 (downloaded) |
  | 580.178.04 | 397214534 | 325655993 |
  | 595.71.05 | 423133031 | 315963253 |
  | 595.91.07 | 423651201 | - |
  | 595.99.02 | 423333820 | - |
  | 595.112.07 | 423351953 | - |
  All seven host versions return 200 on the full .run.
- Checksum: the `.sha256sum` file contains `<hash>  NVIDIA-Linux-x86_64-<v>.run`. 580.178.04:
  `5975a86ee45bffcb626f51ae33d1169b108186a2ea47ad651e72f13fa4b6d6f9`; local sha256sum of the download matched it.
- Open vs proprietary: these URLs have no separate "open" or "production" path. One .run holds both kernel
  module trees (extracted dir has `kernel/` and `kernel-open/`), and the userspace libs ship once.
  `latest.txt` currently reads `595.112.07 595.112.07/NVIDIA-Linux-x86_64-595.112.07.run`.
- The `-no-compat32` variant drops the 32-bit libs and is about 70-110 MB smaller. Its 580.173.02 extraction
  contained the same 64-bit libs used in the mismatch test.

## 2. Extracting without root
- `sh NVIDIA-Linux-x86_64-580.178.04.run --help` (makeself wrapper) lists `-x, --extract-only`
  ("Extract the contents ... but do not run 'nvidia-installer'"), `--check`, `--info`. `--advanced-options`
  lists `--target [NewDirectory]` ("Extract contents in 'NewDirectory'"). `--no-kernel-modules` is an
  nvidia-installer *install* option, so it does not apply to extraction.
- Ran as an unprivileged user: `sh NVIDIA-Linux-x86_64-580.178.04.run --extract-only --target <dir>` -> exit OK, 2.28 s,
  1.5 GB extracted (full .run, with compat32). Nothing was installed. `-x --target` on the 580.173.02 no-compat32 .run gave 1.2 GB.
- `<dir>/.manifest` maps each file to a type and module, e.g. `nvidia_icd.json 0644 VULKAN_ICD_JSON icd.d/ MODULE:opengl`,
  `libEGL_nvidia.so.580.178.04 0755 OPENGL_LIB NATIVE MODULE:egl`, `libGLX_nvidia.so.0 0000 OPENGL_SYMLINK NATIVE libGLX_nvidia.so.580.178.04`.
  The `.so.0` symlinks are NOT in the extracted tree. The installer creates them, so the caller has to create them.

## 3. File set and the ICD json
- README "Installed Components" (580.178.04): "libGLX_nvidia.so.0 and libEGL_nvidia.so.0 can be used as Vulkan ICDs.
  The Vulkan ICD configuration file included with the driver references libGLX_nvidia.so.0, however in environments
  where X11 client libraries are not available, libEGL_nvidia.so.0 should be used." The ICD json goes to
  /etc/vulkan/icd.d/nvidia_icd.json and the layers json to /etc/vulkan/implicit_layer.d/nvidia_layers.json. The README
  lists libnvidia-eglcore/glcore/glsi/glvkspirv/gpucomp/rtcore/allocator as "Various libraries that are used internally".
  The README index has no dedicated Vulkan chapter (README/vulkan.html returns 404).
- Shipped `nvidia_icd.json` (580.178.04):
  `{"file_format_version":"1.0.1","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.4.312"}}`.
  `nvidia_layers.json` defines VK_LAYER_NV_optimus (libGLX_nvidia.so.0, enabled only with __NV_PRIME_RENDER_OFFLOAD=1)
  and VK_LAYER_NV_present (libnvidia-present.so.580.178.04, enabled only with NVPRESENT_ENABLE_SMOOTH_MOTION=1).
  The live test did not need either.
- ELF NEEDED (readelf -d, 580.178.04):
  - libEGL_nvidia: libpthread, librt, libc, libdl, libnvidia-glsi.so.<v>
  - libGLX_nvidia: libnvidia-glsi, libnvidia-tls, libnvidia-glcore (.so.<v>), libX11.so.6, libXext.so.6, libc, libdl
  - libnvidia-eglcore: libnvidia-glsi, libnvidia-gpucomp (+libc/libm/libdl/ld-linux)
  - libnvidia-glcore: libnvidia-tls, libnvidia-gpucomp
  - libnvidia-glsi, libnvidia-glvkspirv, libnvidia-gpucomp, libnvidia-tls: libc/libm/libdl/librt/libpthread only
  - String references (dlopen candidates) in eglcore/glcore: libnvidia-glvkspirv.so.<v>, libnvidia-rtcore.so.<v>,
    libnvidia-allocator.so.1, libcuda.so.1, libdrm.so.2, libwayland-client.so.0.
- Libraries actually initialised by vulkaninfo, EGL manifest (LD_DEBUG=libs "calling init"):
  /nvx/libEGL_nvidia.so.0, /nvx/libnvidia-eglcore.so.<v>, /nvx/libnvidia-glsi.so.<v>, /nvx/libnvidia-glvkspirv.so.<v>,
  /nvx/libnvidia-gpucomp.so.<v>, and /usr/lib/x86_64-linux-gnu/libnvidia-allocator.so.1 (injected by the toolkit).
- The toolkit already injects these under compute/utility/video: libnvidia-gpucomp.so.580.178.04, libnvidia-allocator.so.1,
  libcuda.so.1, libnvidia-ml.so.1 and others. They are not graphics libs. libnvidia-container nvc_info.c puts eglcore, glcore,
  tls and glsi in `graphics_libs`, and libGLX_nvidia, libEGL_nvidia and libnvidia-glvkspirv in `graphics_libs_glvnd`. Those
  are injected only with the graphics capability. The toolkit docs say: "`graphics` | Rendering with OpenGL, EGL, and Vulkan",
  and the default is utility,compute.
- The pod image provides libvulkan.so.1, libX11.so.6, libXext.so.6 and /usr/bin/vulkaninfo. /etc/vulkan/icd.d exists and is empty.
- Sizes (bytes, 580.178.04):
  | file | size |
  |---|---|
  | libEGL_nvidia.so.580.178.04 | 1296640 |
  | libnvidia-eglcore.so.580.178.04 | 35076592 |
  | libnvidia-glsi.so.580.178.04 | 603384 |
  | libnvidia-glvkspirv.so.580.178.04 | 9553072 |
  | libnvidia-gpucomp.so.580.178.04 | 72265336 (toolkit already injects it) |
  | libGLX_nvidia.so.580.178.04 | 1211968 |
  | libnvidia-glcore.so.580.178.04 | 37062312 |
  | libnvidia-tls.so.580.178.04 | 18632 |
  - Tested EGL set of 4 files (EGL, eglcore, glsi, glvkspirv; gpucomp from the toolkit): 45 MB on disk.
  - Full 8-lib set (EGL + GLX paths, gpucomp included): 157088232 bytes (150 MB).

## 4. Version-match requirement
- libnvidia-glsi.so.580.178.04 contains: "NVIDIA: API mismatch: the NVIDIA kernel module has version %s, but this NVIDIA
  driver component has version %s.  Please make sure that the kernel module and all NVIDIA driver components have the
  same version."
- Live test, 580.173.02 userspace (EGL, eglcore, glsi, glvkspirv, gpucomp) against the 580.178.04 kernel:
  - VK_LOADER_DEBUG: `loader_scanned_icd_add: Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD /nvx/libEGL_nvidia.so.0`
  - ffmpeg: `Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER`
  - host kernel log: `NVRM: API mismatch: the client 'vulkaninfo' (pid ...) has the version 580.173.02, but this kernel module has the version 580.178.04.`
  - Result: the patch version has to match exactly too; matching only the branch is not enough.
- Same version, mixed sources: 580.178.04 extracted EGL/eglcore/glsi/glvkspirv ran next to toolkit-injected 580.178.04
  libcuda, libnvidia-ml, libnvidia-gpucomp and libnvidia-allocator. ffmpeg Vulkan init, libplacebo and vulkaninfo all
  succeeded. I found no primary-source statement about a known issue with that mix. The only failure seen was the
  version mismatch.

## 5. Live check outputs (laptop RTX 2060, host 580.178.04)
- Command shape: `docker run --rm --memory 4g --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video -v <set>:/nvx:ro
  -e LD_LIBRARY_PATH=/nvx -e VK_ICD_FILENAMES=a manifest under the mounted lib dir --entrypoint sh ghcr.io/formobr/monty-pod:b004d54202d589841a72878a413b4a5900a28807 <probe>`
- Manifest used: `{"file_format_version":"1.0.1","ICD":{"library_path":"/nvx/libEGL_nvidia.so.0","api_version":"1.4.312"}}`
  plus a symlink libEGL_nvidia.so.0 -> libEGL_nvidia.so.580.178.04.
- Baseline (no /nvx): ffmpeg `Instance creation failure: VK_ERROR_INCOMPATIBLE_DRIVER ... Device creation failed: -542398533.`.
  vulkaninfo `vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER`.
- EGL manifest, 8-lib dir: ffmpeg rc=0, `Device 0 selected: NVIDIA GeForce RTX 2060 (discrete) (0x1f15)`, `frame=1`.
  vulkaninfo: apiVersion 1.4.312, driverID DRIVER_ID_NVIDIA_PROPRIETARY, driverInfo 580.178.04, conformanceVersion 1.4.1.3.
  Container wall time 1.63 s.
- GLX manifest (library_path=/nvx/libGLX_nvidia.so.0), 8-lib dir: identical result, 1.41 s.
- EGL manifest, 4-lib dir: ffmpeg rc=0, device selected, driverInfo 580.178.04.
- libplacebo: `-init_hw_device vulkan=vk -filter_hw_device vk -f lavfi -i testsrc2=s=320x240 -frames:v 3 -vf
  format=yuv420p,hwupload,libplacebo=w=160:h=120:format=yuv420p,hwdownload,format=yuv420p -f null -` -> rc=0, frame=3.
- 4-lib dir with VK_ICD_FILENAMES set but without LD_LIBRARY_PATH (absolute library_path) -> VK_ERROR_INCOMPATIBLE_DRIVER.
  The libs' NEEDED sonames are resolved through the linker search path, so an absolute library_path alone does not work.
- Vulkan-Loader docs: VK_ICD_FILENAMES is "deprecated", "replaced by VK_DRIVER_FILES" (loader >= 1.3.207), "still active
  and supported", and ignored when running with elevated privileges. Absolute paths are recommended.

## 6. Cost
- Download of the full 580.178.04 .run (397 MB) on this laptop: 44.5 s. The no-compat32 variant is about 326 MB.
- `--extract-only`: 2.28 s, 1.5 GB extracted (full) or 1.2 GB (no-compat32).
- Minimal set kept: 45 MB (4 files, EGL path) or about 117 MB if libnvidia-gpucomp is carried too.
  The full 8-lib set is 150 MB.
