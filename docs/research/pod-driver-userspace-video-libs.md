---
covers: [podagent/main.py, podagent/driver_userspace.py, nvenc, nvdec, cuvid, libnvidia-encode, libnvcuvid, LD_LIBRARY_PATH, NVIDIA-Linux-x86_64]
sources: [https://raw.githubusercontent.com/NVIDIA/libnvidia-container/main/src/nvc_info.c, https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavcodec/nvenc.c, https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/master/include/ffnvcodec/dynlink_loader.h, https://download.nvidia.com/XFree86/Linux-x86_64/580.178.04/README/installedcomponents.html, https://download.nvidia.com/XFree86/Linux-x86_64/580.178.04/, https://download.nvidia.com/XFree86/Linux-x86_64/580.173.02/]
verified: "2026-11-06 laptop RTX 2060, host kernel driver 580.178.04, image ghcr.io/formobr/monty-pod:f00f5e3e (ffmpeg n6.1.3), docker --gpus all --memory=4g -e NVIDIA_DRIVER_CAPABILITIES=compute,utility (NO video). Baseline NVENC probe (`ffmpeg -f lavfi -i color=c=black:s=256x256:d=0.1 -c:v h264_nvenc -preset p5 -frames:v 1 -f null -`) rc=255 '[h264_nvenc] Cannot load libnvidia-encode.so.1' + 'The minimum required Nvidia driver for nvenc is 520.56.06 or newer' (the same two lines as prod monty-api log 2026-11-06 06:39:12 CET, A10 clore/123844 worker fleet-worker-506d2170: 'REFUSING work: h264_nvenc will not open on this pod — exit 255: [h264_nvenc] Cannot load libnvidia-encode.so.1 [h264_nvenc] The minimum required Nvidia driver for nvenc is 520.56.06 or newer'); h264_cuvid decode rc=255 'Cannot load libnvcuvid.so.1'; -hwaccel cuda decode rc=0 with 'Failed setup for format cuda' (SILENT software fallback). With ONLY libnvidia-encode.so.580.178.04 + libnvcuvid.so.580.178.04 (+ hand-made .so.1 symlinks) from NVIDIA-Linux-x86_64-580.178.04-no-compat32.run bind-mounted ro at /nvx and LD_LIBRARY_PATH=/nvx:/usr/local/cuda/lib64: NVENC probe rc=0, 1 s nvenc encode rc=0, h264_cuvid decode rc=0 25 frames 'CUVID capabilities ... 8 bit: supported: 1', -hwaccel cuda decode 'NVDEC capabilities' pix_fmt cuda rc=0. Same two libs from 580.173.02 on the 580.178.04 kernel: NVENC rc=187 'OpenEncodeSessionEx failed: unsupported device (2)', cuvid rc=187 'CUDA_ERROR_NO_DEVICE', kernel 'NVRM: API mismatch: the client ffmpeg has the version 580.173.02, but this kernel module has the version 580.178.04'."
critics:
  - {family: claude, verdict: PASS-WITH-CONDITIONS, receipt: "reviewer 2026-11-06: (1) _verified readlink-checks libnvidia-encode.so.1 and libnvcuvid.so.1 like ALIAS; (2) re-probe timeout bounded by the remaining cap min(120, max(10, left)); (3) NVDEC trigger has no evidence against the real hwdownload probe — scope NVENC only; (4) prepend idempotent, driver version queried once; (5) comments say graphics+video, kept ~181 MB vs KEPT_BYTES 200 MB, opticalflow left out on purpose; the Vulkan note says whether the dir was reused"}
  - {family: codex, verdict: UNAVAILABLE, receipt: "codex-review door 2026-11-06 ~11:10Z exit 3: quota until 16:15"}
---

# NVENC/NVDEC userspace from the host-exact NVIDIA .run: research receipt (draft)

Continuation of `pod-driver-userspace-from-nvidia.md` (MISC-998: Vulkan graphics userspace). Board row MISC-1026.

## 1. Who injects the video libs
- libnvidia-container `src/nvc_info.c`: `video_libs` = libvdpau_nvidia, libnvidia-encode, libnvidia-opticalflow, libnvcuvid
  (l.97-102), added only under the `video` capability (`OPT_VIDEO_LIBS`, `match_library_flags` l.780-785). `compute_libs`
  (l.81-95) carry libcuda, libnvidia-ptxjitcompiler, libnvidia-gpucomp, libnvidia-allocator.
- So a host whose container runtime does not grant `video` (our create body asks NVIDIA_DRIVER_CAPABILITIES=all; the host's
  toolkit/config decides) gives a container with CUDA but no encoder/decoder libs. That is the prod A10 shape.

## 2. What ffmpeg says
- nv-codec-headers `dynlink_loader.h`: NVENC_LIBNAME "libnvidia-encode.so.1" (l.66), NVCUVID_LIBNAME "libnvcuvid.so.1",
  CUDA_LIBNAME "libcuda.so.1" (l.64-65); `LOAD_LIBRARY` logs "Cannot load %s" (l.89-92).
- FFmpeg `libavcodec/nvenc.c` `nvenc_load_libraries` (l.303-345): CUDA functions first (a failure there returns without the
  driver line); then NVENC functions — any failure prints `nvenc_print_driver_requirement` "The minimum required Nvidia
  driver..." (l.290, 317), also printed when the driver's max NVENC API is below the build's (l.332). So the min-driver line
  is NOT evidence of an old driver when it follows "Cannot load libnvidia-encode.so.1".
- `cuInit(0) failed` (prod class, 37 refusals) is nvenc.c l.793: CUDA itself is broken on that host — the video-lib fix
  does not apply to it.

## 3. What the .run holds
- `.manifest` (580.178.04 and 580.173.02 alike): `libnvcuvid.so.<v> NVCUVID_LIB MODULE:nvcuvid`, `libnvcuvid.so.1
  NVCUVID_LIB_SYMLINK`, `libnvidia-encode.so.<v> ENCODEAPI_LIB MODULE:encodeapi`, `libnvidia-encode.so.1
  ENCODEAPI_LIB_SYMLINK`. The symlinks are NOT created by `--extract-only`; the caller makes them.
- libnvidia-encode (285464 B): SONAME libnvidia-encode.so.1, NEEDED libnvcuvid.so.1, libc, ld-linux, libdl.
- libnvcuvid (20749912 B): SONAME libnvcuvid.so.1, NEEDED libm, libc, ld-linux, libdl; dlopens libcuda.so.1, which the
  toolkit already injects under `compute`.
- README installedcomponents: "/usr/lib/libnvcuvid.so.<v>" (NVCUVID hardware decoding), "/usr/lib/libnvidia-encode.so.<v>"
  (NvEncodeAPI); "The Optical Flow library depends on the NVCUVID library, which in turn depends on the CUDA library."

## 4. Scope decided with the critic
- Trigger: ONLY the NVENC probe failing with "Cannot load libnvidia-encode.so.1". The NVDEC probe (hwdownload, -v error) was
  not shown to print "Cannot load libnvcuvid.so.1", and libnvidia-container injects encode and cuvid together (video_libs),
  so an NVDEC-only trigger has no evidence; libnvcuvid is still kept because libnvidia-encode NEEDS libnvcuvid.so.1 and the
  NVDEC probe then runs on the same LD_LIBRARY_PATH.
- libnvidia-opticalflow is left out on purpose (no op uses it).

## 5. Consequences for the design (facts, not choices)
- The version must equal the kernel module's exactly (section verified: 173.02 on 178.04 fails with API mismatch).
- `-hwaccel cuda` exits 0 while silently decoding on the CPU when libnvcuvid is absent: an NVDEC probe must test for
  hardware decode explicitly (h264_cuvid, or the absence of "Failed setup for format cuda"), never rc alone.
- The existing fetch (sha256 check, -no-compat32 first, wall/stall caps, fresh target) already downloads the archive that
  holds both libs; keeping two more files (~21 MB) in the kept set adds no download.
- The env must reach fresh subprocesses (ffmpeg children), as for Vulkan: LD_LIBRARY_PATH in os.environ before the re-probe.
