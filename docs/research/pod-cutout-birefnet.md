---
covers:
  - "podagent/main.py"
  - "podagent/infer_cutout.py"
  - "podagent/models.py"
  - "podagent/infer_lanes.py"
  - "Dockerfile"
  - "constraints.txt"
  - "contracts/infer_request.schema.json"
  - "contracts/infer_result.schema.json"
  - "contracts/cutout.schema.json"
  - "the cutout infer kind: BiRefNet_dynamic background removal on the pod GPU, TorchScript fp16 at 1024, needs torchvision's deform_conv2d op"
  - "cutout"
  - "BiRefNet"
  - "torchvision"
sources:
  - "https://github.com/ZhengPeng7/BiRefNet — the model's code and README (MIT); inference recipe: resize to the model size, ToTensor, Normalize(mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]), model(x)[-1].sigmoid(), resize back"
  - "https://huggingface.co/ZhengPeng7/BiRefNet_dynamic — model card license: mit; revision 280306042f57b7a33854319da62fd86aaa89ec4c pinned here: model.safetensors 444 473 596 bytes sha256 e3d2e4884e51ff30f0cd630edc6b1e41b06b7f23a0a2a5169f7b7cb33a711c2d, https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/blob/280306042f57b7a33854319da62fd86aaa89ec4c/birefnet.py sha256 2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831, https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/blob/280306042f57b7a33854319da62fd86aaa89ec4c/BiRefNet_config.py sha256 e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52, config.json sha256 557f461de22ede5b0a2ff60967e579180194ded47587f6e30a3a09ad08ae248b. The remote code imports torchvision (vgg/resnet backbones, torchvision.ops.deform_conv2d), timm, kornia, einops, transformers"
  - "https://github.com/pytorch/vision (README compatibility table) — torch 2.11 ↔ torchvision 0.26 (Python >=3.10, <=3.14); this image pins torch==2.11.0+cu128 (constraints.txt)"
  - "https://download.pytorch.org/whl/cu128/torchvision/ — torchvision-0.26.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl exists (checked 2026-11-06); the image is python3.11 on nvidia/cuda:12.8.1"
verified: >-
  2026-11-06 on a desktop RTX 2060 6 GB (torch 2.8.0+cu128 / torchvision 0.23.0, scratch venv). (1) Quality: BiRefNet_dynamic fp16 at 1024
  vs the paid background-removal provider's cut-outs on 16 generated pictures: alpha IoU mean 0.994 / min 0.982, boundary F@2px 0.981 —
  best of 11 models in the operator's bench. (2) Cost: resident 996 MiB (process), peak 2670 MiB at 1024, 390 ms per image warm; load
  1.6-1.8 s cold, 0.45 s warm; unload 0.07-0.15 s returns the memory. (3) TorchScript: torch.jit.trace + freeze of a wrapper
  `model(x)[-1].sigmoid()` on a [1,3,1024,1024] fp16 input → a 463 442 136-byte .pt; max abs difference vs eager 0.0008. (4) Loading that
  .pt in a FRESH process with only torch imported FAILS: «Unknown builtin op: torchvision::deform_conv2d»; with `import torchvision` it
  loads and runs. timm/kornia/einops/transformers are NOT needed at run time once traced. NOT verified here: tracing under torch 2.11 (the
  artifact is built inside this image so trace and runtime match) and the rented GPU's latency.
critics:
  - {family: first-reviewer, verdict: PASS-WITH-CONDITIONS, receipt: "reviewer 2026-11-06 ~13:05 — conditions folded into «Conditions» below"}
  - {family: codex, verdict: UNAVAILABLE, receipt: "codex-review door exit 3 — quota until 16:15, 2026-11-06"}
---

# Background removal on the pod: BiRefNet_dynamic as the `cutout` infer kind

The RGBA cut of a picture (a stock photo or a generated opaque picture) is made on the pod GPU instead of a paid background-removal call.

**Shape: an inference kind** (the person_matte / clean_plate precedent): only the inference path has weights delivery (WeightsRef →
weights.ensure, content-addressed cache) and per-kind VRAM booking (infer_lanes); an op has no weights field.

**Runtime settings:**
- Weights: BiRefNet_dynamic at HF revision 280306042f57b7a33854319da62fd86aaa89ec4c (MIT), TRACED to TorchScript (fp16, input
  [1,3,1024,1024], output = the last decoder map after sigmoid, [1,1,1024,1024]) by the operator's build tool running INSIDE this image,
  shipped as a weights tar whose config.json manifest records {hf_repo, hf_revision, input, dtype, mean, std, torch, torchvision} and which
  carries the upstream MIT LICENSE text. No trust_remote_code and no HF access at run time.
- Image dependency: torchvision==0.26.0+cu128 from https://download.pytorch.org/whl/cu128 (same index and pip line as torch), only for the
  `torchvision::deform_conv2d` TorchScript op; `import torchvision` before torch.jit.load.
- Pre/post: exif_transpose; cap long side 2048 (LANCZOS); RGB; PIL Image.BILINEAR to 1024×1024 (antialiased on reduce, as benched — never
  F.interpolate without antialias); /255; Normalize(mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]); fp16 CUDA; sigmoid map resized
  back to the source size bilinear (align_corners False) = the soft alpha; RGBA PNG = source RGB + alpha; bbox at alpha ≥ 24/255 (the same
  number as the operator's still-measure threshold).
- VRAM: cached like the other heavy kinds (module cache keyed by weights sha, dropped only by card turns); KIND_VRAM_MIB, heavy_lane_mib /
  coresident_mib and the heavy set include it; the 2670 / 996 MiB figures are placeholders until the traced .pt is re-measured in this image
  at the 2048 cap.
- Named outcomes per item: an input with real transparency → `already_cut` (no model run); no alpha ≥ 24/255 → `empty_cut`; ≥ 98 % of the
  frame ≥ 24/255 → `no_separation`. The kind refuses to load when the manifest's torch/torchvision differ from the runtime's. CUDA missing /
  OOM / weights sha mismatch = a named refusal; never a CPU run, never a fallback.
