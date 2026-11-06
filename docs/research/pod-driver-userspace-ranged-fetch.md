---
covers: [podagent/driver_userspace.py, NVIDIA-Linux-x86_64, Range, accept-ranges, Content-Range, Accept-Encoding]
sources: [https://download.nvidia.com/XFree86/Linux-x86_64/595.112.07/, https://download.nvidia.com/XFree86/Linux-x86_64/595.112.07/license.txt, https://www.rfc-editor.org/rfc/rfc9110#name-range-requests]
verified: "2026-11-06: HEAD https://download.nvidia.com/XFree86/Linux-x86_64/595.112.07/NVIDIA-Linux-x86_64-595.112.07-no-compat32.run -> HTTP/2 200, accept-ranges: bytes, content-length: 316359348, server: AkamaiNetStorage, cache-control: max-age=86400; GET with Range: bytes=0-1023 -> 206, 1024 bytes. Prod stand the same day (a rented RTX 5060 Ti, host driver 595.112.07, host injected no graphics libs): driver_userspace=DOWNLOAD_TIMEOUT elapsed=110s cap=150s (wall = 150 - 40 s reserve), every 10 s window >= 1 MB/s, so the host averaged < ~2.9 MB/s on one stream. 14-day prod outcomes of the fetch: EXTRACTION_FAILED 56 (fixed by 101f73a), DOWNLOAD_TIMEOUT 2, ok 2."
critics:
  - {family: orchestrator, verdict: PASS-WITH-CONDITIONS, receipt: "reviewer 2026-11-06: do NOT raise RANK_WALL_S to 240 s — the pod has no claim during preflight, the unclaimed free-cancel wall is 555 s and the fleet median cold start is 316 s, so 316+250+25 s overruns; keep 150 s and ship parallel ranges alone; stall+wall on a supervisor timer, not per chunk; every range must be 206 with Content-Range == requested start-end/total and total == probed length (a 200 to a ranged GET fails before its body is read); probe with GET Range bytes=0-0 or HEAD allow_redirects=True; one fd + os.pwrite at offsets, never wb per thread; INSUFFICIENT_DISK on the probed length before preallocating, ENOSPC -> INSUFFICIENT_DISK; on wall/stall/failure set a stop event, close responses, join every thread with a deadline before rmtree/return; single-stream fallback only after the joins, into a fresh file name"}
  - {family: codex, verdict: PASS-WITH-CONDITIONS, receipt: "codex-review 2026-11-06: validate exact Content-Range, shared total and received byte count, reject short/long bodies; Accept-Encoding: identity and reject any content-encoding; partition [0,L) with no gap/overlap, positional writes with short-write handling; fallback once, clean, inside the ORIGINAL deadline — global stall/wall exhaustion, disk failure and checksum mismatch refuse, never restart; join writers and flush before hashing, hash incrementally with deadline checks inside the 40 s reserve; bytes=/streams= trailer on both NVENC and Vulkan refusals, bytes = current-attempt completed bytes"}
---

# Parallel ranged fetch of the host-exact NVIDIA .run: research receipt

Continuation of pod-driver-userspace-from-nvidia.md and pod-driver-userspace-video-libs.md (board MISC-1032).

## Facts
- The download host answers `accept-ranges: bytes` and serves `Range` requests with 206 (verified above). RFC 9110 §14:
  a 206 carries `Content-Range: bytes start-end/complete-length`; a server MAY ignore Range and answer 200 with the whole body.
- One stream on the slow host averaged < ~2.9 MB/s; four streams help only if the limit is per connection, not the host
  link — unproven, so the refusal trailer gains `bytes=<done>/<total> streams=<k>` to measure it on the next slow host.
- The wall stays 150 s: the unclaimed pod is cancelled at 555 s and the fleet median cold start is 316 s (pod pool constants),
  so a longer fetch would turn a slow-but-healthy boot into a cancelled one.
- The checksum is the published `.sha256sum` of the whole file; SHA-256 over the finished file equals the streaming digest
  of the same bytes. Hashing 316 MB from page cache is well under the 40 s reserve.
- License (595.112.07 license.txt §1.1(c)): «Deploy, for your own use, the SOFTWARE on infrastructure you own or lease»; the
  pod downloads from the vendor onto the leased host it runs on — unchanged by this design.
