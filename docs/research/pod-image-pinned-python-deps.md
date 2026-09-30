---
covers:
  - "Dockerfile"
  - "constraints.txt"
  - "pod image Python dependencies: every pip resolve constrained to the exact versions of the last image that served correctly (constraints.txt, pip -c)"
sources:
  - "measured 2026-09-30 on the laptop: `python3 -m pip freeze` inside ghcr.io/formobr/monty-pod:1346c99d… (the image that served for days; pod clore/2214844 held its stream with 1-2 lease losses over hours) vs ghcr.io/formobr/monty-pod:ee3505e1… (built 2026-09-30; source diff vs 16cbd40e: one apt line, libegl1): 8 Python packages drifted because pod-agent/Dockerfile:80 resolves them unpinned at every build - websockets 17.0.1 -> 17.1, pydantic 2.13.4 -> 2.13.5 (pydantic_core 2.46.4 -> 2.46.5), urllib3 2.7.0 -> 2.8.0, idna 3.18 -> 3.20, charset-normalizer 3.5.1 -> 3.5.2, regex 2026.7.19 -> 2026.9.29, tqdm 4.70.0 -> 4.70.1"
  - "measured 2026-09-30: image 16cbd40e… (built 2026-09-30 02:20, served by pod clore/2214844 for 288 min with 1-2 lease losses until the api restart) still had websockets 17.0.1 and the other 7 packages at the 1346c99d versions; ee3505e1 (built ~09:00 the same day) is the first image with the drifted set - so the drift, websockets 17.1 among it, arrived with ee3505e1, the image every looping pod ran"
  - "prod api/pool logs 2026-09-30 (researcher, box-local CEST): every pod on the new images (clore/2215647, 2215682, 2215689) reopened its stream every ~10-15 s within minutes of its first claim - each new socket replayed frames the api had already COMMITTED (the pod never settled their ACKs) and the pool retired it as stream_lease_loss; the same code on the old image did not"
  - "https://websockets.readthedocs.io/en/stable/project/changelog.html — 17.1 (2026-08-26) changes the threading (sync) implementation the pod's EventStream uses: «Added support for reconnecting automatically by using reconnect() as an iterator», «connect() now follows redirects», «connect() can connect to another host and port than those specified in the URI», «Connections are now garbage collected immediately once closed»"
  - "pod-agent/Dockerfile:80 installs `transformers==4.57.6 opencv-python-headless numpy 'requests>=2.32,<3' 'urllib3>=2,<3' pydantic huggingface_hub soundfile Pillow jsonschema websockets` (mostly unpinned); pyproject says websockets>=13; :68 torch/torchaudio from the cu128 index unpinned"
critics:
  - {family: claude, verdict: GO, receipt: "orchestrator session 2026-09-30, reviewer role: re-ran pip freeze in both images (exactly 8 packages moved, torch/nvidia unchanged); dry-ran both pip lines with --ignore-installed -c <freeze>: the cu128 index still serves the frozen torch/nvidia set and the app-deps line resolves the frozen set"}
  - {family: codex, verdict: GO, receipt: "codex exec gpt-5.6-terra high 2026-09-30: evidence supports a reproducibility pin; -c on both resolver lines; harmless on the --no-deps line"}
---
# Pod image: every Python dependency pinned to the last image that worked

A rebuild of the pod image that changed one apt line silently moved 8 Python packages, websockets 17.0.1 -> 17.1 among
them (17.1 reworks the threading client the pod's event stream uses), and from that image on every pod dropped and
reopened its stream every 10-15 s. Whatever the exact line inside websockets, the class is ours: the image resolves
its Python dependencies fresh at every build. The fix pins them: `constraints.txt` = the full `pip freeze` of the last image that served correctly (16cbd40e…, identical to 1346c99d… for these packages), without the self line `monty-pod-agent @ file:///app`, applied with `-c constraints.txt` to every pip line in the Dockerfile (the
torch line and the app-deps line), so a rebuild reproduces the exact versions and a version moves only by an explicit
edit of constraints.txt.
