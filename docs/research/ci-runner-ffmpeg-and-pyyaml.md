---
covers:
  - "pyproject.toml"
  - ".github/workflows/ci.yml"
  - "CI test job: apt ffmpeg on the ubuntu-24.04 runner; pyyaml as a dev dependency so the guard test can yaml.safe_load the workflow"
sources:
  - "https://raw.githubusercontent.com/actions/runner-images/main/images/ubuntu/Ubuntu2404-Readme.md — the ubuntu-24.04 runner image (OS 24.04.5 LTS) lists no ffmpeg; matches the CI failure `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'` (runs 37445428053, 37517280159)"
  - "https://docs.github.com/en/actions/how-tos/manage-runners/github-hosted-runners/customize-runners — example step `sudo apt-get update` then `sudo apt-get install jq`; «Always run `sudo apt-get update` before installing a package. In case the apt index is stale ...»"
  - "https://launchpad.net/ubuntu/noble/+source/ffmpeg — noble (24.04) publishes ffmpeg 7:6.1.1-3ubuntu5 (universe); the Debian/Ubuntu `ffmpeg` binary package ships both ffmpeg and ffprobe"
  - "Dockerfile:107-129 — the pod image pins BtbN ffmpeg n6.1.3 (6.1 line) and names host ffmpeg 6.1.1-3ubuntu5 as the reference that holds the acrossfade join; the runner's apt ffmpeg is the same 6.1.1-3ubuntu5 build as the laptop gate"
  - "https://pypi.org/project/PyYAML/ — latest 6.0.3, requires_python >=3.8"
  - "https://pyyaml.org/wiki/PyYAMLDocumentation — «yaml.load ... is as powerful as pickle.load»; safe_load «supports only standard YAML tags ... does not construct class instances»"
verified: "2026-10-13 laptop: `ffmpeg -version` = 6.1.1-3ubuntu5 (the gate that passed); `pip download --only-binary=:all: --python-version 3.11 pyyaml` resolves pyyaml-6.0.3-cp311-cp311-manylinux2014_x86_64 (binary wheel, no compile on the runner); same for cp312 (the laptop .venv)"
critics:
  - {family: orchestrator, verdict: PASS, receipt: "orchestrator session 2026-10-13, reviewer role, 3 rounds: r1 BLOCK (pyyaml edit pending; step-level if/continue-on-error/exit/function bypasses), r2 PASS-WITH-CONDITIONS (apt flag allow-list, step-key allow-list name/id/run, non-ASCII spacing), r3 PASS; unpinned pyyaml in the dev extra approved as proposed"}
  - {family: codex, verdict: PASS, receipt: "codex exec v0.156.1 gpt-6-astra 2026-10-13, 4 rounds: r1-r3 BLOCK (simulate/download-only flags, inherited defaults.run.shell, pytest-before-install via `pytest;`/paths/quotes, \\v hiding the install in a comment), r4 PASS (20 extra mutations rejected; session 01a1344d-0754-70c0-b571-477cee522f4e); unpinned pyyaml judged sound"}
---
# CI installs the ffmpeg the tests need; pyyaml so the guard can parse the workflow

**Problem.** Tests spawn `ffmpeg`; the GitHub ubuntu runner has none, so CI `test` failed on every push since
2026-10-06 and the `image` job (needs: test) never ran. The local gate passed because the laptop has ffmpeg.

**Settings written.**
- `.github/workflows/ci.yml`, job `test`: `runs-on: ubuntu-24.04` (pinned so the apt ffmpeg stays on noble's 6.1.1
  line, the same line the pod image and the laptop use), and a step before `pytest -q`:
  `sudo apt-get update` / `sudo apt-get install -y ffmpeg` — the shape GitHub's runner docs give. No version pin on
  the apt package: noble publishes one ffmpeg build today (7:6.1.1-3ubuntu5). Pinning the runner release keeps the
  upstream 6.1.1 line; it does NOT freeze future Ubuntu package revisions (a -3ubuntu6 security rebuild could land) —
  accepted, because the tests exercise the CPU filter path, not NVENC, and a pinned apt version would break the
  step the day the archive supersedes it. A setup-ffmpeg action is not used: GitHub's docs show apt for runner packages.
- `pyproject.toml` `[project.optional-dependencies].dev`: add `pyyaml` (dev-only, not in the pod image) — the guard
  test `tests/test_ci_installs_what_tests_need.py` must `yaml.safe_load` the workflow (a text grep counted a
  commented-out install as installed: v1 BLOCK). safe_load, never load: the docs call load pickle-equivalent.
  Unpinned like the other dev deps (`pytest`, `jsonschema`); safe_load's API is stable across PyYAML 5.1-6.0.3.

**Guard scope (critic rounds 1-3).** The guard accepts an install step only when every executable line of its `run`
is `apt-get update` or `apt-get install <flags> ffmpeg` with flags from a fixed allow-list (`-y --yes -q -qq --quiet
--no-install-recommends`; `-s`, `--dry-run`, `--download-only`, `--assume-no` install nothing and are refused), words
split on space/tab only (a non-breaking space does not normalise to a space), the step's keys are a subset of
`name`/`id`/`run` (no `if`, `continue-on-error`, `shell`, `env`, `working-directory`), neither the workflow nor the
job sets `defaults.run.shell` or `env.BASH_ENV` (orchestrator r3 nit), install lines are printable ASCII split on newline only (a VT/CR/FF cannot hide the install in a comment), and a later step mentions `pytest` while no earlier step does (any substring: `.venv/bin/pytest`, `"pytest"`, `python -m 'pytest'` all count — conservative).
Out of scope: a hidden pytest run inside a script before the install step (it fails loudly in CI, not silently).
