"""MISC-309: the local gate and GitHub CI must not diverge on the binaries the tests spawn.

Tests spawn `ffmpeg`. The laptop has it, the ubuntu runner does not, so from 2026-10-06 on every CI run died
(`FileNotFoundError: ... 'ffmpeg'`), the `image` job (needs: test) never ran and no pod image was built while
the local gate stayed green. This test reads the real workflow and fails locally if the CI `test` job stops
installing ffmpeg before it runs pytest.

The workflow is PARSED, not grepped (v1 BLOCK): a commented-out install line, a `false && apt-get install`
that never executes, or an install placed after pytest must not count as installed.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
TESTS = ROOT / "tests"

# The ONE install that counts: the bounded download of the pinned BtbN asset the Dockerfile uses (apt hung CI for
# 25 min and has no bound), both names read from the Dockerfile inside `run:`. The step must be exactly these lines
# in this order — anything else (`exit 0`, a function body, a heredoc, `if ... fi`, `&&`, `||`) could keep the
# install from running, so a step holding it does not count.
_DOWNLOAD_LINES = (
    r'FFMPEG_BUILD=\$\(sed -n "s/\^ARG FFMPEG_BUILD=//p" Dockerfile\)',
    r'FFMPEG_ASSET=\$\(sed -n "s/\^ARG FFMPEG_ASSET=//p" Dockerfile\)',
    r'test -n "\$FFMPEG_BUILD"',
    r'test -n "\$FFMPEG_ASSET"',
    r'curl -fL --retry 3 --max-time 300 -o (\S+) '
    r'"https://github\.com/BtbN/FFmpeg-Builds/releases/download/\$FFMPEG_BUILD/\$FFMPEG_ASSET"',
    r"mkdir (\S+)",
    r"tar -xJf (\S+) -C (\S+) --strip-components=1",
    r"sudo install -m 0755 (\S+)/bin/ffmpeg (\S+)/bin/ffprobe /usr/local/bin",
    r"ffmpeg -version",
)
_DOWNLOAD_RES = tuple(re.compile(f"^{p}$") for p in _DOWNLOAD_LINES)
# Any mention of pytest marks the step as running it — `.venv/bin/pytest`, `"pytest"`, `python -m 'pytest'`,
# `pytest; true`. Conservative on purpose: a false red here costs a reworded step, a false green costs CI.
_PYTEST = re.compile(r"pytest")
# An install line holds printable ASCII only: bash splits lines on \n alone, so a \v, \r or \f that Python's
# splitlines() would break on could hide the install inside a comment.
_PRINTABLE = re.compile(r"^[ -~]*$")
# The keys an install step may have. `if`/`continue-on-error` can skip it or let pytest run after it failed;
# `shell`, `env` (BASH_ENV) and `working-directory` can change what the lines do. `timeout-minutes` only bounds it
# (and is REQUIRED: the step that hung had none).
_SAFE_STEP_KEYS = {"name", "id", "run", "timeout-minutes"}


def _is_bounded_download(cmds: list[str]) -> bool:
    if len(cmds) != len(_DOWNLOAD_RES) or not all(_PRINTABLE.match(c) for c in cmds):
        return False
    ms = [r.match(c) for r, c in zip(_DOWNLOAD_RES, cmds)]
    if not all(ms):
        return False
    archive, = ms[4].groups()
    made, = ms[5].groups()
    tar_archive, tar_dir = ms[6].groups()
    inst_a, inst_b = ms[7].groups()
    # the archive curl wrote is the one tar reads, and tar extracts into the dir the binaries are installed from
    return archive == tar_archive and made == tar_dir == inst_a == inst_b


def _has_timeout(step: dict) -> bool:
    t = step.get("timeout-minutes")
    return isinstance(t, int) and not isinstance(t, bool) and t > 0
# The quoted program name a test hands to subprocess: "ffmpeg" or 'ffmpeg'.
_SPAWNS_FFMPEG = re.compile(r"""["']ffmpeg["']""")


def _strip_comment(line: str) -> str:
    """Drop a shell comment: `#` to end of line, when outside quotes and at a word start."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _commands(run: str) -> list[str]:
    """The executable lines of a `run:` value: comments stripped, blank lines dropped, whitespace normalised."""
    out = []
    for raw in run.split("\n"):
        # Split on what bash splits on (space, tab), not str.split(): a non-breaking space is one word to bash.
        line = " ".join(w for w in re.split(r"[ \t]+", _strip_comment(raw)) if w)
        if line:
            out.append(line)
    return out


def ffmpeg_installed_before_pytest(workflow_text: str, job: str = "test") -> bool:
    """True iff `job` has a step that installs ffmpeg and runs before the first step that runs pytest."""
    wf = yaml.safe_load(workflow_text) or {}
    test_job = (wf.get("jobs") or {}).get(job) or {}
    # `defaults.run.shell` or `env.BASH_ENV` (workflow or job) can turn every `run` into a no-op; the guard cannot
    # read through either.
    if any(((d.get("defaults") or {}).get("run") or {}).get("shell") or "BASH_ENV" in (d.get("env") or {})
           for d in (wf, test_job)):
        return False
    steps = test_job.get("steps") or []
    installed = False
    for step in steps:
        step = step or {}
        cmds = _commands(str(step.get("run") or ""))
        if any(_PYTEST.search(c) for c in cmds):
            return installed
        if _is_bounded_download(cmds) and set(step) <= _SAFE_STEP_KEYS and _has_timeout(step):
            installed = True
    return False


def _tests_that_spawn_ffmpeg() -> list[str]:
    me = Path(__file__).resolve()
    return sorted(
        p.relative_to(ROOT).as_posix()
        for p in TESTS.rglob("*.py")
        if p.resolve() != me and _SPAWNS_FFMPEG.search(p.read_text(encoding="utf-8", errors="replace"))
    )


def test_the_ci_test_job_installs_ffmpeg():
    needers = _tests_that_spawn_ffmpeg()
    assert needers, "no test spawns ffmpeg any more — drop the CI install step and this guard together"
    assert ffmpeg_installed_before_pytest(CI.read_text(encoding="utf-8")), (
        f"{len(needers)} test file(s) spawn ffmpeg (e.g. {needers[:3]}) but the `test` job in "
        f"{CI.relative_to(ROOT)} does not install it before pytest — CI dies with FileNotFoundError and no pod "
        "image is built while the local gate stays green"
    )


def _wf(*steps: str) -> str:
    body = "\n".join(steps)
    return f"jobs:\n  test:\n    runs-on: ubuntu-24.04\n    steps:\n{body}\n"


_CHECKOUT = "      - uses: actions/checkout@v7"
_PYTEST_STEP = "      - run: pytest -q"
_REAL_LINES = (
    'FFMPEG_BUILD=$(sed -n "s/^ARG FFMPEG_BUILD=//p" Dockerfile)',
    'FFMPEG_ASSET=$(sed -n "s/^ARG FFMPEG_ASSET=//p" Dockerfile)',
    'test -n "$FFMPEG_BUILD"',
    'test -n "$FFMPEG_ASSET"',
    'curl -fL --retry 3 --max-time 300 -o ffmpeg.tar.xz '
    '"https://github.com/BtbN/FFmpeg-Builds/releases/download/$FFMPEG_BUILD/$FFMPEG_ASSET"',
    "mkdir ffmpeg-dist",
    "tar -xJf ffmpeg.tar.xz -C ffmpeg-dist --strip-components=1",
    "sudo install -m 0755 ffmpeg-dist/bin/ffmpeg ffmpeg-dist/bin/ffprobe /usr/local/bin",
    "ffmpeg -version",
)


def _body(lines=_REAL_LINES) -> str:
    return "\n".join(f"          {ln}" for ln in lines)


def _step(lines=_REAL_LINES, head: str = "      - timeout-minutes: 10") -> str:
    return f"{head}\n        run: |\n{_body(lines)}"


_REAL_INSTALL = _step()
_CURL = 4


def _mutated(i: int, new: str | None) -> tuple[str, ...]:
    lines = list(_REAL_LINES)
    if new is None:
        del lines[i]
    else:
        lines[i] = new
    return tuple(lines)


def test_a_real_install_step_before_pytest_counts():
    assert ffmpeg_installed_before_pytest(_wf(_CHECKOUT, _REAL_INSTALL, _PYTEST_STEP))
    # a trailing comment is not a command
    assert ffmpeg_installed_before_pytest(_wf(_step(_mutated(8, "ffmpeg -version  # smoke")), _PYTEST_STEP))


def test_the_apt_install_that_hung_ci_does_not_count():
    apt = "      - timeout-minutes: 10\n        run: |\n          sudo apt-get update\n          sudo apt-get install -y ffmpeg"
    assert not ffmpeg_installed_before_pytest(_wf(_CHECKOUT, apt, _PYTEST_STEP))
    assert not ffmpeg_installed_before_pytest(_wf("      - run: apt-get install ffmpeg", _PYTEST_STEP))


def test_a_commented_out_install_does_not_count():
    commented = _step(_mutated(_CURL, "# " + _REAL_LINES[_CURL]))
    assert not ffmpeg_installed_before_pytest(_wf(_CHECKOUT, commented, _PYTEST_STEP))


def test_a_yaml_comment_does_not_count():
    commented = "\n".join("      # " + ln for ln in _REAL_INSTALL.splitlines())
    assert not ffmpeg_installed_before_pytest(_wf(_CHECKOUT, commented, _PYTEST_STEP))


def test_a_never_executed_install_does_not_count():
    curl = _REAL_LINES[_CURL]
    for guarded in (f"false && {curl}", f"true || {curl}", f"echo {curl}", f"{curl} || true"):
        assert not ffmpeg_installed_before_pytest(_wf(_step(_mutated(_CURL, guarded)), _PYTEST_STEP)), guarded


def test_an_install_the_step_may_never_run_does_not_count():
    body = _body()
    for skipped in ("      - timeout-minutes: 10\n        if: false\n        run: |\n" + body,
                    "      - timeout-minutes: 10\n        if: ${{ github.event_name == 'nope' }}\n        run: |\n" + body,
                    "      - timeout-minutes: 10\n        continue-on-error: true\n        run: |\n" + body,
                    "      - timeout-minutes: 10\n        run: |\n          exit 0\n" + body,
                    "      - timeout-minutes: 10\n        run: |\n          f() {\n" + body + "\n          }",
                    "      - timeout-minutes: 10\n        run: |\n          if false; then\n" + body + "\n          fi",
                    "      - timeout-minutes: 10\n        run: |\n          cat <<'EOF'\n" + body + "\n          EOF"):
        assert not ffmpeg_installed_before_pytest(_wf(skipped, _PYTEST_STEP)), skipped


def test_a_download_that_is_not_the_pinned_bounded_one_does_not_count():
    curl = _REAL_LINES[_CURL]
    variants = {
        "no -f (an HTTP error installs nothing and exits 0)": curl.replace("-fL", "-L"),
        "no --max-time": curl.replace(" --max-time 300", ""),
        "no --retry": curl.replace(" --retry 3", ""),
        "a hard-coded URL, not the Dockerfile's pin": curl.replace("$FFMPEG_BUILD/$FFMPEG_ASSET", "latest/ffmpeg.tar.xz"),
        "a smoke check that cannot fail": "ffmpeg -version || true",
    }
    for why, line in variants.items():
        idx = 8 if "-version" in line else _CURL
        assert not ffmpeg_installed_before_pytest(_wf(_step(_mutated(idx, line)), _PYTEST_STEP)), why
    for i in (0, 1, 2, 3, 7):  # a dropped read, dropped `test -n`, or dropped install
        assert not ffmpeg_installed_before_pytest(_wf(_step(_mutated(i, None)), _PYTEST_STEP)), i
    pinned_in_run = _mutated(0, "FFMPEG_BUILD=autobuild-2025-08-31-13-00")
    assert not ffmpeg_installed_before_pytest(_wf(_step(pinned_in_run), _PYTEST_STEP))
    # curl writes one file, tar reads another: nothing is installed from what was downloaded
    assert not ffmpeg_installed_before_pytest(_wf(_step(_mutated(6, _REAL_LINES[6].replace("ffmpeg.tar.xz", "x.tar.xz"))), _PYTEST_STEP))
    # non-breaking space is one word to bash
    assert not ffmpeg_installed_before_pytest(_wf(_step(_mutated(7, _REAL_LINES[7].replace("sudo install", "sudo\u00a0install"))), _PYTEST_STEP))
    for sep in ("\\v", "\\r", "\\f"):  # one bash line: the download is inside the comment
        line = '"# disabled' + sep + curl.replace('"', '\\"') + '"'
        assert not ffmpeg_installed_before_pytest(_wf(f"      - timeout-minutes: 10\n        run: {line}", _PYTEST_STEP)), sep


def test_the_install_step_must_carry_a_timeout():
    no_timeout = _step(head="      - name: Install ffmpeg")
    assert not ffmpeg_installed_before_pytest(_wf(_CHECKOUT, no_timeout, _PYTEST_STEP))
    for bad in ("0", "-1", "true", "'10'"):
        assert not ffmpeg_installed_before_pytest(_wf(_step(head=f"      - timeout-minutes: {bad}"), _PYTEST_STEP)), bad


def test_an_install_step_whose_environment_can_change_it_does_not_count():
    body = _body()
    for key in ("env:\n          BASH_ENV: /tmp/noop", "working-directory: /tmp", "shell: bash"):
        step = f"      - timeout-minutes: 10\n        {key}\n        run: |\n" + body
        assert not ffmpeg_installed_before_pytest(_wf(step, _PYTEST_STEP)), key
    assert not ffmpeg_installed_before_pytest("defaults:\n  run:\n    shell: 'true {0}'\n" + _wf(_REAL_INSTALL, _PYTEST_STEP))
    job_defaults = _wf(_REAL_INSTALL, _PYTEST_STEP).replace(
        "    runs-on:", "    defaults:\n      run:\n        shell: 'true {0}'\n    runs-on:")
    assert not ffmpeg_installed_before_pytest(job_defaults)
    assert not ffmpeg_installed_before_pytest("env:\n  BASH_ENV: /tmp/noop\n" + _wf(_REAL_INSTALL, _PYTEST_STEP))
    job_env = _wf(_REAL_INSTALL, _PYTEST_STEP).replace("    runs-on:", "    env:\n      BASH_ENV: /tmp/noop\n    runs-on:")
    assert not ffmpeg_installed_before_pytest(job_env)
    named = "      - name: Install ffmpeg\n        id: ffmpeg\n        timeout-minutes: 10\n        run: |\n" + _body()
    assert ffmpeg_installed_before_pytest(_wf(named, _PYTEST_STEP))


def test_an_install_after_pytest_does_not_count():
    assert not ffmpeg_installed_before_pytest(_wf(_CHECKOUT, _PYTEST_STEP, _REAL_INSTALL))
    assert not ffmpeg_installed_before_pytest(_wf("      - run: CI=1 pytest -q", _REAL_INSTALL))
    assert not ffmpeg_installed_before_pytest(_wf("      - run: env CI=1 pytest -q", _REAL_INSTALL))
    for early in ("pytest; true", "true && pytest||true", "(pytest)", ".venv/bin/pytest -q || true",
                  '\'"pytest" -q || true\'', "python -m 'pytest' -q || true"):
        assert not ffmpeg_installed_before_pytest(_wf(f"      - run: {early}", _REAL_INSTALL, _PYTEST_STEP)), early
    assert not ffmpeg_installed_before_pytest(_wf(_CHECKOUT, _REAL_INSTALL)), "no pytest step: nothing to guard"


def test_an_install_in_another_job_does_not_count():
    wf = _wf(_CHECKOUT, _PYTEST_STEP) + "  image:\n    runs-on: ubuntu-24.04\n    steps:\n" + _REAL_INSTALL + "\n"
    assert not ffmpeg_installed_before_pytest(wf)
