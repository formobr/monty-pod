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

# Flags an install/update line may carry: each makes apt quieter or non-interactive, none skips the install
# (`-s`, `--dry-run`, `--download-only`, `--assume-no` all exit 0 having installed nothing, so they are not here).
_APT_FLAG = r"(?:-y|--yes|-q|-qq|--quiet|--no-install-recommends)"
# A whole line that installs ffmpeg and nothing else: no `&&`, `||`, `;`, `if`, or other prefix that might stop it.
_APT_INSTALL_FFMPEG = re.compile(rf"^(?:sudo )?apt(?:-get)? install(?: {_APT_FLAG})* ffmpeg$")
# The only other line an install step may carry. Anything else (`exit 0`, a function body, a heredoc, `if ... fi`)
# could keep the install from running, so a step holding it does not count.
_APT_UPDATE = re.compile(rf"^(?:sudo )?apt(?:-get)? update(?: {_APT_FLAG})*$")
# Any mention of pytest marks the step as running it — `.venv/bin/pytest`, `"pytest"`, `python -m 'pytest'`,
# `pytest; true`. Conservative on purpose: a false red here costs a reworded step, a false green costs CI.
_PYTEST = re.compile(r"pytest")
# An install line holds printable ASCII only: bash splits lines on \n alone, so a \v, \r or \f that Python's
# splitlines() would break on could hide the install inside a comment.
_PRINTABLE = re.compile(r"^[ -~]*$")
# The keys an install step may have. `if`/`continue-on-error` can skip it or let pytest run after it failed;
# `shell`, `env` (BASH_ENV) and `working-directory` can change what the lines do.
_SAFE_STEP_KEYS = {"name", "id", "run"}
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
        if (
            any(_APT_INSTALL_FFMPEG.match(c) for c in cmds)
            and all(_PRINTABLE.match(c) and (_APT_INSTALL_FFMPEG.match(c) or _APT_UPDATE.match(c)) for c in cmds)
            and set(step) <= _SAFE_STEP_KEYS
        ):
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
_REAL_INSTALL = "      - run: |\n          sudo apt-get update\n          sudo apt-get install -y ffmpeg"


def test_a_real_install_step_before_pytest_counts():
    assert ffmpeg_installed_before_pytest(_wf(_CHECKOUT, _REAL_INSTALL, _PYTEST_STEP))
    assert ffmpeg_installed_before_pytest(_wf("      - run: apt-get install ffmpeg  # needed by render tests", _PYTEST_STEP))


def test_a_commented_out_install_does_not_count():
    commented = "      - run: |\n          sudo apt-get update\n          # sudo apt-get install -y ffmpeg"
    assert not ffmpeg_installed_before_pytest(_wf(_CHECKOUT, commented, _PYTEST_STEP))


def test_a_yaml_comment_does_not_count():
    assert not ffmpeg_installed_before_pytest(_wf(_CHECKOUT, "      # - run: sudo apt-get install -y ffmpeg", _PYTEST_STEP))


def test_a_never_executed_install_does_not_count():
    for guarded in ("false && sudo apt-get install -y ffmpeg", "true || apt-get install -y ffmpeg",
                    "echo sudo apt-get install -y ffmpeg", "sudo apt-get install -y ffmpeg-doc"):
        assert not ffmpeg_installed_before_pytest(_wf(f"      - run: {guarded}", _PYTEST_STEP)), guarded


def test_an_install_the_step_may_never_run_does_not_count():
    install = "\n".join(_REAL_INSTALL.splitlines()[1:])
    for skipped in ("      - if: false\n        run: |\n" + install,
                    "      - if: ${{ github.event_name == 'nope' }}\n        run: |\n" + install,
                    "      - continue-on-error: true\n        run: |\n" + install,
                    "      - run: |\n          exit 0\n" + install,
                    "      - run: |\n          f() {\n" + install + "\n          }",
                    "      - run: |\n          if false; then\n" + install + "\n          fi",
                    "      - run: |\n          cat <<'EOF'\n" + install + "\n          EOF"):
        assert not ffmpeg_installed_before_pytest(_wf(skipped, _PYTEST_STEP)), skipped


def test_an_install_that_installs_nothing_does_not_count():
    for flag in ("-s", "--simulate", "-d", "--dry-run", "--download-only", "--assume-no", "--print-uris"):
        line = f"sudo apt-get install -y {flag} ffmpeg"
        assert not ffmpeg_installed_before_pytest(_wf(f"      - run: {line}", _PYTEST_STEP)), line
    assert not ffmpeg_installed_before_pytest(_wf("      - run: sudo\u00a0apt-get install -y ffmpeg", _PYTEST_STEP))
    for sep in ("\\v", "\\r", "\\f"):  # one bash line: the install is inside the comment
        step = f'      - run: "# disabled{sep}sudo apt-get install -y ffmpeg"'
        assert not ffmpeg_installed_before_pytest(_wf(step, _PYTEST_STEP)), sep


def test_an_install_step_whose_environment_can_change_it_does_not_count():
    install = "\n".join(_REAL_INSTALL.splitlines()[1:])
    for key in ("env:\n          BASH_ENV: /tmp/noop", "working-directory: /tmp", "shell: bash", "timeout-minutes: 1"):
        step = f"      - {key}\n        run: |\n" + install
        assert not ffmpeg_installed_before_pytest(_wf(step, _PYTEST_STEP)), key
    assert not ffmpeg_installed_before_pytest("defaults:\n  run:\n    shell: 'true {0}'\n" + _wf(_REAL_INSTALL, _PYTEST_STEP))
    job_defaults = _wf(_REAL_INSTALL, _PYTEST_STEP).replace(
        "    runs-on:", "    defaults:\n      run:\n        shell: 'true {0}'\n    runs-on:")
    assert not ffmpeg_installed_before_pytest(job_defaults)
    assert not ffmpeg_installed_before_pytest("env:\n  BASH_ENV: /tmp/noop\n" + _wf(_REAL_INSTALL, _PYTEST_STEP))
    job_env = _wf(_REAL_INSTALL, _PYTEST_STEP).replace("    runs-on:", "    env:\n      BASH_ENV: /tmp/noop\n    runs-on:")
    assert not ffmpeg_installed_before_pytest(job_env)
    named = "      - name: Install ffmpeg\n        id: ffmpeg\n        run: |\n" + install
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
