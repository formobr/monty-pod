"""The pod image never resolves a Python dependency fresh at build time.

A rebuild that only added one apt line moved 8 Python packages (websockets 17.0.1 -> 17.1 among them) and
every pod on that image looped its stream. Every pip line in the Dockerfile is therefore constrained by
constraints.txt — the freeze of the last image that served correctly — and every entry there is an exact
`==` pin. Static check of the Dockerfile; no docker build here. See
docs/research/pod-image-pinned-python-deps.md.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
CONSTRAINTS = ROOT / "constraints.txt"


def _instructions() -> list[str]:
    """Dockerfile instructions, comments dropped and `\\` continuations joined into one line each."""
    text = "\n".join(ln for ln in DOCKERFILE.read_text().splitlines() if not ln.strip().startswith("#"))
    return [ln.strip() for ln in re.sub(r"\\\n", " ", text).splitlines() if ln.strip()]


def _pins() -> dict[str, str]:
    pins = {}
    for ln in CONSTRAINTS.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        name, sep, version = ln.partition("==")
        assert sep and name and version and not re.search(r"[<>=!~;@ ]", name + version), \
            f"constraints.txt line is not an exact pin: {ln!r}"
        pins[re.sub(r"[-_.]+", "-", name).lower()] = version
    return pins


def test_every_pip_line_is_constrained_and_every_constraint_is_exact():
    pins = _pins()
    assert pins, "constraints.txt pins nothing"

    constraints_copied_in_stage = False
    pip_lines = 0
    for ins in _instructions():
        if ins.upper().startswith("FROM "):
            constraints_copied_in_stage = False  # every stage is its own filesystem: copy it again
        elif ins.upper().startswith("COPY ") and re.search(r"\bconstraints\.txt\s+/tmp/constraints\.txt\b", ins):
            constraints_copied_in_stage = True
        for cmd in re.findall(r"pip install\b[^;&]*", ins):
            pip_lines += 1
            assert re.search(r"(?:^|\s)-c\s+/tmp/constraints\.txt\b", cmd), f"pip line without -c: {cmd!r}"
            assert constraints_copied_in_stage, f"constraints.txt not COPYed into this stage before: {cmd!r}"
    # torch cu128 line, app-deps line, --no-deps app line
    assert pip_lines == 3, f"expected 3 pip install lines, found {pip_lines}"


def test_constraints_pin_the_websockets_the_stream_was_proven_on():
    assert _pins().get("websockets") == "17.0.1"
