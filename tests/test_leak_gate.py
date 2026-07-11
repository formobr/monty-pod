"""Secret scan over every git-tracked file. This is a PUBLIC repo — nothing that
looks like a live credential may ever land in it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
MAX_SIZE = 1_000_000

PATTERNS = [re.compile(p) for p in (
    r"sk-ant-[A-Za-z0-9-]{8,}",
    r"sk-or-v1-[A-Za-z0-9]{8,}",
    r"gsk_[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"ghp_[A-Za-z0-9]{36}",
    r"github_pat_[A-Za-z0-9_]{22,}",
    r"AIza[0-9A-Za-z_-]{35}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
)]

_FALLBACK_EXCLUDE_DIRS = {".git", ".venv", "dist", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = ""

    files = [REPO_ROOT / line for line in out.splitlines() if line.strip()]
    if files:
        return files

    # Nothing committed yet (or git unavailable): walk the tree with the same excludes.
    fallback = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(REPO_ROOT).parts
        if any(part in _FALLBACK_EXCLUDE_DIRS or part.endswith(".egg-info") for part in parts):
            continue
        fallback.append(path)
    return fallback


def test_no_secrets_in_tracked_files() -> None:
    hits: list[str] = []

    for path in _tracked_files():
        if path.resolve() == THIS_FILE:
            continue  # this file contains the patterns themselves
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_SIZE:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in PATTERNS:
                if pattern.search(line):
                    rel = path.relative_to(REPO_ROOT)
                    hits.append(f"{rel}:{lineno}: matched /{pattern.pattern}/")

    assert not hits, "secret-like strings found:\n" + "\n".join(hits)
