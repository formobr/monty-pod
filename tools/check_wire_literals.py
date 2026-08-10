#!/usr/bin/env python3
"""Reject hand-owned shared-wire versions and valid frame dictionaries in pod tests."""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "vendor" / "monty-contracts" / "wire_bundle.json"
GENERATED = ROOT / "podagent" / "wire_generated.py"


@dataclass(frozen=True)
class Problem:
    path: Path
    line: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.message}"


def _bundle() -> dict:
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def _fingerprints() -> tuple[tuple[str, frozenset[str]], ...]:
    schemas = _bundle()["schemas"]
    found: list[tuple[str, frozenset[str]]] = []
    for surface in ("pod_stream", "pod_stream_server"):
        schema = schemas[surface]
        for branch in schema["oneOf"]:
            definition = schema["$defs"][branch["$ref"].rsplit("/", 1)[-1]]
            found.append((surface, frozenset(definition["required"])))
    return tuple(found)


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _trusted_constructor(node: ast.Call) -> bool:
    names = {
        "PodStreamFrame", "PodStreamServerFrame", "StreamAck", "StreamEventFrame",
        "StreamJob", "StreamJobAckFrame", "StreamResultFrame",
    }
    if _call_name(node) in {"valid_wire", "invalid_wire", *names}:
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"model_validate", "model_validate_json"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in names
    )


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, version: int, fingerprints: tuple[tuple[str, frozenset[str]], ...]) -> None:
        self.path = path
        self.version = version
        self.fingerprints = fingerprints
        self.fixture_depth = 0
        self.problems: list[Problem] = []

    def visit_Call(self, node: ast.Call) -> None:
        if _trusted_constructor(node):
            self.fixture_depth += 1
            self.generic_visit(node)
            self.fixture_depth -= 1
            return
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        if not self.fixture_depth:
            keys = frozenset(
                str(key.value) for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
            for surface, fingerprint in self.fingerprints:
                if fingerprint <= keys:
                    self.problems.append(Problem(
                        self.path, node.lineno,
                        f"hand-built valid {surface} frame; use valid_wire()",
                    ))
                    break
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "Literal":
            members = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if any(isinstance(item, ast.Constant) and item.value == self.version for item in members):
                self.problems.append(Problem(
                    self.path, node.lineno,
                    "hand-owned shared-wire Literal version; import WIRE_CONTRACT_VERSION/generated fields",
                ))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if isinstance(node.value, ast.Constant) and node.value.value == self.version:
            for name in names:
                if "VERSION" in name and ("WIRE" in name or "SCHEMA" in name):
                    self.problems.append(Problem(
                        self.path, node.lineno,
                        "hand-owned numeric shared-wire version; import the generated constant",
                    ))
        self.generic_visit(node)


def problems(paths: Iterable[Path]) -> list[Problem]:
    bundle = _bundle()
    fingerprints = _fingerprints()
    found: list[Problem] = []
    for path in sorted(set(paths)):
        if path.resolve() == GENERATED.resolve():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        visitor = _Visitor(path, int(bundle["contract_version"]), fingerprints)
        visitor.visit(tree)
        found.extend(visitor.problems)
    return found


def live_paths(root: Path = ROOT) -> Iterable[Path]:
    yield from (root / "podagent").rglob("*.py")
    yield from (root / "tests").rglob("*.py")


if __name__ == "__main__":
    import sys

    found = problems(live_paths())
    for problem in found:
        print(problem.render(), file=sys.stderr)
    raise SystemExit(1 if found else 0)
