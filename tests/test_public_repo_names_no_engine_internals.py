"""This is a PUBLIC repo: it names no engine-internal provider, brand, person, engine module or engine path
anywhere in its tracked tree. The engine runs the same check at its pod pin; this copy makes the pod refuse
its own leak at landing instead of after.

Two kinds of rule, and NO exemption list (no path, line or file is excused by name or hash):
- NAMES (provider, brand, agent, person, the engine's own repo name) are a closed list. A plain list would
  leak the very names it guards, so each is held as (first two chars, length, sha256 prefix); the two-char
  prefix only narrows where a window is hashed, the hash decides. The engine's repo name is also checked on
  a separator-stripped copy of the line, so its hyphen, underscore, joined, spaced and CamelCase spellings
  all hash the same. A handful of engine module names are held the same way, and are a leak however they
  are written: a bare import, a path into the engine's own scripts package, or a bare dotted attribute —
  `mod.fn` names a module exactly as much as `from mod import fn` does — a REFERENCE SHAPE, not python call
  syntax, decides this, so a backtick or a string literal around the name changes nothing.
- FILE REFERENCES are a RULE, not a list: any `<dir>/…/<name>.<ext>` path (whatever its top directory, a
  ticket's plan doc cited by its ticket dir included) and any bare `<name>.<ext>` citation, for a source/config/doc
  extension, is a leak UNLESS it is something this repo itself owns. What it owns is DERIVED, never typed
  by hand: the paths `git ls-files` lists (and every dir-boundary suffix of them), the file names the pod's
  own code spells as a whole string constant (a runtime file it reads or writes — `outbox.json`), the data
  files named at an operating-system location anywhere in the tree, and the handler modules its own op
  contracts declare. Two shapes are not repo paths at all and so never a
  candidate: a URL or host-led path (`https://…`, `cp.example/o/1.json`) and an absolute path under an
  operating-system root (`/tmp/…`, `/etc/…`). Describe the role instead («the engine's resolver», «the
  engine op», «the engine's queue plan»).

vendor/ is scanned like every other path: the full NAME list, the MODULE shapes and the FILE rule, with no
skip, no narrowed list and no hashed list of excused lines — not even for a value that is a schema
`enum`/`const` member of the vendored bundle itself. What the pod vendors from the producer is therefore
trimmed to the two surfaces it actually consumes (`pod_stream`, `pod_stream_server`, per
vendor/monty-contracts/README.md); a surface this pod never reads is never vendored in the first place,
so its prose and its closed vocabulary never reach this tree to need an exception.

Two SHAPES below carry an opaque, caller-chosen string that is conventionally path-shaped by design, never
a citation — and each is narrowed a different way, never by field name alone:
- a MEDIA-asset wire field (`id`/`asset`/`clip`/…) is narrowed by the matched file's own EXTENSION: it is
  recognized only when that file is a media/data payload, never a source/config/doc one, so it works
  anywhere in the tree (this repo's own fixtures build specs with it far outside contracts/ or vendor/).
- a PROVENANCE field (`source_path`/`generated_by`/…) and the bundle's own `source_sha256` hash-manifest
  are narrowed by LOCATION instead: recognized only on a JSON/YAML line already inside contracts/ or
  vendor/, never in a .py or .md file anywhere, so neither can be used to wave an engine path through
  ordinary code or prose.

This file is scanned like every other file; it reads nothing outside this repo.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
MAX_SIZE = 1 << 20

# A NAME is a leak wherever it appears, as a case-insensitive substring. This mirrors the engine's own
# deny-names (hashed — see the module docstring).
_NAMES: tuple[tuple[str, int, str], ...] = (
    ("cr", 13, "09f40b04503b6fe8"), ("ne", 8, "89877c5726fdc819"), ("ви", 5, "6e9d47cd62406572"),
    ("ру", 6, "217b01db1c752838"), ("ru", 6, "fa0777bf024e9b7f"), ("re", 16, "befe179fa551c92a"),
    ("du", 13, "6770fd1b72133e85"), ("pi", 13, "cbb32b037c32f4af"), ("tw", 8, "0315dc6e935ecfbb"),
    ("op", 10, "7f871cbf905f6e0c"), ("de", 8, "f8849d864257947c"), ("an", 9, "c70eca6b0f88f44d"),
    ("cl", 6, "c857d09db23e6822"), ("te", 8, "3f40462915a3e602"), ("t.", 4, "cdf01902dc74865b"),
    ("ru", 6, "31d1b3d540ecf852"), ("cl", 5, "9e7ea5b45eeefced"), ("ne", 6, "9b29d4791aab15a6"),
    ("ma", 9, "2d94f8d9bd18c0ff"), ("ne", 9, "eb64b6b15a2e866a"), ("sl", 6, "a2e7b25e3c545f45"),
)
# Pod-side-only: the engine's own repo name, hashed in its JOINED form and matched against the line with
# every separator stripped (_joined) — so `a-b`, `a_b`, `ab`, `a b`, `a.b` and `AB` are one name. Naming the
# engine BY NAME is exactly what this gate exists to refuse, so this repo forbids it locally even though the
# engine's list does not carry it.
_JOINED_NAMES: tuple[tuple[str, int, str], ...] = (
    ("vi", 11, "e2c6f5323bad9592"),
)
_SEPARATORS = re.compile(r"[\s_.\-]+")
# An engine MODULE is a leak only when referred to AS a module (an import, a path into the engine's own
# scripts package, a directory named X, or a bare dotted attribute) — a BARE word with no path, no dot and no
# extension names nothing by itself (a local variable spelled like one, e.g. `head_frame = next(...)`,
# names nothing of the engine's) and is not flagged; a `<name>.<ext>` FILE citation is the separate rule
# below instead.
_MODULES = frozenset((
    "6893290451c7faf3", "45e168db28dad734", "6cf676ec36e42f5c", "c399472034b3617b", "4c759eb0af1c1ddb",
    "23e22b442878df6e", "6a0d80e0ffbfc6b9", "0fa895451d1f178b", "34446dc5929c9dc8", "ccb538d7e031d1c2",
    "952b306c4be606fa", "0d77ecdcced74b4b", "b2f7f26289c74cdf", "d9ad3cac9c1a6bd2",
    "8f26d101dad2b641", "8eff21ef6f91ba33", "6503dfe19e08dc57", "2f974e90e6c2e153", "95dfe95dfe375cb2",
    "be51ed885eda040e", "db010deac94eb2b5", "ebd3682953b049f2", "f2e9e54ba788953c", "baaba080e6ce66db",
    "288b9dbc8ce71488",
))
_MODULE_REF_SHAPES = (
    re.compile(r"\b(?:from|import)\s+\(?\s*((?:[\w.]+\s*,\s*)*[\w.]+)"),
    re.compile(r"\bscripts[./](\w+)"),
    re.compile(r"\b(?:__import__|import_module)\(\s*['\"]([\w.]+)"),
    # A bare dotted attribute access (`mod.fn`, inside backticks, a string, or plain prose — the shape is
    # lexical, not syntactic): group 1 is the first segment only, so a chain like `a.b.c` is found through
    # its first dot same as a plain `a.b`.
    re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_]*\b"),
)
# FILE REFERENCES. The extensions are a category ("a source, config or doc file a repo keeps"), not a list
# of anyone's names. A BARE name (no slash) stays source/config/doc only — a bare `master.mp4` is a dynamic
# OUTPUT name, not a citation of a tracked location, and flagging it was noise with nothing to fix. A full
# MULTI-SEGMENT path is a different claim: a path citing a specific directory and file IS a location
# citation whatever its extension — a real engine fixture under a media extension names a real location
# exactly as much as a `.py` module under the same directory would — so the slash-path shape ALSO covers
# common media/data payload exts,
# EXCEPT inside a JSON/YAML DATA file: this wire format's own `inputs[].id` / `asset` fields are opaque,
# caller-chosen identifiers that are conventionally path-shaped BY DESIGN (`broll/clip-01.mp4`,
# contracts/spec.schema.json `id`) — never a citation of a tracked location — so a media/data extension
# there is data, not a path, while a source/config/doc extension (this repo's own prose embedded in the
# same JSON, a `description` naming a real file) is scanned exactly as elsewhere.
_EXTS = "py|pyi|md|yaml|yml|json|jsonl|toml|tsx|ts|jsx|js|sh|go"
_MEDIA_EXTS = "mp4|mov|mkv|wav|mp3|png|jpg|jpeg|webp|gif|woff2?|ttf"
_FILE = rf"[\w-][\w.-]*\.(?:{_EXTS})(?![\w-])"
_PATH_FILE = rf"[\w-][\w.-]*\.(?:{_EXTS}|{_MEDIA_EXTS})(?![\w-])"
_DOTFILE = r"\.[\w-]+(?:\.[\w-]+)*"
# group 1: a leading `/` or `~/` (absolute), group 2: the repo-relative-looking path itself. The lookbehind
# keeps a match from starting mid-token or mid-URL (`://host/…` never yields a candidate). Two variants:
# the DATA one (JSON/YAML) stays extension-narrow; the other also matches media/data payload extensions.
_SLASH_PATH_REF = re.compile(
    rf"(?<![\w.:/~-])(~?/)?((?:\.{{1,2}}/)*(?:[\w.-]+/)+(?:{_FILE}|{_DOTFILE}))")
_SLASH_PATH_REF_MEDIA = re.compile(
    rf"(?<![\w.:/~-])(~?/)?((?:\.{{1,2}}/)*(?:[\w.-]+/)+(?:{_PATH_FILE}|{_DOTFILE}))")
_DATA_EXTS = frozenset(("json", "jsonl", "yaml", "yml"))
_BARE_FILE_REF = re.compile(rf"(?<![\w/.~-])(\.?{_FILE})")
_WHOLE_FILE_NAME = re.compile(rf"^\.?{_FILE}$")
# A dotted-or-slash member reference into a `scripts` package names no single file with an extension (and
# may carry whitespace around a prose slash), so it is its own shape: group 1 is the member name. The
# trailing lookaheads exclude a path with MORE segments after it, or a recognized extension right after it
# — either means the slash-path shape above already owns this match, extension and all.
_SCRIPTS_MEMBER_REF = re.compile(rf"\bscripts\s*[./]\s*(\w+)\b(?!/)(?!\.(?:{_EXTS})\b)")
# An import of the `scripts` package proper names no single file either: group 1 is the imported names
# (comma-split), present only for the `from scripts import …` form.
_SCRIPTS_IMPORT_REF = re.compile(
    r"\bfrom\s+\.*scripts\s+import\s+\(?\s*((?:\w+\s*,\s*)*\w+)"
    r"|\bimport\s+scripts\b()")
# A name built fresh for ONE throwaway test file (`tmp_path / "whatever.py"`) names nothing of the engine's
# — it is a disposable fixture, not a citation — so it is read off the source text and excused by name,
# the same way a real citation is read off the source text and flagged by name.
_TMP_FIXTURE_REF = re.compile(rf'tmp_path\s*/\s*"({_FILE})"')
# The wire SPEC's own `inputs[].id` / staged-input-id / overlay `clip`/`track`/`sound` fields, and the
# pinned vendor bundle's own manifest fields (`source_path`, `key`, `result_key`, `corr_id`, `schema`,
# `generated_by`, `$id`), are closed, PUBLIC wire vocabulary — named in the open in the schema that owns
# each one (contracts/spec.schema.json's own prose for the pod spec; the bundle's own tracked, readable
# shape for the rest) — never hidden. A `"field": "value"` dict entry (JSON, or the identical Python
# dict-literal syntax a test fixture builds a spec with) under one of these field names carries an OPAQUE,
# caller-chosen id or a same-repo fixture path by the schema's own contract: never a citation of a tracked
# location, whatever namespace word or spelling that string happens to use. Two DIFFERENT groups, gated
# two different ways (see `_path_leaks`):
# - a MEDIA-asset field (an overlay clip, a watermark sting, a music track, …) always holds a clip, an
#   image or a font wherever this repo's own fixtures build one — never a source/config/doc file — so this
#   group is gated by the matched file's own EXTENSION, and stays available anywhere in the tree.
_WIRE_MEDIA_ID_FIELDS = ("id", "clip", "track", "cold", "burn", "clicks", "asset", "sound", "sting", "idle")
# - a PROVENANCE field is the pinned bundle's (or a contracts/ example's) own self-referential metadata —
#   which upstream example this fixture came from (`source_path`), which tool produced it
#   (`generated_by`), an opaque result/correlation key that happens to look path-shaped in exactly one
#   fixture (`result_key`, `corr_id`) — never a field a test elsewhere in this repo builds a spec with, so
#   this group is gated by LOCATION instead: a JSON/YAML line already under contracts/ or vendor/.
_WIRE_PROVENANCE_FIELDS = ("source_path", "result_key", "corr_id", "schema", "generated_by", "$id")
_WIRE_MEDIA_ID_FIELD_PREFIX = re.compile(
    rf'"(?:{"|".join(re.escape(f) for f in _WIRE_MEDIA_ID_FIELDS)})"\s*:\s*"$')
_WIRE_MEDIA_ID_FIELD_VALUE = re.compile(
    rf'"(?:{"|".join(re.escape(f) for f in _WIRE_MEDIA_ID_FIELDS)})"\s*:\s*"([^"]*)"')
_WIRE_PROVENANCE_FIELD_PREFIX = re.compile(
    rf'"(?:{"|".join(re.escape(f) for f in _WIRE_PROVENANCE_FIELDS)})"\s*:\s*"$')
# The NAMESPACE a caller picks for a media-id field's value is free-form (this repo's examples use
# `brand`/`broll`/`music`; a test fixture is just as free to invent its own), so it is derived from every
# such VALUE this repo already owns — never hand-typed — rather than guessed: a later reference to that
# SAME namespace (its own namespace, not the whole path) reads clean wherever it appears, while a real
# source-tree segment (`dev`, `remotion`, `scripts`, …) was never produced this way. The FILE at the end is
# always a MEDIA/data payload (a clip, an image, a font) — an asset id never points at a source/config/doc
# file — so only a media extension is eligible here too; a source or doc extension can never pass as one,
# whatever namespace happens to sit in front of it.
_WIRE_ASSET_REF = re.compile(rf"\b([A-Za-z][\w-]*)/(?:[\w.-]+/)*[\w-][\w.-]*\.(?:{_MEDIA_EXTS})(?![\w-])")
# A `"name.ext": "<hex digest>"` entry is a hash MANIFEST (this exact bundle's own `source_sha256` map,
# keyed by the upstream contracts repo's own file names) — the shape proves it, no field name needed: the
# value right after the matched name is a bare hex digest, nothing a repo-path citation is ever paired
# with. Gated by LOCATION, same as a provenance field: it only exists inside contracts/ or vendor/.
_HASH_MANIFEST_VALUE = re.compile(r'^"\s*:\s*"[0-9a-f]{32,}"')


def _h(s: str) -> str:
    # casefold, not lower: re.IGNORECASE in the engine's own regex case-folds (e.g. U+017F LONG S -> 's'),
    # and a plain .lower() would miss that and let a deliberately folded spelling through.
    return hashlib.sha256(s.casefold().encode()).hexdigest()[:16]


def _joined(line: str) -> str:
    return _SEPARATORS.sub("", line.casefold())


def _name_index(names: tuple[tuple[str, int, str], ...]) -> tuple[re.Pattern[str], dict[str, set[tuple[int, str]]]]:
    by_prefix: dict[str, set[tuple[int, str]]] = {}
    for prefix, length, digest in names:
        by_prefix.setdefault(prefix, set()).add((length, digest))
    starts = re.compile("(?=(" + "|".join(re.escape(p) for p in sorted(by_prefix)) + "))")
    return starts, by_prefix


_NAME_INDEX = _name_index(_NAMES)
_JOINED_INDEX = _name_index(_JOINED_NAMES)


def _windows(low: str, index: tuple[re.Pattern[str], dict[str, set[tuple[int, str]]]]) -> list[str]:
    starts, by_prefix = index
    found: list[str] = []
    for m in starts.finditer(low):
        i = m.start()
        for length, digest in by_prefix[m.group(1)]:
            if _h(low[i:i + length]) == digest:
                found.append(low[i:i + length])
    return found


def _name_leaks(line: str) -> list[str]:
    """All NAME leaks on the line, not just the first — two different names on one line must both surface."""
    return _windows(line.casefold(), _NAME_INDEX) + _windows(_joined(line), _JOINED_INDEX)


def _name_leak(line: str) -> str | None:
    found = _name_leaks(line)
    return found[0] if found else None


def _is_module(segment: str) -> bool:
    return _h(segment) in _MODULES


def _module_leaks(line: str) -> list[str]:
    """All MODULE leaks on the line (see _name_leaks for why 'all' rather than 'first')."""
    found: list[str] = []
    for shape in _MODULE_REF_SHAPES:
        for m in shape.finditer(line):
            for segment in re.split(r"[\s.,]+", m.group(1)):
                if segment and _is_module(segment):
                    found.append(m.group(0))
    return found


def _module_leak(line: str) -> str | None:
    found = _module_leaks(line)
    return found[0] if found else None


def _py_import_leaks(text: str) -> list[tuple[int, str]]:
    """Wrapped, aliased and relative imports are read by the interpreter's own parser."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = ([node.module] if node.module else []) + [a.name for a in node.names]
        hits += [(node.lineno, dotted) for dotted in names if any(map(_is_module, dotted.split(".")))]
    return hits


def _git_files(*args: str) -> list[str]:
    out = subprocess.run(["git", "ls-files", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                         check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def _tracked_files() -> list[str]:
    """What lands: the index plus every not-ignored new file (a file this very change adds is tracked the
    moment it lands, and must be scanned — and resolvable — before that)."""
    files = _git_files() + [f for f in _git_files("--others", "--exclude-standard")
                            if not {".venv", "__pycache__"} & set(f.split("/"))]
    # this file is scanned even before it is first committed — it must not be the one file that leaks
    own = THIS_FILE.relative_to(REPO_ROOT).as_posix()
    return list(dict.fromkeys(files if own in files else files + [own]))


def _suffixes(path: str) -> tuple[str, ...]:
    """Every dir-boundary suffix of a relative path: three segments give the path itself, its last two
    segments joined, and its last segment alone."""
    parts = path.split("/")
    return tuple("/".join(parts[i:]) for i in range(len(parts)))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _runtime_file_names(text: str) -> set[str]:
    """File names the pod's own code spells as a WHOLE string constant (not inside prose, not a docstring):
    the runtime files it reads and writes. A constant with a slash in it is not a name — it is a path, and
    gets checked like any other path."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    docs = _docstring_nodes(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs):
            continue
        value = node.value.strip()
        if _WHOLE_FILE_NAME.match(value):
            names.add(value)
        elif (m := _SLASH_PATH_REF.fullmatch(value)) and _is_runtime_location(m.group(1), m.group(2)):
            names.add(value.rsplit("/", 1)[-1])  # `/var/cache/…/outbox.json` names the runtime `outbox.json`
    return names


def _handler_modules(text: str) -> set[str]:
    """`"handler": "pkg.mod:fn"` in an op contract: the modules this repo's own wire contract names."""
    try:
        doc = json.loads(text)
    except ValueError:
        return set()
    handler = doc.get("handler") if isinstance(doc, dict) else None
    if not isinstance(handler, str):
        return set()
    return set(handler.partition(":")[0].split("."))


class _LocalTree:
    """What this repo itself owns, all of it derived: exact tracked paths, every dir-boundary SUFFIX of each
    (so a reference that drops a leading mount/package prefix — `ops/runner.py` for our own
    `podagent/ops/runner.py`, or the reverse, one that ADDS one — `app/podagent/ops/runner.py` — still
    resolves to ours), every tracked basename and `.py` stem, the runtime file names its code spells, and the
    handler modules its op contracts declare."""

    def __init__(self, files: tuple[str, ...], runtime_names: frozenset[str] = frozenset(),
                 handler_modules: frozenset[str] = frozenset(), asset_namespaces: frozenset[str] = frozenset()):
        self.full = frozenset(files)
        self.suffixes = frozenset(s for f in files for s in _suffixes(f))
        self.names = frozenset(f.rsplit("/", 1)[-1] for f in files) | runtime_names
        self.runtime_names = runtime_names
        self.stems = frozenset(Path(f).stem for f in files if f.endswith(".py")) | handler_modules
        self.asset_namespaces = asset_namespaces

    def has_path(self, path: str) -> bool:
        # direction A: `path` is a (suffix-of / equal to) something we track; direction B: something we
        # track is a suffix of `path` (an added prefix in front of a path that is, further in, truly ours).
        return path in self.suffixes or any(s in self.full for s in _suffixes(path))

    def has_name(self, name: str) -> bool:
        if name.startswith("."):  # a suffix (`*.schema.json`, a `{stem}`-built sidecar): some name we own ends so
            return any(n.endswith(name) for n in self.names)
        return name in self.names or (name.endswith(".py") and name[:-3] in self.stems)


def _os_data_file_names(text: str) -> set[str]:
    """A data file named at an operating-system location (`/etc/…/x.json`) is a file on a box — a driver's
    manifest, a cache — never a repo file; its bare name is the same runtime file wherever it is cited."""
    return {m.group(2).rsplit("/", 1)[-1] for m in _SLASH_PATH_REF.finditer(text)
            if m.group(1) and _is_runtime_location(m.group(1), m.group(2)) and m.group(2).endswith(".json")}


def _local_tree(files: list[str], read=lambda rel: (REPO_ROOT / rel).read_text(encoding="utf-8")) -> _LocalTree:
    runtime: set[str] = set()
    handlers: set[str] = set()
    namespaces: set[str] = set()
    for rel in files:
        try:
            text = read(rel)
        except (OSError, UnicodeDecodeError):
            continue
        runtime |= _os_data_file_names(text)
        if rel.startswith("podagent/") and rel.endswith(".py"):
            runtime |= _runtime_file_names(text)
        elif rel.startswith("contracts/ops/") and rel.endswith(".json"):
            handlers |= _handler_modules(text)
        if rel.startswith("vendor/"):
            # the bundle's own `"schema": "<name>.schema.json"` registry entries name its OWN sibling
            # schema surfaces — DERIVED here, not hand-typed — so a later bare mention of that same exact
            # name elsewhere in the SAME bundle (cross-referencing one surface from another's prose) is a
            # self-reference, not a leak.
            runtime |= set(re.findall(r'"schema"\s*:\s*"([\w.-]+\.schema\.json)"', text))
        # the asset-id NAMESPACE vocabulary is derived strictly from this repo's OWN schema/contracts and
        # the pinned vendor bundle's OWN self-referential fields — never from a test fixture that merely
        # uses it, so a test cannot invent its own exemption
        if rel.startswith("contracts/") or rel.startswith("vendor/"):
            namespaces |= {v.split("/", 1)[0] for v in _WIRE_MEDIA_ID_FIELD_VALUE.findall(text)
                           if _WIRE_ASSET_REF.fullmatch(v)}
    return _LocalTree(tuple(files), frozenset(runtime), frozenset(handlers), frozenset(namespaces))


_OS_ROOTS = frozenset(("tmp", "var", "etc", "opt", "usr", "proc", "sys", "run", "dev", "root", "mnt", "srv"))
_MEDIA_ONLY_PATH = re.compile(rf"\.(?:{_MEDIA_EXTS})\b", re.IGNORECASE)


def _is_runtime_location(rooted: str | None, path: str) -> bool:
    first = path.split("/", 1)[0]
    if rooted:
        # a recognized OS directory (`/etc/…`), a dotfile mount (`~/.cache/…`), OR an absolute path to a
        # MEDIA/data payload (`/w/seq0.mov`): a repo never ships a real source/doc file at an absolute
        # path, so only a non-media extension absolute path is still a candidate worth checking (an
        # absolute `.py`/`.md`/… path is unusual enough on its own to be worth a second look).
        return first in _OS_ROOTS or first.startswith(".") or bool(_MEDIA_ONLY_PATH.search(path))
    # a host-led path: its first segment is a dotted host name, not a dotfile or `.`/`..`
    return "." in first and not first.startswith(".")


def _path_leaks(line: str, local: _LocalTree, *, data_file: bool = False, wire_data: bool = False) -> list[str]:
    # Two DIFFERENT gates, neither of them "anywhere in any file" (see the module docstring):
    # - a MEDIA-id field (`"sound": "…"`, `"clip": "…"`) is gated by the matched file's own EXTENSION, not
    #   by location: it fires only when the matched path ends in a media extension, so a source-extension
    #   path keeps failing wherever it is written, while the identical dict-literal shape a real test
    #   fixture builds a spec with, holding a real media-extension id, anywhere under tests/, reads clean.
    # - a PROVENANCE field (`"source_path": "…"`, `"generated_by": "…"`) and the HASH-MANIFEST shape
    #   (`"name.ext": "<hex digest>"`) are gated by LOCATION instead: both are this exact pinned bundle's
    #   own self-referential metadata, so they are only recognized on a JSON/YAML line already under
    #   contracts/ or vendor/, never on an ordinary .py or .md line anywhere else.
    found: list[str] = []
    slash_ref = _SLASH_PATH_REF if data_file else _SLASH_PATH_REF_MEDIA
    for m in slash_ref.finditer(line):
        rooted, path = m.group(1), m.group(2)
        rel = re.sub(r"^(?:\.{1,2}/)+", "", path)
        if _is_runtime_location(rooted, rel):
            continue
        if _WIRE_ASSET_REF.fullmatch(rel) and rel.split("/", 1)[0] in local.asset_namespaces:
            continue
        if _MEDIA_ONLY_PATH.search(rel) and _WIRE_MEDIA_ID_FIELD_PREFIX.search(line[:m.start(2)]):
            continue
        if wire_data and (_WIRE_PROVENANCE_FIELD_PREFIX.search(line[:m.start(2)])
                          or _HASH_MANIFEST_VALUE.match(line[m.end(2):])):
            continue
        if not (local.has_path(rel) or rel.rsplit("/", 1)[-1] in local.runtime_names):
            found.append(path)
    tmp_fixtures = frozenset(_TMP_FIXTURE_REF.findall(line))
    for m in _BARE_FILE_REF.finditer(line):
        name = m.group(1)
        if _MEDIA_ONLY_PATH.search(name) and _WIRE_MEDIA_ID_FIELD_PREFIX.search(line[:m.start(1)]):
            continue
        if wire_data and (_WIRE_PROVENANCE_FIELD_PREFIX.search(line[:m.start(1)])
                          or _HASH_MANIFEST_VALUE.match(line[m.end(1):])):
            continue
        if not (local.has_name(name) or name in tmp_fixtures):
            found.append(name)
    for m in _SCRIPTS_MEMBER_REF.finditer(line):
        if m.group(1) not in local.stems:
            found.append(m.group(0))
    for m in _SCRIPTS_IMPORT_REF.finditer(line):
        targets = re.split(r"\s*,\s*", m.group(1)) if m.group(1) is not None else [""]
        if any(t not in local.stems for t in targets):
            found.append(m.group(0))
    return found


def _scan(rel: str, text: str | None, local: _LocalTree = _LocalTree(())) -> list[str]:
    parts = tuple(rel.split("/"))
    data_file = rel.rsplit(".", 1)[-1].lower() in _DATA_EXTS if "." in rel.rsplit("/", 1)[-1] else False
    wire_data = data_file and parts[0] in ("contracts", "vendor")
    hits: list[str] = []
    path_leaks = (_name_leaks(rel) + _module_leaks(rel)
                  + [seg for seg in parts[:-1] if _is_module(seg)])
    hits += [f"{rel}:0: {leak!r} (path)" for leak in dict.fromkeys(path_leaks)]
    if text is None:
        return hits
    for i, line in enumerate(text.splitlines(), 1):
        # vendor/ included: the full NAME list, the MODULE shapes and the FILE rule apply to every tracked
        # file, with NO exemption of any kind — `wire_data` only narrows the id-field/hash-manifest shapes
        # inside _path_leaks, and only for a JSON/YAML line already inside contracts/ or vendor/.
        leaks = (_name_leaks(line) + _module_leaks(line)
                 + _path_leaks(line, local, data_file=data_file, wire_data=wire_data))
        hits += [f"{rel}:{i}: {leak!r}" for leak in dict.fromkeys(leaks)]
    if rel.endswith(".py"):
        hits += [f"{rel}:{n}: import of {dotted!r}" for n, dotted in _py_import_leaks(text)]
    return hits


def test_the_tree_names_no_engine_internals() -> None:
    files = _tracked_files()
    local = _local_tree(files)
    hits: list[str] = []
    for rel in files:
        path = REPO_ROOT / rel
        text: str | None = None
        try:
            if path.is_file() and path.stat().st_size <= MAX_SIZE:
                text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                # not valid utf-8, but still a tracked text-ish file — scan it byte-lossy rather than give up
                # on its contents outright (the engine gate gives up here; this copy closes that gap).
                text = path.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                pass
        except OSError:
            pass
        hits += _scan(rel, text, local)
    assert not hits, (
        "engine internals named in this PUBLIC repo — describe the role (the provider, the engine's EDL step,"
        " the brand, the engine's resolver), never the name or the path:\n  " + "\n  ".join(hits[:60]))


def test_the_scanner_flags_reference_shapes_not_lookalike_identifiers(monkeypatch) -> None:
    # a synthetic, non-leaking module name hashed on the fly: the reference SHAPES are what is under test
    fake = "zz_engine_mod"
    monkeypatch.setattr(sys.modules[__name__], "_MODULES", _MODULES | {_h(fake)})
    assert _module_leak(f"from {fake} import x") == f"from {fake}"
    assert _module_leak(f"from .util import (ensure_dir, {fake})")
    assert _module_leak(f'importlib.import_module("pkg.{fake}")')
    assert _module_leak(f"{fake} = next(iter(outbox))") is None
    assert _module_leak(f"key = {fake}_key(frame)") is None
    assert _py_import_leaks(f"from .util import (\n    a,\n    {fake} as b,\n)\n") == [(1, f"{fake}")]
    assert _scan(f"{fake}/core.py", None) == [f"{fake}/core.py:0: {fake!r} (path)"]
    # a BARE dotted attribute — no import, no `scripts/`, no backtick-call syntax required — names the
    # module exactly as much as an import does, in code, in a comment, or inside backticks/a string
    assert _module_leak(f"see `{fake}.resolve_sfx` here") == f"{fake}.resolve_sfx"
    assert _module_leak(f"# `{fake}.bus_fits_under` folds that") == f"{fake}.bus_fits_under"
    assert _module_leak(f'"{fake}.plan_cues made the kind decision"') == f"{fake}.plan_cues"
    assert _module_leak(f"it needs {fake} to build a real thing") is None  # no dot: still a bare mention


def test_an_engine_path_is_a_leak_at_any_extension_once_it_names_a_full_location() -> None:
    # assembled from PARTS, same discipline as the test above: a REAL untracked engine directory + a media
    # extension names a real location exactly as much as a `.py` one would; a bare dynamic output name
    # like `master.mp4` must stay clean — it is never a citation of a tracked location, just a filename
    # this process itself produced at runtime.
    z, mp4_, mov_ = "zz_any", ".mp4", ".mov"
    dev_, lp_, fixtures_, rm_, src_ = "dev", "localpod", "fixtures", "remotion", "src"
    local = _local_tree(["podagent/main.py"])
    assert _path_leaks(f"{dev_}/{lp_}/{fixtures_}/{z}-smoke{mp4_}", local)
    assert _path_leaks(f"{rm_}/{src_}/{z}_clip{mov_}", local)
    for clean in (f"master{mp4_}", f"short1{mp4_}"):
        assert not _path_leaks(clean, local), clean


def _sources(files: dict[str, str]) -> _LocalTree:
    return _local_tree(list(files), read=files.__getitem__)


def test_any_engine_file_reference_is_a_leak_by_rule_not_by_list() -> None:
    # every example below is assembled from PARTS at runtime — dir, stem and extension never sit together
    # as one contiguous literal — so this file's own source carries no file-reference shape for the scan
    # above to find.
    s, t_, d_, z, dot = "scripts", "tests", "docs", "zz_any", "."
    py, yaml_, tsx_, md_, sh_, go_, js_ = ".py", ".yaml", ".tsx", ".md", ".sh", ".go", ".json"
    local = _sources({
        f"{s}/release_image{py}": "",
        "podagent/main" + py: f'"""{z}_doc{js_}"""\nOUT = "{z}_out{js_}"\n',
        f"{t_}/test_main{py}": "",
        "contracts/ops/x" + js_: '{"handler": "zzpack.zz_op:run"}',
    })
    assert {"release_image", "main", "test_main", "zzpack", "zz_op"} <= local.stems
    assert f"{z}_out{js_}" in local.runtime_names and f"{z}_doc{js_}" not in local.runtime_names
    # WHATEVER the top directory — not only `scripts` — a path to a file this repo does not itself track is
    # a leak, and so is a bare file name of any source/config/doc kind, a plan ref and a dotfile path
    for leak in (f"see {s}/{z}{py}:306", f"{s}/zz_pkg/zz_mod{py}", f"{s}.{z}",
                 f"python -m {s}.{z} --slug x", f"the engine's {s} / {z}",
                 f"from {s} import zz_a, zz_b", f"from ..{s} import zz_a", f"import {s}",
                 f'importlib.import_module("{s}.{z}")',
                 f"see {z}{py}:306", f"`{z}{py}`'s budget",
                 f"{t_}/test_{z}{py}", f"registry/{z}{yaml_}", f"remotion/src/ZzAny{tsx_}",
                 f"{d_}/ZZ_ANY{md_}", f"zz_top/{z}{py}", f"app/zz_top/{z}{py}", f"TRK-1/PLAN{md_}",
                 f"PLAN{md_}", f"{z}{yaml_} key", f"{z}{sh_}", f"{z}_test{go_}", f"zz_dev/zz/{dot}env.example",
                 f"/zz_home/{z}{py}", f"{z}_doc{js_}"):
        assert _path_leaks(leak, local), leak
    for clean in (f"python {s}/release_image{py} verify", f"from {s} import release_image",
                  "subscripts = 1", "the engine's resolver", "the engine's queue plan",
                  f"{t_}/test_main{py}", "podagent/main" + py, f"app/podagent/main{py}", f"main{py}",
                  f"{s}/release_image{py}:1 is read by {t_}/test_main{py}",
                  f"{z}_out{js_}", f"work/{z}_out{js_}", f"zz_op{py}", "master.mp4",
                  f"https://zz.example/zz/{z}{sh_}", f"zz.example/o/1{js_}", f"/tmp/{z}{js_}",
                  f"/etc/zz/{z}{js_}", f"~/.cache/{z}{js_}"):
        assert not _path_leaks(clean, local), clean
    # a data file named at an operating-system location is a runtime file wherever its bare name is cited
    assert _os_data_file_names(f"cat /etc/zz/{z}_drv{js_} | /zz_home/{z}_x{js_}") == {f"{z}_drv{js_}"}
    assert _scan("podagent/x" + py, f"# mirrors {s}/{z}{py}:12\n", local) == [
        f"podagent/x{py}:1: '{s}/{z}{py}'"]


def test_the_engine_repo_name_is_caught_in_every_spelling(monkeypatch) -> None:
    # a synthetic two-word name hashed on the fly: the SPELLINGS are what is under test
    joined = "zzleakrepo"
    monkeypatch.setattr(sys.modules[__name__], "_JOINED_INDEX",
                        _name_index(((joined[:2], len(joined), _h(joined)),)))
    for spelling in ("zzleak-repo", "zzleak_repo", "zzleakrepo", "zzleak repo", "ZzleakRepo", "zzleak.repo",
                     "see the ZZLEAK-REPO tree"):
        assert _name_leaks(spelling) == [joined], spelling
    assert _name_leaks("zzleak and a repo") == []


def test_this_file_is_scanned_like_every_other() -> None:
    files = _tracked_files()
    own = THIS_FILE.relative_to(REPO_ROOT).as_posix()
    assert own in files
    assert _scan(own, THIS_FILE.read_text(encoding="utf-8"), _local_tree(files)) == []


def test_every_listed_name_is_caught_in_any_case() -> None:
    # the hashes are only as good as the window search: a hashed name must hit at any casing and offset
    synth = "zzleakname"
    starts, by_prefix = _name_index(((synth[:2], len(synth), _h(synth)),))
    low = f"x {synth.upper()} y".casefold()
    assert any(_h(low[m.start():m.start() + n]) == d
               for m in starts.finditer(low) for n, d in by_prefix[m.group(1)])


def test_the_vendored_bundle_is_scanned_with_the_full_list(monkeypatch) -> None:
    # a synthetic provider name hashed on the fly: vendor/ gets the SAME name list as every other path, and
    # NO value is excused for being a schema enum/const member — not even inside the pinned bundle itself.
    leak = "zzleakprov"
    monkeypatch.setattr(sys.modules[__name__], "_NAME_INDEX", _name_index(((leak[:2], len(leak), _h(leak)),)))
    bundle = "vendor/monty-contracts/zz" + ".json"
    doc = (f'{{"provider": {{"enum": ["subscription", "{leak}_api"]}},\n'
           f' "brand": {{"description": "the {leak} brand"}},\n'
           f' "x": {{"const": "{leak}_api", "default": "{leak}"}}}}\n')
    local = _sources({bundle: doc})
    assert _scan(bundle, doc, local) == [
        f"{bundle}:1: {leak!r}", f"{bundle}:2: {leak!r}", f"{bundle}:3: {leak!r}"]
    # prose under vendor/ has no wire excuse at all, and outside vendor/ the same leak is caught the same way
    prose, own = "vendor/monty-contracts/x" + ".md", "contracts/zz" + ".json"
    assert _scan(prose, f"person: {leak}\n") == [f"{prose}:1: {leak!r}"]
    assert _scan(own, doc, local)[:1] == [f"{own}:1: {leak!r}"]


def test_the_id_field_shape_never_excuses_a_source_or_config_path() -> None:
    # the id-FIELD bypass (`"sound": "…"`, `"schema": "…"`) is gated by the matched file's own EXTENSION,
    # not by location or field name alone: a SOURCE/config/doc extension is never a media payload, so the
    # bypass never fires for one, in or out of contracts/vendor, however known-looking the namespace in
    # front of it is (e.g. one this repo's own contracts/vendor trees happen to derive, like "work" or
    # "tools" or "monty") or whatever field name introduces it.
    s, dev_, z, py = "scripts", "dev", "zz_any", ".py"
    local = _local_tree(["podagent/main.py"])
    for line, data_file in (
        (f'x = {{"key": "{s}/zz_resolver{py}"}}', True),
        (f'x = {{"key": "{s}/zz_resolver{py}"}}', False),
        (f'# see "schema": "{dev_}/zz/{z}{py}"', True),
        (f'# mirrors tools/{z}_resolver{py}:12', False),
        (f'# see work/{z}{py}', False),
    ):
        assert _path_leaks(line, local, data_file=data_file), (line, data_file)
    # the identical dict-literal shape with a real MEDIA extension still reads clean, because the field
    # IS what this wire spec's own id fields are for — a real test fixture builds a spec this way
    wav = ".wav"
    assert not _path_leaks(f'x = {{"sound": "sfx/missing{wav}"}}', local, data_file=False)


def test_the_hash_manifest_shape_only_excuses_a_path_under_contracts_or_vendor() -> None:
    # the HASH-MANIFEST bypass (`"name.ext": "<hex digest>"`) is this exact pinned bundle's own
    # `source_sha256` map, keyed by the upstream repo's OWN schema/doc file names — it is gated by
    # LOCATION, and never fires on an ordinary line of source or prose anywhere else, whatever name it keys on.
    z, json_ = "zz_any", ".json"
    digest = "0123456789abcdef0123456789abcdef"
    local = _local_tree(["podagent/main.py"])
    assert _path_leaks(f'"{z}{json_}": "{digest}"', local, data_file=True)
    assert not _path_leaks(f'"{z}{json_}": "{digest}"', local, data_file=True, wire_data=True)
