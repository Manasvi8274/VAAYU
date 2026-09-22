import os
from pathlib import Path

from assistant.config_schema import load_config
from assistant.tools.registry import registry

_config = load_config()
_SAFE_ROOTS = _config.files.resolved_roots()
_MAX_RESULTS = 20
_MAX_READ_BYTES = 50_000
_ALLOWED_READ_EXTENSIONS = {".txt", ".md", ".csv", ".log"}


def _is_within_safe_roots(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved == root or resolved.is_relative_to(root) for root in _SAFE_ROOTS)


@registry.register(
    name="search_files",
    description=(
        "Search for files by (partial) name within the user's safe folders (Desktop, Documents, "
        "Downloads) - use this for 'search my files for X', 'find a file called X', 'do I have a "
        "file named X'. Returns real full paths - call this BEFORE read_file whenever you don't "
        "already have a confirmed real path from an earlier step; never guess/invent a path or "
        "filename yourself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Filename or partial filename to search for."}
        },
        "required": ["query"],
    },
)
def search_files(query: str) -> list[str]:
    query_lower = query.lower()
    matches: list[str] = []
    for root in _SAFE_ROOTS:
        if not root.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if query_lower in filename.lower():
                    matches.append(str(Path(dirpath) / filename))
                    if len(matches) >= _MAX_RESULTS:
                        return matches
    return matches


@registry.register(
    name="read_file",
    description=(
        "Read the contents of a small text file (.txt, .md, .csv, .log) within the "
        "user's safe folders. Large files are truncated. path should be a real full path - if you "
        "don't already have one confirmed (e.g. from search_files), call search_files first rather "
        "than guessing a filename/path yourself."
    ),
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Full path to the file to read."}},
        "required": ["path"],
    },
)
def read_file(path: str) -> str:
    target = Path(path)
    if not target.is_absolute() and target.parent == Path("."):
        # A bare filename with no directory component resolves against the
        # process's own working directory by default, which almost never
        # contains the user's actual files - found live: "read the X.txt
        # file" failed with "File not found" even though X.txt genuinely
        # existed in ~/Documents, because Path("X.txt") never looked there.
        # Try each safe root before giving up - if none has it, this is
        # genuinely "not found" (not "outside the safe folders", which
        # would be a misleading message for a bare name that named no
        # folder at all), so raise that directly rather than falling
        # through to the safe-roots check below.
        for root in _SAFE_ROOTS:
            candidate = root / target.name
            if candidate.exists():
                target = candidate
                break
        else:
            raise FileNotFoundError(f"File not found: {path}")

    if not _is_within_safe_roots(target):
        raise PermissionError(
            "That file is outside the allowed safe folders (Desktop, Documents, Downloads)."
        )
    if target.suffix.lower() not in _ALLOWED_READ_EXTENSIONS:
        raise ValueError(f"Only these file types can be read: {sorted(_ALLOWED_READ_EXTENSIONS)}")
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = target.read_bytes()[: _MAX_READ_BYTES + 1]
    text = data[:_MAX_READ_BYTES].decode("utf-8", errors="replace")
    if len(data) > _MAX_READ_BYTES:
        text += "\n...[truncated]"
    return text
