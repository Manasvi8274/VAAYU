"""
Small disk-persisted cache for tool lookups that are slow (a real page
navigation + DOM query) but stable (the answer doesn't change often) -
e.g. a YouTube channel name -> channel URL mapping. Explicitly requested:
"cache data frequently, so that many commands can be done faster without
[the] whole process."

Not a general-purpose cache - deliberately simple (one JSON file, load/save
whole), since the data volume here is small (a few hundred cached lookups at
most) and this only needs to save repeated slow navigations within/across
sessions, not serve as a high-throughput store.
"""

import json
import time
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CACHE_FILE = _CACHE_DIR / "tool_cache.json"


def _load() -> dict[str, Any]:
    if not _CACHE_FILE.exists():
        return {}
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, Any]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")


def get(namespace: str, key: str, max_age_seconds: float) -> Any | None:
    """Returns the cached value for namespace/key if present and fresher
    than max_age_seconds, else None (cache miss or expired)."""
    entry = _load().get(namespace, {}).get(key.lower())
    if entry is None:
        return None
    if time.time() - entry["cached_at"] > max_age_seconds:
        return None
    return entry["value"]


def set(namespace: str, key: str, value: Any) -> None:
    data = _load()
    data.setdefault(namespace, {})[key.lower()] = {"value": value, "cached_at": time.time()}
    _save(data)
