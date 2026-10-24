"""
Everything Vaayu hears, decides and does, one JSON object per line in
data/interactions.jsonl - so a later review (by the user, or a Claude fix
session) can see exactly what happened in real use: what was heard, what
state Vaayu was in, which path handled it (fast route / LLM / multi-step),
every tool call with its arguments and result, the reply, and timings.

Written and flushed per event (not buffered), so a kill mid-command loses
nothing. Never raises - logging must not be able to break Vaayu.
"""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "interactions.jsonl"
_MAX_FIELD_CHARS = 2000


def _clip(value):
    if isinstance(value, str):
        return value if len(value) <= _MAX_FIELD_CHARS else value[:_MAX_FIELD_CHARS] + "...[cut]"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _clip(str(value))


def log_event(kind: str, **fields) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now().isoformat(timespec="milliseconds"), "kind": kind, **_clip(fields)}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def log_error(where: str, exc: BaseException, **fields) -> None:
    log_event("error", where=where, error=repr(exc),
              traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-3000:], **fields)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self.start) * 1000)
