"""
Commands Vaayu learned from the user's own way of saying them (2026-11-05, user: "try anything and everything to get
the response accurate but faster and faster").

When a sentence the router didn't know was worked out by the brain (the slow LLM path, 3-15 s) into ONE tool call
that worked, the exact sentence and that call are kept (data/learned_commands.json): next time the same sentence is
routed at once, like a built-in command. "chrome chalao" is slow the first time only.

Kept only when it's safe to repeat without thinking: a single call, it worked, nothing outward or dangerous
(guardian/action_rules.py tier < 2), no confirmation needed. If the user's next words within CORRECTION_SECONDS
say it was wrong ("no", "wrong", "not that", "galat"), it is forgotten again.
"""

import json
import re
import threading
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "learned_commands.json"
CORRECTION_SECONDS = 40   # calibration knob
MAX = 500
_NEVER = {"think_hard", "do_task", "look_at_screen", "click_on_screen", "remember_name", "review_logs"}
_CORRECTION = re.compile(r"^(?:no|nope|nahi|nahin|wrong|galat|not that|that'?s wrong|that'?s not (?:it|what i (?:said|meant|asked))|"
                         r"i (?:didn'?t|did not) (?:say|ask|mean) that|undo that|why did you do that)\b", re.I)
_lock = threading.Lock()
_last: tuple[str, float] | None = None  # (key, when) - the one learned most recently


def _key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _load() -> dict:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def match(text: str) -> list | None:
    global _last
    key = _key(text)
    entry = _load().get(key)
    if not entry:
        return None
    _last = (key, time.time())  # used now - "no, not that" right after still unlearns it
    return [(entry["tool"], dict(entry["args"]))]


def from_turn(text: str, trace: list[dict], registry=None) -> bool:
    """After a turn: learn it if the brain turned it into one safe tool call that worked. True if learned."""
    global _last
    calls = [t for t in trace if t.get("tool")]
    if len(calls) != 1 or calls[0].get("path") != "llm" or not calls[0].get("ok"):
        return False
    if any(t.get("path") == "chat" or t.get("rerouted") or t.get("name_corrected") for t in trace):
        return False
    name, args = calls[0]["tool"], calls[0].get("args") or {}
    from guardian.action_rules import OUTWARD, tier
    if registry is None:
        from assistant.tools.registry import registry
    spec = registry._tools.get(name)
    if name in _NEVER or spec is None or spec.requires_confirmation or tier(name, args) >= OUTWARD:
        return False
    key = _key(text)
    if not key or len(key.split()) > 12:
        return False
    with _lock:
        data = _load()
        data[key] = {"tool": name, "args": args, "learned": time.strftime("%Y-%m-%dT%H:%M:%S")}
        while len(data) > MAX:
            data.pop(next(iter(data)))
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        _last = (key, time.time())
    return True


def correction(text: str) -> bool:
    """The user says the last learned command was wrong: forget it. True if one was forgotten."""
    global _last
    if not _last or time.time() - _last[1] > CORRECTION_SECONDS or not _CORRECTION.match(_key(text)):
        return False
    with _lock:
        data = _load()
        gone = data.pop(_last[0], None) is not None
        if gone:
            PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        _last = None
    return gone
