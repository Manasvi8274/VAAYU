"""
Routines (user, 2026-11-05): one phrase, several things. "When I say I'm leaving, pause the music and lock the
laptop" - from then on "I'm leaving" does both, at once, through the router (no model, no wait).

Made by voice:  "when I say <phrase>, <steps>"  /  "create a routine called <phrase>: <steps>"
                "what are my routines"  /  "delete the <phrase> routine"
Each step must be a command Vaayu already knows (router.fast_route) - so a routine is checked when it's made, and
runs exactly what was checked: data/routines.json keeps the tool calls themselves. Built in: "good morning" (the
briefing, briefing.py), "I'm leaving" (pause, lock) until the user makes their own.
"""

import json
import re
import threading
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "routines.json"
_DEFAULTS = {"i'm leaving": [["control_media", {"action": "pause"}], ["lock_computer", {}]]}
_lock = threading.Lock()


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower().replace("i am ", "i'm ")))


def load() -> dict:
    try:
        saved = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {**_DEFAULTS, **saved}


def _save(data: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps({k: v for k, v in data.items() if _DEFAULTS.get(k) != v}, indent=1), encoding="utf-8")


def match(text: str) -> list | None:
    """The tool calls of the routine `text` names, or None."""
    calls = load().get(_norm(text))
    return [(name, dict(args)) for name, args in calls] if calls else None


def _steps(steps_text: str) -> list | None:
    from assistant.llm import router
    routed = router.fast_route(steps_text)
    if routed and routed[0] == "tools":
        return [[n, a] for n, a in routed[1]]
    return None


_VERB = re.compile(r"^(?:open|close|play|pause|resume|stop|lock|set|turn|mute|unmute|start|show|take|type|search|go|"
                   r"increase|decrease|raise|lower|volume|tell|give|remind|check|brief|read|minimi[sz]e|maximi[sz]e|"
                   r"switch|launch|run|put|make|next|skip|shut|sleep)\b", re.I)


def create(said: str) -> str | None:
    """'when i say <phrase>, <steps>': the phrase ends at the first comma; without one, it is the shortest start whose
    rest begins with a command word and is all known commands ("work mode open vs code ..." -> "work mode")."""
    said = re.sub(r"\s+", " ", said).strip()
    if "," in said:
        head, tail = said.split(",", 1)
        splits = [(head, tail)]
    else:
        words = said.split()
        splits = [(" ".join(words[:k]), " ".join(words[k:])) for k in range(1, min(8, len(words)))]
    for phrase, rest in splits:
        phrase = phrase.strip(" ,:")
        rest = re.sub(r"^(?:you should |please |then |do |to )", "", rest.strip(" ,:"))
        if phrase and _VERB.match(rest) and (calls := _steps(rest)):
            with _lock:
                data = load()
                data[_norm(phrase)] = calls
                _save(data)
            return f"Done. When you say '{phrase}', I'll {rest}."
    return None


def delete(phrase: str) -> bool:
    with _lock:
        data = load()
        key = _norm(phrase)
        if key not in data:
            return False
        data.pop(key)
        _DEFAULTS.pop(key, None)  # a built-in one the user didn't want
        _save(data)
        return True


def describe() -> str:
    names = list(load())
    return ("Your routines: " + ", ".join(f"'{n}'" for n in names) + ".") if names else "You have no routines yet."
