"""
What Vaayu remembers about the user, kept only on this laptop:

  data/conversations.jsonl  every turn (what was said, what Vaayu replied)
  data/memory_facts.json    lasting facts pulled out of personal talk (name,
                            people in their life, goals, worries, dates...)

Facts are learned after each chat turn by one small, tool-free LLM call in a
background thread, so learning never slows the spoken reply. Nothing here
may raise into the caller - memory must not be able to break Vaayu.
"""

import json
import re
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from assistant import interaction_log

_DATA = Path(__file__).resolve().parent.parent / "data"
CONVERSATIONS_PATH = _DATA / "conversations.jsonl"
FACTS_PATH = _DATA / "memory_facts.json"
_MAX_FACTS = 300
_RECENT_MINUTES = 30  # older turns are a different conversation - the facts carry those
_lock = threading.Lock()

_EXTRACT_PROMPT = (
    "You maintain a memory of facts about the USER. From what the USER said below, list any lasting facts "
    "worth remembering for future conversations: their name, family and friends, work or studies, goals, "
    "likes and dislikes, worries, plans, important dates. One short fact per line, written about the user in "
    "the third person in English (e.g. 'Has a sister named Riya.'). Only facts the user actually stated - "
    "never guess. Ignore commands to the computer. If there is nothing lasting, reply exactly NONE."
)


def log_turn(user: str, reply: str) -> None:
    try:
        CONVERSATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "user": user, "reply": reply}
        with open(CONVERSATIONS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        interaction_log.log_error("memory.log_turn", exc)


def recent_turns(n: int = 6) -> list[dict]:
    """The last n turns of the current conversation (within _RECENT_MINUTES)."""
    try:
        with open(CONVERSATIONS_PATH, encoding="utf-8") as f:
            last = deque(f, maxlen=n)  # ponytail: reads the whole file once per chat turn; tail-seek if it gets huge
        since = datetime.now() - timedelta(minutes=_RECENT_MINUTES)
        turns = [json.loads(line) for line in last]
        return [t for t in turns if datetime.fromisoformat(t["ts"]) >= since]
    except (OSError, ValueError, KeyError):
        return []


def _load() -> list[dict]:
    try:
        data = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save(items: list[dict]) -> None:
    FACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FACTS_PATH.write_text(json.dumps(items[-_MAX_FACTS:], indent=1, ensure_ascii=False), encoding="utf-8")


def facts(n: int = 40) -> list[str]:
    return [item["fact"] for item in _load()[-n:]]


def add_facts(new: list[str]) -> int:
    with _lock:
        items = _load()
        known = {item["fact"].lower() for item in items}
        added = 0
        for fact in new:
            if fact.lower() not in known:
                items.append({"fact": fact, "ts": datetime.now().isoformat(timespec="seconds")})
                known.add(fact.lower())
                added += 1
        if added:
            _save(items)
        return added


_STOP = {"the", "a", "an", "my", "me", "i", "about", "that", "this", "of", "to", "and", "mera", "meri", "mere", "ke",
         "ki", "ka", "baare", "mein"}


def forget(about: str | None = None) -> int:
    """Remove every fact (about=None) or the ones mentioning `about`. Returns how many."""
    with _lock:
        items = _load()
        if not about:
            keep = []
        else:
            words = [w for w in re.findall(r"[a-z0-9']+", about.lower()) if w not in _STOP]
            if not words:
                return 0
            keep = [i for i in items if not all(re.search(rf"\b{re.escape(w)}", i["fact"].lower()) for w in words)]
        _save(keep)
        return len(items) - len(keep)


def _parse(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line and line.upper().rstrip(".") != "NONE" and 4 <= len(line) <= 200:
            out.append(line)
    return out


def learn(client, model: str, user_text: str, reply: str) -> int:
    from assistant.llm.ollama_client import _KEEP_ALIVE, _THINK  # lazy: ollama_client imports this module
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": _EXTRACT_PROMPT},
                  {"role": "user", "content": f"USER said: {user_text}\nAssistant replied: {reply}"}],
        keep_alive=_KEEP_ALIVE,
        think=_THINK,
        options={"num_ctx": 2048, "num_predict": 120, "temperature": 0.2},
    )
    return add_facts(_parse(response["message"].get("content", "")))


def learn_async(client, model: str, user_text: str, reply: str) -> threading.Thread:
    def run():
        try:
            learn(client, model, user_text, reply)
        except Exception as exc:
            interaction_log.log_error("memory.learn", exc, text=user_text)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
