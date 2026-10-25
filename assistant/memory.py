"""
What Vaayu remembers about the user, kept only on this laptop:

  data/conversations.jsonl  every turn (what was said, what Vaayu replied)
  data/knowledge/facts.*    lasting facts pulled out of personal talk (name,
                            people in their life, goals, worries, dates...),
                            searchable by meaning (knowledge.py)

Mem0-style: after each chat turn one small, tool-free LLM call in a
background thread sees what the user said plus the 3 most similar facts
already remembered, and answers ADD / UPDATE <n> / NONE - so "moved to Pune"
replaces "lives in Gurugram" instead of piling up next to it. The chat
prompt gets the facts relevant to what's being said plus the newest few
(facts_for), not a fixed last-40. Nothing here may raise into the caller -
memory must not be able to break Vaayu.
"""

import json
import re
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from assistant import interaction_log, knowledge

_DATA = Path(__file__).resolve().parent.parent / "data"
CONVERSATIONS_PATH = _DATA / "conversations.jsonl"
FACTS_PATH = _DATA / "memory_facts.json"  # the old flat list - migrated into knowledge once
_KIND = "facts"
_SAME = 0.92  # calibration knob: this similar to a remembered fact = the same fact
# calibration knobs, measured with bge-small: unrelated facts score 0.43-0.58 against what's said,
# related ones 0.60-0.70, a reworded fact 0.98 ("lives in Gurugram" vs "lives in Pune": 0.74)
_RELEVANT = 0.58  # how similar a fact must be to what's said to come up in chat
_SHOWN = 0.55  # facts shown to the ADD/UPDATE decision
_MAX_FACTS = 300
_RECENT_MINUTES = 30  # older turns are a different conversation - the facts carry those
_lock = threading.Lock()

_EXTRACT_PROMPT = (
    "You maintain a memory of facts about the USER. From what the USER said below, list any lasting facts "
    "worth remembering for future conversations: their name, family and friends, work or studies, goals, "
    "likes and dislikes, worries, plans, important dates. One short fact per line, written about the user in "
    "the third person in English (e.g. 'Has a sister named Riya.'). Only facts the user actually stated - "
    "never guess. Ignore commands to the computer. Write each one as 'ADD: <fact>'. If it changes or corrects "
    "one of the ALREADY REMEMBERED facts (moved city, new job, changed plan), write 'UPDATE <number>: <new fact>' "
    "instead. If there is nothing lasting, reply exactly NONE."
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


_migrated: set[Path] = set()


def _rows() -> list[dict]:
    """Remembered facts, oldest first. The old memory_facts.json is moved in on first use."""
    if FACTS_PATH not in _migrated:
        _migrated.add(FACTS_PATH)
        try:
            old = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
            if isinstance(old, list) and old and not knowledge.all(_KIND):
                for item in old:
                    knowledge.add(_KIND, item["fact"], dedupe=1.01)
            FACTS_PATH.rename(FACTS_PATH.with_suffix(".json.migrated"))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return knowledge.all(_KIND)


def facts(n: int = 40) -> list[str]:
    return [r["text"] for r in _rows()[-n:]]


def facts_for(query: str, k: int = 8, recent: int = 5) -> list[str]:
    """The facts that matter for what's being said, plus the newest few."""
    try:
        relevant = [row["text"] for _s, row in knowledge.search(_KIND, query, k=k, min_score=_RELEVANT)]
    except Exception as exc:  # embedding model unavailable: newest facts still work
        interaction_log.log_error("memory.facts_for", exc)
        relevant = []
    return list(dict.fromkeys(relevant + facts(recent)))


def _trim() -> None:
    rows = _rows()
    if len(rows) > _MAX_FACTS:
        knowledge.remove(_KIND, [r["id"] for r in rows[:len(rows) - _MAX_FACTS]])


def add_facts(new: list[str]) -> int:
    """Add facts not already remembered (by meaning, not spelling); how many were new."""
    with _lock:
        added = 0
        for fact in new:
            if knowledge.search(_KIND, fact, k=1, min_score=_SAME):
                continue
            knowledge.add(_KIND, fact, dedupe=1.01)
            added += 1
        if added:
            _trim()
        return added


_STOP = {"the", "a", "an", "my", "me", "i", "about", "that", "this", "of", "to", "and", "mera", "meri", "mere", "ke",
         "ki", "ka", "baare", "mein"}


def forget(about: str | None = None) -> int:
    """Remove every fact (about=None) or the ones mentioning `about`. Returns how many."""
    with _lock:
        rows = _rows()
        if not about:
            return knowledge.remove(_KIND, [r["id"] for r in rows])
        words = [w for w in re.findall(r"[a-z0-9']+", about.lower()) if w not in _STOP]
        if not words:
            return 0
        gone = [r["id"] for r in rows if all(re.search(rf"\b{re.escape(w)}", r["text"].lower()) for w in words)]
        return knowledge.remove(_KIND, gone)


_UPDATE = re.compile(r"^UPDATE\s*#?(\d+)\s*[:.-]\s*(.+)$", re.I)
_ADD = re.compile(r"^ADD\s*[:.-]\s*(.+)$", re.I)


def _parse(text: str) -> list[tuple[int | None, str]]:
    """[(None, fact) to add | (n, fact) to replace remembered fact n]. A plain line counts as an add."""
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if not line or line.upper().rstrip(".") == "NONE":
            continue
        if m := _UPDATE.match(line):
            n, fact = int(m.group(1)), m.group(2).strip()
        else:
            n, fact = None, (_ADD.match(line).group(1).strip() if _ADD.match(line) else line)
        if 4 <= len(fact) <= 200 and fact.upper().rstrip(".") != "NONE":
            out.append((n, fact))
    return out


def learn(client, model: str, user_text: str, reply: str) -> int:
    from assistant.llm.ollama_client import _KEEP_ALIVE, _THINK  # lazy: ollama_client imports this module
    shown = [row for _s, row in knowledge.search(_KIND, user_text, k=3, min_score=_SHOWN)]
    known = "".join(f"\n{i}. {row['text']}" for i, row in enumerate(shown, 1))
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": _EXTRACT_PROMPT},
                  {"role": "user", "content": (f"ALREADY REMEMBERED:{known}\n\n" if known else "")
                                              + f"USER said: {user_text}\nAssistant replied: {reply}"}],
        keep_alive=_KEEP_ALIVE,
        think=_THINK,
        options={"num_ctx": 2048, "num_predict": 120, "temperature": 0.2},
    )
    changed, new = 0, []
    for n, fact in _parse(response["message"].get("content", "")):
        if n is not None and 1 <= n <= len(shown):
            with _lock:
                changed += knowledge.replace(_KIND, shown[n - 1]["id"], fact)
        else:
            new.append(fact)
    return changed + add_facts(new)


_LEARN_WHEN_QUIET_SECONDS = 30  # Ollama answers one call at a time: learning mid-conversation made the
_LEARN_GIVE_UP_SECONDS = 900    # user's next reply wait (real log: 20-44 s turns). Learn once they pause.


def _wait_until_quiet() -> bool:
    import time
    deadline = time.time() + _LEARN_GIVE_UP_SECONDS
    while time.time() < deadline:
        try:
            quiet = time.time() - interaction_log.LOG_PATH.stat().st_mtime
        except OSError:
            quiet = _LEARN_WHEN_QUIET_SECONDS
        if quiet >= _LEARN_WHEN_QUIET_SECONDS:
            return True
        time.sleep(5)
    return False


def learn_async(client, model: str, user_text: str, reply: str) -> threading.Thread:
    def run():
        try:
            if not _wait_until_quiet():
                return
            learn(client, model, user_text, reply)
        except Exception as exc:
            interaction_log.log_error("memory.learn", exc, text=user_text)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
