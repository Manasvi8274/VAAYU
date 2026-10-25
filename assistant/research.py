"""
Learning about the world while idle - Vaayu reads, and remembers what he read.

When nobody has spoken for a while (config research.idle_minutes) and
nothing is playing, a background thread picks a topic - something the user
asked about recently (from data/knowledge/asked.jsonl, to go deeper), or one
of their interests from what they watch (youtube_taste) - searches it, reads
the top page or two, and has the same local model write 3-4 factual
sentences from that text only. The summary goes into knowledge "notes" with
its source URL, so later answers (and offline answers, web_answer.py) can
use it. Budgeted: research.per_hour / per_day topics at most.

Every note is also appended to data/knowledge/research_log.jsonl, so the
user can see what he learned and delete anything bad.
"""

import json
import re
import threading
import time
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import httpx

from assistant import interaction_log, knowledge

_PAGE_BYTES = 200_000
_PAGE_CHARS = 6000  # what the 3B model reads per topic (~1500 tokens)
_TIMEOUT = 6
_CHECK_SECONDS = 60
_REDO_DAYS = 7  # a topic researched within this many days isn't picked again
_SUMMARY_PROMPT = ("Summarise the text below for a personal assistant's notes: 3 or 4 short factual sentences "
                   "about the topic, using ONLY facts stated in the text - no opinions, no guesses, no "
                   "markdown. If the text has nothing useful about the topic, reply exactly NOTHING.")


def _log_path() -> Path:
    return knowledge.DIR / "research_log.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


# --- reading a page ------------------------------------------------------------------------------

class _Text(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        self._skip += tag in self._SKIP

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def html_to_text(html: str) -> str:
    parser = _Text()
    parser.feed(html)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def _fetch(url: str) -> str:
    with httpx.stream("GET", url, timeout=_TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (Vaayu personal assistant)"}) as r:
        r.raise_for_status()
        if "html" not in r.headers.get("content-type", "html"):
            return ""
        body = b""
        for chunk in r.iter_bytes():
            body += chunk
            if len(body) >= _PAGE_BYTES:
                break
    return html_to_text(body.decode(r.encoding or "utf-8", errors="ignore"))


# --- what to read about -----------------------------------------------------------------------------

def _recent_topics(days: int = _REDO_DAYS) -> set[str]:
    since = (datetime.now() - timedelta(days=days)).isoformat()
    return {row["topic"].lower() for row in _read_jsonl(_log_path()) if row.get("ts", "") >= since}


def next_topic() -> str | None:
    done = _recent_topics()
    asked = [row["q"] for row in reversed(_read_jsonl(knowledge.DIR / "asked.jsonl")[-50:])]
    for q in asked:  # newest question first: what they care about right now
        if q.lower() not in done:
            return q
    from assistant.tools import youtube_taste
    channels, topics = youtube_taste.profile()
    for name, _w in channels.most_common(5) + topics.most_common(5):
        topic = f"latest news about {name}"
        if topic.lower() not in done:
            return topic
    return None


# --- one topic -------------------------------------------------------------------------------------------

def research(topic: str, client, model: str, still_idle=lambda: True) -> dict | None:
    """Search, read, summarise, remember. The note row, or None if nothing useful (or no longer idle)."""
    from assistant.tools.web_answer import _ddg_text
    results = _ddg_text(topic)  # ConnectionError when offline - the caller backs off
    text, url = "", ""
    for r in results[:3]:
        href = r.get("href") or ""
        if not href.startswith("http") or "youtube.com" in href:
            continue
        try:
            page = _fetch(href)
        except Exception:
            continue
        if len(page) > 300:
            text, url = (text + " " + page)[:_PAGE_CHARS], url or href
        if len(text) >= _PAGE_CHARS:
            break
    if not text:  # no readable page: the search snippets are still something
        text = " ".join(f"{r.get('title', '')}: {r.get('body', '')}" for r in results)[:_PAGE_CHARS]
        url = (results[0].get("href") if results else "") or ""
    if not text.strip() or not still_idle():  # the user started talking - the GPU is theirs
        return None
    response = client.chat(
        model=model,
        messages=[{"role": "system", "content": _SUMMARY_PROMPT},
                  {"role": "user", "content": f"Topic: {topic}\n\nText:\n{text}"}],
        keep_alive=-1,
        think=False,
        options={"num_ctx": 4096, "num_predict": 180, "temperature": 0.2},  # chat's context size: no model reload
    )
    summary = (response["message"].get("content") or "").strip()
    if not summary or summary.upper().startswith("NOTHING") or len(summary) < 40:
        _log({"ts": _now(), "topic": topic, "url": url, "note": None})
        return None
    row, updated = knowledge.add("notes", summary, {"url": url, "topic": topic, "date": _now()[:10]})
    _log({"ts": _now(), "topic": topic, "url": url, "note": summary, "id": row["id"], "updated": updated})
    return row


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(entry: dict) -> None:
    knowledge.DIR.mkdir(parents=True, exist_ok=True)
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- the background loop ---------------------------------------------------------------------------------

def _budget_left(per_hour: int, per_day: int, now: float | None = None) -> bool:
    now = now or time.time()
    stamps = []
    for row in _read_jsonl(_log_path()):
        try:
            stamps.append(datetime.fromisoformat(row["ts"]).timestamp())
        except (KeyError, ValueError):
            continue
    today = datetime.fromtimestamp(now).date()
    return (sum(now - t < 3600 for t in stamps) < per_hour
            and sum(datetime.fromtimestamp(t).date() == today for t in stamps) < per_day)


def idle(idle_minutes: float, now: float | None = None) -> bool:
    """Nobody has spoken (nothing logged) for idle_minutes, and no media is playing."""
    from assistant import orchestrator
    now = now or time.time()
    try:
        last = interaction_log.LOG_PATH.stat().st_mtime
    except OSError:
        last = 0.0
    return now - last >= idle_minutes * 60 and not orchestrator._media_playing()


def start(config, client) -> threading.Thread | None:
    """The idle-time research thread (Orchestrator.run_forever starts it)."""
    knowledge.warm_up()  # memory and the answer cache need the embedding model too, research or not
    cfg = config.research
    if not cfg.enabled:
        return None

    def loop():
        backoff = 0.0
        while True:
            time.sleep(_CHECK_SECONDS + backoff)
            try:
                if not (idle(cfg.idle_minutes) and _budget_left(cfg.per_hour, cfg.per_day)):
                    continue
                topic = next_topic()
                if topic:
                    research(topic, client, config.llm.model, still_idle=lambda: idle(cfg.idle_minutes))
                backoff = 0.0
            except ConnectionError:
                backoff = min(backoff + 300, 1800)  # offline: check back less often
            except Exception as exc:
                interaction_log.log_error("research", exc)
                backoff = min(backoff + 300, 1800)
    thread = threading.Thread(target=loop, daemon=True, name="research")
    thread.start()
    return thread
