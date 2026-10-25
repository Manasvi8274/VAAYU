"""
What the user watches on YouTube, and their taste from it - like YouTube's
own "because you watched": every video that plays in the browser Vaayu
controls is logged (title, channel, when), whoever started it, and
recommendations are ranked by the channels and topics they come back to.

Also matches what the user says ("the stand-up one by Pranav Sharma", "the
one posted 12 hours ago", "the third one") to the videos on the page.

Stored in data/watch_history.jsonl (per user, gitignored).
"""

import difflib
import json
import math
import re
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "watch_history.jsonl"
_HALF_LIFE_DAYS = 14  # calibration knob: how fast old viewing stops counting toward taste
_POLL_SECONDS = 30
_lock = threading.Lock()

_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "with", "from", "at", "is", "it", "this",
         "that", "which", "was", "video", "videos", "one", "play", "open", "start", "click", "right", "side", "sidebar",
         "recommended", "recommendation", "recommendations", "suggested", "posted", "uploaded", "ago", "currently",
         "played", "playing", "watch", "show", "me", "please", "vaayu", "official", "full", "new", "hd", "4k", "ka",
         "ki", "ke", "wala", "wali", "wale", "hai", "jo", "se", "par", "pe", "chalao", "lagao", "kholo"}
_ORDINALS = {"first": 1, "1st": 1, "pehla": 1, "pehli": 1, "second": 2, "2nd": 2, "doosra": 2, "dusra": 2, "doosri": 2,
             "third": 3, "3rd": 3, "teesra": 3, "teesri": 3, "fourth": 4, "4th": 4, "chautha": 4, "fifth": 5, "5th": 5,
             "sixth": 6, "6th": 6, "last": -1}
_AGE = [(r"(\d+)\s*(?:hours?|hrs?|h)\b", "h"), (r"(\d+)\s*(?:days?|d)\b", "d"), (r"(\d+)\s*(?:weeks?|w)\b", "w"),
        (r"(\d+)\s*(?:months?|mo)\b", "mo"), (r"(\d+)\s*(?:years?|y)\b", "y"), (r"(\d+)\s*(?:minutes?|mins?|m)\b", "min")]


# --- history ----------------------------------------------------------------------

def record(video_id: str, title: str, channel: str) -> bool:
    """Log a watched video; False if it's the same as the last one logged."""
    if not video_id or not title:
        return False
    with _lock:
        last = history(1)
        if last and last[-1]["id"] == video_id:
            return False
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "id": video_id, "title": title,
                                "channel": channel or ""}, ensure_ascii=False) + "\n")
    return True


def history(n: int | None = None) -> list[dict]:
    try:
        lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-n:] if n else lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _words(text: str) -> list[str]:
    t = (text or "").lower()
    for pattern, unit in _AGE:  # "12 hours ago" and YouTube's "12h ago" become the same token
        t = re.sub(pattern, lambda m: m.group(1) + unit, t)
    return [w for w in re.findall(r"[a-z0-9]+", t) if w not in _STOP and len(w) > 1]


def profile(now: float | None = None) -> tuple[Counter, Counter]:
    """(channel weights, topic-word weights), recent viewing counting most."""
    now = now or time.time()
    channels, topics = Counter(), Counter()
    for row in history(2000):
        try:
            age_days = (now - datetime.fromisoformat(row["ts"]).timestamp()) / 86400
        except (KeyError, ValueError):
            continue
        w = 0.5 ** (max(age_days, 0) / _HALF_LIFE_DAYS)
        if row.get("channel"):
            channels[row["channel"].lower()] += w
        for word in set(_words(row.get("title", ""))):
            topics[word] += w
    return channels, topics


def rank(candidates: list[dict]) -> list[dict]:
    """Candidates the user would most likely pick first; YouTube's own order breaks ties.
    Already-watched videos go last."""
    channels, topics = profile()
    watched = history(2000)
    # By title too: live, it suggested another upload of the song that was playing.
    seen = {row["id"] for row in watched} | {_title_key(row.get("title", "")) for row in watched}
    top_ch = max(channels.values(), default=1) or 1
    top_t = max(topics.values(), default=1) or 1

    def score(c):
        s = 3 * channels.get((c.get("channel") or "").lower(), 0) / top_ch
        words = set(_words(c.get("title", "")))
        if words:
            s += sum(topics.get(w, 0) for w in words) / top_t / math.sqrt(len(words))
        return s - (10 if c.get("id") in seen or _title_key(c.get("title", "")) in seen else 0)
    return sorted(candidates, key=score, reverse=True)


def _title_key(title: str) -> str:
    """Same song/video across uploads: "Cheri Cheri Lady (Lyrics) | ..." ~ "Cheri Cheri Lady (Official Video)"."""
    # Episode numbers kept ("vlog 1" != "vlog 2"); only upload-style words dropped.
    words = re.findall(r"[a-z0-9]+", re.split(r"[|(\[]", title or "")[0].lower())
    return " ".join(w for w in words if w not in _UPLOAD_WORDS)[:80]


_UPLOAD_WORDS = {"official", "video", "lyrics", "lyric", "audio", "hd", "4k", "full", "music", "new", "the"}


# --- matching what the user said to a video on the page ----------------------------------------------

def ordinal(description: str) -> int | None:
    words = re.findall(r"[a-z0-9]+", (description or "").lower())
    picked = [w for w in words if w in _ORDINALS or (w.isdigit() and len(w) == 1)]
    rest = [w for w in words if w not in _ORDINALS and not w.isdigit() and w not in _STOP and w != "number"]
    if len(picked) == 1 and not rest:
        return _ORDINALS.get(picked[0]) or int(picked[0])
    return None


def _similar(a: str, b: str) -> float:
    from assistant.tools.names import sound_key
    return max(difflib.SequenceMatcher(None, a, b).ratio(),
               difflib.SequenceMatcher(None, sound_key(a), sound_key(b)).ratio() if len(a) > 3 else 0)


def match(description: str, candidates: list[dict]) -> tuple[dict | None, float]:
    """Best candidate for the words said, and how sure (0-1: share of the said words found)."""
    said = _words(description)
    if not said or not candidates:
        return None, 0.0
    best, best_score = None, 0.0
    for c in candidates:
        have = _words(f"{c.get('title', '')} {c.get('channel', '')} {c.get('age', '')}")
        if not have:
            continue
        got = 0.0
        for w in said:
            s = max(_similar(w, h) for h in have)
            got += 1.0 if s >= 0.8 else 0.6 if s >= 0.7 else 0.0
        score = got / len(said)
        if score > best_score:
            best, best_score = c, score
    return best, best_score


# --- logging what plays, whoever started it ---------------------------------------------------------------

def start_watch_logger() -> None:
    """Every 30 s: if a YouTube video is playing in the browser Vaayu controls, log it."""
    def loop():
        from assistant.tools import youtube_player
        while True:
            time.sleep(_POLL_SECONDS)
            try:
                tab = youtube_player.youtube_tab(require_watch=True)
                if tab and youtube_player.is_playing(tab):
                    youtube_player.record_current(tab)
            except Exception:
                pass  # browser closed / not Vaayu's - nothing to log
    threading.Thread(target=loop, daemon=True, name="watch-logger").start()


if __name__ == "__main__":
    import tempfile
    HISTORY_PATH = Path(tempfile.mkdtemp()) / "h.jsonl"
    recs = [{"id": "a", "title": "Sinking In | Stand-up Comedy by Pranav Sharma", "channel": "Pranav Sharma", "age": "12h ago"},
            {"id": "b", "title": "Arijit Singh Top 10", "channel": "T-Series", "age": "2y ago"},
            {"id": "c", "title": "OG Crew new vlog", "channel": "OG Crew", "age": "3d ago"}]
    assert match("syncing in stand up comedy by pranav sharma", recs)[0]["id"] == "a"
    assert match("the one posted 12 hours ago", recs)[0]["id"] == "a"
    assert ordinal("second video") == 2 and ordinal("the 3") == 3 and ordinal("pranav sharma") is None
    record("c", "OG Crew vlog 1", "OG Crew")
    assert rank(recs)[0]["id"] == "c" or rank(recs)[-1]["id"] == "c"  # watched: last
    record("x", "OG Crew vlog 2", "OG Crew")
    assert [r["id"] for r in rank(recs)][-1] == "c"
    print("ok")
