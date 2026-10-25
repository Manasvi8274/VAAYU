"""
Alarms and timers - "wake me up at 1:30 pm" / "set a timer for 10 minutes".

When one is due, Vaayu keeps calling out (every ~10 s, speakers turned up if they were low, any song
paused) until the user answers - "I'm awake", "I'm up", "uth gaya", "okay stop" - or 30 minutes pass.
"5 more minutes" snoozes. Kept in data/alarms.json, so they survive a restart.
"""

import json
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "alarms.json"
_RING_EVERY = 10.0       # seconds between wake-up calls
_RING_FOR = 30 * 60      # give up after this long
_MIN_VOLUME = 60         # an alarm nobody can hear is no alarm: speakers at least this loud while ringing
_lock = threading.Lock()
_ringing: dict | None = None
_ring_started = 0.0

_ACK = re.compile(r"\b(?:i'?m|i am|am) (?:awake|up|getting up)\b|\b(?:i'?m|i am) wake\b|\bwoke up\b|\bawake\b|"
                  r"\b(?:uth|utth) (?:gaya|gayi|gaye|raha|rahi)\b|\bjaag (?:gaya|gayi)\b|"
                  r"\b(?:stop|off|cancel|band karo|dismiss)(?: the)?(?: alarm| it| timer)?\b|^(?:okay|ok|haan|ha|yes|bas)\b",
                  re.I)
_SNOOZE = re.compile(r"\b(?P<n>\d+|five|ten|two|few) (?:more )?(?:minutes?|mins?)\b|\bsnooze\b", re.I)
_WORD_NUM = {"two": 2, "five": 5, "ten": 10, "few": 5}


def _load() -> list[dict]:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _save(rows: list[dict]) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def parse_time(said: str, now: datetime | None = None) -> datetime:
    """"1:30 pm", "13:30", "1:30" (the next 1:30 - am or pm, whichever comes first), "7", "7 am", "half past 6"."""
    now = now or datetime.now()
    s = said.strip().lower().replace(".", "")
    s = re.sub(r"\bhalf past (\d{1,2})", r"\1:30", s)
    s = re.sub(r"\b(?:baje|o'?clock)\b", "", s)
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a m|p m|in the morning|in the evening|at night|subah|shaam|raat)?", s)
    if not m:
        raise ValueError(f"I didn't catch a time in '{said}' - say it like 'wake me up at 7:30 am'.")
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(" ", "")
    if hour > 23 or minute > 59:
        raise ValueError(f"{said} isn't a time.")
    if half in ("pm", "intheevening", "atnight", "shaam", "raat") and hour < 12:
        hour += 12
    elif half in ("am", "inthemorning", "subah") and hour == 12:
        hour = 0
    candidates = [hour] if half or hour > 12 or hour == 0 else [hour, (hour + 12) % 24]
    times = []
    for h in candidates:
        t = now.replace(hour=h, minute=minute, second=0, microsecond=0)
        times.append(t if t > now else t + timedelta(days=1))
    return min(times)


def set_alarm(when: str, label: str = "") -> str:
    at = parse_time(when)
    return _add(at, "alarm", label)


def set_timer(seconds: int, label: str = "") -> str:
    if seconds <= 0:
        raise ValueError("How long should the timer be?")
    return _add(datetime.now() + timedelta(seconds=seconds), "timer", label)


def _add(at: datetime, kind: str, label: str) -> str:
    with _lock:
        rows = _load()
        rows.append({"id": f"{at:%Y%m%d%H%M%S}-{len(rows)}", "at": at.isoformat(timespec="seconds"), "kind": kind,
                     "label": label})
        _save(rows)
    if kind == "timer":
        left = int((at - datetime.now()).total_seconds() + 0.5)
        mins, secs = divmod(left, 60)
        return f"Timer set for {mins} minute{'s' * (mins != 1)}" + (f" {secs} seconds." if secs else ".")
    day = "today" if at.date() == datetime.now().date() else "tomorrow"
    return f"Alarm set for {at:%I:%M %p} {day}.".replace(" 0", " ")


def list_alarms() -> str:
    rows = sorted(_load(), key=lambda r: r["at"])
    if not rows:
        return "You have no alarms or timers set."
    parts = [f"{r['kind']} at {datetime.fromisoformat(r['at']):%I:%M %p}".replace(" 0", " ") for r in rows]
    return "You have " + ", ".join(parts) + "."


def cancel(which: str = "all") -> str:
    with _lock:
        rows = _load()
        if not rows:
            return "There's no alarm or timer to cancel."
        if which in ("", "all"):
            _save([])
            return f"Cancelled {len(rows)} alarm{'s' * (len(rows) != 1)}."
        try:
            at = parse_time(which)
            keep = [r for r in rows if datetime.fromisoformat(r["at"]).strftime("%H:%M") != at.strftime("%H:%M")]
        except ValueError:
            keep = [r for r in rows if r["kind"] not in which]
        _save(keep)
        return "Cancelled." if len(keep) < len(rows) else "I couldn't find that alarm."


# --- ringing ---------------------------------------------------------------------------

def is_ringing() -> bool:
    return _ringing is not None


def is_ack(text: str) -> bool:
    return bool(_ACK.search(text or ""))


def snooze_minutes(text: str) -> int | None:
    m = _SNOOZE.search(text or "")
    if not m:
        return None
    n = m.group("n")
    return int(n) if n and n.isdigit() else _WORD_NUM.get(n or "", 5)


def dismiss(text: str = "") -> str:
    """The user answered. Snooze if they asked for more minutes, else stop."""
    global _ringing
    alarm, _ringing = _ringing, None
    if alarm is None:
        return "No alarm is ringing."
    mins = snooze_minutes(text)
    if mins:
        set_timer(mins * 60, "snooze")
        return f"Okay, {mins} more minutes."
    return "Good, you're up! Have a great day." if alarm["kind"] == "alarm" else "Okay, timer stopped."


def due() -> dict | None:
    """Pops and returns the next alarm whose time has come."""
    with _lock:
        rows = _load()
        now = datetime.now()
        for r in sorted(rows, key=lambda r: r["at"]):
            if datetime.fromisoformat(r["at"]) <= now:
                rows.remove(r)
                _save(rows)
                return r
    return None


def start_scheduler(speak, set_volume=None) -> None:
    """Background loop: when an alarm is due, call out until dismissed (or 30 min)."""
    def loop():
        global _ringing, _ring_started
        while True:
            time.sleep(1)
            if _ringing is None:
                alarm = due()
                if alarm is None:
                    continue
                _ringing, _ring_started = alarm, time.time()
                restore = set_volume(_MIN_VOLUME) if set_volume else None
                try:
                    while _ringing is alarm and time.time() - _ring_started < _RING_FOR:
                        at = datetime.fromisoformat(alarm["at"])
                        what = (f"Wake up! It's {at:%I:%M %p}." if alarm["kind"] == "alarm"
                                else "Time's up! Your timer is done.").replace(" 0", " ")
                        speak(f"{what} {alarm.get('label') or ''} Say 'I'm awake' to stop.".replace("  ", " "))
                        for _ in range(int(_RING_EVERY * 10)):
                            if _ringing is not alarm:
                                break
                            time.sleep(0.1)
                finally:
                    if _ringing is alarm:
                        _ringing = None
                    if restore:
                        restore()
    threading.Thread(target=loop, daemon=True, name="alarms").start()


if __name__ == "__main__":
    base = datetime(2026, 10, 25, 9, 0)
    assert parse_time("1:30 pm", base).hour == 13
    assert parse_time("13:30", base).strftime("%H:%M") == "13:30"
    assert parse_time("1:30", base).strftime("%H:%M") == "13:30"  # next 1:30 after 9 am is 1:30 pm
    assert parse_time("7", base).strftime("%H:%M") == "19:00"
    assert parse_time("7 am", base).day == 26
    assert parse_time("half past 6 in the morning", base).strftime("%d %H:%M") == "26 06:30"
    assert is_ack("I'm awake") and is_ack("ok I am up") and is_ack("uth gaya") and not is_ack("what time is it")
    assert snooze_minutes("5 more minutes") == 5 and snooze_minutes("I'm awake") is None
    print("ok")
