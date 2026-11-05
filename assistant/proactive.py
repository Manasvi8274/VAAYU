"""
Vaayu speaking up on his own (#12) - each kind off until the user turns it on ("turn on proactive mode" for all,
"turn on break reminders" for one; data/proactive.json):

  welcome  back after an hour away -> "Welcome back. It's 3:40. You have 2 things due today: ..."
  breaks   two hours at the laptop without a 5-minute pause -> a nudge to stretch
  night    still at it after 1 AM -> once a night, a nudge to sleep

"Away" and "at it" come from the keyboard and mouse (Windows' last-input time), not the microphone.
"""

import ctypes
import json
import threading
import time
from datetime import datetime
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "proactive.json"
KINDS = {"welcome": "welcome back greetings", "breaks": "break reminders", "night": "late night reminders"}
AWAY = 60 * 60          # calibration knobs (seconds)
PAUSE = 5 * 60
WORK = 2 * 60 * 60
NIGHT_HOUR = 1
_lock = threading.Lock()


def settings() -> dict:
    try:
        saved = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    return {k: bool(saved.get(k, False)) for k in KINDS}


def switch(kind: str | None, on: bool) -> str:
    """kind None = all of them."""
    with _lock:
        now = settings()
        for k in ([kind] if kind else KINDS):
            now[k] = on
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(now), encoding="utf-8")
    what = KINDS[kind] if kind else "speaking up on my own - welcome backs, breaks and late nights"
    return f"Okay, {what} {'on' if on else 'off'}."


def idle_seconds() -> float:
    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
    info = LASTINPUTINFO(ctypes.sizeof(LASTINPUTINFO), 0)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    return max(0.0, (ctypes.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0)


def _due_today() -> list[str]:
    try:
        tasks = json.loads((PATH.parent / "planner" / "tasks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    today = datetime.now().date().isoformat()
    return [t["task"] for t in tasks if not t.get("done") and t.get("due") and t["due"] <= today]


class Watcher:
    """What to say, given time and idleness - no threads or clocks of its own, so it can be tested."""

    def __init__(self):
        self.away_since = None       # when the current absence started (None = present)
        self.working_since = None    # start of the current stretch without a PAUSE
        self.night_said = ""         # the date a night nudge was last said

    def tick(self, now: datetime, idle: float, enabled: dict) -> str | None:
        t = now.timestamp()
        if idle >= AWAY:
            self.away_since = self.away_since or t - idle
            self.working_since = None
            return None
        if idle >= PAUSE:
            self.working_since = None
            return None
        came_back = self.away_since is not None
        self.away_since = None
        self.working_since = self.working_since or t - idle
        if came_back and enabled.get("welcome"):
            due = _due_today()
            said = f"Welcome back. It's {now:%I:%M %p}".replace(" 0", " ") + "."
            if due:
                said += f" You have {len(due)} thing{'s' if len(due) > 1 else ''} due today: " + ", ".join(due[:3]) + "."
            return said
        if enabled.get("breaks") and t - self.working_since >= WORK:
            self.working_since = t
            return "You've been at it for two hours, bhai. Stand up and stretch for a minute - I'll be here."
        if enabled.get("night") and NIGHT_HOUR <= now.hour < 5 and self.night_said != now.date().isoformat():
            self.night_said = now.date().isoformat()
            return f"It's {now:%I:%M}".replace(" 0", " ").lstrip("0") + " already, bhai. Get some sleep soon - tomorrow will thank you."
        return None


def start(speak, busy=lambda: False, every: float = 30.0) -> threading.Thread:
    watcher = Watcher()

    def run():
        while True:
            time.sleep(every)
            try:
                enabled = settings()
                if not any(enabled.values()):
                    continue
                said = watcher.tick(datetime.now(), idle_seconds(), enabled)
                if said and not busy():
                    speak(said)
            except Exception:
                pass  # never let this break Vaayu
    thread = threading.Thread(target=run, daemon=True, name="proactive")
    thread.start()
    return thread
