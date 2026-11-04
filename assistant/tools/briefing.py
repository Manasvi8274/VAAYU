"""
Jarvis-style: Vaayu speaks up on his own.

  daily_briefing  "good morning" / "what's my day" - the time, the weather where you are, your to-dos due today
                  (the planner's list, data/planner/tasks.json) and your alarms, in one go.
  watch_battery   a background check: says once that the battery is low (unplugged, 20% or less), and once more at 10%.
"""

import json
import threading
import time
from datetime import date, datetime
from pathlib import Path

import psutil

from assistant.tools.registry import registry

TASKS = Path(__file__).resolve().parent.parent.parent / "data" / "planner" / "tasks.json"
LOW, CRITICAL = 20, 10  # calibration knobs: battery % warnings


def _tasks_today() -> list[str]:
    try:
        rows = json.loads(TASKS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    today = date.today().isoformat()
    return [r["task"] for r in rows if not r.get("done") and r.get("due") and r["due"] <= today]


@registry.register(
    name="daily_briefing",
    description="A morning briefing: the time, today's weather, the to-dos due today and the alarms set. For 'good "
                "morning', 'what's my day', 'brief me', 'aaj ka kya plan hai'.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def daily_briefing() -> str:
    now = datetime.now()
    hour = now.hour
    hello = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    parts = [f"{hello}! It's {now.strftime('%I:%M %p').lstrip('0')}, {now.strftime('%A %d %B')}."]
    try:
        from assistant.tools import web_answer
        parts.append(web_answer.get_weather(""))
    except Exception:
        pass  # no internet - the rest of the briefing still helps
    tasks = _tasks_today()
    parts.append(f"You have {len(tasks)} thing{'s' if len(tasks) != 1 else ''} to do today: " + "; ".join(tasks[:5]) + "."
                 if tasks else "Nothing on your to-do list for today.")
    try:
        from assistant import alarms
        said = alarms.list_alarms()
        if said and "no alarm" not in said.lower():
            parts.append(said)
    except Exception:
        pass
    return " ".join(parts)


def battery_warning(percent: float, plugged: bool, said: set) -> str | None:
    """What to say about the battery now, once per level; said remembers what was said (reset when charging)."""
    if plugged:
        said.clear()
        return None
    for level in (CRITICAL, LOW):
        if percent <= level and level not in said:
            said.update({level, LOW})
            return (f"Bhai, the battery is at {int(percent)} percent - plug in the charger soon."
                    if level == LOW else f"Battery is at {int(percent)} percent. Plug in now or the laptop will shut down.")
    return None


def watch_battery(speak, every: float = 60.0) -> None:
    def loop():
        said: set = set()
        while True:
            try:
                b = psutil.sensors_battery()
                if b is not None and (msg := battery_warning(b.percent, b.power_plugged, said)):
                    speak(msg)
            except Exception:
                pass
            time.sleep(every)
    threading.Thread(target=loop, daemon=True, name="battery-watch").start()
