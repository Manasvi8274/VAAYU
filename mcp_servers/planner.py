"""
"planner": Vaayu's own MCP server for planning and thinking - plans for a goal, a to-do list with due days,
"plan my day", and thinking a hard question through. The plans and steps are written by the bigger local
model (mcp_servers/common.py); the lists are kept in data/planner/ (this folder only).

    .venv/Scripts/python.exe -m mcp_servers.planner         # stdio server (Vaayu starts it)
"""

import difflib
import json
import re
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from mcp_servers.common import ask_json, ask_thinking, said  # noqa: E402

DATA = REPO / "data" / "planner"
PLANS, TASKS = DATA / "plans.json", DATA / "tasks.json"
server = MCPServer("planner", instructions="Plans, to-do lists, the day's schedule, and thinking things through.")

_DAYS = {"today": 0, "aaj": 0, "tomorrow": 1, "kal": 1, "day after tomorrow": 2, "parso": 2}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _load(path: Path) -> list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _save(path: Path, rows: list) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


def due_date(when: str, today: date | None = None) -> str:
    """'today' / 'tomorrow' / 'friday' / '2026-11-02' / '' -> an ISO date ('' = no date)."""
    today, w = today or date.today(), (when or "").strip().lower()
    if not w:
        return ""
    if w in _DAYS:
        return (today + timedelta(days=_DAYS[w])).isoformat()
    for i, name in enumerate(_WEEKDAYS):
        if name in w:
            return (today + timedelta(days=(i - today.weekday()) % 7 or 7)).isoformat()
    try:
        return date.fromisoformat(w).isoformat()
    except ValueError:
        return ""


def _spoken(day: str) -> str:
    if not day:
        return ""
    d = date.fromisoformat(day)
    return {0: "today", 1: "tomorrow"}.get((d - date.today()).days, d.strftime("%A %d %B"))


@server.tool()
@said
def make_plan(goal: str, timeframe: str = "") -> str:
    """Make a step-by-step plan for a goal (study for an exam, a trip, learning a skill, a project) and keep it."""
    spec = ask_json('You make practical plans. JSON: {"title": short title, "steps": [{"step": what to do, '
                    '"when": when or how long (e.g. "Day 1", "Week 2", "30 min")}]} with 4-10 concrete steps.',
                    f"Goal: {goal}" + (f"\nTimeframe: {timeframe}" if timeframe else ""))
    steps = [s for s in spec.get("steps") or [] if isinstance(s, dict) and s.get("step")]
    if not steps:
        raise RuntimeError("I couldn't come up with a plan for that - say the goal a bit differently?")
    plans = _load(PLANS)
    plan = {"id": uuid.uuid4().hex[:8], "title": str(spec.get("title") or goal)[:80], "goal": goal,
            "made": datetime.now().isoformat(timespec="seconds"), "steps": steps}
    _save(PLANS, plans + [plan])
    lines = "; ".join(f"{s.get('when', '')}: {s['step']}".strip(": ") for s in steps[:6])
    return f"Plan '{plan['title']}', {len(steps)} steps - {lines}" + ("; and more." if len(steps) > 6 else ".")


@server.tool()
@said
def show_plan(name: str = "") -> str:
    """Read out a saved plan - the newest, or the one named."""
    plans = _load(PLANS)
    if not plans:
        return "There are no plans yet."
    plan = plans[-1]
    if name:
        plan = max(plans, key=lambda p: difflib.SequenceMatcher(None, name.lower(), p["title"].lower()).ratio())
    return f"{plan['title']}: " + "; ".join(f"{s.get('when', '')}: {s['step']}".strip(": ") for s in plan["steps"])


@server.tool()
@said
def list_plans() -> str:
    """The saved plans, newest first."""
    plans = _load(PLANS)
    return ("Plans: " + "; ".join(p["title"] for p in reversed(plans[-10:]))) if plans else "There are no plans yet."


@server.tool()
@said
def add_task(task: str, due: str = "") -> str:
    """Add something to the to-do list - due 'today', 'tomorrow', a weekday or a date (optional)."""
    tasks = _load(TASKS)
    row = {"id": uuid.uuid4().hex[:8], "task": task.strip(), "due": due_date(due), "done": False,
           "added": datetime.now().isoformat(timespec="seconds")}
    _save(TASKS, tasks + [row])
    return f"Added '{row['task']}'" + (f", due {_spoken(row['due'])}." if row["due"] else " to your to-do list.")


@server.tool()
@said
def list_tasks(when: str = "") -> str:
    """The to-do list - everything open, or what's due on a day ('today', 'tomorrow', a weekday)."""
    open_tasks = [t for t in _load(TASKS) if not t["done"]]
    day = due_date(when)
    if day:
        open_tasks = [t for t in open_tasks if t["due"] and t["due"] <= day]  # due then, or overdue
    if not open_tasks:
        return "Nothing on your list" + (f" for {when}." if when else ".")
    return f"{len(open_tasks)} to do: " + "; ".join(
        t["task"] + (f" ({_spoken(t['due'])})" if t["due"] else "") for t in open_tasks[:15])


@server.tool()
@said
def complete_task(task: str) -> str:
    """Tick off a task on the to-do list (by what it says)."""
    tasks = _load(TASKS)
    open_tasks = [t for t in tasks if not t["done"]]
    if not open_tasks:
        raise RuntimeError("There's nothing open on your list.")
    best = max(open_tasks, key=lambda t: difflib.SequenceMatcher(None, task.lower(), t["task"].lower()).ratio())
    if difflib.SequenceMatcher(None, task.lower(), best["task"].lower()).ratio() < 0.45 and \
            task.lower() not in best["task"].lower():
        raise RuntimeError(f"I can't find '{task}' on your list.")
    best["done"] = True
    _save(TASKS, tasks)
    return f"Done: {best['task']}."


@server.tool()
@said
def plan_my_day(notes: str = "") -> str:
    """A schedule for the rest of today from the to-do list (and anything said, e.g. 'gym at 6')."""
    today = date.today().isoformat()
    due = [t["task"] for t in _load(TASKS) if not t["done"] and (not t["due"] or t["due"] <= today)]
    if not due and not notes:
        return "Your list is empty for today - tell me what you want to get done and I'll plan it."
    spec = ask_json('You plan a realistic day. JSON: {"schedule": [{"time": "HH:MM", "do": str}]} - start from the '
                    'time given, keep breaks and meals, most important first.',
                    f"It is {datetime.now():%H:%M}. To do: {'; '.join(due) or 'nothing listed'}. Notes: {notes or '-'}")
    items = [s for s in spec.get("schedule") or [] if isinstance(s, dict) and s.get("do")]
    if not items:
        raise RuntimeError("I couldn't make a schedule - try telling me the tasks again.")
    return "Your day: " + "; ".join(f"{s.get('time', '')} {s['do']}".strip() for s in items[:12])


@server.tool()
@said
def think_deeply(question: str) -> str:
    """Think a hard question, decision or problem through carefully before answering (slow: a minute or two)."""
    answer = ask_thinking(question)
    if not answer:
        raise RuntimeError("I thought about it but couldn't reach an answer.")
    return re.sub(r"\s+", " ", answer)[:1500]


if __name__ == "__main__":
    server.run("stdio")
