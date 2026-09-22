"""
Append-only log of real problems Vaayu hits during normal use (tool
failures, the grounding-retry firing, etc.) - one JSON object per line in
data/issues_log.jsonl, so a missed or skipped daily check-in never loses
anything (per explicit instruction: "if he misses some updates time and
shut off they are not deleted, but instead he picks them up in next
update"). Entries are marked acknowledged (never deleted) once a check-in
actually happens and the user says yes - declined or missed check-ins leave
them exactly as they were for the next attempt.
"""

import json
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = _REPO_ROOT / "data" / "issues_log.jsonl"
REPORTS_DIR = _REPO_ROOT / "data" / "daily_reports"
LAST_CHECKIN_PATH = _REPO_ROOT / "data" / "last_checkin_date.txt"


def read_last_checkin_date() -> str | None:
    """The date (YYYY-MM-DD, in whatever timezone the caller used) the daily
    check-in was last OFFERED - written regardless of whether the user said
    yes or no, so a decline doesn't get re-asked every turn the rest of the
    same day, but a missed day (Vaayu asleep/off at check-in time) still
    gets asked the next time it's actually online, even hours late."""
    if not LAST_CHECKIN_PATH.exists():
        return None
    return LAST_CHECKIN_PATH.read_text(encoding="utf-8").strip() or None


def write_last_checkin_date(date_str: str) -> None:
    LAST_CHECKIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_CHECKIN_PATH.write_text(date_str, encoding="utf-8")


def log_issue(user_said: str, tool_name: str | None, error: str, kind: str = "tool_failure") -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "kind": kind,
        "user_said": user_said,
        "tool": tool_name,
        "error": error,
        "acknowledged": False,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _read_all_entries() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupt line rather than losing the whole log
    return entries


def read_unacknowledged() -> list[dict]:
    return [e for e in _read_all_entries() if not e.get("acknowledged")]


def acknowledge_all() -> int:
    """Marks every currently-unacknowledged entry as acknowledged (rewrites
    the file in place) - call only after a successful, confirmed check-in,
    never on a missed or declined one. Returns how many were acknowledged."""
    entries = _read_all_entries()
    count = 0
    for entry in entries:
        if not entry.get("acknowledged"):
            entry["acknowledged"] = True
            count += 1
    if count:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    return count


def summarize_for_speech(issues: list[dict]) -> str:
    """Short, spoken-friendly summary - "3 issues today, mostly close_app
    and send_whatsapp_message" rather than reading every entry aloud."""
    if not issues:
        return "no issues"
    tool_counts: dict[str, int] = {}
    for issue in issues:
        name = issue.get("tool") or "general"
        tool_counts[name] = tool_counts.get(name, 0) + 1
    top_tools = sorted(tool_counts, key=tool_counts.get, reverse=True)[:3]
    count = len(issues)
    noun = "issue" if count == 1 else "issues"
    return f"{count} {noun} today, mostly around {', '.join(top_tools)}"


def write_daily_report(issues: list[dict]) -> Path:
    """Writes a human-readable markdown report for a real fix session -
    this is the actual output of a check-in, not a code change."""
    import datetime

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    report_path = REPORTS_DIR / f"{today}.md"

    lines = [f"# Vaayu issues - {today}", "", f"{len(issues)} issue(s) logged.", ""]
    for issue in issues:
        when = datetime.datetime.fromtimestamp(issue["timestamp"]).strftime("%H:%M:%S")
        lines.append(f"## {when} - {issue.get('kind', 'issue')} ({issue.get('tool') or 'general'})")
        if issue.get("user_said"):
            lines.append(f"- **User said:** {issue['user_said']}")
        lines.append(f"- **Error:** {issue.get('error', '(none given)')}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
