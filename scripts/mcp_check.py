"""
Test cases for every MCP server Vaayu has - each one doing real work, checked.

    .venv/Scripts/python.exe scripts/mcp_check.py                  # every server, started directly
    .venv/Scripts/python.exe scripts/mcp_check.py designer planner # just these
    .venv/Scripts/python.exe scripts/mcp_check.py --via-vaayu      # only the APPROVED ones, through Vaayu's own
                                                                   # loader and tool registry (action rules too) -
                                                                   # exactly what Vaayu runs
Results: data/mcp_check.md (and printed). Files made by the checks go to designs/mcp-check/; designs open
nothing on screen (open_it=False).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from assistant import mcp_tools  # noqa: E402

CHECK = ROOT / "designs" / "mcp-check"
REPORT = ROOT / "data" / "mcp_check.md"
R = str(ROOT)


def _file_made(result: str) -> bool:
    m = re.search(r"saved as (\S+?)(?:,|\.$| -|$)", result)
    return bool(m) and (ROOT / m.group(1).rstrip(".")).exists() and (ROOT / m.group(1).rstrip(".")).stat().st_size > 0


# server -> [(tool, arguments, what must be in the answer (text, case-insensitive) or a check function)]
CASES = {
    "windows-apps": [
        ("app_controls", {"app": "calculator"}, "equals"),
        ("app_click", {"app": "calculator", "target": "Clear"}, "clicked"),
        ("app_click", {"app": "calculator", "target": "Five"}, "clicked"),
        ("app_click", {"app": "calculator", "target": "Plus"}, "clicked"),
        ("app_click", {"app": "calculator", "target": "Three"}, "clicked"),
        ("app_click", {"app": "calculator", "target": "Equals"}, "clicked"),
        ("app_read", {"app": "calculator"}, "display is 8"),
    ],
    "sequential-thinking": [
        ("sequentialthinking", {"thought": "To make tea I need water, tea leaves, milk and sugar.", "thoughtNumber": 1,
                                "totalThoughts": 2, "nextThoughtNeeded": True}, "thoughtnumber"),
        ("sequentialthinking", {"thought": "So: boil water, add leaves, milk and sugar, strain.", "thoughtNumber": 2,
                                "totalThoughts": 2, "nextThoughtNeeded": False}, "thoughtnumber"),
    ],
    "filesystem": [
        ("list_allowed_directories", {}, "documents"),
        ("create_directory", {"path": str(CHECK)}, "success"),
        ("write_file", {"path": str(CHECK / "hello.txt"), "content": "Hello from Vaayu"}, "success"),
        ("read_text_file", {"path": str(CHECK / "hello.txt")}, "hello from vaayu"),
        ("search_files", {"path": str(CHECK), "pattern": "hello"}, "hello.txt"),
    ],
    "playwright": [
        ("browser_navigate", {"url": "https://example.com"}, "example domain"),
        ("browser_snapshot", {}, "example domain"),
        ("browser_close", {}, ""),
    ],
    "fetch": [("fetch", {"url": "https://example.com"}, "example domain")],
    "duckduckgo": [("search", {"query": "Python programming language", "max_results": 3}, "python")],
    "wikipedia": [("search_wikipedia", {"query": "Taj Mahal", "limit": 3}, "taj"),
                  ("get_summary", {"title": "Taj Mahal"}, "agra")],
    "time": [("get_current_time", {"timezone": "Asia/Kolkata"}, "asia/kolkata"),
             ("convert_time", {"source_timezone": "Asia/Kolkata", "time": "14:30", "target_timezone": "Europe/London"},
              "europe/london")],
    "calculator": [("calculate", {"expression": "2*(3+4)**2"}, "98")],
    "git": [("git_status", {"repo_path": R}, "branch"), ("git_log", {"repo_path": R, "max_count": 2}, "commit")],
    "word": [("create_document", {"filename": str(CHECK / "word-check.docx"), "title": "MCP check"}, "creat"),
             ("add_paragraph", {"filename": str(CHECK / "word-check.docx"), "text": "Hello from Vaayu"}, "add"),
             ("get_document_text", {"filename": str(CHECK / "word-check.docx")}, "hello from vaayu")],
    "powerpoint": [("create_presentation", {}, "presentation"),
                   ("save_presentation", {"file_path": str(CHECK / "ppt-check.pptx")}, "sav")],
    "excel": [("create_workbook", {"filepath": str(CHECK / "excel-check.xlsx")}, "creat"),
              ("write_data_to_excel", {"filepath": str(CHECK / "excel-check.xlsx"), "sheet_name": "Sheet",
                                       "data": [["Item", "Price"], ["Tea", 10]], "start_cell": "A1"}, "writ"),
              ("read_data_from_excel", {"filepath": str(CHECK / "excel-check.xlsx"), "sheet_name": "Sheet"}, "tea")],
    "designer": [
        ("design_diagram", {"description": "flowchart of making tea: boil water, add tea leaves, add milk and sugar, "
                                           "strain, serve", "open_it": False}, _file_made),
        ("design_image", {"description": "a cute robot reading a book in a cozy library", "open_it": False}, _file_made),
        ("design_logo", {"description": "logo for a tea shop called Chai Point", "open_it": False}, _file_made),
        ("design_poster", {"description": "poster for a Diwali party on Saturday at 7 pm at the community hall",
                           "open_it": False}, _file_made),
        ("design_presentation", {"description": "solar energy for school students", "slides": 4, "open_it": False},
         _file_made),
        ("design_document", {"description": "a one page resume for a software tester with 3 years of Selenium and "
                                            "Python experience", "open_it": False}, _file_made),
        ("list_designs", {}, "latest designs"),
    ],
    "planner": [
        ("add_task", {"task": "Vaayu MCP check task", "due": "today"}, "added"),
        ("list_tasks", {"when": "today"}, "vaayu mcp check task"),
        ("complete_task", {"task": "vaayu mcp check task"}, "done"),
        ("make_plan", {"goal": "learn basic Spanish in one month"}, "plan"),
        ("plan_my_day", {"notes": "gym at 6 pm, call mom at 8 pm"}, "your day"),
        ("think_deeply", {"question": "A train leaves at 9:40 and the trip takes 2 hours 35 minutes, but it runs "
                                      "15 minutes late. When does it arrive?"}, "12:30"),
    ],
}
SETUP = {"windows-apps": "calculator"}  # an app the checks open first (and close after)


def _ok(result: str, expect) -> bool:
    return expect(result) if callable(expect) else expect.lower() in (result or "").lower()


def _run_cases(name: str, call) -> list[dict]:
    rows = []
    for tool, args, expect in CASES.get(name, []):
        t = time.time()
        try:
            result, ok = call(tool, args), None
            ok = _ok(result, expect)
        except Exception as exc:
            result, ok = f"ERROR: {exc}", False
        rows.append({"server": name, "tool": tool, "ok": ok, "seconds": round(time.time() - t, 1),
                     "result": re.sub(r"\s+", " ", str(result))[:220]})
        print(f"  {'PASS' if ok else 'FAIL'} {tool:<24} {rows[-1]['seconds']:>6}s  {rows[-1]['result'][:120]}", flush=True)
    return rows


def direct(names: list[str]) -> list[dict]:
    from mcp.client.stdio import StdioServerParameters
    rows = []
    for entry in json.loads(mcp_tools.LIST.read_text(encoding="utf-8")):
        name = entry["name"]
        if names and name not in names:
            continue
        print(f"{name}:", flush=True)
        app = SETUP.get(name)
        if app:
            from assistant.tools import apps
            apps.open_app(app, foreground=False)
            time.sleep(6)  # Calculator lays out its scientific mode for a moment after opening
        try:
            cmd = [c.replace("{python}", mcp_tools._python()) for c in entry["command"]]
            server = mcp_tools.Server(name, StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=R,
                                                                  env=dict(os.environ)),
                                      timeout=float(entry.get("timeout", 120)))
            rows += _run_cases(name, server.call and (lambda tool, args, s=server: s.call(tool, **args)))
        except Exception as exc:
            rows.append({"server": name, "tool": "(start)", "ok": False, "seconds": 0, "result": f"didn't start: {exc}"})
            print(f"  FAIL (start) {exc}", flush=True)
        finally:
            if app:
                apps.close_app(app)
    return rows


def via_vaayu(names: list[str]) -> list[dict]:
    from assistant.tools.registry import ToolRegistry
    registry = ToolRegistry()
    loaded = mcp_tools.load_all(registry)
    print(f"Vaayu's loader started {len(loaded)} approved server(s): {', '.join(loaded) or 'none'}", flush=True)
    rows = []
    for name in loaded:
        if names and name not in names:
            continue
        print(f"{name}:", flush=True)
        app = SETUP.get(name)
        if app:
            from assistant.tools import apps
            apps.open_app(app, foreground=False)
            time.sleep(6)  # Calculator lays out its scientific mode for a moment after opening

        def call(tool, args):
            out = registry.dispatch(tool, args, path="mcp_check")
            if not out.get("ok"):
                raise RuntimeError(out.get("error"))
            return out["result"]
        try:
            rows += _run_cases(name, call)
        finally:
            if app:
                apps.close_app(app)
    return rows


def report(rows: list[dict], mode: str) -> None:
    passed = sum(bool(r["ok"]) for r in rows)
    servers = sorted({r["server"] for r in rows})
    by = {s: [r for r in rows if r["server"] == s] for s in servers}
    lines = [f"# MCP check ({mode}) - {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"**{passed}/{len(rows)} checks passed** across {len(servers)} server(s).", "",
             "| Server | Passed | Tool | Result | Time |", "|---|---|---|---|---|"]
    for s in servers:
        ok = sum(bool(r["ok"]) for r in by[s])
        for i, r in enumerate(by[s]):
            lines.append(f"| {s if i == 0 else ''} | {f'{ok}/{len(by[s])}' if i == 0 else ''} | "
                         f"{'PASS' if r['ok'] else '**FAIL**'} {r['tool']} | {r['result'][:100].replace('|', '/')} | "
                         f"{r['seconds']} s |")
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{passed}/{len(rows)} checks passed - report: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    CHECK.mkdir(parents=True, exist_ok=True)
    if "--via-vaayu" in sys.argv:
        report(via_vaayu(args), "through Vaayu's loader")
    else:
        report(direct(args), "servers started directly")
    os._exit(0)  # the servers' threads are daemons; don't wait on their pipes
