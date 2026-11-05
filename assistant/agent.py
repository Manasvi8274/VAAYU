"""
Jobs that need several steps (user, 2026-11-05: "find 3 cheap flights to Goa and put them in a document").

"Research X and put it in a document" starts a job in the background - Vaayu says "On it, I'll tell you when it's
done" and stays usable. Each step is chosen by Claude (the user's Pro login, as tools/think.py - never API credits;
the local 4B if Claude isn't there) from Vaayu's own tools, run through the registry - so the Guardian's action
rules apply to every step - and its result goes back into the next choice. When Claude says it's done, Vaayu says
the summary out loud.

Never inside a job: anything outward or dangerous (send, post, pay, delete - guardian/action_rules.py tier 2+), and
anything that only makes sense said by the user (alarms, volume, music, voice). The summary says what to do instead
("say 'send it to Raman' if you want it sent").
"""

import json
import re
import threading
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "data" / "agent_jobs.jsonl"
MAX_STEPS = 8          # calibration knob
RESULT_CHARS = 1500    # of each step's result shown to the planner
ANNOUNCE = None        # the orchestrator sets this to Vaayu's voice
# Tools a job may use: reading, looking up, writing things down, making documents. (MCP tools join by name once loaded.)
ALLOWED = {
    "answer_question", "get_weather", "read_page", "read_file", "search_files", "list_notes", "add_note",
    "list_open_windows", "list_browser_tabs", "think_hard",  # nothing that grabs the screen: jobs run behind
    "search", "fetch", "fetch_content", "search_wikipedia", "get_summary", "get_article", "calculate",
    "get_current_time", "convert_time", "design_document", "design_presentation", "design_diagram", "design_image",
    "make_plan", "add_task", "list_tasks", "show_plan", "write_data_to_excel", "create_workbook",
    "read_data_from_excel", "get_document_text", "list_directory", "read_text_file", "directory_tree",
}
_jobs: list[dict] = []
_lock = threading.Lock()


def _tools(registry) -> dict:
    from guardian.action_rules import OUTWARD, tier
    return {n: s for n, s in registry._tools.items() if n in ALLOWED and tier(n) < OUTWARD and not s.requires_confirmation}


def _menu(tools: dict) -> str:
    lines = []
    for name, spec in sorted(tools.items()):
        props = ", ".join(f"{k}" for k in (spec.parameters or {}).get("properties", {}))
        lines.append(f"- {name}({props}): {spec.description[:160]}")
    return "\n".join(lines)


def _prompt(task: str, menu: str, steps: list[dict]) -> str:
    done = "\n".join(f"Step {i + 1}: {s['tool']}({json.dumps(s['args'], ensure_ascii=False)}) -> "
                     f"{s['result'][:RESULT_CHARS]}" for i, s in enumerate(steps)) or "(nothing yet)"
    return (f"You are the planner inside Vaayu, a voice assistant on the user's Windows laptop. Do this job for the "
            f"user, one step at a time, using ONLY these tools:\n{menu}\n\n"
            f"Job: {task}\n\nDone so far:\n{done}\n\n"
            "Reply with ONE JSON object and nothing else - either the next step "
            '{"tool": "<name>", "args": {...}} or, when the job is finished (or can\'t be done with these tools), '
            '{"done": "<2-3 short spoken sentences for the user: what you found/made and where it is>"}. '
            "Never repeat a step that already worked. You cannot send messages, pay, or delete anything - if the "
            f"job needs that, finish and tell the user what to say. At most {MAX_STEPS} steps.")


def _ask(prompt: str) -> str:
    from assistant.tools import think
    answer = think.ask_claude(prompt)
    return _local_plan(prompt) if answer is None else answer


def _local_plan(prompt: str) -> str:
    """No Claude (usage limit, not installed): the local 4B plans - slower and simpler, but the job still gets done."""
    from mcp_servers.common import ask_json
    return json.dumps(ask_json("You plan the steps of a job for a voice assistant.", prompt))


def _parse(answer: str) -> dict:
    m = re.search(r"\{.*\}", answer or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def run(task: str, registry=None, ask=None) -> dict:
    """The whole job, start to finish (blocking). Returns its record: task, steps, summary."""
    from assistant.tools.registry import registry as default
    from guardian.action_rules import CONTENT_TOOLS
    registry, ask = registry or default, ask or _ask
    tools = _tools(registry)
    job = {"task": task, "started": datetime.now().isoformat(timespec="seconds"), "steps": [], "summary": ""}
    with _lock:
        _jobs.append(job)
    for _ in range(MAX_STEPS):
        step = _parse(ask(_prompt(task, _menu(tools), job["steps"])))
        if "done" in step or not step:
            job["summary"] = str(step.get("done") or "I couldn't work out how to do that one, sorry.")
            break
        name, args = step.get("tool"), step.get("args") or {}
        if name not in tools:
            result = f"error: {name} is not one of the tools you may use"
        else:
            read_before = any(s["tool"] in CONTENT_TOOLS for s in job["steps"])
            out = registry.dispatch(name, args, after_content=read_before, path="agent")
            result = str(out.get("result") if out.get("ok") else f"error: {out.get('error')}")
        job["steps"].append({"tool": name, "args": args, "result": result})
    else:
        job["summary"] = f"I did {MAX_STEPS} steps and stopped there - " + (job["steps"][-1]["result"][:200]
                                                                              if job["steps"] else "")
    job["finished"] = datetime.now().isoformat(timespec="seconds")
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return job


def start(task: str) -> str:
    """Runs the job in the background; Vaayu says the summary when it's done."""
    def go():
        try:
            job = run(task)
            summary = job["summary"]
        except Exception as exc:
            summary = f"Sorry, the job '{task[:60]}' went wrong: {exc}"
        if ANNOUNCE:
            ANNOUNCE(f"Done with your task. {summary}")
    threading.Thread(target=go, daemon=True, name="agent-job").start()
    return "On it - I'll work on that in the background and tell you when it's done."


def status() -> str:
    with _lock:
        running = [j for j in _jobs if "finished" not in j]
    if not running:
        return "No task running right now."
    j = running[-1]
    return f"Still working on '{j['task'][:60]}' - {len(j['steps'])} steps done so far."
