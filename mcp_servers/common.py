"""Shared by Vaayu's own MCP servers (designer, planner): say every failure plainly, and ask the local model."""

import functools
import json
import re
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
WRITER = "qwen3.5:4b-q4_K_M"  # calibration knob: the bigger local model that writes plans, slides, documents


def said(fn):
    """The SDK hides an unexpected error's message ("Error executing tool x") - Vaayu should say what went wrong,
    so every failure goes back as a ToolError. (windows_apps.py has its own copy: changing that file would need
    a new Guardian report.)"""
    from mcp.server.mcpserver.exceptions import ToolError

    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(str(exc) or type(exc).__name__) from exc
    return run


def _chat(body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["message"]
    except OSError as exc:
        raise RuntimeError(f"My writing model isn't answering ({exc}) - is Ollama running?") from exc


def ask_json(system: str, prompt: str, timeout: float = 300) -> dict:
    """The writing model's answer as JSON (Ollama's JSON mode keeps a small model to the format)."""
    msg = _chat({"model": WRITER, "stream": False, "think": False, "format": "json", "keep_alive": "2m",
                 "options": {"num_ctx": 4096, "temperature": 0.5},
                 "messages": [{"role": "system", "content": system + " Reply with JSON only."},
                              {"role": "user", "content": prompt}]}, timeout)
    try:
        return json.loads(msg.get("content") or "{}")
    except ValueError as exc:
        raise RuntimeError("My writing model's answer wasn't usable - try asking again.") from exc


THINK_TOKENS = 2000  # calibration knob: ~2 minutes of thinking on the GTX 1650


def ask_thinking(prompt: str, timeout: float = 600) -> str:
    """A considered answer: the model thinks first (slow - a few minutes - but deeper than Vaayu's quick brain).
    If its thinking uses up the budget before it answers (found: 8 minutes of thinking, no answer), the answer is
    asked for from that thinking."""
    system = "Think it through carefully, then answer clearly and briefly - the answer is spoken aloud. No markdown."
    msg = _chat({"model": WRITER, "stream": False, "think": True, "keep_alive": "2m",
                 "options": {"num_ctx": 8192, "num_predict": THINK_TOKENS, "temperature": 0.6},
                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}, timeout)
    answer = (msg.get("content") or "").strip()
    if not answer and msg.get("thinking"):
        msg = _chat({"model": WRITER, "stream": False, "think": False, "keep_alive": "2m",
                     "options": {"num_ctx": 8192, "num_predict": 300, "temperature": 0.3},
                     "messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": f"Question: {prompt}\n\nYour reasoning so far:\n"
                                                              f"{msg['thinking'][-6000:]}\n\nNow the final answer, "
                                                              "in one or two sentences."}]}, timeout)
        answer = (msg.get("content") or "").strip()
    return re.sub(r"[*#`]+", "", answer)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40] or "design"
