"""
Real thinking, on request: "think about X", "think deeply about X", "ask Claude X".

The local brain (3B) answers in 1-2 s but can't reason deeply; the 4B with its thinking mode takes minutes on the
GTX 1650 (measured 2026-11-05). So a question the user explicitly asks Vaayu to think about goes to Claude through
the user's own Claude Pro login (the `claude` CLI, as the daily review uses it - never API credits): ~10 s, said
"Let me think about that properly." first (orchestrator ACK). Only these questions leave the laptop. If Claude isn't
there (not installed, usage limit), the local 4B thinks instead (mcp_servers/common.py ask_thinking).
"""

import os
import re
import shutil
import subprocess
import tempfile

from assistant.tools.registry import registry

MODEL = "sonnet"   # calibration knob: "haiku" answers in about the same time on this laptop
TIMEOUT = 90
_LIMIT = re.compile(r"usage limit|rate limit|hit your .*limit", re.I)


def _prompt(question: str) -> str:
    from assistant.llm.ollama_client import is_hindi
    lang = ("Answer in simple spoken Hindi, in Devanagari script." if is_hindi(question)
            else "Answer in clear, simple spoken English.")
    return ("You are the reasoning engine of Vaayu, a voice assistant who talks like a caring elder brother (bhai). "
            "Think the question through carefully, then give ONLY the final answer: 2-4 short sentences that will be "
            f"spoken aloud - no markdown, no lists, no preamble. {lang}\n\nQuestion: {question}")


def _local(question: str) -> str:
    from mcp_servers.common import ask_thinking
    return ask_thinking(question)


@registry.register(
    name="think_hard",
    description="Think a hard question, decision or problem through properly before answering (about 10 s) - only "
                "when the user asks Vaayu to think about something, think deeply, or ask Claude.",
    parameters={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
)
def think_hard(question: str) -> str:
    claude = shutil.which("claude")
    if claude:
        try:
            # the prompt on stdin: `claude` is a .cmd on Windows, and a line break in an argument cut it off there
            # (live: Claude answered "I didn't catch your question")
            run = subprocess.run([claude, "-p", "--model", MODEL, "--output-format", "text"], input=_prompt(question),
                                 capture_output=True, text=True, encoding="utf-8", errors="replace",
                                 timeout=TIMEOUT, cwd=tempfile.gettempdir(),
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), env=dict(os.environ))
            answer = (run.stdout or "").strip()
            if run.returncode == 0 and answer and not _LIMIT.search(answer):
                return re.sub(r"[*#`]+", "", answer)
        except (OSError, subprocess.TimeoutExpired):
            pass
    answer = _local(question)  # minutes on this GPU, but an answer
    if not answer:
        raise RuntimeError("I thought about it but couldn't reach an answer.")
    return answer
