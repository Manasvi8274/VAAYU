"""
Wraps the local Ollama chat API with a multi-step tool-calling loop:
1. send messages + tool schemas -> model may return a tool_call
2. dispatch the tool (gated by a confirmation check for sensitive tools)
3. send the tool result back -> model may call another tool, or give a
   final natural-language reply
Repeats up to _MAX_TOOL_STEPS times, so a single spoken command can chain
several tool calls (e.g. read a page, click something, read again, click
again) instead of needing a separate command per step - upgraded from an
original one-tool-call-per-turn cap once browser_automation.py's
read_page/click_on_page made multi-step web tasks a real, common case.
"""

import re
from typing import Callable

import ollama

from assistant.self_review import issue_log
from assistant.tools.registry import ToolRegistry

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_NO_TOOL_FALLBACK = (
    "I don't actually have a tool to do that yet - I can only act through the "
    "tools I've been given."
)
# Found live, comparing qwen2.5:3b against qwen2.5:7b-instruct on the exact
# bug-case commands from this session: after the grounding nudge above, the
# 3B model sometimes attempts a tool call by generating Qwen's own raw
# <tool_call>{...}</tool_call> text format itself instead of using the
# structured tool-calling API - and when that JSON is malformed (observed:
# a missing closing quote, {"position: 3} instead of {"position": 3}),
# Ollama's parser can't extract it as a real tool_calls entry and returns it
# as plain message content instead. Without this check that garbage would
# have been spoken to the user verbatim - reproduced with a clean,
# single-model state (not a symptom of GPU/VRAM contention from testing two
# models at once, confirmed by re-running isolated).
_FAKE_TOOL_CALL_RE = re.compile(r"</?tool_call>|^\s*\{\s*[\"']name[\"']\s*:", re.IGNORECASE)
# Found live (real-usage bug hunt): a real, successful tool call (a video
# genuinely started playing) got a fabricated "[play](chrome://dino)" link
# grafted onto an otherwise-correct reply - a new hallucination shape not
# covered by _FAKE_TOOL_CALL_RE, which only guards missing/malformed tool
# calls, not garbage appended to a legitimate one. This assistant has no
# tool that returns a URL for it to relay, so any markdown link in a reply
# is by construction fabricated - strip the markup, keep the link's own
# text (usually still a genuine part of the sentence).
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPTY_REPLY_FALLBACK = (
    "Sorry, something went wrong there and I don't have a real answer - please try that again."
)
_MAX_TOOL_STEPS = 5

# Ollama unloads a model from memory after 5 minutes idle by default, and
# reloading it is slow - measured ~20s wall time on this machine for a 3B
# model vs ~0.3s once warm, entirely separate from actual inference time.
# Vaayu is meant to sit idle between commands for normal conversational use
# (minutes at a time), so without this every command after a pause paid that
# reload penalty - this is very likely the real cause behind "responses are
# taking too long". keep_alive=-1 keeps it resident until Ollama itself
# restarts, trading ~2GB of continuously-held RAM for eliminating that spike.
_KEEP_ALIVE = -1

# The default context window (4096 tokens) was measured as nearly exhausted
# by the tool schemas ALONE (3214 tokens for system prompt + one short user
# message + this project's tool set, confirmed via prompt_eval_count) -
# leaving under 900 tokens for the rest of any real conversation before
# Ollama silently drops earlier messages (including the system prompt) out
# of context. That's a second, independent latency/reliability problem, not
# just slow - it can silently degrade tool selection as a conversation grows
# or more tools get added. Raised well past current measured usage for
# headroom; not set arbitrarily high since a larger context window also
# costs more compute per call on this CPU-only setup.
_NUM_CTX = 8192

# Weak local models occasionally answer an action request in plain text
# ("Brave is now closed") without ever actually calling the tool that would
# make it true - confirmed in real use (e.g. "close brave" reported as done
# when it wasn't). If the user's own message looks like an action request and
# the model's first-pass reply called no tool at all, we give it exactly one
# corrective nudge before accepting the reply, rather than trusting free-form
# text alone to mean the action happened.
_ACTION_VERBS = (
    "open", "close", "shut", "quit", "exit", "play", "pause", "stop", "resume",
    "mute", "unmute", "send", "message", "type", "write", "click", "switch",
    "go to", "navigate", "increase", "decrease", "raise", "lower", "turn up",
    "turn down", "volume", "minimize", "maximize", "fullscreen", "full screen",
    "set", "delete", "remove", "search", "find",
)
_GROUNDING_NUDGE = (
    "You did not call any tool just now. If the user's last message asked you "
    "to perform an action, you must call the matching tool now - do not say an "
    "action happened unless a tool result actually confirms it. If there is "
    "genuinely no tool for what they asked, say so plainly instead."
)


def _looks_like_action_request(text: str) -> bool:
    lowered = text.lower()
    return any(verb in lowered for verb in _ACTION_VERBS)


def cli_confirm(name: str, arguments: dict) -> bool:
    """Default confirmation prompt for text-mode milestones (M1/M2).
    Swapped for a voice-based confirmation once the assistant can speak (M3+)."""
    answer = input(f"Confirm action '{name}' with arguments {arguments}? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def _sanitize_reply(text: str) -> str:
    """This assistant only acts through real registered tools - it must never
    present code (or a hallucinated fake tool call) as if it were executing
    something. Weak local models sometimes hallucinate a code block instead
    of correctly saying they have no tool for a request, or - confirmed
    live - a malformed <tool_call>{...}</tool_call> attempt that Ollama's
    parser couldn't extract into a real tool_calls entry; strip both out
    rather than showing/speaking fake "action" text or raw JSON. Also strips
    markdown links outright (see _MARKDOWN_LINK_RE) and unconditionally
    falls back to a real sentence if nothing usable is left - an originally
    empty reply used to pass straight through as "", which is silent dead
    air in voice mode and just a blank line in text mode, with no
    indication anything happened at all (found live)."""
    cleaned = _CODE_BLOCK_RE.sub("", text or "").strip()
    if _FAKE_TOOL_CALL_RE.search(cleaned):
        return _NO_TOOL_FALLBACK
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned).strip()
    if not cleaned:
        return _EMPTY_REPLY_FALLBACK
    return cleaned


class OllamaBrain:
    def __init__(self, host: str, model: str):
        self.client = ollama.Client(host=host)
        self.model = model

    def respond(
        self,
        messages: list[dict],
        tool_registry: ToolRegistry,
        confirm: Callable[[str, dict], bool] = cli_confirm,
    ) -> tuple[str, list[dict]]:
        any_tool_called = False
        nudged = False
        # Tracks the most recent real tool attempt's outcome, distinguishing
        # a genuine failure from the user simply declining a confirmation -
        # found live (qwen2.5:7b-instruct, "send hey there to vedant on
        # whatsapp" with no browser running): send_whatsapp_message failed
        # with a real ConnectionError, and the model's very next reply
        # claimed "Sending 'hey there' to Vedant on WhatsApp" anyway, with no
        # second tool call. any_tool_called being True by then meant the
        # nudge above never even triggered - it only guards the very first,
        # no-tool-at-all case, not "a real action was attempted and failed,
        # then falsely reported as done" - a materially worse version of the
        # same bug class, and in the single highest-stakes category (sending
        # a real message). Declining a confirmation must NOT trigger this -
        # the model correctly saying "okay, not sending that" is the wanted
        # behavior there, not something to override.
        last_action_failed = False
        last_user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_text = m.get("content", "")
                break

        for _step in range(_MAX_TOOL_STEPS):
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tool_registry.get_ollama_tools(),
                keep_alive=_KEEP_ALIVE,
                options={"num_ctx": _NUM_CTX},
            )
            assistant_msg = response["message"]
            messages = messages + [assistant_msg]

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                if last_action_failed:
                    # Never trust free-form text here, no matter what it
                    # says - the model chose not to retry with a real tool
                    # call, so the ground truth is exactly what the tool
                    # actually returned, not whatever it generates next.
                    return last_failure_reply, messages
                if (
                    not any_tool_called
                    and not nudged
                    and _looks_like_action_request(last_user_text)
                ):
                    # First pass claimed something with no tool backing it -
                    # give the model one grounded retry instead of trusting it.
                    # Also worth a daily-check-in mention even though it
                    # self-corrects here - it's a real reliability signal.
                    nudged = True
                    issue_log.log_issue(
                        user_said=last_user_text,
                        tool_name=None,
                        error="Model answered an action request with no tool call on the first pass.",
                        kind="grounding_nudge_fired",
                    )
                    messages = messages + [{"role": "system", "content": _GROUNDING_NUDGE}]
                    continue
                return _sanitize_reply(assistant_msg.get("content", "")), messages

            any_tool_called = True
            call = tool_calls[0]  # one tool call per LLM step, but steps can chain (see loop)
            name = call["function"]["name"]
            arguments = call["function"].get("arguments") or {}

            if tool_registry.requires_confirmation(name) and not confirm(name, arguments):
                tool_result = {"ok": False, "error": "User declined to confirm this action."}
                last_action_failed = False  # declining is a normal outcome, not a failure to correct
            else:
                tool_result = tool_registry.dispatch(name, arguments)
                last_action_failed = not tool_result.get("ok")
                if last_action_failed:
                    # A real tool failure, not the user declining a
                    # confirmation (that's expected behavior, not a bug).
                    error_text = str(tool_result.get("error", "something went wrong"))
                    last_failure_reply = f"That didn't work - {error_text}"
                    issue_log.log_issue(
                        user_said=last_user_text,
                        tool_name=name,
                        error=error_text,
                    )

            messages = messages + [{"role": "tool", "content": str(tool_result), "name": name}]
            # loop back: the model sees the tool result and decides whether
            # to call another tool or give a final reply

        # Hit the step cap without a final text reply - force one more call
        # with tools withheld, so the model must summarize instead of
        # attempting yet another action.
        final_response = self.client.chat(
            model=self.model, messages=messages, keep_alive=_KEEP_ALIVE, options={"num_ctx": _NUM_CTX}
        )
        final_msg = final_response["message"]
        messages = messages + [final_msg]
        return _sanitize_reply(final_msg.get("content", "")), messages
