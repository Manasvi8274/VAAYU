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
from datetime import datetime
from typing import Callable

import ollama

from assistant import memory
from assistant.llm import router
from assistant.llm.prompts import BROTHER_PROMPT
from assistant.self_review import issue_log
from assistant.tools import names, notes, web_answer
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
_TOOL_NAME_RE = re.compile(r"\b[a-z]+_[a-z_]+\b")
_DIDNT_GET_IT = "Sorry, I didn't get that - could you say it again?"
_MAX_SPOKEN_CHARS = 220

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

# Qwen3.5 reasons in a <think> block before every reply by default - several
# seconds of dead air per step in a voice assistant. Qwen3.5 dropped Qwen3's
# /no_think prompt switch, so this must go through Ollama's own `think` field.
_THINK = False

# A benchmark run hung for 8+ minutes on one ordinary command (not
# reproducible on retry - sampling randomness or an Ollama stall). No call
# may block Vaayu forever: cap reply length, and time out the HTTP call.
_MAX_REPLY_TOKENS = 300
# models/Modelfile samples at temperature 1.5 - far too random for picking tools
# and arguments. Commands run near-deterministic; chat keeps some warmth.
_TOOL_TEMPERATURE = 0.2
_CHAT_TEMPERATURE = 0.7
_CALL_TIMEOUT_SECONDS = 60

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


# Conversation vs command, for whatever fast_route didn't handle. Whole words
# here ("upset" is not "set"). Strong verbs are commands even inside personal
# talk ("I'm bored, play something"); weak ones aren't ("I want to quit my
# job" is life advice, not close_app).
_STRONG_VERBS = re.compile(r"\b(?:open|play|pause|resume|mute|unmute|volume|increase|decrease|raise|lower|turn up|"
                           r"turn down|minimi[sz]e|maximi[sz]e|full ?screen|search|go to|navigate|click|switch|press|"
                           r"keys?|hotkey|summari[sz]e)\b|^\w+ chahiye$", re.I)
_WEAK_VERBS = re.compile(r"\b(?:close|shut|quit|exit|stop|send|message|type|write|set|delete|remove|find)\b", re.I)
_TOOL_TOPIC = re.compile(r"\b(?:battery|charg\w*|wi-?fi|internet|bluetooth|awaa?z|videos?|youtube|tabs?|windows?|apps?|"
                         r"screen\w*|files?|folders?|notes?|whatsapp|browser|brave|chrome|edge|music|songs?|gaana|mic|"
                         r"speakers?|usb|printers?|monitors?|cpu|ram|disk|memory|storage|wallpaper|page|website|"
                         r"spreadsheet|documents?|pdf|excel|firefox|auto|quality|\d+p)\b", re.I)
_PERSONAL = re.compile(r"\b(?:i|i'm|im|i've|i'd|my|me|myself|mujhe|mujhko|mera|meri|mere|main|mai|hum|life|feel|"
                       r"feeling|bhai|tum|tu|aap|you think|advice|opinion)\b", re.I)
_HINDI_WORDS = re.compile(r"\b(?:hai|hain|kya|aap|aapko|ko|ka|ki|ke|mein|nahi|karo|kar|hum|apna|fir|phir|yeh|ye|toh|bhi|kuch|kaise|accha|achha|raha|rahi|karne|karna|hoga|hogi|wala|sakte|sakta)\b", re.I)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Few-shot turns: told only in words, the 3B model answered Hinglish in Devanagari. Its Roman
# Hinglish was tested too (and after training on Claude-written Hinglish): fluent-sounding but
# meaningless, so Hinglish is understood and answered in plain English (prompts.BROTHER_PROMPT).
_CHAT_EXAMPLES = [
    {"role": "user", "content": "bhai aaj padhai ka bilkul mann nahi hai"},
    {"role": "assistant", "content": "Bhai, that happens to everyone. Set a 25 minute timer and start with one small "
                                     "topic - starting is the hardest part, the rest follows."},
    {"role": "user", "content": "I keep putting off my workouts"},
    {"role": "assistant", "content": "Then make it too small to skip - ten minutes, same time every day. Once it's a "
                                     "habit you can make it longer. Consistency beats intensity."},
]


def _romanize(text: str) -> str:
    """Last resort for a Devanagari reply: Roman letters the voice can read ("क्या हाल है" -> "kya hal hai")."""
    import unicodedata
    try:
        from indic_transliteration import sanscript
    except ImportError:
        return "Sorry bhai, I got my words mixed up - ask me once more?"
    out = re.sub(r"(?<=[^\Waāiīuū])a\b", "", sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST))
    out = out.replace("ṃ", "n").replace("|", ".").replace("।", ".")  # anusvara, danda
    return "".join(c for c in unicodedata.normalize("NFKD", out) if not unicodedata.combining(c))


_FORGET_ALL = re.compile(r"^(?:forget|bhool jao?) (?:everything|all of it|sab kuch|sab)\b.*$|"
                         r"^forget (?:what|everything) i (?:told|said to) you$", re.I)
_FORGET_ABOUT = re.compile(r"^forget (?:what i (?:told|said to) you |everything i (?:told|said to) you )?about (?P<a>.+)$|"
                           r"^(?P<a2>.+?) (?:ke baare mein |wali baat )?bhool jao?$", re.I)
_CHAT_TURNS = 6
_CHAT_FACTS = 8


def _is_conversation(text: str) -> bool:
    t = router.normalize(text)
    if not t or _STRONG_VERBS.search(t) or _TOOL_TOPIC.search(t):
        return False
    if any(re.search(rf"\b{re.escape(app.lower())}\b", t, re.I) for app in names.load()["app"]):
        return False
    # Starts like a command ("close it", "check this") - "tell me..."/"...batao" are questions, fine to chat.
    if router._ACTION_START.search(t) and not re.search(r"^tell me\b|\bbatao$", t, re.I):
        return False
    return bool(_PERSONAL.search(t)) or not _WEAK_VERBS.search(t)


def cli_confirm(name: str, arguments: dict) -> bool:
    """Default confirmation prompt for text-mode milestones (M1/M2).
    Swapped for a voice-based confirmation once the assistant can speak (M3+)."""
    answer = input(f"Confirm action '{name}' with arguments {arguments}? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


_PLACEHOLDER_RE = re.compile(r"\[(?:current|today|insert|your|date|time|day|name|location|city)[^\]]{0,30}\]", re.I)


def _now_line() -> str:
    """The real date and time for every model call. Real log: with no clock, the chat model said
    "today is Monday" on a Sunday, then flip-flopped, then answered "The current time is [current time]"."""
    now = datetime.now()
    return (f"\nRight now it is {now.strftime('%A')}, {now.day} {now.strftime('%B %Y')}, "
            f"{now.strftime('%I:%M %p').lstrip('0')} on this laptop's clock (India). Use exactly this for any "
            "question about today's date, day or time - never guess it.")


def _with_tool(registry: ToolRegistry, name: str) -> ToolRegistry:
    """The registry the call goes to, with `name` in it - the LLM loop only offers a top-5 subset."""
    if name in registry._tools:
        return registry
    from assistant.tools.registry import registry as full
    return full


def _with_now(messages: list[dict]) -> list[dict]:
    """messages with the clock added to the system prompt, for this call only (history stays clean)."""
    if messages and messages[0].get("role") == "system":
        return [{**messages[0], "content": messages[0]["content"] + _now_line()}] + messages[1:]
    return [{"role": "system", "content": _now_line().strip()}] + messages


def _sanitize_reply(text: str, max_sentences: int = 2) -> str:
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
    # Real log: replies read the system prompt back ("use either close_app or
    # open_app - neither is about closing the browser itself") and ran 15-20 s
    # of speech. A snake_case tool name is never meant for the user's ears.
    if _PLACEHOLDER_RE.search(cleaned):  # "The current time is [current time]" - a template, not an answer
        return _DIDNT_GET_IT
    if _TOOL_NAME_RE.search(cleaned):
        return _DIDNT_GET_IT
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    if len(cleaned) > _MAX_SPOKEN_CHARS and len(sentences) > max_sentences:
        cleaned = " ".join(sentences[:max_sentences])
    return cleaned


class OllamaBrain:
    def __init__(self, host: str, model: str, fast_paths: bool = True):
        self.client = ollama.Client(host=host, timeout=_CALL_TIMEOUT_SECONDS)
        self.model = model
        # router.py's fast route / tool subset / direct replies. False keeps
        # the plain every-tool LLM loop (the old behavior, used as the
        # benchmark baseline and by the LLM-loop unit tests).
        self.fast_paths = fast_paths
        # What the last respond() did, for data/interactions.jsonl: which path
        # handled it, every LLM call and every tool call with its outcome.
        # The caller resets it per turn; multi-step sub-turns append to it.
        self.trace: list[dict] = []

    def _dispatch(self, registry: ToolRegistry, name: str, arguments: dict, path: str) -> dict:
        routed = router.speakers_or_browser(name, arguments, getattr(self, "_user_text", ""))
        if routed[0] != name:  # the model picked YouTube for a song (real log) - songs play through the speakers
            self.trace.append({"path": path, "rerouted": {"from": name, "to": routed[0]}})
            name, arguments = routed
            registry = _with_tool(registry, name)
        heard = arguments
        arguments = names.correct_arguments(name, arguments)  # "Ogru" -> "OG Crew" (see tools/names.py)
        if arguments != heard:
            self.trace.append({"path": path, "name_corrected": {"from": heard, "to": arguments}})
        from guardian.action_rules import CONTENT_TOOLS
        read_content = path == "llm" and any(t.get("tool") in CONTENT_TOOLS and t.get("ok") for t in self.trace)
        result = _with_tool(registry, name).dispatch(name, arguments, after_content=read_content, path=path)
        self.trace.append({"path": path, "tool": name, "args": arguments, "ok": bool(result.get("ok")),
                           "result" if result.get("ok") else "error": str(result.get("result" if result.get("ok") else "error"))})
        return result

    def _phrase_result(self, user_text: str, name: str, result) -> str:
        """One small, tool-free LLM call that turns a data result (wifi
        details, mic level...) into a spoken sentence."""
        self.trace.append({"path": "llm_call", "purpose": "phrase_result", "tool": name})
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer the user's question in one or two short spoken sentences "
                                              "using only this tool result - state the answer directly. No markdown. "
                                              "Reply in the same language as the question: English, or Hinglish "
                                              "written in Roman letters." + _now_line()},
                {"role": "user", "content": f"Question: {user_text}\nResult of {name}: {result}"},
            ],
            keep_alive=_KEEP_ALIVE,
            think=_THINK,
            options={"num_ctx": 2048, "num_predict": _MAX_REPLY_TOKENS, "temperature": _TOOL_TEMPERATURE},
        )
        reply = _sanitize_reply(response["message"].get("content", ""))
        return _romanize(reply) if _DEVANAGARI.search(reply) else reply  # live: "Droupadi Murmu India ka 15वीं राष्ट्रपति"

    def _chat(self, messages: list[dict], user_text: str) -> tuple[str, list[dict]]:
        """Personal talk, opinions, advice: the big-brother persona, no tools,
        with what Vaayu remembers about the user. Learning new facts from this
        turn is the orchestrator's job (real turns only, never benchmarks)."""
        t = router.normalize(user_text)
        forgetting = bool(_FORGET_ALL.match(t) or _FORGET_ABOUT.match(t))
        self.trace.append({"path": "chat_forget" if forgetting else "chat"})  # nothing to learn from a forget
        if _FORGET_ALL.match(t):
            memory.forget()
            reply = "Done, I've forgotten everything you told me."
            return reply, messages + [{"role": "assistant", "content": reply}]
        if m := _FORGET_ABOUT.match(t):
            removed = memory.forget(m.group("a") or m.group("a2"))
            reply = "Done, I've forgotten that." if removed else "I didn't have anything saved about that."
            return reply, messages + [{"role": "assistant", "content": reply}]
        system = BROTHER_PROMPT + _now_line()
        if known := memory.facts_for(user_text, k=_CHAT_FACTS):  # the relevant ones + the newest few
            system += "\nWhat you remember about them:\n" + "\n".join(f"- {f}" for f in known)
        try:
            saved = [n["text"] for n in notes.list_notes(5)]
        except Exception:  # notes DB unreadable - chat still works
            saved = []
        if saved:
            system += "\nTheir recent notes:\n" + "\n".join(f"- {n}" for n in saved)
        history = []
        for turn in memory.recent_turns(_CHAT_TURNS):
            history += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["reply"]}]
        convo = [{"role": "system", "content": system}, *_CHAT_EXAMPLES, *history, {"role": "user", "content": user_text}]
        for attempt in range(2):
            response = self.client.chat(
                model=self.model,
                messages=convo,
                keep_alive=_KEEP_ALIVE,
                think=_THINK,
                options={"num_ctx": 4096, "num_predict": 200, "temperature": _CHAT_TEMPERATURE if attempt == 0 else 0.3},
            )
            reply = _sanitize_reply(response["message"].get("content", ""), max_sentences=4)
            if not _DEVANAGARI.search(reply) and len(_HINDI_WORDS.findall(reply)) < 3:
                break
            # Live: the 3B model answered in Devanagari (the voice can't speak it), or in garbled Hinglish
            # ("Bhai, kya aapko kisi problem hai ya update karne hai?") - replies are meant to be English.
            self.trace.append({"path": "chat", "language_retry": True})
            convo = convo + [{"role": "system", "content": "Write that reply again in clear, simple English only - "
                                                           "you may start with 'Bhai,'."}]
        if _DEVANAGARI.search(reply):
            reply = _romanize(reply)
        return reply, messages + [{"role": "assistant", "content": reply}]

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
        # Found in data/issues_log.jsonl: after a failure the model re-issued
        # the exact same failing call (switch_to_window 'Brave' 5x,
        # control_media 'pause' 5x), burning every step at ~3-4s each -
        # ~15-20s of silence for one command. One identical retry is kept
        # (transient failures do recover - see the flaky-close test), a
        # second one ends the turn with the real failure instead.
        last_failed_call = None
        identical_retries = 0
        last_user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_text = m.get("content", "")
                break
        self._user_text = last_user_text  # for _dispatch's song -> speakers rule

        if self.fast_paths:
            routed = router.fast_route(last_user_text)
            if routed and routed[0] == "reply":
                self.trace.append({"path": "fast_reply"})
                return routed[1], messages + [{"role": "assistant", "content": routed[1]}]
            if routed and all(name in tool_registry._tools for name, _args in routed[1]):
                replies = []
                for name, arguments in routed[1]:  # multi-step: in order, stop at the first failure
                    if tool_registry.requires_confirmation(name, arguments) and not confirm(name, arguments):
                        self.trace.append({"path": "fast", "tool": name, "args": arguments, "declined": True})
                        replies.append("Okay, I won't do that.")
                        break
                    tool_result = self._dispatch(tool_registry, name, arguments, "fast")
                    if not tool_result.get("ok"):
                        error_text = str(tool_result.get("error", "something went wrong"))
                        issue_log.log_issue(user_said=last_user_text, tool_name=name, error=error_text)
                        replies.append(f"That didn't work - {error_text}")
                        break
                    spoken = router.format_result(name, tool_result.get("result"))
                    if not spoken:
                        spoken = self._phrase_result(last_user_text, name, tool_result.get("result"))
                        if name == "answer_question":  # instant next time (weather/time never reach here)
                            web_answer.remember_answer(arguments.get("question", last_user_text), spoken)
                    replies.append(spoken)
                reply = " ".join(replies)
                return reply, messages + [{"role": "assistant", "content": reply}]
            steps = router.split_steps(last_user_text)
            if steps:
                # "open spotify and play some music": each step as its own
                # command, in order - a step that fails stops the rest.
                # (Steps can't split again: split_steps needs a connector.)
                base = messages[:-1] if messages and messages[-1].get("role") == "user" else messages
                self.trace.append({"path": "steps", "steps": steps})
                replies = []
                for step in steps:
                    step_reply, _ = self.respond(base + [{"role": "user", "content": step}], tool_registry, confirm)
                    replies.append(step_reply)
                    if step_reply.startswith("That didn't work"):
                        break
                reply = " ".join(replies)
                return reply, messages + [{"role": "assistant", "content": reply}]
            if _is_conversation(last_user_text):
                return self._chat(messages, last_user_text)
            tool_registry = router.select_tools(last_user_text, tool_registry)

        for _step in range(_MAX_TOOL_STEPS):
            self.trace.append({"path": "llm_call", "tools_offered": list(tool_registry._tools)})
            response = self.client.chat(
                model=self.model,
                messages=_with_now(messages),
                tools=tool_registry.get_ollama_tools(),
                keep_alive=_KEEP_ALIVE,
                think=_THINK,
                options={"num_ctx": _NUM_CTX, "num_predict": _MAX_REPLY_TOKENS, "temperature": _TOOL_TEMPERATURE},
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
                    self.trace.append({"path": "llm", "grounding_nudge": True, "model_said": assistant_msg.get("content", "")})
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

            if self.fast_paths and (blocked := router.blocked_reply(name)):
                self.trace.append({"path": "llm", "tool": name, "args": arguments, "blocked": True})
                # e.g. WhatsApp: the model swapped "SAVED MESSAGES" for "mom"
                # in the benchmark - recipients only ever come from fast_route.
                return blocked, messages + [{"role": "assistant", "content": blocked}]

            if self.fast_paths and (unsure := router.ungrounded(name, arguments, last_user_text)):
                self.trace.append({"path": "llm", "tool": name, "args": arguments, "ungrounded": True})
                return unsure, messages + [{"role": "assistant", "content": unsure}]

            if last_action_failed and (name, arguments) == last_failed_call:
                identical_retries += 1
                if identical_retries > 1:
                    return last_failure_reply, messages

            if _with_tool(tool_registry, name).requires_confirmation(name, arguments) and not confirm(name, arguments):
                tool_result = {"ok": False, "error": "User declined to confirm this action."}
                self.trace.append({"path": "llm", "tool": name, "args": arguments, "declined": True})
                last_action_failed = False  # declining is a normal outcome, not a failure to correct
            else:
                tool_result = self._dispatch(tool_registry, name, arguments, "llm")
                last_action_failed = not tool_result.get("ok")
                if last_action_failed:
                    # A real tool failure, not the user declining a
                    # confirmation (that's expected behavior, not a bug).
                    error_text = str(tool_result.get("error", "something went wrong"))
                    last_failure_reply = f"That didn't work - {error_text}"
                    last_failed_call = (name, arguments)
                    issue_log.log_issue(
                        user_said=last_user_text,
                        tool_name=name,
                        error=error_text,
                    )

            messages = messages + [{"role": "tool", "content": str(tool_result), "name": name}]
            if self.fast_paths and tool_result.get("ok") and name in router.DIRECT_REPLY_TOOLS:
                reply = router.format_result(name, tool_result.get("result"))
                if reply:
                    return reply, messages + [{"role": "assistant", "content": reply}]
            # loop back: the model sees the tool result and decides whether
            # to call another tool or give a final reply

        # Hit the step cap without a final text reply - force one more call
        # with tools withheld, so the model must summarize instead of
        # attempting yet another action.
        self.trace.append({"path": "llm_call", "purpose": "forced_summary"})
        final_response = self.client.chat(
            model=self.model,
            messages=_with_now(messages),
            keep_alive=_KEEP_ALIVE,
            think=_THINK,
            options={"num_ctx": _NUM_CTX, "num_predict": _MAX_REPLY_TOKENS, "temperature": _TOOL_TEMPERATURE},
        )
        final_msg = final_response["message"]
        messages = messages + [final_msg]
        return _sanitize_reply(final_msg.get("content", "")), messages
