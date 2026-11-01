"""
Detects the assistant's wake/sleep/shutdown voice commands ("wake up Vaayu",
"sleep Vaayu", "shutdown Vaayu"). Keyword-based, not exact-phrase, matching -
STT output for a short command varies run to run ("wake up vaayu" / "wake up
the AI" / etc.) - requiring the name "Vaayu" alongside a control word keeps
this robust to that noise while avoiding false triggers from ordinary
conversation (e.g. "I need to sleep soon" alone must never put the assistant
to sleep - it only reacts when its own name is said too).
"""

import re
from enum import Enum


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


def _name_present(normalized: str) -> bool:
    # Accept both spellings ("Vayu" and "Vaayu" are both legitimate
    # romanizations) since STT output for this word could not be reliably
    # validated in testing (see PLAN.md) - no reason to be stricter than
    # necessary about which one comes out.
    return "vaayu" in normalized or "vayu" in normalized or bool(_MISHEARD_NAME.search(normalized))


# Real log (2026-10-25): "Vaayu, pause the audio" came out as "Why you pause the audio" / "Why you post
# the audio" / "Why you lower the volume" at least 5 times - and, the name missed, it was dropped as the
# song's own audio. "why you" counts as the name only when a command follows it ("why you did that" doesn't).
_COMMAND_AFTER = (r"(?=[\s,.!?]+(?:pause|post|pose|play|stop|resume|open|close|lower|reduce|increase|raise|turn|set|"
                  r"change|mute|unmute|skip|next|start|search|send|show|tell|go|take|lock|volume|switch|put|bring)\b)")
_MISHEARD_NAME = re.compile(r"\b(?:why|wai|vai|whai|bye) (?:you|u|yu)\b" + _COMMAND_AFTER, re.IGNORECASE)
_NAME_WORD_RE = re.compile(r"(?:\b(?:vaa?yu)\b|\b(?:why|wai|vai|whai|bye) (?:you|u|yu)\b" + _COMMAND_AFTER +
                           r")[\s,.!?:;-]*", re.IGNORECASE)


def text_after_name(text: str) -> str:
    """What was said after the LAST mention of Vaayu's name - "...sponsored
    by X. Vaayu, pause the video" -> "pause the video". Used while media is
    playing, when everything before the name is likely the video's audio."""
    last = None
    for last in _NAME_WORD_RE.finditer(text):
        pass
    return text[last.end():].strip() if last else text.strip()


def command_part(text: str) -> str:
    """The command in a sentence with Vaayu's name in it: what came after the name, or - when the name came
    last - what came before it, if that's short (real log: "Not able to hear the song Vaayu." was taken as a
    bare "Vaayu" and ignored). '' if it's only the name."""
    rest = text_after_name(text)
    if has_command_content(rest):
        return rest
    before = _NAME_WORD_RE.sub(" ", text).strip(" ,.!?")
    return before if has_command_content(before) and len(before.split()) <= 12 else ""


def is_wake_command(text: str) -> bool:
    normalized = _normalize(text)
    return _name_present(normalized) and "wake" in normalized


def is_sleep_command(text: str) -> bool:
    normalized = _normalize(text)
    return _name_present(normalized) and "sleep" in normalized


def is_shutdown_command(text: str) -> bool:
    normalized = _normalize(text)
    has_shutdown = "shutdown" in normalized or ("shut" in normalized and "down" in normalized)
    return _name_present(normalized) and has_shutdown


_WAKE_FILLER_WORDS = {"wake", "up", "vaayu", "vayu", "hey", "hi", "hello", "ok", "okay", "key"}


def is_name_mentioned(text: str) -> bool:
    """Broader than is_wake_command - true whenever Vaayu's name is said at
    all, regardless of whether "wake" is also present. Used to support
    "Vaayu, open brave" as a single combined wake-and-command utterance,
    not just a bare "wake up Vaayu"."""
    return _name_present(_normalize(text))


def has_command_content(text: str) -> bool:
    """True if there are meaningful words beyond the wake trigger/name
    itself - distinguishes "Vaayu, open brave" (wake + command in one
    breath) from a bare "wake up Vaayu" (just waking, nothing else to do)."""
    normalized = _normalize(text)
    remaining = [w for w in normalized.split() if w not in _WAKE_FILLER_WORDS]
    return len(remaining) > 0


class ControlCommand(Enum):
    NONE = "none"
    WAKE = "wake"
    SLEEP = "sleep"
    SHUTDOWN = "shutdown"


def detect_control_command(text: str) -> ControlCommand:
    # Checked in this order so a phrase that could ambiguously match more
    # than one doesn't silently pick the less consequential interpretation.
    if is_shutdown_command(text):
        return ControlCommand.SHUTDOWN
    if is_wake_command(text):
        return ControlCommand.WAKE
    if is_sleep_command(text):
        return ControlCommand.SLEEP
    return ControlCommand.NONE


_AFFIRMATIVE_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed",
    "correct", "right", "affirmative", "please", "go",
}
_AFFIRMATIVE_PHRASES = ("go ahead", "do it", "send it", "sounds good")
_NEGATIVE_WORDS = {
    "no", "nope", "nah", "dont", "stop", "cancel", "wait", "negative", "never", "not",
}


def is_affirmative(text: str) -> bool:
    """Loose yes/no detection for spoken confirmations ("should I send this?"
    -> "yes" / "go ahead" / "no, don't"). Checked negative-first so a reply
    like "no, don't send it" (which also contains no clearly-affirmative
    words) is correctly read as a decline, not a false affirmative from
    something incidental matching. "not" is deliberately in the negative set
    even though it can misfire on rare phrasing like "yes, why not, go
    ahead" - found via testing that without it, "that's not right"/"not
    correct" (clear declines) both matched the affirmative word "right"/
    "correct" and returned True. For a gate that sends a real message on the
    user's behalf, a false affirmative is much worse than a false negative
    (worst case there, the user just says yes again) - biased accordingly."""
    normalized = _normalize(text)
    if not normalized.strip():
        return False
    words = set(normalized.split())
    if words & _NEGATIVE_WORDS:
        return False
    if any(phrase in normalized for phrase in _AFFIRMATIVE_PHRASES):
        return True
    return bool(words & _AFFIRMATIVE_WORDS)
