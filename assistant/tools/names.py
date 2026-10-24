"""
The user's own names - YouTube channels, WhatsApp chats, apps - and sound-
alike correction of what speech-to-text made of them.

Found live: "OG Crew" came out as "OG Grill", "Ogru", "Ogkru", "Odin krue",
"OGCREO", and spelled "O-G-C-R-E-W". Whisper has no idea what "OG Crew" is,
but Vaayu can: an argument that sounds like a known name becomes that name.
Only close matches count, so a genuinely new name is left alone, and the
reply says the corrected name so a wrong correction is easy to notice.

Stored in data/known_names.json (per user, gitignored); add by voice with
"remember the channel X" / "remember the contact X".
"""

import difflib
import json
import re
from pathlib import Path

from assistant.tools.registry import registry

PATH = Path(__file__).resolve().parent.parent.parent / "data" / "known_names.json"
DEFAULTS = {
    "channel": ["OG Crew", "Carryminati", "Dhruv Rathee", "Technical Guruji", "Samay Raina", "MrBeast", "Lofi Girl",
                "T-Series", "BB Ki Vines", "Total Gaming", "Mortal", "Linus Tech Tips", "Ranveer Allahbadia"],
    "contact": ["SAVED MESSAGES", "Raman Garg"],
    "app": ["Tekken 7", "Armoury Crate", "WhatsApp", "Brave", "VS Code", "Spotify", "Discord", "Telegram",
            "Notepad", "Calculator", "File Explorer", "Task Manager", "Steam", "Zoom", "Outlook"],
}
# Which tool argument holds which kind of name.
ARG_KINDS = {
    ("play_latest_video_from_channel", "channel_name"): "channel",
    ("send_whatsapp_message", "contact_name"): "contact",
    ("open_app", "app_name"): "app",
    ("close_app", "app_name"): "app",
}
MATCH_THRESHOLD = 0.75  # calibration knob: sound-alike similarity (0-1) needed to correct a name


def load() -> dict[str, list[str]]:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {kind: list(dict.fromkeys(data.get(kind, []) + DEFAULTS[kind])) for kind in DEFAULTS}


def _save(names: dict[str, list[str]]) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    extra = {k: [n for n in v if n not in DEFAULTS[k]] for k, v in names.items()}
    PATH.write_text(json.dumps(extra, indent=1, ensure_ascii=False), encoding="utf-8")


_SOUNDS = [("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "s"), ("w", "u"), ("ew", "u"),
           ("oo", "u"), ("ee", "i"), ("y", "i"), ("v", "b"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
           ("gh", "g"), ("sh", "s"), ("ch", "k")]


_DIGITS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
           "eight": "8", "nine": "9", "ten": "10"}


def sound_key(text: str) -> str:
    """Spelling-independent key: "OG Crew", "Ogkru", "o-g-c-r-e-w" all -> "ogkru"."""
    t = (text or "").lower()
    t = " ".join(_DIGITS.get(w, w) for w in t.split())  # "tekken seven" == "tekken 7"
    t = re.sub(r"\b([a-z])(?:[\s.-]+(?=[a-z]\b))", r"\1", t)  # spelled letters: "o g c r e w" -> "ogcrew"
    t = re.sub(r"[^a-z0-9]", "", t)
    for a, b in sorted(_SOUNDS, key=lambda p: -len(p[0])):
        t = t.replace(a, b)
    t = re.sub(r"(.)\1+", r"\1", t)  # double letters
    return re.sub(r"(?<=[^aeiou])[aeiou]+$", lambda m: m.group(0)[0], t)  # trailing vowel clusters


def best_match(heard: str, kind: str) -> str | None:
    """The known name that `heard` sounds like, or None if nothing is close."""
    key = sound_key(heard)
    if not key:
        return None
    scored = []
    for name in load()[kind]:
        if name.lower() == heard.strip().lower():
            return name
        score = difflib.SequenceMatcher(None, key, sound_key(name)).ratio()
        scored.append((score, name))
    score, name = max(scored, default=(0, None))
    return name if score >= MATCH_THRESHOLD else None


def correct_arguments(tool: str, arguments: dict) -> dict:
    fixed = dict(arguments)
    for (tool_name, arg), kind in ARG_KINDS.items():
        if tool == tool_name and isinstance(fixed.get(arg), str):
            match = best_match(fixed[arg], kind)
            if match and match.lower() != fixed[arg].strip().lower():  # only real mishearings, not just capitals
                fixed[arg] = match
    return fixed


@registry.register(
    name="remember_name",
    description="Remember a name the user uses - a YouTube channel, WhatsApp contact/chat, or app - so it's "
                "recognised correctly even when speech-to-text mishears it.",
    parameters={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(DEFAULTS)},
            "name": {"type": "string"},
        },
        "required": ["kind", "name"],
    },
)
def remember_name(kind: str, name: str) -> str:
    if kind not in DEFAULTS:
        raise ValueError(f"kind must be one of {list(DEFAULTS)}")
    name = name.strip()
    names = load()
    if name not in names[kind]:
        names[kind].append(name)
        _save(names)
    return f"Got it - I'll remember the {kind} {name}."
