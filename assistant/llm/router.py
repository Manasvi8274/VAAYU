"""
Cuts how much the local model has to do per command - measured as the
biggest accuracy/latency lever on this GTX 1650 (12-command dry run,
vaayu-finetuned): all 37 tools in the prompt ~1/11 correct, only 5
relevant tools 10/12 correct.

1. fast_route(): unambiguous everyday commands (English + Hinglish) map
   straight to tool calls with no LLM at all - well under a second. Multi-
   step commands ("X and Y", "X aur Y") run step by step when every step is
   itself unambiguous.
2. select_tools(): everything else goes to the LLM with only the top-k
   most relevant tools instead of the whole registry.
3. DIRECT_REPLY_TOOLS / format_result(): a successful action tool's own
   result string is spoken as the reply, skipping the second LLM call that
   only re-worded "Opened notepad." into a sentence.

WhatsApp contact/message are parsed here deliberately: the 504-case
benchmark caught the model sending "SAVED MESSAGES" requests to "Mom" and
inventing message text - pulling both verbatim from the user's own words is
the only reliable way at this model size.
"""

import datetime
import re

from assistant.tools.registry import ToolRegistry

_LEAD = re.compile(
    r"^(?:(?:hey|ok|okay|so|okay so|ok so|and|um+|uh+|hmm+)\s+)*(?:vaa?yu\s+)?"
    r"(?:(?:please|pls|can you|could you|would you|will you|zara|just)\s+)*", re.I)
_TAIL = re.compile(r"(?:\s+(?:please|pls|for me|na|yaar|now|right now|bhai|jaldi))+$", re.I)
_FILLER_WORDS = re.compile(r"\b(?:um+|uh+|hmm+|erm)\b", re.I)
# Speech-to-text punctuation: drop it, except inside "google.com" or "10:30".
_PUNCT = re.compile(r"\.\.\.|(?<![a-z0-9])[.:]|[.:](?![a-z0-9])|[,!?;\"]", re.I)

# Words that mean the target is a tab/page/file, not an installed app.
_NOT_AN_APP = re.compile(
    r"\b(tab|tabs|page|website|site|folder|video|song|music|link|url|channel|window|message|"
    r"note|notes|vaa?yu|fullscreen|full screen|volume|sound|screen|laptop|computer)\b|"
    r"\bfile\b(?! explorer| manager)|\.com|\.in|\.org| on | to | and | then ",
    re.I,
)
_SITE = r"(?:youtube|hotstar|netflix|prime video|spotify|jiocinema|zee5|sonyliv)"

_MEDIA = [
    (r"(?:pause|play|resume|stop)(?: (?:it|the music|music|the song|song|media|playback))?|"
     r"(?:gaana |music )?(?:ruko|rok do|roko|chalao|chala do)", "play_pause"),
    (r"(?:play )?(?:the )?next(?: song| track)?|skip(?: this| the)?(?: song| track)?|agla gaana(?: chalao| lagao)?", "next"),
    (r"(?:play )?(?:the )?(?:previous|prev)(?: song| track)?|(?:go back to |play )?the (?:last|previous) (?:song|track)|"
     r"pichla gaana(?: chalao| lagao)?", "previous"),
    (r"(?:increase|raise|turn up|boost)(?: the)?(?: system)? (?:volume|sound)|(?:system )?volume (?:up|badhao|badha do|tez karo|increase karo)|"
     r"awaa?z (?:badhao|badha do|tez karo|badhado)|(?:make it )?louder", "volume_up"),
    (r"(?:decrease|lower|reduce|turn down)(?: the)?(?: system)? (?:volume|sound)|(?:system )?volume (?:down|kam karo|kam kar do|ghatao)|"
     r"awaa?z (?:kam karo|kam kar do|dheere karo|dheemi karo)|(?:make it )?(?:quieter|softer)", "volume_down"),
    (r"mute(?: the)?(?: (?:system|volume|sound|audio|laptop|computer|speakers?))?|(?:awaa?z|sound|volume) band (?:karo|kar do)", "mute"),
]
_MEDIA = [(re.compile(rf"^(?:{p})$", re.I), a) for p, a in _MEDIA]

_OPEN_VERB = r"(?:open|launch|start|fire up|bring up|run)"
_OPEN = [
    re.compile(rf"^{_OPEN_VERB}(?: up)? (?:the )?(?P<app>.+?)(?: app| application| browser)?(?P<bg> in (?:the )?background)?$", re.I),
    re.compile(r"^get (?:the )?(?P<app>.+?)(?: app)? (?:running|started|open|up)$", re.I),
    re.compile(r"^(?P<app>.+?)(?: app| browser)?(?: ko)? (?:kholo|khol do|kholdo|khol de|open karo|open kro|open kar do|"
               r"chalu karo|chalu kar do|start karo|start kro)$", re.I),
]
_CLOSE = [
    re.compile(r"^(?:close|quit|exit|kill|shut|shut down|get rid of|terminate)(?: the)? (?P<app>.+?)(?: app| application| browser)?$", re.I),
    re.compile(r"^(?P<app>.+?)(?: app| browser)?(?: ko)? (?:band karo|band kar do|band kardo|band karde|band kro|close karo|"
               r"close kro|close kar do|hata do|hatao)$", re.I),
]

_INFO = [
    (re.compile(r"\b(battery|charge|charging|charger|plugged|cpu|ram|memory usage|disk space|storage|system status)\b", re.I),
     "get_system_status", {}),
    (re.compile(r"\b(wi-?fi|internet|online|network)\b", re.I), "get_network_status", {}),
    (re.compile(r"\bbluetooth\b", re.I), "list_connected_devices", {"category": "bluetooth"}),
    (re.compile(r"\busb\b", re.I), "list_connected_devices", {"category": "usb"}),
    (re.compile(r"\bprinters?\b", re.I), "list_connected_devices", {"category": "printer"}),
    (re.compile(r"\b(displays?|monitors?)\b", re.I), "list_connected_devices", {"category": "display"}),
    (re.compile(r"\b(mic|microphone|speaker volume|speakers?)\b", re.I), "get_audio_status", {}),
]
_QUESTION = re.compile(r"\b(is|are|am|what|what's|how|how's|which|kitni|kitna|kya|check|batao|status|level|hai)\b", re.I)

_SCREENSHOT = re.compile(r"^(?:take|capture|grab|le lo|lo)?\s*(?:a )?(?:screen ?shot|screen capture)(?: (?:lo|le lo|lelo|karo))?$|"
                         r"^capture (?:the |my )?screen$", re.I)
_LOCK = re.compile(r"^lock(?: (?:my|the))? (?:computer|laptop|pc|screen|system)$|^(?:laptop|computer|pc|screen) lock (?:karo|kar do)$", re.I)
_MINIMIZE = re.compile(r"^minimi[sz]e(?: (?:this|it|this window|the window|current window))?$|^(?:ye|isko|this) minimi[sz]e (?:karo|kar do)$", re.I)
_TIME = re.compile(r"^(?:what(?:'s| is) the time|what time is it|time kya hai|kitne baje hai[n]?|time batao|kya time hai|"
                   r"tell me the time)(?: now)?$", re.I)
_DATE = re.compile(r"^(?:what(?:'s| is) (?:the )?(?:date|day)(?: today)?|aaj (?:kya date hai|kaunsa din hai)|"
                   r"what day is it(?: today)?|today's date)$", re.I)

# In-page video control (any site). Only wording that clearly says video/site.
_VIDEO_PLAY = re.compile(rf"^(?P<act>pause|play|resume)(?: the)?(?: (?P<site>{_SITE}))? video$|^video (?P<hi>roko|rok do|chalao|chala do)$", re.I)
_FULL_ON = re.compile(r"^(?:make (?:it|this|the video) |go |put it (?:in )?)?full ?screen(?: (?:this|this video|the video|mode))?$|"
                      r"^full ?screen (?:karo|kar do)$", re.I)
_FULL_OFF = re.compile(r"^(?:exit|leave|close|get out of) full ?screen(?: mode)?$|^shrink (?:the |this )?video$|"
                       r"^full ?screen (?:hatao|band karo)$", re.I)
_NUM_WORDS = {"a": 1, "one": 1, "two": 2, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}
_SEEK = re.compile(r"^(?:(?P<fw>skip ahead|skip forward|fast forward|forward|jump ahead)|(?P<bw>go back|rewind|jump back|skip back))"
                   r"(?: by)? (?P<n>\d+|a|one|two|five|ten|fifteen|twenty|thirty) ?(?P<unit>seconds?|secs?|minutes?|mins?)$|"
                   r"^(?:video )?(?P<n2>\d+) (?P<unit2>second|seconds|sec|minute|minutes|min) (?P<dir2>aage|peeche|piche) (?:karo|kar do)$",
                   re.I)
_VIDEO_VOL = re.compile(rf"^(?P<verb>turn up|increase|raise|turn down|decrease|lower|reduce)(?: the)? (?:(?P<site>{_SITE}) )?"
                        rf"(?:video )?volume$|^(?P<mute>mute|unmute)(?: the)? (?:(?P<site2>{_SITE}) )?video$|"
                        rf"^set(?: the)? (?:video|(?P<site3>{_SITE})) volume to (?P<level>\d+)(?: percent|%)?$", re.I)

_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5,
             "5th": 5, "last": -1}
_TAB_CLOSE = re.compile(r"^close (?:the )?(?P<q>.+?) tab$|^(?:ye|is) tab (?:ko )?band karo$", re.I)
_TAB_SWITCH = re.compile(r"^(?:go|switch|move) (?:back )?to (?:the )?(?P<q>.+?) tab$|^(?P<q2>.+?) (?:wale |wala )?tab (?:pe|par) jao$", re.I)
_LIST_TABS = re.compile(r"^(?:what|which) tabs (?:are|do i have) open$|^(?:list|show)(?: me)?(?: all)?(?: my)?(?: browser)? tabs$", re.I)
_WEBSITE = re.compile(r"^(?:open|go to|visit|launch)(?: the)? (?P<url>[a-z0-9-]+\.(?:com|in|org|net|io|co|ai|dev)(?:/\S*)?)$|"
                      r"^open (?:the )?(?P<site>[a-z0-9 ]+?) (?:website|site)$", re.I)
_WEB_SEARCH = re.compile(r"^(?:search(?: the web| online| google)?(?: for)?|google|look up) (?P<q>.+)$", re.I)
_LOCAL_HINT = re.compile(r"\b(file|files|folder|documents|desktop|downloads|my computer|on my laptop|pdf|docx|spreadsheet)\b", re.I)

_TYPE = re.compile(r"^(?:type|write)(?: out)? (?P<text>.+?)(?: in (?:notepad|word|here|the document))?$|^likho (?P<t2>.+)$|^(?P<t3>.+) likho$", re.I)
_NOTE = re.compile(r"^(?:note that|note down|make a note(?: that)?|add a note(?: that| saying)?|save a note(?: that| saying)?|"
                   r"remember (?:that|to)) (?P<text>.+)$|^yaad rakhna (?P<t2>.+)$", re.I)
_LIST_NOTES = re.compile(r"^(?:show|read|list)(?: me)?(?: all)? my(?: last \d+)? notes$|^what are my notes$|^mere notes (?:dikhao|batao)$", re.I)

# WhatsApp: "send M to W on whatsapp", "W ko whatsapp pe M bhejo", "whatsapp W saying M", "tell W on whatsapp that M"
_WA = [
    re.compile(r"^(?:send|whatsapp) (?P<m>.+?) to (?P<w>.+?) (?:on|via|over|through) whatsapp$", re.I),
    re.compile(r"^(?P<w>.+?) ko whatsapp (?:pe|par|pr) (?P<m>.+?) (?:bhejo|bhej do|send karo|send kro)$", re.I),
    re.compile(r"^whatsapp (?P<w>.+?) (?:saying|that) (?P<m>.+)$", re.I),
    re.compile(r"^(?:tell|message|text) (?P<w>.+?) on whatsapp (?:that|saying) (?P<m>.+)$", re.I),
]

_SMALL_TALK = [
    (re.compile(r"^(?:thanks|thank you|thank you so much|thanks a lot|shukriya|dhanyavaad|thnx)$", re.I), "You're welcome!"),
    (re.compile(r"^(?:ok|okay|okk|cool|fine|alright|got it|theek hai|achha|accha)$", re.I), "Okay."),
    (re.compile(r"^(?:never ?mind|nothing|forget it|nothing forget it|kuch nahi|rehne do|cancel)$", re.I), "Okay, no problem."),
    (re.compile(r"^(?:how are you|how are you doing|kya haal hai|kaise ho|kaisa hai|what's up|sup)$", re.I),
     "I'm doing great! What can I do for you?"),
    (re.compile(r"^(?:are you there|you there|hello|hi|hey|namaste)$", re.I), "Yes, I'm here. What can I do for you?"),
]


def normalize(text: str) -> str:
    t = _PUNCT.sub(" ", text.strip())
    t = _FILLER_WORDS.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _LEAD.sub("", t)
    t = _TAIL.sub("", t)
    return t.strip()


def _app_ok(app: str) -> bool:
    return bool(app) and len(app.split()) <= 3 and not _NOT_AN_APP.search(f" {app} ")


def _tab_target(q: str, tool: str):
    q = q.strip()
    if q.lower() in _ORDINALS:
        return ("tool", tool, {"position": _ORDINALS[q.lower()]})
    if q.lower() in ("this", "current", "the current", "this current"):
        return ("tool", tool, {}) if tool == "close_browser_tab" else None
    return ("tool", tool, {"query": q})


def _route_one(t: str):
    """One unambiguous command -> ("tool", name, args) / ("reply", text) / None."""
    if not t:
        return None
    if _TIME.match(t):
        return ("reply", datetime.datetime.now().strftime("It's %I:%M %p.").replace(" 0", " "))
    if _DATE.match(t):
        return ("reply", datetime.datetime.now().strftime("Today is %A, %d %B %Y."))
    for pattern, reply in _SMALL_TALK:
        if pattern.match(t):
            return ("reply", reply)
    for pattern in _WA:
        m = pattern.match(t)
        if m and m.group("w").strip() and m.group("m").strip():
            return ("tool", "send_whatsapp_message", {"contact_name": m.group("w").strip(), "message": m.group("m").strip()})
    for pattern, action in _MEDIA:
        if pattern.match(t):
            return ("tool", "control_media", {"action": action})
    if m := _VIDEO_PLAY.match(t):
        act = (m.group("act") or "").lower()
        hi = (m.group("hi") or "").lower()
        args = {"action": "pause" if act == "pause" or hi.startswith("rok") else "play"}
        if m.group("site"):
            args["tab_hint"] = m.group("site")
        return ("tool", "control_video_playback", args)
    if _FULL_OFF.match(t):
        return ("tool", "set_video_fullscreen", {"fullscreen": False})
    if _FULL_ON.match(t):
        return ("tool", "set_video_fullscreen", {"fullscreen": True})
    if m := _SEEK.match(t):
        if m.group("n2"):
            n, unit, fw = int(m.group("n2")), m.group("unit2"), m.group("dir2").lower() == "aage"
        else:
            raw = m.group("n").lower()
            n, unit, fw = (int(raw) if raw.isdigit() else _NUM_WORDS[raw]), m.group("unit"), bool(m.group("fw"))
        seconds = n * 60 if unit.lower().startswith("min") else n
        return ("tool", "seek_video", {"direction": "forward" if fw else "backward", "seconds": seconds})
    if m := _VIDEO_VOL.match(t):
        if m.group("level"):
            args = {"level": int(m.group("level"))}
        elif m.group("mute"):
            args = {"direction": m.group("mute").lower()}
        else:
            args = {"direction": "up" if m.group("verb").lower() in ("turn up", "increase", "raise") else "down"}
        site = m.group("site") or m.group("site2") or m.group("site3")
        if site:
            args["tab_hint"] = site
        return ("tool", "set_video_volume", args)
    if _SCREENSHOT.match(t):
        return ("tool", "take_screenshot", {})
    if _LOCK.match(t):
        return ("tool", "lock_computer", {})
    if _MINIMIZE.match(t):
        return ("tool", "minimize_window", {})
    if _LIST_TABS.match(t):
        return ("tool", "list_browser_tabs", {})
    if m := _TAB_CLOSE.match(t):
        return _tab_target(m.group("q") or "this", "close_browser_tab")
    if m := _TAB_SWITCH.match(t):
        routed = _tab_target(m.group("q") or m.group("q2"), "switch_to_browser_tab")
        if routed:
            return routed
    if _LIST_NOTES.match(t):
        return ("tool", "list_notes", {})
    if m := _NOTE.match(t):
        return ("tool", "add_note", {"text": m.group("text") or m.group("t2")})
    if m := _WEBSITE.match(t):
        return ("tool", "open_website", {"url_or_query": m.group("url") or m.group("site")})
    if (m := _WEB_SEARCH.match(t)) and not _LOCAL_HINT.search(t):
        return ("tool", "open_website", {"url_or_query": m.group("q")})
    if m := _TYPE.match(t):
        return ("tool", "type_text", {"text": m.group("text") or m.group("t2") or m.group("t3")})
    if len(t.split()) <= 7 and _QUESTION.search(t) and not re.search(r"\b(open|close|band|kholo|turn on|turn off)\b", t, re.I):
        for pattern, tool, args in _INFO:
            if pattern.search(t):
                return ("tool", tool, dict(args))
    for name, patterns in (("close_app", _CLOSE), ("open_app", _OPEN)):
        for pattern in patterns:
            m = pattern.match(t)
            if m and _app_ok(m.group("app").strip()):
                args = {"app_name": m.group("app").strip()}
                if "bg" in pattern.groupindex and m.group("bg"):
                    args["foreground"] = False
                return ("tool", name, args)
    return None


_SPLIT = re.compile(r"\s+(?:and then|and also|and|then|after that|aur phir|aur fir|aur|phir|fir)\s+", re.I)


def fast_route(text: str):
    """("reply", text) / ("tools", [(name, args), ...]) for an unambiguous
    command, or None to fall back to the LLM. Deliberately conservative:
    a miss only costs LLM latency, a wrong hit does the wrong thing."""
    t = normalize(text)
    # Split first: a greedy single rule ("<anything> likho") would otherwise
    # swallow "notepad kholo aur hello likho" whole. A message that merely
    # contains "and" doesn't split cleanly into routable parts, so it falls
    # through to the single-command rules intact.
    parts = _SPLIT.split(t)
    if len(parts) > 1:
        routed = [_route_one(p) for p in parts]
        if all(r and r[0] == "tool" for r in routed):
            return ("tools", [r[1:] for r in routed])
    one = _route_one(t)
    if one:
        return one if one[0] == "reply" else ("tools", [one[1:]])
    return None


# Hinglish/colloquial words -> the English words tool names/descriptions use.
_SYNONYMS = {
    "kholo": "open", "khol": "open", "chalu": "open", "launch": "open", "start": "open", "running": "open",
    "band": "close", "quit": "close", "exit": "close", "hata": "close", "rid": "close",
    "bhejo": "send whatsapp message", "bhej": "send whatsapp message", "msg": "message", "text": "message",
    "likho": "type text", "likh": "type text", "gaana": "play song youtube", "song": "play youtube",
    "songs": "play youtube", "music": "media play", "awaz": "volume", "awaaz": "volume", "sound": "volume",
    "loud": "volume", "louder": "volume", "quieter": "volume", "track": "media next previous",
    "skip": "seek video", "forward": "seek video", "rewind": "seek video", "aage": "seek video", "peeche": "seek video",
    "fullscreen": "fullscreen video", "tab": "browser tab", "tabs": "browser tab", "chrome": "app browser",
    "brave": "app browser", "battery": "system status", "charge": "system status", "plugged": "system status",
    "wifi": "network status", "internet": "network status", "online": "network status",
    "bluetooth": "devices connected", "mic": "audio status", "speaker": "audio status",
    "remember": "note", "yaad": "note", "file": "files search read", "dhundo": "search", "find": "search",
    "orb": "status orb", "ball": "status orb move position", "corner": "status orb move position",
    "update": "daily time", "check": "daily time", "password": "sensitive field",
    "window": "window", "minimize": "minimize window", "chhota": "minimize window", "website": "open website",
    "google": "open website search", "dikhao": "show open website", "spreadsheet": "files search",
}
_WORD = re.compile(r"[a-z]+")
_STOP = {"the", "a", "an", "to", "of", "on", "in", "is", "it", "and", "or", "for", "this", "that", "my",
         "me", "please", "karo", "kro", "do", "de", "hai", "ko", "se", "pe", "par", "use", "not", "with",
         "by", "be", "are", "you", "if", "only", "any", "e", "g", "eg", "s"}


def _words(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        out.add(w.rstrip("s") if len(w) > 3 else w)
        for s in _SYNONYMS.get(w, "").split():
            out.add(s.rstrip("s") if len(s) > 3 else s)
    return out


def select_tools(text: str, registry: ToolRegistry, k: int = 5) -> ToolRegistry:
    """Top-k tools by word overlap with the request; name words count 3x
    (names are short and precise, descriptions long and noisy)."""
    query = _words(normalize(text))
    scored = []
    for name, spec in registry._tools.items():
        name_words = _words(name.replace("_", " "))
        desc_words = _words(spec.description)
        score = 3 * len(query & name_words) + len(query & desc_words) / (1 + len(desc_words) ** 0.5)
        scored.append((score, name))
    scored.sort(key=lambda x: -x[0])
    subset = ToolRegistry()
    subset._tools = {name: registry._tools[name] for _score, name in scored[:k]}
    return subset


# Action tools whose result string already says what happened - spoken
# as-is instead of a second LLM call. Read/list/get tools are left out on
# purpose: their result is data the model still has to act on or summarize.
DIRECT_REPLY_TOOLS = {
    "open_app", "close_app", "control_media", "take_screenshot", "lock_computer", "minimize_window",
    "restore_window", "switch_to_window", "play_video_on_youtube", "play_latest_video_from_channel",
    "add_note", "set_video_volume", "set_video_fullscreen", "seek_video", "control_video_playback",
    "close_browser_tab", "switch_to_browser_tab", "type_text", "send_whatsapp_message",
    "move_status_orb", "set_daily_update_time", "open_website", "get_system_status",
}


def format_result(name: str, result) -> str | None:
    """Speakable reply for a tool result, or None when it needs the LLM to
    phrase it (data results from info tools)."""
    if name == "get_system_status" and isinstance(result, dict):
        parts = []
        if result.get("battery_percent") is not None:
            plug = ", charging" if result.get("battery_plugged_in") else ""
            parts.append(f"Battery is at {round(result['battery_percent'])}%{plug}")
        parts.append(f"CPU {round(result.get('cpu_percent', 0))}%")
        parts.append(f"RAM {round(result.get('ram_percent', 0))}%")
        return ". ".join(parts) + "."
    if name == "control_media" and isinstance(result, str):
        return "Done."
    return result if isinstance(result, str) and result.strip() else None
