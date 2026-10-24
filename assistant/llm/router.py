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
    r"^(?:(?:hey|ok|okay|so|okay so|ok so|and|um+|uh+|hmm+|hold on|let's|lets)\s+)*(?:vaa?yu\s+)?"
    r"(?:(?:please|pls|can you|could you|would you|will you|zara|just)\s+)*", re.I)
_TAIL = re.compile(r"(?:\s+(?:please|pls|for me|na|yaar|now|right now|bhai|jaldi|quickly|a bit|a little|"
                   r"jaldi se|vaa?yu))+$", re.I)
_FILLER_WORDS = re.compile(r"\b(?:um+|uh+|hmm+|erm|thoda|zara)\b", re.I)
# Speech-to-text punctuation: drop it, except inside "google.com"/"10:30" and
# a ": " separator, which the WhatsApp "to X: message" shape needs.
_PUNCT = re.compile(r"\.\.\.|(?<![a-z0-9])\.|\.(?![a-z0-9])|[,!?;\"]", re.I)

# Words that mean the target is a tab/page/file, not an installed app.
_NOT_AN_APP = re.compile(
    r"\b(tab|tabs|page|website|site|folder|video|song|music|link|url|channel|window|message|"
    r"note|notes|vaa?yu|fullscreen|full screen|volume|sound|screen|laptop|computer|wi-?fi|bluetooth|it|"
    r"gaana|song|music)\b|"
    r"\bfile\b(?! explorer| manager)|\.com|\.in|\.org| on | to | and | then ",
    re.I,
)
_SITE = r"(?:youtube|hotstar|netflix|prime video|spotify|jiocinema|zee5|sonyliv)"

_SONG = r"(?:song|track|music|gaana|gana)"
_KARO = r"(?:karo|kar do|kardo|kar de|karde|kro)"
_MEDIA = [
    (rf"(?:pause|play|resume|stop|continue|keep playing|continue playing)(?: (?:it|the music|music|the song|song|media|playback|playing))?|"
     rf"(?:{_SONG} )?(?:ruko|rok do|roko|chalao|chala do)|{_SONG} (?:band {_KARO}|pause {_KARO}|rok do|roko|stop {_KARO})|"
     rf"(?:{_SONG}) (?:wapas|phir se|fir se) (?:chalao|chala do|play {_KARO})", "play_pause"),
    (rf"(?:play )?(?:the )?next(?: {_SONG})?|skip(?: this| the| it)?(?: {_SONG})?|agla {_SONG}(?: chalao| lagao| bajao)?|"
     rf"change(?: this| the)? {_SONG}|{_SONG} (?:badlo|change {_KARO}|badal do)|(?:koi )?(?:aur|dusra|doosra) {_SONG}(?: chalao| lagao| bajao)?",
     "next"),
    (rf"(?:play |go to )?(?:the )?(?:previous|prev)(?: {_SONG})?|(?:go back to |play |back to )?(?:the )?(?:last|previous) {_SONG}|"
     rf"pichla(?: wala)? {_SONG}(?: chalao| lagao| bajao)?|play the {_SONG} before this", "previous"),
    (rf"(?:increase|raise|turn up|boost|bump up)(?: the)?(?: system)? (?:volume|sound)|turn the (?:volume|sound) up|"
     rf"(?:system )?volume (?:up|badhao|badha do|tez {_KARO}|increase {_KARO})|awaa?z (?:badhao|badha do|tez {_KARO}|badhado)|"
     rf"(?:.* )?louder|(?:it's |its |it is )?too (?:quiet|soft)|i can't hear(?: anything| it)?", "volume_up"),
    (rf"(?:decrease|lower|reduce|turn down)(?: the)?(?: system)? (?:volume|sound)|turn the (?:volume|sound) down|"
     rf"(?:system )?volume (?:down|kam {_KARO}|ghatao)|awaa?z (?:kam {_KARO}|dheere {_KARO}|dheemi {_KARO})|"
     rf"(?:make it )?(?:quieter|softer)|(?:it's |its |it is )?too loud", "volume_down"),
    (rf"(?:un)?mute(?: the| it| everything)?(?: (?:system|volume|sound|audio|laptop|computer|speakers?))?|"
     rf"silence(?: the| it| everything)?(?: (?:laptop|computer|sound|audio))?|(?:awaa?z|sound|volume) band {_KARO}", "mute"),
]
_MEDIA = [(re.compile(rf"^(?:{p})$", re.I), a) for p, a in _MEDIA]

# Well-known sites: "open youtube" means the website, not an installed app
# (found live: open_app('youtube') -> "YOUTUBE is not running"). Desktop apps
# that share a name (whatsapp, spotify, telegram...) are deliberately absent.
_SITES = {
    "youtube": "youtube.com", "yt": "youtube.com", "youtube music": "music.youtube.com", "gmail": "mail.google.com",
    "google": "google.com", "google maps": "maps.google.com", "maps": "maps.google.com",
    "google drive": "drive.google.com", "drive": "drive.google.com", "instagram": "instagram.com", "insta": "instagram.com",
    "facebook": "facebook.com", "twitter": "x.com", "linkedin": "linkedin.com", "netflix": "netflix.com",
    "hotstar": "hotstar.com", "jiohotstar": "hotstar.com", "prime video": "primevideo.com", "amazon": "amazon.in",
    "flipkart": "flipkart.com", "github": "github.com", "chatgpt": "chatgpt.com", "reddit": "reddit.com",
    "wikipedia": "wikipedia.org", "whatsapp web": "web.whatsapp.com", "zomato": "zomato.com", "swiggy": "swiggy.com",
    "myntra": "myntra.com", "irctc": "irctc.co.in", "claude": "claude.ai", "perplexity": "perplexity.ai",
}
_BROWSERS = r"(?:brave|chrome|google chrome|edge|microsoft edge|firefox|comet)"
_SITE_OPEN = re.compile(
    rf"^(?:open|launch|start|go to|visit)(?: up)? (?:the )?(?P<site>{'|'.join(sorted(map(re.escape, _SITES), key=len, reverse=True))})"
    rf"(?: website| site| page)?(?: (?:in|on) (?:a )?new tab)?(?: (?:in|on|using) (?:the )?(?P<browser>{_BROWSERS})(?: browser)?)?"
    rf"(?: (?:in|on) (?:a )?new tab)?$|"
    rf"^(?P<site2>{'|'.join(sorted(map(re.escape, _SITES), key=len, reverse=True))})(?: website| site)? "
    rf"(?:kholo|khol do|open karo|open kro|chalao)$", re.I)

_OPEN_VERB = r"(?:open|launch|start|fire up|bring up|run)"
_OPEN = [
    re.compile(rf"^{_OPEN_VERB}(?: up)? (?:the )?(?P<app>.+?)(?: app| application| browser)?(?P<bg> in (?:the )?background)?$", re.I),
    re.compile(r"^get (?:the )?(?P<app>.+?)(?: app)? (?:running|started|open|up)$", re.I),
    re.compile(r"^(?P<app>.+?)(?: app| browser)?(?: ko)? (?:kholo|khol do|kholdo|khol de|open karo|open kro|open kar do|"
               r"chalu karo|chalu kar do|chalu kar|start karo|start kro)$", re.I),
    re.compile(r"^(?:mujhe |i need |i want )(?P<app>.+?) (?:chahiye|open|opened)$", re.I),
]
_CLOSE = [
    re.compile(r"^(?:close|quit|exit|kill|shut|shut down|get rid of|terminate)(?: the)? (?P<app>.+?)(?: app| application| browser)?$", re.I),
    re.compile(r"^(?P<app>.+?)(?: app| browser)?(?: ko)? (?:band karo|band kar do|band kardo|band karde|band kar de|band kro|close karo|"
               r"close kro|close kar do|hata do|hatao)$", re.I),
    re.compile(r"^(?:i'?m|i am) done with (?:the )?(?P<app>.+?)(?: close it)?$", re.I),
    re.compile(r"^(?P<app>.+?) ki (?:zarurat|zaroorat|jarurat) nahi(?: hai)?(?: band (?:karo|kar do))?$", re.I),
]

# Wi-Fi/Bluetooth on-off. Checked before the close-app rules: "wifi band karo"
# means turn it off, not "close an app called wifi".
_RADIO = re.compile(
    r"^(?:turn|switch) (?P<st>on|off) (?:the )?(?P<k>wi-?fi|bluetooth)$|"
    r"^(?:turn|switch) (?:the )?(?P<k2>wi-?fi|bluetooth) (?P<st2>on|off)$|"
    r"^(?P<v>enable|disable) (?:the )?(?P<k3>wi-?fi|bluetooth)$|"
    r"^(?P<k4>wi-?fi|bluetooth) (?P<st4>on|off|band|chalu|start|bandh) (?:karo|kar do|kardo|kar de|karde|kro)$", re.I)

# Things Vaayu must refuse outright rather than let the model guess at (the
# benchmark caught "shut down the laptop" -> close_app('Windows Terminal')).
# Payment apps/money transfer are a permanent block by explicit request.
_PC = r"(?:the |my |this )?(?:laptop|computer|pc|system|windows|machine)"
_REFUSE = [
    (re.compile(rf"\b(?:shut ?down|restart|reboot|hibernate|power off|sleep|log ?out of|log ?off|sign out of) {_PC}\b|"
                rf"\bput {_PC} to sleep\b|\b{_PC} (?:band|shutdown|restart) (?:karo|kar do)\b", re.I),
     "I don't shut down, restart or sleep the laptop - you can do that yourself, or say 'Vaayu die' to stop me."),
    (re.compile(r"\b(?:delete|erase|wipe|format|remove)\b.*\b(?:file|files|folder|folders|everything|drive|downloads|documents|desktop|all)\b",
                re.I),
     "I don't delete or format files - that's something you should do yourself."),
    (re.compile(r"^(?:install|uninstall|download and install)\b", re.I),
     "I can't install or uninstall software yet."),
    # Capabilities Vaayu has no tool for - a plain "can't" instead of the
    # model's nearest-looking guess (benchmark: "turn on the lights" ->
    # Wi-Fi on, "send an email to my boss" -> opened Outlook).
    (re.compile(r"\b(?:turn|switch) (?:on|off) (?:the |my )?(?:lights?|fans?|ac|air conditioner|tv|geyser)\b|"
                r"\b(?:lights?|fans?|ac|tv|geyser) (?:on|off|band|chalu) (?:karo|kar do)\b", re.I),
     "I can't control lights or home appliances - I only work on this laptop."),
    (re.compile(r"\b(?:send|write|compose|draft) (?:an? )?e-?mail\b|\be-?mail (?:to|my)\b", re.I),
     "I can't send emails yet."),
    (re.compile(r"^(?:call|phone|dial|ring|make a (?:phone )?call)\b|\bko (?:call|phone) (?:karo|kar do|lagao)\b", re.I),
     "I can't make phone calls."),
    (re.compile(r"\b(?:set|create) (?:an? )?(?:alarm|timer|reminder)\b|\balarm (?:laga|set) (?:do|karo)\b", re.I),
     "I can't set alarms or timers yet - I can save a note instead."),
    (re.compile(r"\b(?:paytm|phonepe|phone pe|gpay|google pay|upi|bhim|send money|transfer money|transfer \d+|"
                r"pay \d+|pay (?:rs|rupees|money)|make (?:a )?payment|net ?banking|bank account|credit card|debit card)\b"
                r"|\bpay (?:my |the )?\w* ?bill\b", re.I),
     "I don't do payments or open payment apps - that's blocked permanently."),
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
_QUESTION = re.compile(r"\b(is|are|am|what|what's|how|how's|which|kitni|kitna|kya|check|batao|status|level|hai|tell)\b", re.I)

_SCREENSHOT = re.compile(r"^(?:take|capture|grab|le lo|lo)?\s*(?:a )?(?:quick )?(?:screen ?shot|screen capture|ss)(?: (?:lo|le lo|lelo|karo))?$|"
                         r"^capture (?:the |my )?screen$|^screen ki (?:photo|picture|pic) (?:lo|le lo|khicho)$", re.I)
_LOCK = re.compile(r"^lock(?: (?:my|the))? (?:computer|laptop|pc|screen|system)$|^(?:laptop|computer|pc|screen) lock (?:karo|kar do)$|"
                   r"^lock (?:karo|kar do|it|it up)$", re.I)
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
                   r"^(?:video )?(?P<n2>\d+) (?P<unit2>second|seconds|sec|minute|minutes|min) (?P<dir2>aage|peeche|piche) (?:karo|kar do)$|"
                   r"^video (?P<dir3>aage|peeche|piche) (?:badhao|karo|kar do) (?P<n3>\d+) (?P<unit3>second|seconds|sec|minute|minutes|min)$",
                   re.I)
_VIDEO_VOL = re.compile(rf"^(?P<verb>turn up|increase|raise|turn down|decrease|lower|reduce)(?: the)? (?:(?P<site>{_SITE}) )?"
                        rf"(?:video )?volume$|^(?P<mute>mute|unmute)(?: the)? (?:(?P<site2>{_SITE}) )?video$|"
                        rf"^set(?: the)? (?:video|(?P<site3>{_SITE})) volume to (?P<level>\d+)(?: percent|%)?$", re.I)

_ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5,
             "5th": 5, "last": -1}
_TAB_CLOSE = re.compile(r"^close (?:the )?(?P<q>.+?) tab$|^(?:ye|is) tab (?:ko )?band karo$", re.I)
_TAB_SWITCH = re.compile(r"^(?:go|switch|move) (?:back )?to (?:the )?(?P<q>.+?) tab$|^(?P<q2>.+?) (?:wale |wala )?tab (?:pe|par) jao$", re.I)
_LIST_TABS = re.compile(r"^(?:what|which) tabs (?:are|do i have) open$|^(?:list|show)(?: me)?(?: all)?(?: my)?(?: open)?(?: browser)? tabs$", re.I)
_YOUTUBE = re.compile(r"^(?:search|find|look up) (?:on )?youtube (?:for )?(?P<q>.+)$|^(?:play|search|put on) (?P<q2>.+?) on youtube$|"
                      r"^youtube (?:pe|par|pr) (?P<q3>.+?) (?:chalao|lagao|bajao|search karo|dikhao)$", re.I)
_CHANNEL_VIDEO = re.compile(
    r"^(?:open youtube (?:and )?)?play (?:the )?(?:latest|newest|new|most recent|recent) video (?:from|of|by|on) (?:the )?(?P<c>.+?)(?: channel| youtube channel)?(?: on youtube)?$|"
    r"^(?P<c2>.+?) (?:ka|ki|ke) (?:latest|naya|new|nayi) video (?:chalao|lagao|dikhao|play karo)$|"
    r"^play (?P<c3>.+?)'s (?:latest|newest|new) video$", re.I)
_CHANNEL_HINT = re.compile(r"\b(?:latest|newest|new|naya|recent) video\b|\bchannel\b", re.I)
_WEBSITE = re.compile(r"^(?:open|go to|visit|launch)(?: the)? (?P<url>[a-z0-9-]+\.(?:com|in|org|net|io|co|ai|dev)(?:/\S*)?)$|"
                      r"^open (?:the )?(?P<site>[a-z0-9 ]+?) (?:website|site)$", re.I)
_WEB_SEARCH = re.compile(r"^(?:search(?: the web| online| google)?(?: for)?|google|look up) (?P<q>.+)$", re.I)
_LOCAL_HINT = re.compile(r"\b(file|files|folder|documents|desktop|downloads|my computer|on my laptop|pdf|docx|spreadsheet)\b", re.I)

_TYPE = re.compile(r"^(?:type|write)(?: out)? (?P<text>.+?)(?: in (?:notepad|word|here|the document))?$|^likho (?P<t2>.+)$|^(?P<t3>.+) likho$", re.I)
_NOTE = re.compile(r"^(?:note that|note down|make a note(?: that)?|add a note(?: that| saying)?|save a note(?: that| saying)?|"
                   r"remember (?:that|to)) (?P<text>.+)$|^yaad rakhna (?P<t2>.+)$|^note (?P<t3>.+)$", re.I)
_LIST_NOTES = re.compile(r"^(?:show|read|list)(?: me)?(?: all)? my(?: last \d+)? notes$|^what are my notes$|^mere notes (?:dikhao|batao)$", re.I)

# WhatsApp - the ONLY way a recipient gets chosen: parsed verbatim from the
# user's own words, never picked by the model (see _LLM_BLOCKED_TOOLS).
# Message is matched greedily so "send going to office to X on whatsapp"
# keeps "going to office" as the message and X as the chat.
_WA_MSG_PREFIX = r"(?:(?:a )?(?:whatsapp )?(?:message|msg|text) (?:saying |that )?)?"
_WA = [
    re.compile(rf"^(?:send|whatsapp) {_WA_MSG_PREFIX}(?P<m>.+) to (?P<w>.+?) (?:on|in|via|over|through|par|pe) whatsapp$", re.I),
    re.compile(rf"^(?:on|in) whatsapp send {_WA_MSG_PREFIX}(?P<m>.+) to (?P<w>.+)$", re.I),
    re.compile(rf"^send {_WA_MSG_PREFIX}(?P<m>.+) on whatsapp to (?P<w>.+)$", re.I),
    re.compile(r"^(?P<w>.+?) ko whatsapp (?:pe|par|pr) (?P<m>.+?) (?:bhejo|bhej do|send karo|send kro)$", re.I),
    re.compile(r"^whatsapp (?:pe|par|pr) (?P<w>.+?) ko (?P<m>.+?) (?:bhejo|bhej do|send karo|send kro)$", re.I),
    re.compile(r"^(?P<w>.+?) ko whatsapp (?:karo|kar do) (?:ki|ke) (?P<m>.+)$", re.I),
    re.compile(r"^whatsapp (?:message )?to (?P<w>.+?) ?: (?P<m>.+)$", re.I),
    re.compile(r"^whatsapp (?P<w>.+?) (?:saying|that) (?P<m>.+)$", re.I),
    re.compile(r"^(?:tell|message|text) (?P<w>.+?) on whatsapp (?:that|saying) (?P<m>.+)$", re.I),
]
_MULTI_RECIPIENT = re.compile(r"\s(?:and|aur|&)\s|,", re.I)
WHATSAPP_HOWTO = ("I only send WhatsApp messages when you say the chat and the message clearly, like: "
                  "send hello to SAVED MESSAGES on WhatsApp.")
# Tools the model may never call on its own - their arguments must come
# from the user's exact words via fast_route, or the request is refused.
_LLM_BLOCKED_TOOLS = {
    "send_whatsapp_message": WHATSAPP_HOWTO,
    # Benchmark: "turn on the lights" -> the model chose set_radio(wifi, on).
    "set_radio": "Say it directly, like: turn wifi off, or turn bluetooth on.",
}

_SMALL_TALK = [
    (re.compile(r"^(?:thanks|thank you|thank you so much|thanks a lot|shukriya|dhanyavaad|thnx)$", re.I), "You're welcome!"),
    (re.compile(r"^(?:ok|okay|okk|cool|fine|alright|got it|(?:acha |achha |accha |chalo |ok )?theek hai|achha|accha|acha|bas|"
                r"bas karo|that's all|that's it)$", re.I), "Okay."),
    (re.compile(r"^(?:never ?mind|nothing|forget it|nothing forget it|kuch nahi|rehne do|cancel|sorry|my bad)$", re.I), "Okay, no problem."),
    (re.compile(r"^(?:good job|well done|nice|great|awesome|you're awesome|shabash|bahut badhiya)$", re.I), "Thank you!"),
    (re.compile(r"^(?:good night|good morning|good evening|good afternoon|bye|see you|goodbye)$", re.I), "Same to you!"),
    (re.compile(r"^(?:how are you|how are you doing|kya haal hai|kaise ho|kaisa hai|what's up|sup)$", re.I),
     "I'm doing great! What can I do for you?"),
    (re.compile(r"^(?:are you there|you there|hello|hi|hey|namaste)$", re.I), "Yes, I'm here. What can I do for you?"),
]


def normalize(text: str, keep_commas: bool = False) -> str:
    raw = text.strip()
    if keep_commas:  # for splitting steps: "open brave, play X" is two commands
        raw = raw.replace(",", " , ")
        t = re.sub(r"\.\.\.|(?<![a-z0-9])\.|\.(?![a-z0-9])|[!?;\"]", " ", raw, flags=re.I)
    else:
        t = _PUNCT.sub(" ", raw)
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
            who = m.group("w").strip()
            if _MULTI_RECIPIENT.search(f" {who} "):
                return ("reply", "I can only send to one WhatsApp chat at a time - tell me one chat first.")
            return ("tool", "send_whatsapp_message", {"contact_name": who, "message": m.group("m").strip()})
    t = re.sub(r"\s?:(?=\s)", "", t)  # the ": " separator only matters to the WhatsApp shapes above
    if m := _RADIO.match(t):
        kind = (m.group("k") or m.group("k2") or m.group("k3") or m.group("k4")).lower().replace("-", "")
        raw = (m.group("st") or m.group("st2") or m.group("v") or m.group("st4")).lower()
        state = "on" if raw in ("on", "enable", "chalu", "start") else "off"
        return ("tool", "set_radio", {"kind": kind, "state": state})
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
        elif m.group("n3"):
            n, unit, fw = int(m.group("n3")), m.group("unit3"), m.group("dir3").lower() == "aage"
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
        return ("tool", "add_note", {"text": m.group("text") or m.group("t2") or m.group("t3")})
    if m := _CHANNEL_VIDEO.match(t):
        name = (m.group("c") or m.group("c2") or m.group("c3")).strip()
        return ("tool", "play_latest_video_from_channel", {"channel_name": name})
    if (m := _YOUTUBE.match(t)) and not _CHANNEL_HINT.search(t):
        return ("tool", "play_video_on_youtube", {"query": m.group("q") or m.group("q2") or m.group("q3")})
    if m := _SITE_OPEN.match(t):
        url = _SITES[(m.group("site") or m.group("site2")).lower()]
        if m.group("browser"):  # "open youtube in brave": that browser first, then the site in it
            return ("multi", [("open_app", {"app_name": m.group("browser")}), ("open_website", {"url_or_query": url})])
        return ("tool", "open_website", {"url_or_query": url})
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


_SPLIT = re.compile(r"\s*,\s*(?:and |then |and then )?|\s+(?:and then|and also|and|then|after that|aur phir|aur fir|aur|phir|fir)\s+", re.I)


def fast_route(text: str):
    """("reply", text) / ("tools", [(name, args), ...]) for an unambiguous
    command, or None to fall back to the LLM. Deliberately conservative:
    a miss only costs LLM latency, a wrong hit does the wrong thing."""
    t = normalize(text)
    # A whole-utterance WhatsApp send is checked before the refusals, so a
    # message that merely mentions e.g. "call" or "email" still goes out.
    if (one := _route_one(t)) and one[0] == "tool" and one[1] == "send_whatsapp_message":
        return ("tools", [one[1:]])
    for pattern, reply in _REFUSE:  # whole utterance, before splitting: "open phonepe and send money"
        if pattern.search(t):
            return ("reply", reply)
    # Split first: a greedy single rule ("<anything> likho") would otherwise
    # swallow "notepad kholo aur hello likho" whole. A message that merely
    # contains "and" doesn't split cleanly into routable parts, so it falls
    # through to the single-command rules intact.
    parts = [normalize(p) for p in _SPLIT.split(normalize(text, keep_commas=True)) if normalize(p)]
    if len(parts) > 1:
        routed = _route_parts(parts)
        if all(r and r[0] in ("tool", "multi") for r in routed):
            return ("tools", [call for r in routed for call in _calls(r)])
    one = _route_one(t)
    if one:
        return one if one[0] == "reply" else ("tools", _calls(one))
    return None


def _calls(routed) -> list:
    """("tool", name, args) or ("multi", [(name, args), ...]) -> [(name, args), ...]"""
    return routed[1] if routed[0] == "multi" else [routed[1:]]


# After "open whatsapp", a later "send hi to X" / "X ko hi bhejo" means WhatsApp.
_WA_IN_CONTEXT = [
    re.compile(rf"^send {_WA_MSG_PREFIX}(?P<m>.+) to (?P<w>.+)$", re.I),
    re.compile(r"^(?P<w>.+?) ko (?P<m>.+?) (?:bhejo|bhej do|send karo)$", re.I),
]
_ACTION_START = re.compile(
    r"^(?:open|close|play|send|type|write|search|take|lock|mute|pause|show|set|turn|go|switch|minimi[sz]e|note|find|read|"
    r"check|tell|start|launch|put|skip|make|save|add|move|change|google)\b|"
    r"\b(?:karo|kro|kar do|kholo|chalao|lagao|bhejo|likho|dikhao|batao|band karo)$", re.I)


def _route_parts(parts: list[str]) -> list:
    routed, whatsapp_open = [], False
    for p in parts:
        r = _route_one(p)
        if r is None and whatsapp_open:
            for pattern in _WA_IN_CONTEXT:
                if (m := pattern.match(p)) and not _MULTI_RECIPIENT.search(f" {m.group('w')} "):
                    r = ("tool", "send_whatsapp_message", {"contact_name": m.group("w").strip(), "message": m.group("m").strip()})
                    break
        if r and r[0] == "tool" and r[1] == "open_app" and r[2]["app_name"].lower() == "whatsapp":
            whatsapp_open = True
        routed.append(r)
    return routed


def split_steps(text: str) -> list[str] | None:
    """For a multi-step command the fast path can't fully handle ("open
    spotify and play some music"): the steps to run one by one - routed
    ones directly, the rest through the LLM - or None if it doesn't
    cleanly split into separate commands (a WhatsApp message containing
    "and" must stay whole)."""
    parts = [normalize(p) for p in _SPLIT.split(normalize(text, keep_commas=True)) if normalize(p)]
    if len(parts) < 2:
        return None
    routed = _route_parts(parts)
    if not any(r and r[0] in ("tool", "multi") for r in routed):
        return None
    if not all(r or _ACTION_START.search(p) for p, r in zip(parts, routed)):
        return None
    return parts


def blocked_reply(tool_name: str) -> str | None:
    """Reply to give instead of letting the model call this tool itself."""
    return _LLM_BLOCKED_TOOLS.get(tool_name)


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
    "wifi": "network status radio", "internet": "network status", "online": "network status",
    "bluetooth": "devices connected radio", "off": "radio", "disable": "radio", "enable": "radio", "mic": "audio status", "speaker": "audio status",
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
    "move_status_orb", "set_daily_update_time", "open_website", "get_system_status", "set_radio",
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
