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

# Fillers and the name in any order: "okay Vaayu, hold on, pause" (suite found
# "hold on" was only stripped when it came before the name).
_LEAD = re.compile(
    r"^(?:(?:hey|ok|okay|so|and|um+|uh+|hmm+|hold on|let's|lets|vaa?yu|please|pls|can you|could you|would you|"
    r"will you|zara|just)\s+)*", re.I)
# Whisper adds a stray "You" before a command (real log: "You open youtube." -> 15 s LLM detour, failed).
_STRAY_YOU = re.compile(r"^you\s+(?=(?:open|close|play|pause|resume|stop|mute|unmute|search|show|turn|set|skip|go|take|lock|send|type)\b)", re.I)
_TAIL = re.compile(r"(?:\s+(?:please|pls|for me|na|yaar|now|right now|bhai|jaldi|quickly|a bit|a little|"
                   r"jaldi se|vaa?yu|for a (?:second|sec|moment|minute|bit)|for now|real quick))+$", re.I)
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
    # Pause and play are separate on purpose: "pause" must never START something
    # (control_media aims them at a playing YouTube video, see media.py).
    (rf"(?:pause|stop|hold)(?: (?:it|the music|music|the song|song|media|playback|the playback|playing|everything))?(?: {_KARO})?|"
     rf"(?:{_SONG} )?(?:ruko|rok do|roko)|{_SONG} (?:band {_KARO}|pause {_KARO}|rok do|roko|stop {_KARO})", "pause"),
    (rf"(?:play|resume|continue|unpause|keep playing|continue playing|start playing|carry on|keep going|go on)"
     rf"(?: (?:it|the music|music|the song|song|media|playback|playing|with it|with the video|with the music))?|"
     rf"(?:{_SONG} )?(?:chalao|chala do)|(?:{_SONG}) (?:wapas|phir se|fir se) (?:chalao|chala do|play {_KARO})", "play"),
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
    (rf"mute(?: the| it| everything)?(?: (?:system|volume|sound|audio|laptop|computer|speakers?))?|"
     rf"silence(?: the| it| everything)?(?: (?:laptop|computer|sound|audio))?|(?:awaa?z|sound|volume) band {_KARO}", "mute"),
    # Found live: "unmute" used to match the mute pattern above and muted again.
    (rf"unmute(?: the| it| everything)?(?: (?:system|volume|sound|audio|laptop|computer|speakers?))?|"
     rf"(?:awaa?z|sound|volume) (?:wapas )?(?:chalu|on) {_KARO}", "unmute"),
]
_MEDIA = [(re.compile(rf"^(?:{p})$", re.I), a) for p, a in _MEDIA]
# Speaker-player audio (tools/music.py) in looser words; "post"/"pose" = how Whisper writes "pause".
_AUDIO_CTRL = re.compile(r"^(?P<a>pause|post|pose|stop|hold|close|end|resume|start|continue|unpause)"
                         r"(?: the| this| that| your| my)? (?:audio|song|music|gaana|track)\b.*$|"
                         r"^(?:audio|song|music|gaana) (?P<a2>band|khatam)(?: karo| kar do)?$", re.I)

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
    rf"^play (?P<site3>youtube)$|"
    rf"^(?P<site2>{'|'.join(sorted(map(re.escape, _SITES), key=len, reverse=True))})(?: website| site)? "
    rf"(?:kholo|khol do|open karo|open kro|chalao)$", re.I)

# --- YouTube (youtube_control, verified against YouTube's own player) ---------
_NTH = {"first": 1, "1st": 1, "pehla": 1, "pehli": 1, "second": 2, "2nd": 2, "doosra": 2, "dusra": 2, "doosri": 2,
        "third": 3, "3rd": 3, "teesra": 3, "teesri": 3, "fourth": 4, "4th": 4, "chautha": 4, "fifth": 5, "5th": 5,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_SPEED_WORDS = {"double": 2, "twice": 2, "half": 0.5, "normal": 1, "regular": 1}
_THE_VIDEO = r"(?: the| this| that| it)?(?: youtube)?(?: video| one)?"
_ON = r"(?:turn on|switch on|enable|show|start|put on)"
_OFF = r"(?:turn off|switch off|disable|hide|stop|remove)"


def _yt(action, value=None):
    return ("tool", "youtube_control", {"action": action, **({"value": str(value)} if value is not None else {})})


_YT_RULES = [
    (rf"^{_ON}(?: the)? (?:captions|subtitles|subtitle|cc)$|^(?:captions|subtitles|subtitle|cc) (?:on|chalu)(?: karo| kar do)?$",
     lambda m: _yt("captions_on")),
    (rf"^{_OFF}(?: the)? (?:captions|subtitles|subtitle|cc)$|^(?:captions|subtitles|subtitle|cc) (?:off|band|hatao)(?: karo| kar do)?$",
     lambda m: _yt("captions_off")),
    (rf"^(?:{_ON}|go to|switch to)(?: the)? theat(?:er|re)(?: mode)?$|^theat(?:er|re) mode(?: on)?$", lambda m: _yt("theater_on")),
    (rf"^(?:{_OFF}|exit|leave)(?: the)? theat(?:er|re)(?: mode)?$|^theat(?:er|re) mode off$|^default view$", lambda m: _yt("theater_off")),
    (rf"^(?:loop|repeat){_THE_VIDEO}$|^put it on (?:loop|repeat)$|^(?:loop|repeat) (?:on|karo)$", lambda m: _yt("loop_on")),
    (r"^(?:stop|turn off) (?:looping|repeating|the loop|repeat)$|^(?:loop|repeat) off$", lambda m: _yt("loop_off")),
    (rf"^like{_THE_VIDEO}$|^video (?:ko )?like (?:karo|kar do)$|^hit (?:the )?like(?: button)?$", lambda m: _yt("like")),
    (rf"^(?:unlike|remove (?:the|my) like){_THE_VIDEO}$", lambda m: _yt("unlike")),
    (rf"^dislike{_THE_VIDEO}$", lambda m: _yt("dislike")),
    (r"^subscribe(?: to)?(?: (?:this|the|their))?(?: channel)?$|^channel (?:ko )?subscribe (?:karo|kar do)$", lambda m: _yt("subscribe")),
    (r"^skip(?: the| this)? ads?$|^ads? skip (?:karo|kar do)$|^skip the advertisement$", lambda m: _yt("skip_ad")),
    (rf"^{_ON}(?: the)? autoplay$|^autoplay on$", lambda m: _yt("autoplay_on")),
    (rf"^{_OFF}(?: the)? autoplay$|^autoplay off$|^don't autoplay$", lambda m: _yt("autoplay_off")),
    (r"^(?:set |change |put )?(?:the )?(?:video )?quality(?: to| at)? (?P<q>\d{3,4}p|4k|auto)$|^(?:play|watch|switch to)(?: it)?(?: in)? (?P<q2>\d{3,4}p|4k|auto)$|"
     r"^(?P<q3>\d{3,4}p|4k)(?: quality)?(?: karo)?$", lambda m: _yt("quality", m.group("q") or m.group("q2") or m.group("q3"))),
    (r"^(?:play(?: it)?|set(?: the)? speed|speed)(?: at| to)? (?P<r>\d(?:\.\d+)?|0?\.\d+) ?(?:x|times)?(?: speed)?$|^(?P<r2>\d(?:\.\d+)?) ?x(?: speed)?$",
     lambda m: _yt("speed", m.group("r") or m.group("r2"))),
    (r"^(?:play(?: it)?(?: at)? )?(?P<w>double|twice|half|normal|regular)(?: the)? speed$|^back to normal speed$",
     lambda m: _yt("speed", _SPEED_WORDS.get((m.group("w") or "normal").lower(), 1))),
    (r"^(?:speed (?:it |the video )?up|speed badhao|faster)$", lambda m: _yt("speed", 1.5)),
    (r"^(?:slow (?:it |the video )?down|slower|speed kam karo)$", lambda m: _yt("speed", 0.75)),
    (r"^(?:what(?:'s| is)(?: this)? (?:playing|video|song)|what(?:'s| is) this|what (?:video|song) is (?:this|playing)|which (?:video|song) is (?:this|playing)|"
     r"what am i (?:watching|listening to)|ye kaunsa (?:gaana|video) hai|kya chal raha hai)$", lambda m: _yt("now_playing")),
    (r"^(?:play )?(?:the )?next video$|^agla video(?: chalao| lagao)?$|^skip(?: this)? video$", lambda m: _yt("next")),
    (r"^(?:play |go back to )?(?:the )?previous video$|^pichla video(?: chalao| lagao)?$", lambda m: _yt("previous")),
    (rf"^(?:restart|replay|start over){_THE_VIDEO}$|^(?:play|start)(?: it| the video)? from the (?:start|beginning)$|"
     r"^shuru se (?:chalao|lagao)$|^start (?:it |the video )?(?:again|over)$", lambda m: _yt("restart")),
    (r"^(?:go|jump|skip|seek)(?: to)? (?P<mm>\d{1,2}):(?P<ss>\d{2})$", lambda m: _yt("seek_to", int(m.group("mm")) * 60 + int(m.group("ss")))),
    (r"^(?:go|jump|skip|seek) to (?P<n>\d+) (?P<u>minutes?|mins?|seconds?|secs?)(?: mark)?$",
     lambda m: _yt("seek_to", int(m.group("n")) * (60 if m.group("u").lower().startswith("m") else 1))),
    (r"^(?:search|find|look up|look for)(?: on)? youtube(?: for)? (?P<q>.+)$|^(?:search|find|look up|look for) (?P<q2>.+?) on youtube$|"
     r"^youtube (?:pe|par|pr) (?P<q3>.+?) (?:search|dhundo|search karo)$", lambda m: _yt("search", m.group("q") or m.group("q2") or m.group("q3"))),
    (r"^(?:play|open|click)(?: the)? (?P<n>first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|one|two|three|four|five)(?: one| video| result| link)?$|"
     r"^(?:play|open)(?: result| video)(?: number)? (?P<d>\d)$|^(?P<h>pehla|pehli|doosra|dusra|doosri|teesra|teesri|chautha) (?:video|wala) (?:chalao|lagao|kholo)$",
     lambda m: _yt("play_result", _NTH.get((m.group("n") or m.group("h") or "").lower()) or m.group("d"))),
    (r"^(?:open|go to|show me)(?: the| this| their)?(?: video's)? channel$|^channel (?:kholo|dikhao)$", lambda m: _yt("open_channel")),
    (r"^(?:go to |open )?(?:the )?youtube home(?: ?page)?$|^go home on youtube$", lambda m: _yt("home")),
    (r"^(?:next|agla|skip(?: this)?) short$", lambda m: _yt("next_short")),
    (r"^copy(?: the| this)?(?: video| youtube)? (?:link|url)$|^link copy (?:karo|kar do)$", lambda m: _yt("copy_link")),
    # Suggestions ranked by watch history (tools/youtube_taste.py)
    (r"^(?:recommend|suggest)(?: me)?(?: a| some| any)?(?: good)?(?: youtube)?(?: something| videos?| video| one)?"
     r"(?: to watch)?(?: on youtube)?(?: for me)?$|^what should i watch(?: next| now| on youtube)?$|"
     r"^(?:give me|any|show me)(?: some| a)?(?: video| youtube)? (?:recommendations?|suggestions?)$|"
     r"^(?:kuch|koi)(?: accha| achha)?(?: video)? (?:recommend|suggest) karo$|^kya dekhu$", lambda m: _yt("recommend")),
    # A video on the page - real log: "Play the video syncing in stand-up comedy by Pranav Sharma which is
    # on the right side of the currently played videos." got "Playing." and nothing happened.
    (r"^(?:play|open|start|click(?: on)?)(?: the)?(?: video| one)?(?: called| named| titled)? (?P<d>.+?),?"
     r"(?: (?:which|that) is| that's)? (?:on|in|from|at) (?:the )?(?:right(?: hand)?(?: side)?|side ?bar|"
     r"recommend(?:ed|ations?)|suggest(?:ed|ions?)|up next)\b.*$|"
     r"^(?:recommendations?|suggestions?|right side) (?:me|mein|se) (?:se )?(?P<d2>.+?) (?:wala|wali|wale)? ?(?:video )?"
     r"(?:chalao|lagao|kholo)$|"
     r"^(?:play|open)(?: the)? video (?!(?:in|on|from|again|at|now|for|to|full|ko)\b)(?P<d3>\S+(?: \S+)+)$",
     lambda m: _yt("play_recommended", (m.group("d") or m.group("d2") or m.group("d3")).strip())),
]
_YT_RULES = [(re.compile(p, re.I), build) for p, build in _YT_RULES]

# "volume to 75", "volume down to 75%", "awaz 40 par karo", "Picture volume level, 75%" (a real
# mis-transcription of "set volume level 75%"): a volume word plus exactly one number means set it.
_VOLUME_WORD = re.compile(r"\b(?:volume|sound|awaa?z)\b", re.I)
_BARE_LEVEL = re.compile(r"^(?:reduce|lower|decrease|drop|increase|raise|bring|turn|set|make|put)(?: it| that| this)?"
                         r"(?: down| up| back)?(?: to| at)? (?P<n>\d{1,3})(?: ?%| percent)$", re.I)
_VOLUME_NUMBER = re.compile(r"\b(\d{1,3})(?:\.\d+)?\s?(?:%|percent)?(?!\s*(?:seconds?|secs?|minutes?|mins?|x\b))", re.I)

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

# Internet answers (tools/web_answer.py) - checked after every local rule, so
# "what's playing", battery, tabs, notes, time/date never reach the web.
_WEATHER = re.compile(r"\b(?:weather|temperature|mausam|forecast|humidity)\b|\bbaa?rish\b|\bhow (?:hot|cold|warm)\b|"
                      r"\b(?:will it|is it|going to|gonna)\b.*\b(?:rain|snow|hot|cold|sunny)\b", re.I)
_WEATHER_PLACE = re.compile(r"\b(?:in|at|of|for) (?:the )?(?P<p>[a-z][a-z .'-]*?)(?: (?:today|tomorrow|tonight|now|this week|kal|aaj))?$|"
                            r"^(?:aaj |kal )?(?P<p2>[a-z][a-z .'-]*?) (?:ka|ki|me|mein|mai) (?:mausam|weather|temperature|baa?rish)", re.I)
_NOT_A_PLACE = {"it", "today", "tomorrow", "outside", "here", "the day", "the week", "now", "aaj", "kal"}
_KNOWLEDGE = re.compile(r"^(?:who|what|what's|whats|when|where|which|whose|why|how (?:many|much|old|far|tall|big|long|deep|"
                        r"high|fast|to|does|do|did|is|are|was)|tell me about|kaun|kab|kahan|kitne|kitna|kitni)\b|"
                        r"\b(?:kaun hai|kaun hain|kaun tha|kaun thi|kya hai|kab hai|kab tha|kahan hai)$", re.I)
# Personal/opinion questions go to chat, questions about Vaayu or this laptop
# stay local - only the outside world is looked up.
_NOT_WEB = re.compile(r"\b(?:i|i'm|im|me|my|mine|myself|we|our|you|your|yours|yourself|you're|u|ur|mujhe|mujhko|mera|meri|"
                      r"mere|main|hum|humara|tum|tumhara|tera|teri|aap|aapka|apna|apni|this|that|these|those|here|ye|yeh|"
                      r"yah|iska|page|screen|windows?|tabs?|playing|open|opened|running|installed|connected|files?|folders?|"
                      r"notes?|downloads|desktop|laptop|computer|pc|system|wi-?fi|bluetooth|battery|charge|cpu|ram|disk|"
                      r"storage|volume|mic|microphone|speakers?|usb|displays?|monitors?|printers?|video|song|music|time|"
                      r"vaayu|vayu)\b", re.I)
_WEB_OK = re.compile(r"^tell me about |\b(?:this|next|last) (?:year|month|week|weekend|season|summer|winter)\b|"
                     r"\b(?:time|date|day) (?:is it )?(?:in|at) (?!india\b)", re.I)  # "time in london": look it up
_MATH = re.compile(r"\d\s*(?:times|plus|minus|into|divided by|multiplied by|x|\*|\+|-|/)\s*\d", re.I)

_SCREENSHOT = re.compile(r"^(?:take|capture|grab|le lo|lo)?\s*(?:a )?(?:quick )?(?:screen ?shot|screen capture|ss)(?: (?:lo|le lo|lelo|karo))?$|"
                         r"^capture (?:the |my )?screen$|^screen ki (?:photo|picture|pic) (?:lo|le lo|khicho)$", re.I)
_LOCK = re.compile(r"^lock(?: (?:my|the))? (?:computer|laptop|pc|screen|system)$|^(?:laptop|computer|pc|screen) lock (?:karo|kar do)$|"
                   r"^lock (?:karo|kar do|it|it up)$", re.I)
_MINIMIZE = re.compile(r"^minimi[sz]e(?: (?:this|it|this window|the window|current window))?$|^(?:ye|isko|this) minimi[sz]e (?:karo|kar do)$", re.I)
# Date/time come from the clock, never the model: real log - "what is date today monday tuesday..." went
# to the chat model, which said Monday on a Sunday, flip-flopped, and once answered "[current time]".
_TIME = re.compile(r"^(?:what(?:'s| is)(?: the)?(?: current| exact)? time(?: now| right now| currently| it is| is it)?|"
                   r"what time is it(?: now| right now)?|(?:the )?current time|time now|"
                   r"(?:tell me|say)(?: the)?(?: current)? time|time (?:kya hai|kya hua|batao|please|bata do)|"
                   r"kitne baje(?: hai[n]?| hue| hain)?|kya time (?:hai|hua)|abhi kya time hai|samay kya hai|"
                   r"(?:can you )?(?:check|tell me) (?:the )?time)(?: now| please)?$", re.I)
_DATE = re.compile(r"^(?:what(?:'s| is)(?: the)? (?:date|day)(?: (?:today|to ?day|of today|is it|is it today))?|"
                   r"what(?:'s| is) today(?:'s date| date| day)?|what(?:'s| is) (?:the )?today'?s date|"
                   r"(?:what|which) day (?:is (?:it|today)|it is)(?: today)?|what is date today|"
                   r"(?:today'?s|today) date|date today|day today|(?:tell me )?(?:the )?date(?: please| today)?|"
                   r"aaj (?:kya|kaunsa|kaun sa|konsa) (?:date|din|tareekh|tarikh) hai|aaj ki (?:date|tareekh|tarikh)(?: kya hai)?|"
                   r"aaj (?:kya )?din hai|kaun ?sa din hai(?: aaj)?|date (?:kya hai|batao))$", re.I)
_DATE_AND_TIME = re.compile(r"^(?:what(?:'s| is) )?(?:the )?(?:date and time|time and date|day and time)(?: now| today)?$", re.I)
# The question buried in a longer sentence (real log: "I mean what is date today monday tuesday wednesday
# what is that date today").
_DATE_IN = re.compile(r"\b(?:what(?:'s| is)(?: the| that)? (?:date|day) (?:today|to ?day)|(?:what|which) day is (?:it )?today|"
                      r"today'?s date|aaj (?:kya|kaunsa|kaun sa) (?:date|din) hai)\b", re.I)
_ELSEWHERE = re.compile(r"\b(?:time|date|day)\b.*\b(?:in|at)\s+(?!india\b|my\b|this\b|the morning\b)[a-z]", re.I)
_TIME_IN = re.compile(r"\b(?:what(?:'s| is) the (?:current )?time|what time is it)\b", re.I)


def _now_reply(date: bool, time: bool) -> str:
    now = datetime.datetime.now()
    clock = now.strftime("%I:%M %p").lstrip("0")
    day = now.strftime("%A, %d %B %Y").replace(" 0", " ")
    if date and time:
        return f"It's {clock} on {day}."
    return f"It's {clock}." if time else f"Today is {day}."

# In-page video control (any site). Only wording that clearly says video/site.
# Includes Whisper mishearings seen in data/interactions.jsonl: "pause the
# video" -> "post the video", "resume that video" -> "assume that video".
_VIDEO_PLAY = re.compile(rf"^(?:i said )?(?P<act>pause|post|pose|paws|pauze|stop|play|resume|assume|presume|unpause|continue)"
                         rf"(?: the| that| this)?(?: (?P<site>{_SITE}))? video$|^video (?P<hi>roko|rok do|chalao|chala do|pause karo|play karo)$",
                         re.I)
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
    r"^(?:open youtube (?:and )?)?play (?:the )?(?:latest|newest|new|most recent|recent) video (?:from|of|by|on) (?:the )?(?P<c>.+?)(?: channel| youtube channel)?(?: (?:on|in) youtube)?$|"
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
_REMEMBER_NAME = re.compile(r"^(?:remember|save|add|learn)(?: the| my)? (?P<k>channel|contact|app) (?:name )?(?P<n>.+?)"
                            r"(?: to my (?:channels|contacts|apps))?$", re.I)
_LIST_NOTES = re.compile(r"^(?:show|read|list)(?: me)?(?: all)? my(?: last \d+)? notes$|^what are my notes$|^mere notes (?:dikhao|batao)$", re.I)
# The daily log review on demand (assistant/daily_review.py)
_REVIEW_LOGS = re.compile(r"^(?:review|scan|check|go through|analy[sz]e)(?: all)?(?: your| the| today'?s| my)? logs?(?: now)?$|"
                          r"^(?:do|run|start)(?: the| your| a)? (?:daily )?(?:log )?review$|"
                          r"^logs? (?:check|scan|review) (?:karo|kar do|kro)$|^apne logs (?:check|dekh) (?:karo|lo)$", re.I)

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
_WA_BARE = re.compile(r"^(?:message|text|whatsapp) (?P<rest>.+?) (?:on|in|via|par|pe) whatsapp$", re.I)


def _split_known_contact(rest: str):
    """(chat, message) when `rest` starts with a chat the user has named before
    (tools/names.py), longest name first - otherwise None, and the request goes
    to the usual "say it clearly" reply. Never guesses where an unknown name ends."""
    from assistant.tools import names
    for name in sorted(names.load()["contact"], key=len, reverse=True):
        if rest.lower().startswith(name.lower() + " ") and rest[len(name):].strip():
            message = rest[len(name):].strip()
            if _ANOTHER_RECIPIENT.search(message):  # "SAVED MESSAGES call mom": is mom the chat? Don't guess.
                return None
            return name, message
    return None


_ANOTHER_RECIPIENT = re.compile(r"\b(?:call|tell|message|text|ask|ping|whatsapp)\s+(?!(?:the|a|an|me|us|it|back|later)\b)\w+",
                                re.I)


_READ_PAGE = re.compile(r"^(?:summari[sz]e|read(?: out)?|explain) (?:this|the|current) (?:page|article|website|site|tab)$|"
                        r"^what(?:'s| is| does) (?:this|the) (?:page|article)(?: say| about)?$|^(?:ye|is) page (?:me|mein) kya hai$",
                        re.I)
_ORB = re.compile(r"^(?:move|put|place|shift|send)(?: the)? (?:status )?(?:orb|ball|circle|bubble)(?: to| in| at| into)?(?: the)? "
                  r"(?P<pos>.+?)(?: corner| of the screen| side)?$", re.I)
_UPDATE_TIME = re.compile(r"^(?:set|change|move|make)(?: my| the)?(?: daily)? (?:update|check-in|check in|checkin) time"
                          r"(?: to| at)? (?P<t>\d{1,2}(?::\d{2})?(?: ?[ap]\.?m\.?)?)$", re.I)
def _clock(said: str) -> str:
    """"9 pm" -> "21:00", "10:30" -> "10:30" (no am/pm: as said)."""
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*([ap])?", said.strip().lower())
    hour, minute, half = int(m.group(1)), m.group(2) or "00", m.group(3)
    if half == "p" and hour < 12:
        hour += 12
    elif half == "a" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute}" if half or m.group(2) else said.strip()


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


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _collapse_repeats(text: str) -> str:
    """Whisper sometimes repeats a sentence ("kya haal hai. kya haal hai.
    kya haal hai." - real log); identical consecutive sentences count once."""
    out = []
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        key = re.sub(r"[^\w]+", " ", sentence.lower()).strip()
        if not out or key != out[-1][0]:
            out.append((key, sentence))
    return " ".join(s for _k, s in out)


_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19, "hundred": 100, "a hundred": 100, "one hundred": 100}
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_NUMBER_WORDS_RE = re.compile(
    r"\b(?:(?P<tens>" + "|".join(_TENS) + r")(?:[ -](?P<unit>" + "|".join(_UNITS) + r"))?|(?P<teen>"
    + "|".join(sorted(_TEENS, key=len, reverse=True)) + r"))\b", re.I)


def _number_words_to_digits(text: str) -> str:
    """"down to thirty" -> "down to 30" (found live: the LLM turned it into 25)."""
    def to_digits(m):
        if m.group("teen"):
            return str(_TEENS[m.group("teen").lower()])
        return str(_TENS[m.group("tens").lower()] + (_UNITS[m.group("unit").lower()] if m.group("unit") else 0))
    return _NUMBER_WORDS_RE.sub(to_digits, text)


def normalize(text: str, keep_commas: bool = False) -> str:
    raw = _number_words_to_digits(_collapse_repeats(text))
    if keep_commas:  # for splitting steps: "open brave, play X" is two commands
        raw = raw.replace(",", " , ")
        t = re.sub(r"\.\.\.|(?<![a-z0-9])\.|\.(?![a-z0-9])|[!?;\"]", " ", raw, flags=re.I)
    else:
        t = _PUNCT.sub(" ", raw)
    t = _FILLER_WORDS.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _LEAD.sub("", t)
    t = _STRAY_YOU.sub("", t)
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
    if _ELSEWHERE.search(t):  # "what is the time in london" - not this laptop's clock
        pass
    elif _DATE_AND_TIME.match(t):
        return ("reply", _now_reply(date=True, time=True))
    elif _TIME.match(t):
        return ("reply", _now_reply(date=False, time=True))
    elif _DATE.match(t):
        return ("reply", _now_reply(date=True, time=False))
    elif len(t.split()) <= 20 and not _ACTION_START.match(t) and (_DATE_IN.search(t) or _TIME_IN.search(t)):
        return ("reply", _now_reply(date=bool(_DATE_IN.search(t)), time=bool(_TIME_IN.search(t))))
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
    for pattern, build in _YT_RULES:
        if m := pattern.match(t):
            return build(m)
    if (_VOLUME_WORD.search(t) and len(t.split()) <= 12 and len(nums := _VOLUME_NUMBER.findall(t)) == 1
            and not re.search(r"\b(?:seconds?|minutes?)\b", t, re.I)):
        level = min(int(nums[0]), 100)
        if re.search(r"\byoutube\b", t, re.I):  # named YouTube: YouTube, even while it's paused (found live)
            return _yt("volume_set", level)
        if re.search(r"\bvideo\b", t, re.I):  # the player's own volume on any site (YouTube handled inside)
            return ("tool", "set_video_volume", {"level": level})
        return ("tool", "control_media", {"action": "volume_set", "level": level})
    # "Reduce it to 20 percent" (real log) - a volume change with no "volume" word in it.
    if m := _BARE_LEVEL.match(t):
        return ("tool", "control_media", {"action": "volume_set", "level": min(int(m.group("n")), 100)})
    if m := _AUDIO_CTRL.match(t):  # real log: "post the audio which you are playing", "pause the song stop the song"
        verb = (m.group("a") or m.group("a2")).lower()  # close/band = pause too: works for a YouTube song as well
        return ("tool", "control_media", {"action": "play" if verb in ("resume", "start", "continue", "unpause") else "pause"})
    for pattern, action in _MEDIA:
        if pattern.match(t):
            return ("tool", "control_media", {"action": action})
    if m := _VIDEO_PLAY.match(t):
        act = (m.group("act") or "").lower()
        hi = (m.group("hi") or "").lower()
        args = {"action": "pause" if act in ("pause", "post", "pose", "paws", "pauze", "stop") or hi.startswith(("rok", "pause")) else "play"}
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
    if _READ_PAGE.match(t):
        return ("tool", "read_page", {})
    if m := _ORB.match(t):
        return ("tool", "move_status_orb", {"position": m.group("pos").strip()})
    if m := _UPDATE_TIME.match(t):
        return ("tool", "set_daily_update_time", {"time": _clock(m.group("t"))})
    if m := _WA_BARE.match(t):  # "message SAVED MESSAGES buy milk on whatsapp" - no word between chat and message
        if split := _split_known_contact(m.group("rest")):
            return ("tool", "send_whatsapp_message", {"contact_name": split[0], "message": split[1]})
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
    if m := _REMEMBER_NAME.match(t):
        return ("tool", "remember_name", {"kind": m.group("k").lower(), "name": m.group("n").strip()})
    if _REVIEW_LOGS.match(t):
        return ("tool", "review_logs", {})
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
        url = _SITES[(m.group("site") or m.group("site2") or m.group("site3")).lower()]
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
    if len(t.split()) <= 12 and _WEATHER.search(t) and not _NOT_WEB.search(t):
        m = _WEATHER_PLACE.search(t)
        place = ((m.group("p") or m.group("p2")) if m else "").strip()
        args = {"place": "" if place.lower() in _NOT_A_PLACE else place}
        if re.search(r"\b(?:tomorrow|kal)\b", t, re.I):
            args["day"] = "tomorrow"
        return ("tool", "get_weather", args)
    if _SONG_QUESTION.search(t) and (_KNOWLEDGE.search(t) or re.search(r"\b(?:tell me|batao|kis|kaun|kaunsi|kisne)\b", t, re.I)):
        return ("tool", "ask_about_song", {"question": t})  # "which movie is this song from?" - the one playing
    if _KNOWLEDGE.search(t) and not _NOT_WEB.search(_WEB_OK.sub("", t)) and not _MATH.search(t):
        return ("tool", "answer_question", {"question": t})
    for name, patterns in (("close_app", _CLOSE), ("open_app", _OPEN)):
        for pattern in patterns:
            m = pattern.match(t)
            if m and _app_ok(m.group("app").strip()):
                args = {"app_name": m.group("app").strip()}
                if "bg" in pattern.groupindex and m.group("bg"):
                    args["foreground"] = False
                return ("tool", name, args)
    if m := _VOICE.match(t):  # "change your voice" / "use the george voice" (tools/voice.py)
        want = (m.group("v") or m.group("v2") or m.group("v3") or "next").strip().lower()
        return ("tool", "set_voice", {"voice": _VOICE_KIND.get(want, want)})
    if m := _PLAY_FOR.match(t):  # "play kesariya for 15 seconds" / "play it for 10 seconds"
        n = int(m.group("n") or m.group("n2")) * (60 if (m.group("u") or m.group("u2")).lower().startswith("m") else 1)
        q = (m.group("q") or "").strip()
        if not q or _THIS_ONE.match(q):
            return ("tool", "music_control", {"action": "play_for", "level": n})
        return ("tool", "play_music", {"query": q, "seconds": n})
    # Songs through the speakers, no browser (tools/music.py) - "on youtube" / "video" still mean YouTube.
    if m := _MUSIC.match(t):
        q = next(g for g in m.groups() if g)
        return ("tool", "play_music", {"query": q.strip()})
    if m := _HINDI_SONG.search(t):  # real log: "It wadya sa gaana chala do ya bollywood" (ek badhiya sa gaana...)
        words = [w for w in re.findall(r"[a-z]+", t.lower()) if w in _GENRES]
        return ("tool", "play_music", {"query": f"{' '.join(words)} songs" if words else "some music"})
    if (m := _PLAY_BARE.match(t)) and not _NOT_A_SONG.search(m.group("q")):
        from assistant.tools import names
        if not names.best_match(m.group("q"), "app"):  # "play tekken" is the game, not a song
            return ("tool", "play_music", {"query": m.group("q").strip()})
    return None


_MUSIC = re.compile(
    r"^(?:play|bajao|chalao|sunao|lagao)(?: me| us)? (?P<q4>some music|music|some songs|a song|songs|koi gaana|kuch gaane)$|"
    r"^(?:play|bajao|chalao|sunao|lagao|put on)(?: me| us)?(?: the| a| some| any)? (?:song|songs|music|gaana|gana|gaane)"
    r"(?: of| by| from| called| named)? (?P<q>.+)$|"
    r"^(?:play|put on)(?: me| us)?(?: the| a| some)? (?P<q2>.+?) (?:song|songs|music|gaana|gaane|track)$|"
    r"^(?P<q3>.+?) (?:ka |ke |ki )?(?:song|songs|gaana|gaane|gana|music) (?:chalao|bajao|lagao|sunao|play karo|chala do|baja do)$",
    re.I)
_VOICE = re.compile(
    r"^(?:change|switch|try)(?: your| the| my)?(?: own)? voice(?: to(?: an?| the)? (?P<v>\w+)(?: (?:voice|one))?.*)?$|"
    r"^(?:use|try|speak in|talk in)(?: an?| the)? (?P<v2>\w+) voice.*$|^(?:try )?(?:a |another |next |different )voice$|"
    r"^(?:awaaz|awaz|voice) (?:badlo|change karo|badal do)$|^(?:become|be) an? (?P<v3>girl|boy|woman|man|female|male)$",
    re.I)
# Real log: "switch your voice to a girl voice female voice" went to the LLM, which picked a browser tab.
_VOICE_KIND = {"girl": "heart", "female": "heart", "woman": "heart", "lady": "heart", "ladki": "heart",
               "boy": "echo", "male": "echo", "man": "echo", "ladka": "echo", "british": "george", "indian": "psi"}
_PLAY_FOR = re.compile(r"^(?:play|bajao|chalao|sunao)(?: me)?(?: only| just)?(?P<q> .+?)? (?:for|only for|just for) "
                       r"(?:only |just )?(?P<n>\d+) (?P<u>seconds?|secs?|minutes?|mins?)(?: only)?$|"
                       r"^(?:play|bajao)(?: only| just)? (?P<n2>\d+) (?P<u2>seconds?|secs?)(?: of it| of this| of the song)?$",
                       re.I)
_SONG_QUESTION = re.compile(r"(?:this|the current|current|that) (?:song|track|gaana)|(?:ye|yeh|is|iss) gaan[ae]|"
                            r"the song (?:that'?s|that is|you'?re) playing", re.I)
_THIS_ONE = re.compile(r"^(?:it|this|that|the song|this song|the music|the audio|the track|again)$", re.I)
_HINDI_SONG = re.compile(r"\b(?:gaana|gaane|gana|song|songs)\b.{0,25}\b(?:chala do|chalao|bajao|baja do|sunao|suna do|"
                         r"laga do|lagao)\b", re.I)
_GENRES = {"bollywood", "punjabi", "romantic", "sad", "party", "old", "lofi", "hindi", "english", "rap", "haryanvi",
           "bhajan", "ghazal", "sufi", "dance", "workout", "arijit"}
_PLAY_BARE = re.compile(r"^(?:play|bajao|sunao) (?P<q>.{3,})$", re.I)
_NOT_A_SONG = re.compile(r"\b(?:video|videos|youtube|channel|tab|it|this|that|next|previous|again|movie|film|trailer|"
                         r"episode|game|match|result|one|first|second|third|last|from|at|speed|"
                         r"latest|newest)\b", re.I)


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
            calls = [call for r in routed for call in _calls(r)]
            # play_video_on_youtube opens YouTube itself - a separate "open youtube" first is a wasted step
            calls = [c for i, c in enumerate(calls) if not (
                c == ("open_website", {"url_or_query": "youtube.com"}) and i + 1 < len(calls)
                and calls[i + 1][0] == "play_video_on_youtube")]
            return ("tools", calls)
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
    routed, whatsapp_open, youtube_open = [], False, False
    for p in parts:
        r = _route_one(p)
        if r is None and whatsapp_open:
            for pattern in _WA_IN_CONTEXT:
                if (m := pattern.match(p)) and not _MULTI_RECIPIENT.search(f" {m.group('w')} "):
                    r = ("tool", "send_whatsapp_message", {"contact_name": m.group("w").strip(), "message": m.group("m").strip()})
                    break
        # "open youtube and play kesariya": after YouTube, a bare "play X" means play it there.
        if r is None and youtube_open and (m := re.match(r"^(?:play|search|put on|chalao|lagao) (?P<q>.+)$", p, re.I)):
            r = ("tool", "play_video_on_youtube", {"query": m.group("q").strip()})
        if r and youtube_open and r[0] == "tool" and r[1] == "play_music":  # they asked for YouTube - play it there
            r = ("tool", "play_video_on_youtube", dict(r[2]))
        if r and r[0] == "tool" and r[1] == "open_app" and r[2]["app_name"].lower() == "whatsapp":
            whatsapp_open = True
        if r and r[0] == "tool" and "youtube" in str(r[2].get("url_or_query") or r[2].get("app_name") or "").lower():
            youtube_open = True
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


# Free-text arguments the model fills in must come from the user's own words.
# Real log: "Open Brave and Play YouTube" -> play_video_on_youtube(query=
# "latest bollywood songs 2026"), which the user never said.
GROUNDED_ARGS = {
    "play_video_on_youtube": ("query", "play"), "play_latest_video_from_channel": ("channel_name", "play"),
    "open_app": ("app_name", "open"), "close_app": ("app_name", "close"), "add_note": ("text", "note down"),
    "type_text": ("text", "type"), "open_website": ("url_or_query", "open"), "search_files": ("query", "find"),
}
_GROUND_STOP = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "my", "me", "app", "com", "www", "http",
                "https", "please", "video", "videos", "song", "songs", "latest", "new"}


def _stem(w: str) -> str:
    return w[:5]


def ungrounded(tool_name: str, arguments: dict, user_text: str) -> str | None:
    """A reply asking the user to repeat, if the model invented this tool
    call's free-text argument instead of taking it from what was said."""
    spec = GROUNDED_ARGS.get(tool_name)
    if not spec:
        return None
    arg, verb = spec
    value = str(arguments.get(arg) or "")
    words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in _GROUND_STOP and len(w) > 1]
    if not words:
        return None
    said = {_stem(w) for w in re.findall(r"[a-z0-9]+", user_text.lower())}
    found = sum(_stem(w) in said for w in words)
    if found / len(words) >= 0.6:
        return None
    return f"I'm not sure what you want me to {verb} - could you say it again?"


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
    "mausam": "weather", "temperature": "weather", "rain": "weather", "barish": "weather",
    "who": "answer question", "kaun": "answer question", "president": "answer question",
    "minister": "answer question", "capital": "answer question",
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
    picked = [name for _score, name in scored[:k]]
    # Word overlap alone missed the needed tool in 75 of 417 distill examples
    # (open_app 27, close_app 13...) - the model can't call a tool it isn't shown.
    lowered = normalize(text).lower()
    for pattern, tool in _MUST_OFFER:
        if tool in registry._tools and tool not in picked and pattern.search(lowered):
            picked = [tool] + picked[:k - 1]
    subset = ToolRegistry()
    subset._tools = {name: registry._tools[name] for name in picked}
    return subset


_MUST_OFFER = [(re.compile(p), tool) for p, tool in [
    (r"\b(?:open|launch|start|run|fire up|boot up|pull up|bring up|kholo|khol|kholna|chalu|on karo|i need|"
     r"i wanna use|i want to use|get me into)\b|\bget .+ up\b", "open_app"),
    (r"\b(?:close|quit|exit|shut|kill|band|hata|hatao|off my screen|finished with|done with)\b", "close_app"),
    (r"\b(?:play|song|songs|gaana|gana|music|youtube)\b", "play_video_on_youtube"),
    (r"\b(?:volume|sound|awaa?z|pause|resume|next|previous|mute|unmute|louder|quieter|dheere|gaana|aage wala)\b",
     "control_media"),
    (r"\b(?:press|shortcut|ctrl|alt|undo|copy|paste|save)\b", "press_hotkey"),
]]


# Action tools whose result string already says what happened - spoken
# as-is instead of a second LLM call. Read/list/get tools are left out on
# purpose: their result is data the model still has to act on or summarize.
DIRECT_REPLY_TOOLS = {
    "open_app", "close_app", "control_media", "take_screenshot", "lock_computer", "minimize_window",
    "restore_window", "switch_to_window", "play_video_on_youtube", "play_latest_video_from_channel",
    "add_note", "set_video_volume", "set_video_fullscreen", "seek_video", "control_video_playback",
    "close_browser_tab", "switch_to_browser_tab", "type_text", "send_whatsapp_message",
    "move_status_orb", "set_daily_update_time", "open_website", "get_system_status", "set_radio", "get_weather", "review_logs",
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
    return result if isinstance(result, str) and result.strip() else None
