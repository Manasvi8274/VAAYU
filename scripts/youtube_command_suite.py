"""
~20,000 generated YouTube voice commands -> did Vaayu understand them?

Every YouTube action x many real ways of saying it (English, Hinglish, and
the speech-to-text mishearings seen in data/interactions.jsonl) x values
(volumes, times, speeds, qualities, result numbers, songs, channels) x
speech noise (name/filler prefixes, trailing words, punctuation, caps).

Each case checks the command reaches the right tool with the right
arguments through the instant (no-LLM) path - a case that would need the
slow LLM fallback counts as a miss. Whether the tool then really works on
YouTube is checked separately, live, by scripts/live_youtube_test.py.

Templates marked HOLDOUT were written up front and are never used to tune
the router: their score is the honest "wording it has never seen" number.

    python scripts/youtube_command_suite.py            # full run + report
    python scripts/youtube_command_suite.py --misses   # also list misses
"""

import re
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant.llm import router  # noqa: E402

H = "HOLDOUT"


def cm(action, **kw):
    return ("control_media", {"action": action, **kw})


def yt(action, value=None):
    return ("youtube_control", {"action": action, **({"value": str(value)} if value is not None else {})})


# intent -> list of (template, holdout?) and an expected-call builder taking the value.
# Templates use {v} for the value. Expected: list of acceptable (tool, args) calls.
SONGS = ["perfect by ed sheeran", "kesariya", "lofi hip hop", "arijit singh songs", "shape of you", "believer",
         "old hindi songs", "punjabi songs", "naatu naatu", "tum hi ho", "blinding lights", "coke studio",
         "relaxing piano music", "workout music", "bollywood party songs"]
CHANNELS = ["MrBeast", "Carryminati", "OG Crew", "Dhruv Rathee", "Technical Guruji", "BB Ki Vines", "T-Series",
            "Total Gaming", "Linus Tech Tips", "Samay Raina", "Mortal", "Lofi Girl"]
QUERIES = ["carryminati", "funny cat videos", "python tutorial", "cricket highlights", "iphone review",
           "street food delhi", "news today", "minecraft", "lofi music", "how to cook pasta"]

INTENTS = {
    "pause": ([("pause", 0), ("pause it", 0), ("pause the video", 0), ("pause the music", 0), ("stop the video", 0),
               ("stop the music", 0), ("hold on, pause", 0), ("pause karo", 0), ("video roko", 0), ("gaana roko", 0),
               ("I said post the video", 0), ("post the video", 0), ("pause this video", H), ("can you pause it", H),
               ("video pause karo", H), ("ruko", H), ("pause the song", H), ("stop", H)],
              lambda v: [cm("pause"), ("control_video_playback", {"action": "pause"})]),
    "play": ([("play", 0), ("resume", 0), ("resume the video", 0), ("play the video", 0), ("continue", 0),
              ("continue playing", 0), ("unpause", 0), ("Assume that video", 0), ("video chalao", 0), ("gaana chalao", 0),
              ("resume it", H), ("keep playing", H), ("play it", H), ("start playing", H), ("music wapas chalao", H),
              ("resume that video", H), ("video play karo", H)],
             lambda v: [cm("play"), ("control_video_playback", {"action": "play"})]),
    "volume_set": ([("volume {v}", 0), ("set volume to {v}", 0), ("set the volume to {v} percent", 0), ("volume {v}%", 0),
                    ("volume to {v}", 0), ("make the volume {v}", 0), ("volume down to {v}", 0), ("volume up to {v}", 0),
                    ("awaz {v} par karo", 0), ("volume {v} percent karo", 0), ("change volume to {v}", H),
                    ("put the volume at {v}", H), ("sound {v} percent", H), ("I want the volume at {v}", H),
                    ("set volume level {v}%", H), ("volume ko {v} pe karo", H)],
                   lambda v: [cm("volume_set", level=v)]),
    "volume_up": ([("volume up", 0), ("increase the volume", 0), ("turn up the volume", 0), ("louder", 0),
                   ("make it louder", 0), ("volume badhao", 0), ("awaz tez karo", 0), ("raise the volume", 0),
                   ("bump up the volume", H), ("thoda volume badhao", H), ("turn the volume up", H), ("it's too quiet", H),
                   ("awaz badhao", H), ("I can't hear it", H)],
                  lambda v: [cm("volume_up")]),
    "volume_down": ([("volume down", 0), ("decrease the volume", 0), ("turn down the volume", 0), ("quieter", 0),
                     ("make it quieter", 0), ("volume kam karo", 0), ("awaz kam karo", 0), ("lower the volume", 0),
                     ("it's too loud", H), ("thoda awaz kam karo", H), ("turn the volume down", H), ("reduce the sound", H),
                     ("awaz dheere karo", H)],
                    lambda v: [cm("volume_down")]),
    "mute": ([("mute", 0), ("mute it", 0), ("mute the video", 0), ("silence", 0), ("sound band karo", 0), ("mute everything", H),
              ("awaz band kar do", H), ("silence it", H)],
             lambda v: [cm("mute"), ("set_video_volume", {"direction": "mute"})]),
    "seek_forward": ([("skip ahead {v} seconds", 0), ("skip forward {v} seconds", 0), ("forward {v} seconds", 0),
                      ("fast forward {v} seconds", 0), ("video {v} second aage karo", 0), ("jump ahead {v} seconds", H),
                      ("skip ahead by {v} seconds", H), ("{v} second aage karo", H)],
                     lambda v: [("seek_video", {"direction": "forward", "seconds": v})]),
    "seek_back": ([("go back {v} seconds", 0), ("rewind {v} seconds", 0), ("skip back {v} seconds", 0),
                   ("video {v} second peeche karo", 0), ("jump back {v} seconds", H), ("rewind by {v} seconds", H),
                   ("{v} second peeche karo", H)],
                  lambda v: [("seek_video", {"direction": "backward", "seconds": v})]),
    "seek_to": ([("go to {mm}:{ss}", 0), ("jump to {mm}:{ss}", 0), ("skip to {mm}:{ss}", 0), ("seek to {mm}:{ss}", H),
                 ("go to {m} minutes", 0), ("jump to {m} minutes", H)],
                None),
    "restart": ([("restart the video", 0), ("replay", 0), ("start over", 0), ("play from the beginning", 0),
                 ("shuru se chalao", 0), ("restart", H), ("replay this video", H), ("start it again", H),
                 ("start the video from the start", H)],
                lambda v: [yt("restart")]),
    "next": ([("next video", 0), ("play the next video", 0), ("agla video", 0), ("skip this video", 0), ("next", 0),
              ("next song", 0), ("change the song", 0), ("agla gaana", H), ("skip it", H), ("play next", H),
              ("agla video chalao", H), ("gaana badlo", H)],
             lambda v: [yt("next"), cm("next")]),
    "previous": ([("previous video", 0), ("play the previous video", 0), ("pichla video", 0), ("previous", 0),
                  ("previous song", H), ("go back to the previous video", H), ("pichla gaana", H)],
                 lambda v: [yt("previous"), cm("previous")]),
    "speed": ([("play at {v}x speed", 0), ("speed {v}", 0), ("set speed to {v}", 0), ("{v}x", 0), ("play it at {v}x", H),
               ("set the speed to {v}x", H), ("{v}x speed", H)],
              lambda v: [yt("speed", v)]),
    "speed_words": ([("double speed", 0), ("normal speed", 0), ("half speed", 0), ("back to normal speed", H),
                     ("play at double speed", H)],
                    None),
    "captions_on": ([("turn on captions", 0), ("turn on subtitles", 0), ("show subtitles", 0), ("captions on", 0),
                     ("subtitles on karo", 0), ("enable captions", H), ("cc on", H), ("switch on subtitles", H)],
                    lambda v: [yt("captions_on")]),
    "captions_off": ([("turn off captions", 0), ("turn off subtitles", 0), ("hide subtitles", 0), ("captions off", 0),
                      ("subtitles band karo", 0), ("disable captions", H), ("remove the subtitles", H), ("cc off", H)],
                     lambda v: [yt("captions_off")]),
    "theater_on": ([("theater mode", 0), ("turn on theater mode", 0), ("theatre mode", 0), ("switch to theater mode", H),
                    ("theater mode on", H)],
                   lambda v: [yt("theater_on")]),
    "theater_off": ([("exit theater mode", 0), ("turn off theater mode", 0), ("theater mode off", H), ("leave theatre mode", H)],
                    lambda v: [yt("theater_off")]),
    "fullscreen_on": ([("full screen", 0), ("make it fullscreen", 0), ("go fullscreen", 0), ("fullscreen karo", 0),
                       ("full screen this video", H), ("go full screen", H)],
                      lambda v: [("set_video_fullscreen", {"fullscreen": True}), yt("fullscreen_on")]),
    "fullscreen_off": ([("exit full screen", 0), ("exit fullscreen", 0), ("shrink the video", 0), ("leave fullscreen", H),
                        ("full screen hatao", H)],
                       lambda v: [("set_video_fullscreen", {"fullscreen": False}), yt("fullscreen_off")]),
    "loop_on": ([("loop this video", 0), ("repeat this video", 0), ("put it on repeat", 0), ("loop the video", H),
                 ("put it on loop", H)],
                lambda v: [yt("loop_on")]),
    "loop_off": ([("stop looping", 0), ("loop off", 0), ("turn off repeat", H), ("repeat off", H)],
                 lambda v: [yt("loop_off")]),
    "like": ([("like this video", 0), ("like the video", 0), ("hit the like button", 0), ("video like karo", 0),
              ("like it", H), ("like this", H)],
             lambda v: [yt("like")]),
    "dislike": ([("dislike this video", 0), ("dislike", H)], lambda v: [yt("dislike")]),
    "subscribe": ([("subscribe", 0), ("subscribe to this channel", 0), ("channel subscribe karo", 0),
                   ("subscribe to the channel", H), ("subscribe to their channel", H)],
                  lambda v: [yt("subscribe")]),
    "skip_ad": ([("skip ad", 0), ("skip the ad", 0), ("skip ads", 0), ("ad skip karo", 0), ("skip this ad", H),
                 ("skip the advertisement", H)],
                lambda v: [yt("skip_ad")]),
    "autoplay_off": ([("turn off autoplay", 0), ("autoplay off", 0), ("disable autoplay", H), ("stop autoplay", H)],
                     lambda v: [yt("autoplay_off")]),
    "autoplay_on": ([("turn on autoplay", 0), ("autoplay on", 0), ("enable autoplay", H)], lambda v: [yt("autoplay_on")]),
    "quality": ([("quality {v}", 0), ("set quality to {v}", 0), ("play in {v}", 0), ("{v}", 0), ("change the quality to {v}", H),
                 ("switch to {v}", H), ("set the video quality to {v}", H)],
                lambda v: [yt("quality", v)]),
    "now_playing": ([("what is playing", 0), ("what's playing", 0), ("which song is this", 0), ("what am i watching", 0),
                     ("ye kaunsa gaana hai", 0), ("what video is this", H), ("which video is playing", H),
                     ("kya chal raha hai", H)],
                    lambda v: [yt("now_playing")]),
    "search": ([("search youtube for {v}", 0), ("search on youtube for {v}", 0), ("find {v} on youtube", 0),
                ("youtube pe {v} search karo", 0), ("look up {v} on youtube", H), ("search {v} on youtube", H),
                ("youtube par {v} dhundo", H)],
               lambda v: [yt("search", v)]),
    "play_result": ([("play the {nth} video", 0), ("play the {nth} one", 0), ("open the {nth} result", 0),
                     ("{hi} video chalao", 0), ("play video number {d}", H), ("play the {nth} result", H),
                     ("click the {nth} video", H)],
                    None),
    "open_channel": ([("open the channel", 0), ("go to the channel", 0), ("open this channel", 0), ("channel kholo", 0),
                      ("show me the channel", H), ("go to their channel", H)],
                     lambda v: [yt("open_channel")]),
    "home": ([("youtube home", 0), ("go to youtube home", 0), ("open youtube home page", H)], lambda v: [yt("home")]),
    "next_short": ([("next short", 0), ("agla short", 0), ("skip this short", H)], lambda v: [yt("next_short")]),
    "copy_link": ([("copy the link", 0), ("copy the video link", 0), ("link copy karo", 0), ("copy this url", H)],
                  lambda v: [yt("copy_link")]),
    "open_youtube": ([("open youtube", 0), ("youtube kholo", 0), ("play youtube", 0), ("go to youtube", 0),
                      ("open youtube in a new tab", 0), ("open the youtube website", H), ("youtube chalao", H),
                      ("visit youtube", H)],
                     lambda v: [("open_website", {"url_or_query": "youtube.com"})]),
    "play_on_youtube": ([("play {v} on youtube", 0), ("youtube pe {v} chalao", 0), ("put on {v} on youtube", 0),
                         ("youtube par {v} lagao", H), ("play {v} on youtube please", H)],
                        # User's rule (2026-10-25): a SONG plays through Vaayu's speakers, even with "youtube"
                        # in the sentence; only a video goes to YouTube in Brave.
                        lambda v: [("play_music" if re.search(r"\b(?:song|songs|music|gaana|gaane)\b", v)
                                    else "play_video_on_youtube", {"query": v})]),
    "latest_from_channel": ([("play the latest video from {v}", 0), ("{v} ka latest video chalao", 0),
                             ("play the newest video from {v}", 0), ("play {v}'s latest video", H),
                             ("play the most recent video from {v}", H), ("{v} ka naya video chalao", H)],
                            lambda v: [("play_latest_video_from_channel", {"channel_name": v})]),
}

NTH = [("first", "pehla", 1), ("second", "doosra", 2), ("third", "teesra", 3), ("fourth", "chautha", 4)]
PREFIXES = ["", "", "", "vaayu ", "Vaayu, ", "hey vaayu ", "please ", "can you ", "um ", "okay vaayu ", "vaayu please ",
            "zara "]
SUFFIXES = ["", "", "", " please", " na", " now", " yaar", "."]


def _values(intent):
    if intent == "volume_set":
        return list(range(0, 101, 5))
    if intent in ("seek_forward", "seek_back"):
        return [5, 10, 15, 20, 30, 45, 60, 90]
    if intent == "speed":
        return ["0.5", "0.75", "1.25", "1.5", "1.75", "2"]
    if intent == "quality":
        return ["144p", "360p", "480p", "720p", "1080p", "auto"]
    if intent == "search":
        return QUERIES
    if intent == "play_on_youtube":
        return SONGS
    if intent == "latest_from_channel":
        return CHANNELS
    return [None]


def _expand(intent, template):
    """(text, acceptable calls) for one template across its values."""
    if intent == "seek_to":
        for mm, ss in [(0, 30), (1, 15), (2, 30), (5, 0), (10, 45), (12, 5)]:
            if "{m}" in template:
                yield template.format(m=mm or 1), [yt("seek_to", (mm or 1) * 60)]
            else:
                yield template.format(mm=mm, ss=f"{ss:02d}"), [yt("seek_to", mm * 60 + ss)]
        return
    if intent == "speed_words":
        word = {"double speed": 2, "normal speed": 1, "half speed": 0.5, "back to normal speed": 1,
                "play at double speed": 2}[template]
        yield template, [yt("speed", word)]
        return
    if intent == "play_result":
        for nth, hi, d in NTH:
            yield template.format(nth=nth, hi=hi, d=d), [yt("play_result", d)]
        return
    for v in _values(intent):
        yield (template.format(v=v) if v is not None else template), INTENTS[intent][1](v)


def _noisy(text, rng):
    t = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
    r = rng.random()
    if r < 0.15:
        t = t.upper()
    elif r < 0.45:
        t = t[:1].upper() + t[1:]
    return t


def build(total=20000, seed=11):
    rng = random.Random(seed)
    base = []
    for intent, (templates, _builder) in INTENTS.items():
        for template, holdout in templates:
            for text, want in _expand(intent, template):
                base.append({"intent": intent, "holdout": holdout == H, "text": text, "want": want})
    cases, seen = [], set()
    for c in base:  # every clean phrasing once
        cases.append({**c, "noisy": False})
        seen.add(c["text"])
    by_intent = defaultdict(list)
    for c in base:
        by_intent[c["intent"]].append(c)
    intents = list(by_intent)
    while len(cases) < total:  # then noisy variants until the target size - intent first, so each gets a fair share
        c = rng.choice(by_intent[rng.choice(intents)])
        t = _noisy(c["text"], rng)
        if t in seen:
            continue
        seen.add(t)
        cases.append({**c, "text": t, "noisy": True})
    return cases


def _norm_args(args):
    return {k: str(v).lower() for k, v in args.items()}


def check(case):
    routed = router.fast_route(case["text"])
    if not routed or routed[0] != "tools" or len(routed[1]) != 1:
        return False, routed
    name, args = routed[1][0]
    got = (name, _norm_args(args))
    for want_name, want_args in case["want"]:
        if got[0] == want_name and all(got[1].get(k) == str(v).lower() for k, v in want_args.items()):
            return True, routed
    return False, routed


def main():
    cases = build()
    by = defaultdict(lambda: [0, 0])
    split = {True: [0, 0], False: [0, 0]}
    misses = []
    for c in cases:
        ok, routed = check(c)
        by[c["intent"]][0] += ok
        by[c["intent"]][1] += 1
        split[c["holdout"]][0] += ok
        split[c["holdout"]][1] += 1
        if not ok:
            misses.append((c, routed))
    total_ok = sum(v[0] for v in by.values())
    print(f"YouTube command suite: {total_ok}/{len(cases)} understood instantly ({100 * total_ok / len(cases):.1f}%)")
    print(f"  tuning templates : {split[False][0]}/{split[False][1]} ({100 * split[False][0] / max(1, split[False][1]):.1f}%)")
    print(f"  HOLDOUT templates: {split[True][0]}/{split[True][1]} ({100 * split[True][0] / max(1, split[True][1]):.1f}%)  <- unseen wording")
    print("\n| intent | understood |\n|---|---|")
    for intent, (ok, n) in sorted(by.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"| {intent} | {ok}/{n} ({100 * ok / n:.0f}%) |")
    if "--misses" in sys.argv:
        seen = set()
        print("\nMisses (one per distinct clean template):")
        for c, routed in misses:
            key = (c["intent"], c["text"].lower().strip(" .,!?"))
            if c["noisy"] or key in seen:
                continue
            seen.add(key)
            print(f"  {'H ' if c['holdout'] else '  '}{c['intent']:20} {c['text']!r:45} -> {routed}")
    return total_ok, len(cases)


if __name__ == "__main__":
    main()
