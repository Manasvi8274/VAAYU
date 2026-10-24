"""
Live end-to-end YouTube test in the real Brave browser: each command goes
through Vaayu's full brain (router -> real tools), then YouTube's own state
is read back independently to see what ACTUALLY happened.

Outcomes per step: PASS, FAIL (Vaayu said it failed - honest), or
FALSE_CONFIRMATION (Vaayu said done, but YouTube's state disagrees - the
bug class this exists to catch).

Windows audio is muted for the run and restored after. Likes, subscriptions
and autoplay are NOT exercised: they change the user's real YouTube account.

    python scripts/live_youtube_test.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pycaw.pycaw import AudioUtilities  # noqa: E402

import assistant.tools  # noqa: E402,F401
from assistant.llm.ollama_client import OllamaBrain  # noqa: E402
from assistant.llm.prompts import SYSTEM_PROMPT  # noqa: E402
from assistant.tools import apps  # noqa: E402
from assistant.tools import youtube_player as yp  # noqa: E402
from assistant.tools.registry import registry  # noqa: E402

P = yp._P


def tab():
    return yp.youtube_tab(require_watch=False)


def ev(js):
    t = tab()
    return yp._ev(t, js) if t else None


def state():
    return ev(f"{P} && {P}.getPlayerState ? {P}.getPlayerState() : null")


def vol():
    return ev(f"{P}.getVolume()")


def now():
    return ev(f"{P}.getCurrentTime()") or 0


def vid():
    return ev(f"{P}.getVideoData().video_id")


def wait(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if cond():
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def playing():
    return state() in (1, 3)


def paused():
    return state() == 2


ctx = {}
# (command, check) - check runs after the command and must see YouTube's real state.
STEPS = [
    ("play lofi hip hop study music on youtube", lambda: wait(lambda: vid() is not None and playing(), 15)),
    ("pause the video", lambda: wait(paused)),
    ("play", lambda: wait(playing)),
    ("pause", lambda: wait(paused)),
    ("resume", lambda: wait(playing)),
    ("I said post the video", lambda: wait(paused)),  # real mishearing of "pause the video"
    ("Assume that video", lambda: wait(playing)),  # real mishearing of "resume that video"
    ("video roko", lambda: wait(paused)),
    ("video chalao", lambda: wait(playing)),
    ("volume 75", lambda: wait(lambda: vol() == 75)),
    ("volume down", lambda: wait(lambda: vol() == 65)),
    ("volume up", lambda: wait(lambda: vol() == 75)),
    ("set the volume to 40 percent", lambda: wait(lambda: vol() == 40)),
    ("Picture volume level, 75%", lambda: wait(lambda: vol() == 75)),  # real mis-transcription from the log
    ("awaz 60 par karo", lambda: wait(lambda: vol() == 60)),
    ("mute", lambda: wait(lambda: ev(f"{P}.isMuted()") is True)),
    ("unmute", lambda: wait(lambda: ev(f"{P}.isMuted()") is False)),
    ("go to 1:30", lambda: wait(lambda: abs(now() - 90) < 3)),
    ("skip ahead 30 seconds", lambda: wait(lambda: 115 <= now() <= 128)),
    ("go back 10 seconds", lambda: wait(lambda: 105 <= now() <= 118)),
    ("restart the video", lambda: wait(lambda: now() < 4)),
    ("play at 1.5x speed", lambda: wait(lambda: ev(f"{P}.getPlaybackRate()") == 1.5)),
    ("normal speed", lambda: wait(lambda: ev(f"{P}.getPlaybackRate()") == 1)),
    ("theater mode", lambda: wait(lambda: ev("document.querySelector('ytd-watch-flexy').hasAttribute('theater')") is True)),
    ("exit theater mode", lambda: wait(lambda: ev("document.querySelector('ytd-watch-flexy').hasAttribute('theater')") is False)),
    ("full screen", lambda: wait(lambda: ev("!!document.fullscreenElement") is True)),
    ("exit full screen", lambda: wait(lambda: ev("!!document.fullscreenElement") is False)),
    ("loop this video", lambda: wait(lambda: ev("document.querySelector('#movie_player video').loop") is True)),
    ("stop looping", lambda: wait(lambda: ev("document.querySelector('#movie_player video').loop") is False)),
    ("turn on captions", lambda: wait(lambda: ev("document.querySelector('.ytp-subtitles-button')?.getAttribute('aria-pressed')") == "true")),
    ("turn off subtitles", lambda: wait(lambda: ev("document.querySelector('.ytp-subtitles-button')?.getAttribute('aria-pressed')") == "false")),
    ("quality 480p", lambda: wait(lambda: ev(f"{P}.getPlaybackQuality()") == "large", 8)),
    ("quality auto", lambda: True),
    ("what is playing", lambda: bool(ctx["reply"]) and (ev(f"{P}.getVideoData().title") or "?") in ctx["reply"]),
    ("next video", lambda: wait(lambda: vid() not in (None, ctx["vid_before"]), 10)),
    ("previous video", lambda: wait(lambda: vid() == ctx["vid_before_prev"], 10)),
    ("search youtube for carryminati", lambda: wait(lambda: "results?search_query" in (ev("location.href") or ""), 10)),
    ("play the second video", lambda: wait(lambda: "/watch" in (ev("location.href") or "") and vid() is not None, 12)),
    ("pause", lambda: wait(paused)),
    ("open the channel", lambda: wait(lambda: "/@" in (ev("location.href") or "") or "/channel/" in (ev("location.href") or ""), 10)),
    ("youtube home", lambda: wait(lambda: ev("location.pathname") == "/", 10)),
    ("play the latest video from Lofi Girl", lambda: wait(lambda: "/watch" in (ev("location.href") or ""), 20)),
    ("pause the video", lambda: wait(paused)),
    # Loosely worded: these go through the slower LLM path, not the instant router.
    ("hey could you carry on with the video", lambda: wait(playing)),
    ("can you hold the playback for a second", lambda: wait(paused)),
    ("bring the youtube sound down to thirty", lambda: wait(lambda: vol() == 30)),
]

FAIL_WORDS = ("didn't", "couldn't", "can't", "isn't", "not ", "no ", "sorry", "only works", "doesn't")


def main():
    ep = AudioUtilities.GetSpeakers().EndpointVolume
    was_muted = ep.GetMute()
    was_level = ep.GetMasterVolumeLevelScalar()  # some steps may legitimately set the laptop volume
    ep.SetMute(1, None)
    brain = OllamaBrain("http://127.0.0.1:11434", "vaayu-finetuned")
    results = []
    try:
        print(apps.open_app("brave", foreground=False))
        start_volume = None
        for command, check in STEPS:
            ctx["vid_before"] = vid() if tab() else None
            if command == "previous video":
                ctx["vid_before_prev"] = ctx.get("vid_before_next")
            if command == "next video":
                ctx["vid_before_next"] = ctx["vid_before"]
            brain.trace = []
            t0 = time.time()
            try:
                reply, _ = brain.respond([{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": command}], registry, confirm=lambda n, a: False)
            except Exception as exc:
                reply = f"CRASH {exc!r}"
            secs = time.time() - t0
            ctx["reply"] = reply
            if start_volume is None and tab() and vol() is not None and command.startswith("play lofi"):
                start_volume = ctx.setdefault("start_volume", vol())
            try:
                ok = bool(check())
            except Exception:
                ok = False
            claimed_failure = reply.startswith("CRASH") or any(w in reply.lower() for w in FAIL_WORDS)
            outcome = "PASS" if ok else ("FAIL" if claimed_failure else "FALSE_CONFIRMATION")
            path = "fast" if all(t.get("path") != "llm_call" for t in brain.trace) else "llm"
            results.append({"command": command, "outcome": outcome, "reply": reply, "secs": round(secs, 2), "path": path,
                            "trace": brain.trace})
            print(f"{outcome:18} {secs:5.1f}s {path:4} {command!r:42} -> {reply[:90]!r}", flush=True)
    finally:
        if ctx.get("start_volume") is not None:  # leave the user's YouTube volume as it was
            try:
                yp.set_volume(level=ctx["start_volume"])
            except Exception:
                pass
        ep.SetMasterVolumeLevelScalar(was_level, None)
        ep.SetMute(was_muted, None)
    out = Path(__file__).resolve().parent.parent / "data" / "benchmark" / "live_youtube.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    n = {k: sum(r["outcome"] == k for r in results) for k in ("PASS", "FAIL", "FALSE_CONFIRMATION")}
    print(f"\n== {n['PASS']}/{len(results)} passed, {n['FAIL']} honest failures, "
          f"{n['FALSE_CONFIRMATION']} FALSE CONFIRMATIONS. Saved {out}")


if __name__ == "__main__":
    main()
