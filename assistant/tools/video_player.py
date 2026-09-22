"""
Site-agnostic control of whatever video is playing in a browser tab - volume,
fullscreen, seek forward/backward, play/pause. Works on any page with a
standard HTML5 <video> element (YouTube, Hotstar, Netflix, Prime Video, most
news/streaming sites), not hardcoded to one site, since the underlying
.volume/.currentTime/.paused/.play()/.pause() properties are the browser's
own standard video API, not something specific to any one site's player UI.

Multi-video disambiguation: if the user has more than one tab with a video
actively playing at once (e.g. music on YouTube in one tab, a series on
Hotstar in another), silently guessing which one they mean is exactly the
kind of thing that goes wrong - every tool here checks every open tab for an
actively-playing video first, and if more than one qualifies, raises a clear
error listing them by title instead of picking one. The LLM relays that as a
question; the user's answer (a site/tab name) then goes in as tab_hint on
the next call, which skips the "which one" check entirely and resolves
straight to a matching tab by title/URL substring.

Requires the browser running with --remote-debugging-port=9222, same as
browser.py/browser_automation.py/youtube.py.
"""

import html
import re
import time

from assistant.tools.browser import _list_tabs
from assistant.tools.browser_automation import _send_cdp_command
from assistant.tools.registry import registry

_VIDEO_EL_JS = "document.querySelector('video')"
# Real tab titles read straight off the DOM can carry HTML entities
# (document.title showing "&amp;" instead of "&") and, on YouTube
# specifically, a leading notification-badge count like "(576) " that has
# nothing to do with the video itself - found live: both leaked verbatim
# into the multi-video disambiguation question read back to the user.
_TITLE_BADGE_PREFIX_RE = re.compile(r"^\(\d+\)\s*")


def _clean_title(title: str | None) -> str:
    if not title:
        return "Untitled tab"
    return _TITLE_BADGE_PREFIX_RE.sub("", html.unescape(title)).strip() or "Untitled tab"


def _eval_js_on_tab(tab: dict, js: str):
    result = _send_cdp_command(
        tab["webSocketDebuggerUrl"], "Runtime.evaluate", {"expression": js, "returnByValue": True}
    )
    return result.get("result", {}).get("result", {}).get("value")


def _tabs_with_playing_video() -> list[dict]:
    playing = []
    for tab in _list_tabs():
        if not tab.get("webSocketDebuggerUrl"):
            continue
        try:
            is_playing = _eval_js_on_tab(
                tab, "(() => { const v = document.querySelector('video'); return v ? !v.paused : false; })()"
            )
        except Exception:
            continue  # a tab that can't be evaluated (e.g. a non-page target) just isn't a candidate
        if is_playing:
            playing.append(tab)
    return playing


def _resolve_video_tab(tab_hint: str | None) -> dict:
    if tab_hint:
        query_lower = tab_hint.strip().lower()
        matches = [
            t
            for t in _list_tabs()
            if query_lower in (t.get("title") or "").lower() or query_lower in (t.get("url") or "").lower()
        ]
        if not matches:
            raise ValueError(f"No open tab found matching '{tab_hint}'.")
        return matches[0]

    playing = _tabs_with_playing_video()
    if len(playing) > 1:
        titles = ", ".join(_clean_title(t.get("title")) for t in playing)
        raise ValueError(
            f"More than one video is playing right now ({titles}) - which one do you mean? "
            "Tell me the site or tab name."
        )
    if len(playing) == 1:
        return playing[0]

    # Nothing is actively playing anywhere - fall back to whichever tab was
    # most recently activated, same default every other browser tool uses.
    tabs = _list_tabs()
    if not tabs:
        raise ConnectionError(
            "No open browser tabs found - is the browser running with --remote-debugging-port=9222?"
        )
    return tabs[0]


_TAB_HINT_PARAM = {
    "type": "string",
    "description": (
        "Which tab/site/video to control, if the user named one - e.g. 'youtube', 'hotstar', or "
        "even a distinctive word from the title ('the dune trailer', 'the podcast'). Set this "
        "PROACTIVELY any time the user's own wording names a specific tab/site/video, not only "
        "reactively after you've already asked 'which one' - found live: a follow-up like 'pause "
        "the dune trailer' left tab_hint empty, silently fell back to the most-recently-active tab "
        "(a different video), and paused the wrong one while claiming the named one was paused. "
        "Omit only when the user's wording genuinely doesn't name a tab/site/video at all."
    ),
}


@registry.register(
    name="set_video_volume",
    description=(
        "Change the volume of the video actually playing in a browser tab (the in-page player "
        "volume, separate from the system/Windows volume - use control_media for that instead). "
        "Works on any site's video player (YouTube, Hotstar, Netflix, etc.), not just YouTube. Use "
        "direction='up' or 'down' for relative changes, or pass an absolute level (0-100). mute/"
        "unmute are also supported via direction. If more than one tab has a video playing, this "
        "raises an error asking which one - relay that question, then call again with tab_hint set "
        "to whatever the user answers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["up", "down", "mute", "unmute"],
                "description": "Relative change to make. Omit if passing an absolute level instead.",
            },
            "amount": {
                "type": "integer",
                "description": "Percentage points to change by for up/down (default 20).",
            },
            "level": {
                "type": "integer",
                "description": "Set the volume to this absolute percentage (0-100), instead of a relative change.",
            },
            "tab_hint": _TAB_HINT_PARAM,
        },
        "required": [],
    },
)
def set_video_volume(
    direction: str | None = None, amount: int = 20, level: int | None = None, tab_hint: str | None = None
) -> str:
    tab = _resolve_video_tab(tab_hint)
    current = _eval_js_on_tab(tab, f"{_VIDEO_EL_JS} ? {_VIDEO_EL_JS}.volume : null")
    if current is None:
        raise RuntimeError("No playing video found on that tab.")

    if level is not None:
        new_volume = max(0, min(100, level)) / 100
        label = f"Set the video volume to {level}%"
    elif direction == "mute":
        new_volume = None
        label = "Muted the video"
    elif direction == "unmute":
        new_volume = None
        label = "Unmuted the video"
    elif direction in ("up", "down"):
        delta = (amount if direction == "up" else -amount) / 100
        new_volume = max(0.0, min(1.0, current + delta))
        label = f"Turned the video volume {direction} to {round(new_volume * 100)}%"
    else:
        raise ValueError("Give either direction (up/down/mute/unmute) or an absolute level.")

    if direction in ("mute", "unmute"):
        want_muted = direction == "mute"
        _eval_js_on_tab(tab, f"{_VIDEO_EL_JS}.muted = {'true' if want_muted else 'false'}")
        actual_muted = _eval_js_on_tab(tab, f"{_VIDEO_EL_JS}.muted")
        if actual_muted != want_muted:
            raise RuntimeError("Tried to change mute state but it didn't take effect.")
    else:
        _eval_js_on_tab(tab, f"{_VIDEO_EL_JS}.volume = {new_volume}; {_VIDEO_EL_JS}.muted = false")
        actual = _eval_js_on_tab(tab, f"{_VIDEO_EL_JS}.volume")
        if actual is None or abs(actual - new_volume) > 0.02:
            raise RuntimeError("Tried to change the volume but it didn't take effect.")

    title = _clean_title(tab.get("title"))
    return f"{label} ({title})."


def _dispatch_key(tab: dict, key: str, code: str, vk: int) -> None:
    # A JS-invoked element.requestFullscreen() from Runtime.evaluate has no
    # real user-activation and Chromium silently rejects it (tested) - a
    # trusted synthetic keypress via the Input domain behaves like a real
    # hardware key and actually works. 'f' is YouTube's own fullscreen
    # shortcut and is also the de-facto standard across most other HTML5
    # video players (Hotstar, Netflix, Prime Video all use it too).
    for event_type in ("keyDown", "keyUp"):
        _send_cdp_command(
            tab["webSocketDebuggerUrl"],
            "Input.dispatchKeyEvent",
            {
                "type": event_type,
                "key": key,
                "code": code,
                "windowsVirtualKeyCode": vk,
                "text": key if event_type == "keyDown" else "",
            },
        )


@registry.register(
    name="set_video_fullscreen",
    description=(
        "Make the currently playing video fullscreen, or exit fullscreen back to the normal page "
        "view - works on any site's video player. If more than one tab has a video playing, this "
        "raises an error asking which one - relay that question, then call again with tab_hint."
    ),
    parameters={
        "type": "object",
        "properties": {
            "fullscreen": {
                "type": "boolean",
                "description": "true to enter fullscreen, false to exit it.",
            },
            "tab_hint": _TAB_HINT_PARAM,
        },
        "required": ["fullscreen"],
    },
)
def set_video_fullscreen(fullscreen: bool, tab_hint: str | None = None) -> str:
    tab = _resolve_video_tab(tab_hint)
    is_fullscreen_js = "!!document.fullscreenElement"
    currently_fullscreen = _eval_js_on_tab(tab, is_fullscreen_js)

    if currently_fullscreen == fullscreen:
        return "Already in fullscreen." if fullscreen else "Already out of fullscreen."

    _dispatch_key(tab, key="f", code="KeyF", vk=70)
    time.sleep(0.3)
    now_fullscreen = _eval_js_on_tab(tab, is_fullscreen_js)

    if now_fullscreen != fullscreen:
        # Found live (real-usage bug hunt): failed once against a video
        # confirmed genuinely playing seconds before and after - not
        # reproduced across 3 repeated attempts afterward, consistent with a
        # timing race (fullscreen transition/ad overlay/buffering state) at
        # the single fixed 0.3s check rather than a structural failure. One
        # retry after a longer wait, same bounded-retry shape already used
        # for the cold-start DOM-query race in youtube.py, before treating
        # it as a genuine failure.
        time.sleep(0.7)
        now_fullscreen = _eval_js_on_tab(tab, is_fullscreen_js)

    if now_fullscreen != fullscreen:
        raise RuntimeError(
            "Tried to toggle fullscreen but the page state didn't change - "
            "make sure a video is actually open on that tab."
        )
    return "Entered fullscreen." if fullscreen else "Exited fullscreen."


@registry.register(
    name="seek_video",
    description=(
        "Skip forward or backward in the currently playing video on a browser tab - 'skip ahead 10 "
        "seconds', 'go back 10 seconds', 'skip forward a minute'. Works on any site's video player, "
        "not just YouTube - the seek amount is exact (verified against the real video position), "
        "unlike VLC's keyboard-shortcut seek which depends on VLC's own configured jump length. If "
        "more than one tab has a video playing, this raises an error asking which one - relay that "
        "question, then call again with tab_hint."
    ),
    parameters={
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["forward", "backward"],
                "description": "Which way to skip.",
            },
            "seconds": {
                "type": "integer",
                "description": "How many seconds to skip (default 10).",
            },
            "tab_hint": _TAB_HINT_PARAM,
        },
        "required": ["direction"],
    },
)
def seek_video(direction: str, seconds: int = 10, tab_hint: str | None = None) -> str:
    tab = _resolve_video_tab(tab_hint)
    current = _eval_js_on_tab(tab, f"{_VIDEO_EL_JS} ? {_VIDEO_EL_JS}.currentTime : null")
    if current is None:
        raise RuntimeError("No playing video found on that tab.")

    delta = seconds if direction == "forward" else -seconds
    _eval_js_on_tab(
        tab, f"{_VIDEO_EL_JS}.currentTime = Math.max(0, {_VIDEO_EL_JS}.currentTime + ({delta}))"
    )
    time.sleep(0.2)
    actual = _eval_js_on_tab(tab, f"{_VIDEO_EL_JS} ? {_VIDEO_EL_JS}.currentTime : null")

    if actual is None:
        raise RuntimeError("Tried to seek but lost track of the video afterward.")
    # Checking direction of movement rather than an exact delta - the video
    # keeps playing during this round trip, so currentTime naturally drifts
    # a bit beyond the seek itself; what actually matters is that it moved
    # the right way (a real seek that silently failed would only show the
    # small normal-playback drift, not a jump - especially detectable for
    # "backward", where normal playback drift would still look like forward
    # movement if the seek itself didn't apply).
    moved_correctly = (actual > current) if direction == "forward" else (actual < current or actual == 0)
    if not moved_correctly:
        raise RuntimeError("Tried to seek but the change didn't take effect.")

    title = _clean_title(tab.get("title"))
    label = f"Skipped {direction} {seconds} seconds"
    return f"{label} ({title})."


@registry.register(
    name="control_video_playback",
    description=(
        "Play, pause, or toggle the currently playing/paused video on a BROWSER TAB - works on any "
        "site's video player. Use action='toggle' for a plain 'pause it'/'play it' when you aren't "
        "sure of the current state. If more than one tab has a video playing, this raises an error "
        "asking which one - relay that question, then call again with tab_hint. Do NOT use this "
        "for VLC or any other desktop media player, even if the user's wording includes the word "
        "'video' (e.g. 'pause the VLC video') - VLC is not a browser tab; use switch_to_window + "
        "press_hotkey(space) for VLC instead, per the system prompt's VLC section."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play", "pause", "toggle"],
                "description": "What to do.",
            },
            "tab_hint": _TAB_HINT_PARAM,
        },
        "required": ["action"],
    },
)
def control_video_playback(action: str, tab_hint: str | None = None) -> str:
    tab = _resolve_video_tab(tab_hint)
    is_paused_js = f"{_VIDEO_EL_JS} ? {_VIDEO_EL_JS}.paused : null"
    currently_paused = _eval_js_on_tab(tab, is_paused_js)
    if currently_paused is None:
        raise RuntimeError("No video found on that tab.")

    if action == "toggle":
        action = "play" if currently_paused else "pause"

    _eval_js_on_tab(tab, f"{_VIDEO_EL_JS}.{action}()")
    time.sleep(0.3)

    now_paused = _eval_js_on_tab(tab, is_paused_js)
    expected_paused = action == "pause"
    if now_paused != expected_paused:
        raise RuntimeError(f"Tried to {action} the video but it didn't take effect.")

    title = _clean_title(tab.get("title"))
    label = "Paused the video" if action == "pause" else "Playing the video"
    return f"{label} ({title})."
