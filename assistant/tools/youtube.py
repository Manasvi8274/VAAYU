"""
Purpose-built YouTube tools (play a channel's latest video, play a
song/video matching a query) - built after testing showed the small local
LLM isn't reliably able to plan the underlying multi-step
search -> read_page -> click_on_page sequence on its own (it kept
hallucinating URLs and reaching for the wrong tools like control_media).
These do the actual multi-step navigation deterministically in Python,
extracting real URLs from YouTube's DOM structure (ytd-channel-renderer /
ytd-video-renderer elements) instead of fuzzy text-matching, so the LLM only
has to make one simple, reliable tool choice instead of planning several.

Requires the browser running with --remote-debugging-port=9222, same as
browser.py/browser_automation.py.
"""

import time
from urllib.parse import quote_plus

from assistant.tools import cache
from assistant.tools.browser import _activate_tab, _new_tab
from assistant.tools.browser_automation import _send_cdp_command, _target_tab
from assistant.tools.registry import registry

_LOAD_WAIT_SECONDS = 1.5
# Channel name -> channel URL is a stable mapping (channels essentially never
# change their handle) - caching it skips a full search-page navigation
# (~1.5s+ page load) on every repeat request for the same channel, directly
# addressing "cache data frequently, so commands can be done faster."
_CHANNEL_CACHE_NAMESPACE = "youtube_channel"
_CHANNEL_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def _resolve_target_tab(new_tab: bool) -> dict:
    """Both tools below resolve their target tab once at the start and
    thread it through every subsequent _navigate/_eval_js call, so a
    new_tab=True request stays on the SAME new tab for its whole multi-step
    navigation (search -> channel's /videos page -> the video itself),
    instead of drifting back to whatever _target_tab() considers "current"
    after the first CDP-level tab creation. Uses the real CDP new-tab
    endpoint (browser._new_tab), not a second browser process - see its
    docstring for why that used to crash the browser / lose tabs."""
    if new_tab:
        tab = _new_tab("about:blank")
        _activate_tab(tab["id"])
        return tab
    return _target_tab()


def _navigate(url: str, tab: dict) -> None:
    _send_cdp_command(tab["webSocketDebuggerUrl"], "Page.navigate", {"url": url})
    time.sleep(_LOAD_WAIT_SECONDS)


def _eval_js(js: str, tab: dict):
    result = _send_cdp_command(
        tab["webSocketDebuggerUrl"], "Runtime.evaluate", {"expression": js, "returnByValue": True}
    )
    return result.get("result", {}).get("result", {}).get("value")


def _eval_js_retry_if_empty(js: str, tab: dict, retries: int = 2, retry_wait: float = 2.0):
    # Found live: a freshly cold-started browser's FIRST navigation to a
    # JS-heavy SPA page (YouTube search results) can still be rendering when
    # _LOAD_WAIT_SECONDS has elapsed - a query that comes back empty right
    # then doesn't necessarily mean "not found", it can just mean "not
    # rendered yet". Reproduced directly: the exact same channel search that
    # failed immediately after a cold browser launch succeeded seconds later
    # with nothing else changed. Retries after a longer wait before treating
    # an empty result as real absence, rather than failing on what might
    # just be a slow first render. Bumped from 1 to 2 retries (real-usage
    # bug hunt, later fix pass): a single retry (one extra 2s wait) still
    # wasn't enough on a genuinely fresh, never-used browser profile's very
    # first-ever page load specifically - confirmed the content was there
    # and fully rendered (46 real watch links) moments after the single
    # retry had already given up.
    value = _eval_js(js, tab)
    for _ in range(retries):
        if value:
            return value
        time.sleep(retry_wait)
        value = _eval_js(js, tab)
    return value


def _find_channel_url(query: str, tab: dict) -> str | None:
    _navigate(f"https://www.youtube.com/results?search_query={quote_plus(query)}", tab)
    return _eval_js_retry_if_empty(
        "(() => { const el = document.querySelector("
        "'ytd-channel-renderer a#main-link, ytd-channel-renderer a[href]'); "
        "return el ? el.href : null; })()",
        tab,
    )


def _find_first_video_url(tab: dict, exclude_shorts: bool = True) -> str | None:
    # Generic href-pattern matching, not YouTube's internal custom element
    # tags (ytd-video-renderer etc.) - tested and found those tags differ
    # between page layouts (search results vs. channel page vs. videos tab)
    # in ways that made tag-based selectors unreliable; a plain watch-URL
    # href match works consistently across all of them.
    js = """
    (() => {
        const els = Array.from(document.querySelectorAll("a[href*='/watch?v=']"))
            .filter(el => el.offsetParent !== null);
        for (const el of els) {
            if (%s && el.href.includes('/shorts/')) continue;
            if (el.href) return el.href;
        }
        return null;
    })()
    """ % ("true" if exclude_shorts else "false")
    return _eval_js_retry_if_empty(js, tab)


@registry.register(
    name="play_latest_video_from_channel",
    description=(
        "Finds a YouTube channel by name and plays its most recent video. Use this whenever the "
        "user asks to open/watch/play something from a specific YouTube channel - handles the "
        "search and navigation itself in one call, do not try to do this step by step with "
        "open_website/read_page/click_on_page. Reuses the current tab by default - pass "
        "new_tab=true if the user says 'in a new tab'/'in another tab'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "channel_name": {
                "type": "string",
                "description": "The channel to find, e.g. 'OG Crew' or 'MrBeast'.",
            },
            "new_tab": {
                "type": "boolean",
                "description": (
                    "Open a new browser tab for this instead of reusing the current one - set true "
                    "only if the user explicitly says 'in a new/another tab'. Default false."
                ),
            },
        },
        "required": ["channel_name"],
    },
)
def play_latest_video_from_channel(channel_name: str, new_tab: bool = False) -> str:
    tab = _resolve_target_tab(new_tab)
    channel_url = cache.get(_CHANNEL_CACHE_NAMESPACE, channel_name, _CHANNEL_CACHE_MAX_AGE_SECONDS)
    from_cache = channel_url is not None

    if channel_url is None:
        channel_url = _find_channel_url(channel_name, tab)
        if not channel_url:
            raise ValueError(f"Couldn't find a YouTube channel matching '{channel_name}'.")

    # The channel's plain home page can show a *featured/pinned* video
    # (tested and confirmed: not necessarily the newest upload) - the
    # dedicated /videos tab is what's actually sorted with the newest first.
    videos_tab_url = channel_url.rstrip("/") + "/videos"
    _navigate(videos_tab_url, tab)
    video_url = _find_first_video_url(tab)

    if not video_url and from_cache:
        # Cached URL might be stale (channel renamed/removed) - fall back to
        # a fresh search once before giving up, rather than failing outright
        # on data that was only ever a speed optimization.
        channel_url = _find_channel_url(channel_name, tab)
        if not channel_url:
            raise ValueError(f"Couldn't find a YouTube channel matching '{channel_name}'.")
        _navigate(channel_url.rstrip("/") + "/videos", tab)
        video_url = _find_first_video_url(tab)
        from_cache = False

    if not video_url:
        raise ValueError(f"Found the channel '{channel_name}' but no videos on it.")

    if not from_cache:
        cache.set(_CHANNEL_CACHE_NAMESPACE, channel_name, channel_url)

    _navigate(video_url, tab)
    return f"Playing the latest video from {channel_name}" + (" in a new tab." if new_tab else ".")


@registry.register(
    name="play_video_on_youtube",
    description=(
        "Searches YouTube for the given query and plays the first matching video - use this for "
        "ANY 'play <song/music/artist/genre>' request, whether or not the user says the word "
        "'youtube' (e.g. 'play fairytale music', 'play some lofi', 'play Ed Sheeran'), as long as "
        "it isn't about a specific channel's latest video (use play_latest_video_from_channel for "
        "that instead). This is the correct tool for playing music/songs in general - reuses the "
        "current browser tab by default; pass new_tab=true if the user says 'in a new tab'/'in "
        "another tab' - do not fall back to open_website for that phrasing, this tool handles it "
        "directly. Construct a good search query yourself: for a genre ('bollywood songs', "
        "'hollywood songs') search that genre plus 'songs' or 'hits'; for an artist, search their "
        "name plus 'songs'; for a specific song, search the song name (plus artist if known)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, e.g. 'latest bollywood songs 2026' or 'Ed Sheeran Perfect'.",
            },
            "new_tab": {
                "type": "boolean",
                "description": (
                    "Open a new browser tab for this instead of reusing the current one - set true "
                    "only if the user explicitly says 'in a new/another tab'. Default false."
                ),
            },
        },
        "required": ["query"],
    },
)
def play_video_on_youtube(query: str, new_tab: bool = False) -> str:
    tab = _resolve_target_tab(new_tab)
    _navigate(f"https://www.youtube.com/results?search_query={quote_plus(query)}", tab)
    video_url = _find_first_video_url(tab)
    if not video_url:
        raise ValueError(f"No video found on YouTube for '{query}'.")

    _navigate(video_url, tab)
    return f"Playing on YouTube: {query}" + (" (new tab)." if new_tab else "")
