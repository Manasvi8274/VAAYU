from urllib.parse import quote_plus
import webbrowser

from assistant.tools import browser_state
from assistant.tools.browser import _activate_tab, _new_tab
from assistant.tools.browser_automation import _send_cdp_command, _target_tab
from assistant.tools.registry import registry

# Longest first: " web site" ends with " site" too, so checking " site"
# first would match it, produce a candidate with a leftover space ("google
# web"), and stop there (break) before ever trying the suffix that actually
# fits - found via the large parametrized browser suite (test_browser_
# suite_large.py), which covers "<name> web site" phrasing directly.
_SITE_SUFFIXES = (" web site", " website", " site")


@registry.register(
    name="open_website",
    description=(
        "Open a URL or run a web search. By default this ALWAYS opens a new browser tab - do not "
        "use it when the user says 'in this tab' / 'on the same tab' / 'in the current tab', pass "
        "same_tab=true instead. Accepts a URL (e.g. 'youtube.com') or a plain search query, opened "
        "as a web search. Do NOT use this to launch a browser application by name (e.g. 'open "
        "chrome', 'open brave') - use open_app for that. Do NOT use this for 'play <song/music/"
        "video>' requests - use play_video_on_youtube or play_latest_video_from_channel instead, "
        "which search and start playback in one call and correctly reuse the current tab. Brought "
        "to the foreground by default - only pass foreground=false if the user explicitly said to "
        "open it in the background."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url_or_query": {
                "type": "string",
                "description": "A URL, or a search query if it isn't a URL.",
            },
            "foreground": {
                "type": "boolean",
                "description": (
                    "Whether to bring the browser to the foreground (default true). Set to false "
                    "only if the user explicitly asked for it to open in the background."
                ),
            },
            "same_tab": {
                "type": "boolean",
                "description": (
                    "If true, navigate the CURRENTLY ACTIVE tab to this URL/search instead of "
                    "opening a new one - use whenever the user says 'this tab', 'same tab', or "
                    "'current tab'. Default false (opens a new tab)."
                ),
            },
        },
        "required": ["url_or_query"],
    },
)
def open_website(url_or_query: str, foreground: bool = True, same_tab: bool = False) -> str:
    text = url_or_query.strip()

    # "open the moviesmod website" -> treat the bare name as a domain
    # (moviesmod.com) rather than running it as a Google search, since
    # "open <name> website" is how people ask to go straight to a site.
    lowered = text.lower()
    for suffix in _SITE_SUFFIXES:
        if lowered.endswith(suffix):
            candidate = text[: -len(suffix)].strip()
            if candidate and " " not in candidate:
                text = candidate if "." in candidate else candidate + ".com"
            break

    looks_like_url = "." in text and " " not in text

    if looks_like_url:
        if not text.lower().startswith(("http://", "https://")):
            text = "https://" + text
        target_url = text
        label = f"Opened {text} in the browser"
    else:
        target_url = "https://www.google.com/search?q=" + quote_plus(text)
        label = f"Searched the web for: {text}"

    if same_tab:
        tab = _target_tab()
        _send_cdp_command(tab["webSocketDebuggerUrl"], "Page.navigate", {"url": target_url})
        return label + " (same tab)."

    # Prefer creating the new tab in the browser instance Vaayu already has
    # CDP access to (the normal case whenever a browser is already running -
    # see _new_tab's docstring for why the old Popen-based approach below
    # could crash the whole browser or silently lose the tab). Only fall
    # back to launching/targeting a browser process directly when no
    # CDP-debugged instance is reachable at all.
    try:
        new_tab = _new_tab(target_url)
        if foreground and new_tab.get("id"):
            _activate_tab(new_tab["id"])
        return label + ("." if foreground else " in the background.")
    except ConnectionError:
        pass  # no CDP-debugged browser reachable - fall back below

    # Prefer the browser the user most recently opened via open_app -
    # webbrowser.open() always goes to the OS default, which silently
    # switched browsers on the user in testing (opened Brave, then a
    # follow-up "open youtube" opened it in Chrome instead).
    if not browser_state.open_url_in_last_browser(target_url, foreground=foreground):
        webbrowser.open(target_url)
    return label + ("." if foreground else " in the background.")
