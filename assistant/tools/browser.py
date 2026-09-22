"""
Browser tab control via the Chrome DevTools Protocol's HTTP endpoint. The OS
has no concept of "tabs" (that's internal to the browser), so window-focus
APIs (windows.py) can't reach into a browser and pick a specific tab - this
talks to Chrome directly instead.

Requires Chrome to be running with --remote-debugging-port=9222 (not the
default). See README for the one-time setup this needs.
"""

import json
import urllib.error
import urllib.request

from assistant.tools.registry import registry

# "127.0.0.1", not "localhost" - found live while testing the open_website
# CDP fix: "localhost" resolves to both ::1 and 127.0.0.1, and Windows tries
# the IPv6 address first, adding a real ~2s connect-refused delay on top of
# the IPv4 one before falling back - doubling the cost of every call this
# makes when no CDP-debugged browser is actually running (measured 4.06s vs
# 2.04s for the identical request, IPv4-only vs "localhost").
_CDP_BASE = "http://127.0.0.1:9222"
_NOT_AVAILABLE_MSG = (
    "Can't reach Chrome's remote debugging port (9222). Chrome must be started "
    "with --remote-debugging-port=9222 for tab control to work - see README."
)


def _cdp_request(path: str) -> bytes:
    try:
        with urllib.request.urlopen(f"{_CDP_BASE}{path}", timeout=3) as resp:
            return resp.read()
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise ConnectionError(_NOT_AVAILABLE_MSG) from exc


def _list_tabs() -> list[dict]:
    tabs = json.loads(_cdp_request("/json"))
    return [t for t in tabs if t.get("type") == "page"]


def _cdp_request_put(path: str) -> bytes:
    req = urllib.request.Request(f"{_CDP_BASE}{path}", method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read()
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise ConnectionError(_NOT_AVAILABLE_MSG) from exc


def _new_tab(url: str) -> dict:
    """Opens a genuinely new tab in the browser instance already reachable
    over CDP - via the DevTools HTTP endpoint itself, not by launching a
    second browser process. Found live (real-usage bug hunt): the previous
    approach for "open in a new tab" (browser_state.open_url_in_last_browser,
    a plain subprocess.Popen([exe, url]) against the SAME profile a
    CDP-debugged instance already has open) could crash the whole browser
    outright, and separately could produce a "tab" that silently never
    existed at all - because "already running with debugging enabled" is
    Vaayu's normal operating condition, not an edge case, so a second
    process against that same profile hits Chromium's single-instance
    hand-off in a broken way essentially every time. Raises ConnectionError
    (via _cdp_request_put) if no CDP-debugged browser is reachable at all -
    callers should fall back to launching one in that case, same as
    everywhere else in this project that checks port 9222."""
    return json.loads(_cdp_request_put(f"/json/new?{url}"))


def _activate_tab(tab_id: str) -> None:
    _cdp_request(f"/json/activate/{tab_id}")


@registry.register(
    name="list_browser_tabs",
    description=(
        "List currently open browser tabs (1-based position, title, URL). Call this first if the "
        "user refers to a tab by position ('the third tab', 'the first tab') or you're not sure "
        "which tab matches what they mean, then pass that position as the index to "
        "switch_to_browser_tab/close_browser_tab."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def list_browser_tabs() -> list[dict]:
    return [
        {"position": i + 1, "title": t.get("title"), "url": t.get("url")}
        for i, t in enumerate(_list_tabs())
    ]


def _resolve_tab(query: str | None, position: int | None) -> dict:
    """Shared lookup for switch/close: match by 1-based position (from
    list_browser_tabs) if given, else by title/URL substring."""
    tabs = _list_tabs()
    if position is not None:
        if not (1 <= position <= len(tabs)):
            raise ValueError(f"There's no tab at position {position} - only {len(tabs)} tab(s) open.")
        return tabs[position - 1]

    if not query:
        raise ValueError("Need either a query (tab title/URL text) or a position to find a tab.")
    query_lower = query.lower()
    matches = [
        t
        for t in tabs
        if query_lower in (t.get("title") or "").lower() or query_lower in (t.get("url") or "").lower()
    ]
    if not matches:
        raise ValueError(f"No open browser tab found matching '{query}'.")
    return matches[0]


@registry.register(
    name="switch_to_browser_tab",
    description=(
        "Switch focus to an already-open browser tab, matching by title/URL text (query) or by "
        "1-based position from list_browser_tabs (position) - e.g. 'the third tab' -> position=3. "
        "Give exactly one of query or position."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to match against tab titles/URLs, e.g. 'youtube' or 'gmail'.",
            },
            "position": {
                "type": "integer",
                "description": "1-based tab position (from list_browser_tabs), e.g. 3 for 'the third tab'.",
            },
        },
        "required": [],
    },
)
def switch_to_browser_tab(query: str | None = None, position: int | None = None) -> str:
    target = _resolve_tab(query, position)
    _cdp_request(f"/json/activate/{target['id']}")
    return f"Switched to tab: {target.get('title')}"


@registry.register(
    name="close_browser_tab",
    description=(
        "Close one specific browser tab (not the whole browser) - matching by title/URL text "
        "(query) or by 1-based position from list_browser_tabs (position), e.g. 'close the third "
        "tab' -> position=3. Give exactly one of query or position. To close the entire browser "
        "application instead, use close_app."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to match against tab titles/URLs, e.g. 'youtube' or 'gmail'.",
            },
            "position": {
                "type": "integer",
                "description": "1-based tab position (from list_browser_tabs), e.g. 3 for 'the third tab'.",
            },
        },
        "required": [],
    },
)
def close_browser_tab(query: str | None = None, position: int | None = None) -> str:
    target = _resolve_tab(query, position)
    _cdp_request(f"/json/close/{target['id']}")
    return f"Closed tab: {target.get('title')}"
