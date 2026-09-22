"""
Real page interaction via the Chrome DevTools Protocol's WebSocket interface
(Page.navigate, Runtime.evaluate) - lets Vaayu do more than just open a URL:
read what's actually on a page and click things by their visible text. This
is what makes a multi-step request like "open youtube, search OG crew,
click the channel, play the latest video" actually followable, instead of
just running a single search and stopping.

Requires the browser running with --remote-debugging-port=9222, same
requirement as browser.py's tab tools (which this reuses for tab listing).
"""

import json

import websocket

from assistant.tools.browser import _cdp_request, _list_tabs
from assistant.tools.registry import registry

_CDP_TIMEOUT = 10


def _target_tab() -> dict:
    """Picks which tab to act on. Chrome/Brave's /json endpoint lists the
    most recently activated tab first in practice (verified empirically) -
    good enough since this is normally called right after open_website
    navigated to a fresh tab."""
    tabs = _list_tabs()
    if not tabs:
        raise ConnectionError(
            "No open browser tabs found - is the browser running with "
            "--remote-debugging-port=9222?"
        )
    return tabs[0]


def _send_cdp_command(ws_url: str, method: str, params: dict | None = None) -> dict:
    ws = websocket.create_connection(ws_url, timeout=_CDP_TIMEOUT)
    try:
        ws.send(json.dumps({"id": 1, "method": method, "params": params or {}}))
        while True:
            data = json.loads(ws.recv())
            if data.get("id") == 1:
                return data
    finally:
        ws.close()


@registry.register(
    name="browse_to",
    description=(
        "Navigate the currently active browser tab to a URL - use to follow a link or go to a "
        "specific page within the already-open tab. Do not use this to open a new tab - use "
        "open_website for that."
    ),
    parameters={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The URL to navigate to."}},
        "required": ["url"],
    },
)
def browse_to(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    tab = _target_tab()
    _send_cdp_command(tab["webSocketDebuggerUrl"], "Page.navigate", {"url": url})
    return f"Navigated to {url}."


# Describes a visible text-entry field by whatever identifying text it
# actually has - found via live testing against a real login form (Hotstar)
# that a plain click_on_page/read_page pass couldn't reach at all: its
# <input> had no placeholder text and no name/id worth matching, but DID
# have a real title attribute ("Mobile number") and matchable parent text
# ("Enter mobile number") - multiple fallbacks needed since real-world forms
# don't consistently expose the same attribute.
_DESCRIBE_INPUT_JS = """
    const describeInput = (el) => {
        const label = el.title || el.getAttribute('aria-label') || el.placeholder || el.name
            || (el.labels && el.labels[0] && el.labels[0].innerText)
            || (el.parentElement ? el.parentElement.innerText.trim().slice(0, 60) : '');
        const kind = el.tagName === 'TEXTAREA' ? 'textarea' : (el.type || 'text');
        return (label || '(unlabeled)') + ' [' + kind + ' field]';
    };
    const isRealTextField = (el) =>
        el.offsetParent !== null &&
        !['hidden', 'submit', 'button', 'checkbox', 'radio', 'file', 'image', 'reset'].includes(el.type);
"""

_READ_PAGE_JS = f"""
(() => {{
    {_DESCRIBE_INPUT_JS}
    const text = document.body.innerText.slice(0, 2000);
    const clickable = Array.from(document.querySelectorAll('a, button'))
        .filter(el => el.offsetParent !== null)
        .slice(0, 40)
        .map(el => (el.innerText || el.getAttribute('aria-label') || '').trim())
        .filter(t => t.length > 0);
    const inputs = Array.from(document.querySelectorAll('input, textarea'))
        .filter(isRealTextField)
        .slice(0, 20)
        .map(describeInput);
    return JSON.stringify({{text, clickable, inputs}});
}})()
"""


@registry.register(
    name="read_page",
    description=(
        "Read the visible text, clickable links/buttons, and fillable input fields on the "
        "currently active browser tab - use this to see what's actually on the page before "
        "deciding what to click or which field to focus (e.g. before logging in somewhere, to see "
        "what the phone number/OTP field is actually called on that site)."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def read_page() -> dict:
    tab = _target_tab()
    result = _send_cdp_command(
        tab["webSocketDebuggerUrl"], "Runtime.evaluate", {"expression": _READ_PAGE_JS, "returnByValue": True}
    )
    value = result.get("result", {}).get("result", {}).get("value")
    if not value:
        raise RuntimeError("Could not read the page content.")
    parsed = json.loads(value)
    return {
        "visible_text": parsed["text"],
        "clickable_elements": parsed["clickable"],
        "input_fields": parsed["inputs"],
    }


@registry.register(
    name="click_on_page",
    description=(
        "Click a link or button on the currently active browser tab, matching by its visible "
        "text (case-insensitive, partial match). Use read_page first if unsure what's clickable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to match against clickable elements, e.g. 'OG Crew' or 'Videos'.",
            }
        },
        "required": ["text"],
    },
)
def click_on_page(text: str) -> str:
    tab = _target_tab()
    click_js = f"""
    (() => {{
        const target = {json.dumps(text)}.toLowerCase();
        const els = Array.from(document.querySelectorAll('a, button'));
        const match = els.find(el => el.offsetParent !== null &&
            ((el.innerText || el.getAttribute('aria-label') || '').toLowerCase().includes(target)));
        if (match) {{ match.click(); return true; }}
        return false;
    }})()
    """
    result = _send_cdp_command(
        tab["webSocketDebuggerUrl"], "Runtime.evaluate", {"expression": click_js, "returnByValue": True}
    )
    found = result.get("result", {}).get("result", {}).get("value")
    if not found:
        raise ValueError(f"No clickable element found matching '{text}'.")
    return f"Clicked: {text}"


@registry.register(
    name="focus_input_field",
    description=(
        "Click/focus a text input or textarea field on the currently active browser tab, matching "
        "by whatever identifying text it has (case-insensitive, partial match) - use this to get "
        "into a phone number/OTP/search/any other text field before typing into it with type_text "
        "or type_sensitive_field. Use read_page first to see the field's actual name under "
        "input_fields - forms don't consistently label fields the same way, so check rather than "
        "guessing. click_on_page is for links/buttons, not fields - use this instead for a field."
    ),
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to match against input fields, e.g. 'mobile number' or 'search'.",
            }
        },
        "required": ["text"],
    },
)
def focus_input_field(text: str) -> str:
    tab = _target_tab()
    focus_js = f"""
    (() => {{
        {_DESCRIBE_INPUT_JS}
        const target = {json.dumps(text)}.toLowerCase();
        const els = Array.from(document.querySelectorAll('input, textarea')).filter(isRealTextField);
        const match = els.find(el => describeInput(el).toLowerCase().includes(target));
        if (!match) return false;
        match.focus();
        match.click();
        return true;
    }})()
    """
    result = _send_cdp_command(
        tab["webSocketDebuggerUrl"], "Runtime.evaluate", {"expression": focus_js, "returnByValue": True}
    )
    found = result.get("result", {}).get("result", {}).get("value")
    if not found:
        raise ValueError(f"No input field found matching '{text}' - try read_page to see what's actually there.")
    return f"Focused: {text}"
