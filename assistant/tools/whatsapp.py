"""
WhatsApp messaging via WhatsApp Web, automated through the same Chrome
DevTools Protocol machinery as the YouTube tools (browser_automation.py).

Explicitly gated by requires_confirmation=True per the project's permanent
safety rule (see PLAN.md) for anything that sends a communication on the
user's behalf: this composes the message but does NOT send it until the
orchestrator has spoken the actual message back and gotten a real spoken
"yes" (see orchestrator.py's _voice_confirm) - the confirmation prompt below
quotes the literal text that will be sent, not a vague "are you sure?".

Requires a browser already logged into web.whatsapp.com (scan the QR code
once, same as normal use) and running with --remote-debugging-port=9222,
same requirement as browser.py/youtube.py.

*** UNVERIFIED END-TO-END ***
This could not be tested against a real, logged-in WhatsApp Web session in
this environment (there's no way to complete WhatsApp's QR login here). The
selectors below use WhatsApp Web's commonly-referenced data-testid/aria-label
attributes with multiple fallbacks, but WhatsApp changes its DOM periodically
without notice. Every step raises a specific error naming exactly which part
failed (search box / contact match / message box / send button / send
verification) rather than silently doing nothing or claiming false success -
if a step doesn't work, that error message is exactly what to report back so
it can be fixed against the real current DOM.
"""

import json
import time

from assistant.tools.browser_automation import _send_cdp_command, _target_tab
from assistant.tools.registry import registry

_LOAD_WAIT_SECONDS = 2.5
_STEP_WAIT_SECONDS = 1.0


def _eval_js(js: str):
    tab = _target_tab()
    result = _send_cdp_command(
        tab["webSocketDebuggerUrl"], "Runtime.evaluate", {"expression": js, "returnByValue": True}
    )
    return result.get("result", {}).get("result", {}).get("value")


def _navigate_current_tab(url: str) -> None:
    tab = _target_tab()
    _send_cdp_command(tab["webSocketDebuggerUrl"], "Page.navigate", {"url": url})


def _ensure_whatsapp_web_open() -> None:
    current = _target_tab()
    if "web.whatsapp.com" not in (current.get("url") or ""):
        _navigate_current_tab("https://web.whatsapp.com")
        time.sleep(_LOAD_WAIT_SECONDS)


_SEARCH_BOX_SELECTOR = (
    "document.querySelector("
    "'[data-testid=\"chat-list-search\"], "
    "div[contenteditable=\"true\"][aria-label*=\"Search\" i], "
    "div[contenteditable=\"true\"][title*=\"Search\" i]')"
)

_MESSAGE_BOX_SELECTOR = (
    "document.querySelector("
    "'[data-testid=\"conversation-compose-box-input\"], "
    "div[contenteditable=\"true\"][aria-label*=\"Type a message\" i], "
    "footer div[contenteditable=\"true\"]')"
)

_SEND_BUTTON_JS = """
(() => {
    const btn = document.querySelector(
        '[data-testid="send"], button[aria-label*="Send" i], span[data-icon="send"]'
    );
    if (!btn) return false;
    (btn.closest('button') || btn).click();
    return true;
})()
"""


def _type_into(selector_js: str, text: str) -> bool:
    # Clears any leftover content first (a stale search query, or an old
    # message draft in the compose box) - without this, execCommand
    # insertText appends at the cursor instead of replacing, and the message
    # actually sent could end up different from the text the user confirmed.
    js = f"""
    (() => {{
        const el = {selector_js};
        if (!el) return false;
        el.focus();
        document.execCommand('selectAll', false, null);
        document.execCommand('delete', false, null);
        document.execCommand('insertText', false, {json.dumps(text)});
        return true;
    }})()
    """
    return bool(_eval_js(js))


def _click_first_search_result(contact_name: str) -> bool:
    js = f"""
    (() => {{
        const target = {json.dumps(contact_name)}.toLowerCase();
        const rows = Array.from(document.querySelectorAll(
            '[data-testid="cell-frame-container"], div[role="listitem"], div[role="row"]'
        ));
        const match = rows.find(el => (el.innerText || '').toLowerCase().includes(target));
        if (!match) return false;
        match.click();
        return true;
    }})()
    """
    return bool(_eval_js(js))


def _last_outgoing_message_text() -> str | None:
    js = """
    (() => {
        const bubbles = Array.from(document.querySelectorAll(
            '.message-out .selectable-text, [data-testid="msg-container"].message-out .selectable-text'
        ));
        if (bubbles.length === 0) return null;
        return bubbles[bubbles.length - 1].innerText || null;
    })()
    """
    return _eval_js(js)


@registry.register(
    name="send_whatsapp_message",
    description=(
        "Compose and send a WhatsApp message to a contact by name, via WhatsApp Web. This is a "
        "sensitive action - the orchestrator automatically speaks the message back and requires a "
        "real spoken 'yes' before it actually sends, so just call this tool directly with the "
        "message you'd send; do not ask for confirmation yourself in text first. Write the message "
        "text naturally based on what the user asked for."
    ),
    parameters={
        "type": "object",
        "properties": {
            "contact_name": {
                "type": "string",
                "description": "The contact/chat to send to, exactly as named by the user.",
            },
            "message": {
                "type": "string",
                "description": "The exact message text to send.",
            },
        },
        "required": ["contact_name", "message"],
    },
    requires_confirmation=True,
    confirmation_prompt=lambda args: (
        f"Send \"{args.get('message', '')}\" to {args.get('contact_name', 'that contact')} on WhatsApp?"
    ),
)
def send_whatsapp_message(contact_name: str, message: str) -> str:
    _ensure_whatsapp_web_open()

    if not _type_into(_SEARCH_BOX_SELECTOR, contact_name):
        raise RuntimeError(
            "Couldn't find WhatsApp Web's search box - make sure the browser is logged into "
            "web.whatsapp.com (scan the QR code once if needed)."
        )
    time.sleep(_STEP_WAIT_SECONDS)

    if not _click_first_search_result(contact_name):
        raise ValueError(f"No WhatsApp contact/chat found matching '{contact_name}'.")
    time.sleep(_STEP_WAIT_SECONDS)

    if not _type_into(_MESSAGE_BOX_SELECTOR, message):
        raise RuntimeError("Found the chat but couldn't find the message box to type into.")
    time.sleep(0.3)

    if not _eval_js(_SEND_BUTTON_JS):
        raise RuntimeError("Typed the message but couldn't find the send button.")
    time.sleep(_STEP_WAIT_SECONDS)

    sent_text = _last_outgoing_message_text()
    if sent_text is None or message.strip() not in sent_text:
        raise RuntimeError(
            "Clicked send but couldn't verify the message actually appears in the chat - "
            "please check WhatsApp directly."
        )
    return f"Sent to {contact_name}: {message}"
