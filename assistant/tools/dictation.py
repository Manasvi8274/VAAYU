"""
Types text into whatever window currently has focus - useful for dictating
into Notepad, a browser field, etc.

Uses clipboard + paste rather than simulating individual keystrokes
(keyboard.write()). Tested and confirmed this matters: character-by-character
simulated typing was too fast for a modern WinUI3-based text control (Windows
11's Notepad) to keep up with, corrupting the text (dropped/repeated
characters). A single paste is atomic and works reliably across both classic
Win32 controls and modern ones.

type_sensitive_field below shares the same paste mechanism but is a
separate, requires_confirmation=True tool for phone numbers/OTPs/passwords/
PINs specifically - per the project's permanent safety rule (see PLAN.md),
anything touching credentials/2FA must be confirmation-gated, and type_text
itself can't be (it's the generic dictation tool used for ordinary,
non-sensitive text - gating every use of it would make plain dictation
prompt for confirmation constantly).
"""

import time

import keyboard
import pywintypes
import win32clipboard
import win32con

from assistant.tools.registry import registry

# `keyboard` (already a dependency, used for media keys elsewhere) is only
# used here to send Ctrl+V, not to type character-by-character.

# Found via live testing (not theoretical): OpenClipboard/SetClipboardData
# can transiently fail with "The handle is invalid" - confirmed genuinely
# intermittent and NOT explained by anything in this module's own control
# flow (isolated every piece: 20 back-to-back sets in a tight loop all
# succeeded; get-then-set immediately after a real CDP browser call
# succeeded; yet the exact same full sequence failed outright, all 5 retries
# included, in another run moments later with nothing else changed). This
# matches Windows' well-documented clipboard contention behavior - its own
# Clipboard History/Cloud Clipboard service, or any other app, can briefly
# hold the clipboard when it changes, for an unpredictable duration. A
# single unlucky failure here would be a real problem for
# type_sensitive_field specifically - an OTP is time-sensitive and
# single-use, so silently failing to type it isn't a minor inconvenience.
# Retrying with backoff is the standard mitigation for this known Windows
# behavior; the budget here (~2s worst case) was sized by testing against a
# real failure that outlasted a shorter one, not picked arbitrarily.
_CLIPBOARD_RETRIES = 8
_CLIPBOARD_RETRY_DELAYS_SECONDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


def _with_clipboard_retry(fn):
    last_error: Exception | None = None
    for attempt in range(_CLIPBOARD_RETRIES):
        try:
            return fn()
        except pywintypes.error as exc:
            last_error = exc
            time.sleep(_CLIPBOARD_RETRY_DELAYS_SECONDS[attempt])
    raise last_error


def _get_clipboard_text() -> str:
    def _do():
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            return ""
        finally:
            win32clipboard.CloseClipboard()

    return _with_clipboard_retry(_do)


def _set_clipboard_text(text: str) -> None:
    def _do():
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
        finally:
            win32clipboard.CloseClipboard()

    _with_clipboard_retry(_do)


def _paste_text_via_clipboard(text: str) -> None:
    original_clipboard = _get_clipboard_text()
    try:
        _set_clipboard_text(text)
        time.sleep(0.05)
        keyboard.send("ctrl+v")
        time.sleep(0.1)
    finally:
        _set_clipboard_text(original_clipboard)


@registry.register(
    name="type_text",
    description=(
        "Types the given text into whatever window currently has focus, as if typed on the "
        "keyboard - useful for dictating text into Notepad, a browser text field, or any other "
        "app the user just opened/clicked into. Does not press Enter or submit anything. Do NOT "
        "use this for a phone number, OTP/verification code, password, or PIN being entered to log "
        "in somewhere - use type_sensitive_field for those instead."
    ),
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The exact text to type."}},
        "required": ["text"],
    },
)
def type_text(text: str) -> str:
    _paste_text_via_clipboard(text)
    return f"Typed: {text}"


@registry.register(
    name="type_sensitive_field",
    description=(
        "Type a phone number, OTP/verification code, password, or PIN into whatever field "
        "currently has focus - use this INSTEAD of type_text for any login/credential value, on "
        "any site (Hotstar, Netflix, banking apps, anything). This always asks for a real spoken "
        "confirmation first, reading back the exact value and which field it's going into, before "
        "actually typing anything - it's a sensitive action per the project's permanent rule that "
        "anything touching credentials/2FA requires confirmation. Just call this tool directly with "
        "the value the user gave you; the confirmation happens automatically, don't ask yourself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "description": "The exact value to type, e.g. the phone number or OTP digits.",
            },
            "field_description": {
                "type": "string",
                "description": "What field this is, e.g. 'phone number field', 'OTP field', 'password field'.",
            },
        },
        "required": ["value", "field_description"],
    },
    requires_confirmation=True,
    confirmation_prompt=lambda args: (
        f"Type {args.get('value', '')} into the {args.get('field_description', 'field')}?"
    ),
)
def type_sensitive_field(value: str, field_description: str) -> str:
    _paste_text_via_clipboard(value)
    return f"Entered the {field_description}."
