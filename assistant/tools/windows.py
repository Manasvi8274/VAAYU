"""
Desktop window switching via the Win32 API (pywin32). This lets the assistant
bring an already-open window (any app) to the foreground - e.g. "switch to my
notepad window" - distinct from browser.py, which handles tabs *within* a
browser window (the OS has no concept of tabs, only windows).
"""

import win32con
import win32gui

from assistant.tools.registry import registry
from assistant.tools.window_utils import force_foreground, list_windows


@registry.register(
    name="list_open_windows",
    description=(
        "List the titles of all currently open desktop windows/applications (OS-level windows, "
        "each a separate app like Notepad, GitHub Desktop, Windows Explorer). Do NOT use this for "
        "'what tabs do I have open'/'what tabs are open' - a browser's tabs are not separate "
        "desktop windows and won't be meaningfully listed here; use list_browser_tabs for any "
        "request that says 'tab(s)' specifically."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def list_open_windows() -> list[str]:
    return [title for _hwnd, title in list_windows()]


@registry.register(
    name="switch_to_window",
    description="Bring an already-open desktop window to the foreground by matching (part of) its title.",
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Text to match against open window titles, e.g. 'notepad' or 'calculator'.",
            }
        },
        "required": ["title"],
    },
)
def switch_to_window(title: str) -> str:
    title_lower = title.lower()
    matches = [(hwnd, t) for hwnd, t in list_windows() if title_lower in t.lower()]
    if not matches:
        raise ValueError(f"No open window found matching '{title}'.")

    hwnd, matched_title = matches[0]
    force_foreground(hwnd)
    return f"Switched to window: {matched_title}"


def _resolve_window(title: str | None) -> tuple[int, str]:
    """No title given -> the current foreground window (e.g. 'minimize this' /
    'minimize the browser' said while it's the active window)."""
    if not title:
        hwnd = win32gui.GetForegroundWindow()
        return hwnd, win32gui.GetWindowText(hwnd)
    title_lower = title.lower()
    matches = [(hwnd, t) for hwnd, t in list_windows() if title_lower in t.lower()]
    if not matches:
        raise ValueError(f"No open window found matching '{title}'.")
    return matches[0]


@registry.register(
    name="minimize_window",
    description=(
        "Minimize a desktop window (send it to the taskbar) by matching (part of) its title, e.g. "
        "'minimize the browser'. Omit title to minimize whichever window is currently focused/"
        "active - use this for 'minimize this' or 'move it to the background'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Text to match against open window titles. Omit for the current window.",
            }
        },
        "required": [],
    },
)
def minimize_window(title: str | None = None) -> str:
    hwnd, matched_title = _resolve_window(title)
    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
    return f"Minimized: {matched_title}"


@registry.register(
    name="restore_window",
    description=(
        "Restore a minimized desktop window and bring it back to the foreground, by matching "
        "(part of) its title, e.g. 'bring back the browser'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Text to match against open window titles.",
            }
        },
        "required": ["title"],
    },
)
def restore_window(title: str) -> str:
    hwnd, matched_title = _resolve_window(title)
    force_foreground(hwnd)
    return f"Restored: {matched_title}"
