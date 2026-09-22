"""
General-purpose OS-level control tools that generalize across every
application rather than needing a bespoke tool per app - a keyboard shortcut
works the same way whether it's typed at Notepad, a browser, or a game.
This is the practical way to cover "every feature of every app" for the
huge fraction of app functionality that's exposed via standard shortcuts
(save, undo, copy/paste, close, switch window, etc.), instead of writing
one-off tools per application per feature, which doesn't scale.
"""

import ctypes
import time
from pathlib import Path

import keyboard

from assistant.tools.registry import registry

_SCREENSHOT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "screenshots"


@registry.register(
    name="press_hotkey",
    description=(
        "Press a keyboard shortcut/hotkey on whatever window currently has focus - e.g. 'ctrl+s' "
        "(save), 'ctrl+z' (undo), 'ctrl+c'/'ctrl+v' (copy/paste), 'alt+tab' (switch window), "
        "'win+d' (show desktop), 'ctrl+shift+esc' (task manager), 'alt+f4' (close window), "
        "'ctrl+t'/'ctrl+w' (new/close tab). Use this for any app action the user names that isn't "
        "covered by a more specific tool - most application features are reachable this way. "
        "Combine keys with '+', e.g. 'ctrl+shift+t'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "keys": {
                "type": "string",
                "description": "The hotkey to press, e.g. 'ctrl+s', 'alt+tab', 'win+d'.",
            }
        },
        "required": ["keys"],
    },
)
def press_hotkey(keys: str) -> str:
    keyboard.send(keys)
    return f"Pressed: {keys}"


@registry.register(
    name="lock_computer",
    description="Lock the computer (Windows lock screen) - same as pressing Win+L.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def lock_computer() -> str:
    ok = ctypes.windll.user32.LockWorkStation()
    if not ok:
        raise RuntimeError("Failed to lock the computer.")
    return "Locked the computer."


@registry.register(
    name="take_screenshot",
    description="Take a screenshot of the whole screen and save it to disk. Returns the file path.",
    parameters={"type": "object", "properties": {}, "required": []},
)
def take_screenshot() -> str:
    from PIL import ImageGrab  # imported lazily - only this tool needs Pillow

    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path = _SCREENSHOT_DIR / filename
    img = ImageGrab.grab()
    img.save(path)
    return f"Saved screenshot to {path}"
