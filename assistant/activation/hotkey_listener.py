"""
Global hotkey detection via the Win32 RegisterHotKey API (through pywin32).

This project originally used the `keyboard` library's wait()/hook(), which
installs a low-level global keyboard hook (WH_KEYBOARD_LL). In testing, that
hook received zero events - not even self-generated synthetic ones - in a
non-elevated process, a known rough edge of that library on Windows.
RegisterHotKey is the lighter-weight, purpose-built Windows API for exactly
this (a single global hotkey, not every keystroke) and was verified working
without Administrator privileges.
"""

import win32con
import win32gui

from assistant.activation.base import Activator

_MODIFIER_MAP = {
    "ctrl": win32con.MOD_CONTROL,
    "control": win32con.MOD_CONTROL,
    "alt": win32con.MOD_ALT,
    "shift": win32con.MOD_SHIFT,
    "win": win32con.MOD_WIN,
    "windows": win32con.MOD_WIN,
}

_HOTKEY_ID = 1


def _parse_hotkey(hotkey: str) -> tuple[int, int]:
    modifiers = 0
    vk = None
    for part in hotkey.lower().split("+"):
        part = part.strip()
        if part in _MODIFIER_MAP:
            modifiers |= _MODIFIER_MAP[part]
        elif len(part) == 1:
            vk = ord(part.upper())
        else:
            raise ValueError(
                f"Unsupported hotkey key '{part}' - only single letters/digits "
                "plus ctrl/alt/shift/win modifiers are supported."
            )
    if vk is None:
        raise ValueError(f"Hotkey '{hotkey}' has no non-modifier key.")
    return modifiers, vk


class HotkeyActivator(Activator):
    def __init__(self, hotkey: str):
        self.hotkey = hotkey
        self.modifiers, self.vk = _parse_hotkey(hotkey)

    def wait_for_trigger(self) -> None:
        win32gui.RegisterHotKey(None, _HOTKEY_ID, self.modifiers, self.vk)
        try:
            while True:
                message = win32gui.GetMessage(None, 0, 0)
                if message[1][1] == win32con.WM_HOTKEY:
                    return
        finally:
            win32gui.UnregisterHotKey(None, _HOTKEY_ID)
