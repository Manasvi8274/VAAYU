"""
Shared Win32 window-foregrounding helpers, used by windows.py (explicit
"switch to X" commands) and by apps.py/web.py (bringing a newly
opened/navigated app or browser to the foreground by default - see
foreground_by_title() below).
"""

import time

import win32api
import win32con
import win32gui
import win32process


def list_windows() -> list[tuple[int, str]]:
    windows: list[tuple[int, str]] = []

    def _callback(hwnd: int, _extra) -> None:
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            windows.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(_callback, None)
    return windows


def _alt_key_tap() -> None:
    """Windows ignores SetForegroundWindow from a background process unless
    the foreground lock is "unlocked" first - a synthetic Alt keypress does
    that without touching the target process at all."""
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)


def force_foreground(hwnd: int) -> None:
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    target_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
    current_thread = win32api.GetCurrentThreadId()

    try:
        # Works for classic Win32 apps, but modern UWP/AppContainer apps
        # (e.g. the Windows 11 Calculator) can deny AttachThreadInput across
        # that security boundary with "Access is denied" - fall back below.
        if target_thread != current_thread:
            win32process.AttachThreadInput(current_thread, target_thread, True)
            try:
                win32gui.SetForegroundWindow(hwnd)
            finally:
                win32process.AttachThreadInput(current_thread, target_thread, False)
        else:
            win32gui.SetForegroundWindow(hwnd)
    except Exception:
        _alt_key_tap()
        win32gui.SetForegroundWindow(hwnd)


def foreground_by_title(
    title_substring: str, retries: int = 5, retry_delay: float = 0.3
) -> bool:
    """Finds a visible window whose title contains title_substring and
    brings it to the foreground. Retries briefly since a just-launched app's
    window sometimes doesn't exist yet the instant the launching call
    returns. Returns True if a matching window was found and foregrounded."""
    needle = title_substring.lower()
    for _ in range(retries):
        matches = [hwnd for hwnd, t in list_windows() if needle in t.lower()]
        if matches:
            force_foreground(matches[0])
            return True
        time.sleep(retry_delay)
    return False
