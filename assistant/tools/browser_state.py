"""
Figures out which browser open_website should target, instead of always
using webbrowser.open() (which goes to the OS default browser regardless of
what's actually open). General by construction - not a hardcoded list of
browser names/paths - so it works for Opera, Vivaldi, or anything else
Windows knows about, not just the ones anticipated in advance.

Resolution order, each falling back to the next:
1. The browser most recently opened via open_app this session (most
   deterministic - reflects Vaayu's own recent action).
2. Whichever registered browser currently owns the focused window (reflects
   what the user is actually looking at right now).
3. Any registered browser with a running process at all.
4. None of the above -> caller falls back to the OS default browser.

"Registered browser" means Windows' own StartMenuInternet registry (the
same list Windows' "choose your default browser" settings page reads from),
queried fresh each time rather than cached, so a newly-installed browser is
picked up without a code change.
"""

import os
import re
import subprocess
import time
import winreg

import psutil
import win32gui
import win32process

from assistant.tools.window_utils import foreground_by_title

_last_browser_name: str | None = None


def _extract_exe_path(command: str) -> str:
    match = re.match(r'^"([^"]+)"', command)
    if match:
        return match.group(1)
    return command.split()[0] if command.split() else command


def _enumerate_registered_browsers() -> dict[str, str]:
    """{lowercased display name: executable path} for every browser
    registered with Windows, merging per-user and machine-wide entries."""
    browsers: dict[str, str] = {}
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            root = winreg.OpenKey(hive, r"SOFTWARE\Clients\StartMenuInternet")
        except OSError:
            continue
        with root:
            i = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(root, subkey_name) as subkey:
                        display_name = winreg.QueryValueEx(subkey, "")[0]
                    with winreg.OpenKey(root, subkey_name + r"\shell\open\command") as cmd_key:
                        command = winreg.QueryValueEx(cmd_key, "")[0]
                    exe_path = _extract_exe_path(command)
                    if os.path.exists(exe_path):
                        browsers[display_name.strip().lower()] = exe_path
                except OSError:
                    continue
    return browsers


def _foreground_browser_exe(registered: dict[str, str]) -> str | None:
    exe_names = {os.path.basename(p).lower(): p for p in registered.values()}
    try:
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        proc_name = psutil.Process(pid).name().lower()
    except Exception:
        return None
    return exe_names.get(proc_name)


def _running_browser_exe(registered: dict[str, str]) -> str | None:
    exe_names = {os.path.basename(p).lower(): p for p in registered.values()}
    try:
        running = {p.info["name"].lower() for p in psutil.process_iter(["name"])}
    except Exception:
        return None
    for exe_name, path in exe_names.items():
        if exe_name in running:
            return path
    return None


def note_app_opened(app_name: str) -> None:
    """Called by open_app whenever it successfully opens something - records
    the name so open_website can prefer it, if it turns out to be a browser."""
    global _last_browser_name
    _last_browser_name = app_name.strip().lower()


def open_url_in_last_browser(url: str, foreground: bool = True) -> bool:
    """Returns True (and launches it) if a specific browser could be
    targeted; False if the caller should fall back to the OS default
    browser (webbrowser.open) instead."""
    registered = _enumerate_registered_browsers()
    if not registered:
        return False

    exe = None
    display_name = None
    if _last_browser_name:
        for name, path in registered.items():
            if _last_browser_name in name or name in _last_browser_name:
                exe, display_name = path, name
                break

    if exe is None:
        exe = _foreground_browser_exe(registered)
    if exe is None:
        exe = _running_browser_exe(registered)
    if exe is None:
        return False

    subprocess.Popen([exe, url])

    if foreground:
        # Sending a URL to an already-running browser doesn't reliably bring
        # its window forward on its own - Windows' foreground-lock can block
        # it, same issue already fixed once for switch_to_window. A brief
        # delay gives the new tab a moment to actually exist before we try.
        time.sleep(0.4)
        search_term = display_name or os.path.splitext(os.path.basename(exe))[0]
        foreground_by_title(search_term)

    return True
