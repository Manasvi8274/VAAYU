import contextlib
import re
import io
import subprocess
import time
import urllib.request

import psutil
from AppOpener import close as app_close
from AppOpener import open as app_open

from assistant.tools import browser_state
from guardian.rules import check_not_payment
from assistant.tools.registry import registry
from assistant.tools.window_utils import foreground_by_title

# AppOpener's own open_things/close_things only reach their `raise
# AppNotFound(...)` line when `output` is truthy (found live, reading
# AppOpener's actual source: the raise is nested inside `if output: if
# throw_error: raise ...` in every code path, including the match_closest
# one this project always uses) - so passing output=False (as this file
# used to, specifically to avoid console spam) silently disabled
# throw_error=True completely, no matter what. Confirmed live: closing/
# opening a nonexistent app with output=False, throw_error=True raised
# nothing at all - meaning open_app/close_app could NEVER detect a real
# failure and always reported success regardless of what actually happened.
# Real-usage bug hunt hit this twice: "close notepad" claimed success while
# Notepad kept running, and "open vlc" claimed success while no vlc.exe
# process ever existed. Fix: pass output=True (so the library's own
# detection actually runs) but capture/discard its prints via
# redirect_stdout, keeping the same quiet console behavior output=False
# was originally meant to provide, without disabling error detection to do it.
def _quiet_stdout():
    return contextlib.redirect_stdout(io.StringIO())

# AppOpener matches "vlc" against its registered app database via
# difflib.get_close_matches with a 0.6 similarity cutoff - a 3-character
# abbreviation scores too low against "VLC Media Player" for that cutoff to
# match, confirmed live (bare "vlc" raises AppNotFound; "vlc media player"
# opens correctly). Not a general problem worth solving generically here
# (that's AppOpener's own matching algorithm) - just the one concrete case
# this project has actually hit real evidence for.
#
# ONLY for opening - close_things() matches against currently-RUNNING
# process names (e.g. real "vlc.exe"), not the registered-app-name
# database open_things() uses, so "vlc" already matches perfectly there on
# its own. Confirmed live: applying this same alias to close_app actively
# broke it - "vlc media player" doesn't fuzzy-match the real "vlc.exe"
# process well enough, causing a real close to fail with AppNotFound.
_OPEN_APP_NAME_ALIASES = {
    "vlc": "vlc media player",
}


def _resolve_open_app_name(app_name: str) -> str:
    return _OPEN_APP_NAME_ALIASES.get(app_name.strip().lower(), app_name)


def _find_matching_process_name(app_name: str) -> str | None:
    """Exact (case/space/'.exe'-insensitive) match of app_name against
    currently running process names - lets close_app verify independently
    that a close genuinely took effect, rather than trusting AppOpener's
    own success signal alone (which, per the output=True fix above, is now
    real but still worth double-checking given real process termination
    isn't always instant). Deliberately NOT the same fuzzy difflib matching
    AppOpener itself uses to guess an app from a vague name - found live
    that fuzzy matching here produces false positives ("brave" scored a
    0.6+ similarity match against the unrelated "BraveCrashHandler.exe",
    which is always present and never what "is brave still running"
    means) - this check needs precision, not a best-effort guess."""
    normalized_target = app_name.strip().lower().removesuffix(".exe").replace(" ", "")
    for p in psutil.process_iter(["name"]):
        proc_name = p.info.get("name") or ""
        normalized_proc = proc_name.lower().removesuffix(".exe").replace(" ", "")
        if normalized_proc == normalized_target:
            return proc_name
    return None


# Chromium browsers are launched directly WITH the DevTools port, instead of
# through AppOpener - the YouTube/tab/page tools all drive the browser over
# that port, and a browser started without it can't be controlled (found
# live: "play the latest OG Crew video" failed after Vaayu itself had just
# opened Brave the normal way). Verified live: Brave with the user's real
# profile exposes the port within 0.5 s.
_CDP_VERSION_URL = "http://127.0.0.1:9222/json/version"
_CHROMIUM_NAMES = {"brave": "brave", "chrome": "google chrome", "google chrome": "google chrome",
                   "edge": "microsoft edge", "microsoft edge": "microsoft edge", "comet": "comet"}


def _cdp_up() -> bool:
    try:
        with urllib.request.urlopen(_CDP_VERSION_URL, timeout=1):
            return True
    except OSError:
        return False


def _open_browser_with_automation(app_name: str, foreground: bool) -> str | None:
    key = _CHROMIUM_NAMES.get(app_name.strip().lower().removesuffix(" browser"))
    exe = browser_state._enumerate_registered_browsers().get(key) if key else None
    if not exe:
        return None
    label = key.title()
    running = any(_proc_exe(p) == exe.lower() for p in psutil.process_iter())
    if running:
        # Launching again would just open another window of the running
        # browser - and the port flag only takes effect at a fresh start.
        if foreground:
            foreground_by_title(app_name)
        browser_state.note_app_opened(app_name)
        if _cdp_up():
            return f"{label} is already open."
        return (f"{label} is already open, but it wasn't opened by me, so I can't control its tabs or YouTube. "
                f"Close {label} and say 'open {app_name}' to let me reopen it.")
    _launch_with_port(exe)
    browser_state.note_app_opened(app_name)
    if foreground:
        foreground_by_title(app_name)
    return f"Opened {label}." + ("" if foreground else " (in the background)")


def ensure_browser(app_name: str = "brave") -> str:
    """A browser Vaayu can drive, for videos: already controllable -> nothing to do; not running -> start it
    with the control port; running WITHOUT the port (opened by the user) -> close it and reopen it with the
    port, restoring its tabs. Real log: "play ... video" kept failing with "it has to be opened by me".
    Returns a note for the reply ("" when nothing had to happen)."""
    if _cdp_up():
        return ""
    key = _CHROMIUM_NAMES.get(app_name.strip().lower())
    exe = browser_state._enumerate_registered_browsers().get(key) if key else None
    if not exe:
        raise RuntimeError(f"I can't find {app_name} on this laptop.")
    procs = [p for p in psutil.process_iter() if _proc_exe(p) == exe.lower()]
    if procs:
        # A normal close (not /F): Brave saves its session first, so --restore-last-session brings the tabs back.
        subprocess.run(["taskkill", "/IM", exe.split("\\")[-1]], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        psutil.wait_procs(procs, timeout=8)
        subprocess.Popen([exe, "--remote-debugging-port=9222", "--restore-last-session"])
    else:
        subprocess.Popen([exe, "--remote-debugging-port=9222"])
    for _ in range(40):
        if _cdp_up():
            browser_state.note_app_opened(app_name)
            return (f" (I restarted {key.title()} so I can control it - your tabs are back.)" if procs
                    else f" (Opened {key.title()}.)")
        time.sleep(0.25)
    raise RuntimeError(f"I opened {key.title()} but couldn't get control of it - try again in a moment.")


def _launch_with_port(exe: str) -> None:
    subprocess.Popen([exe, "--remote-debugging-port=9222"])
    for _ in range(20):  # up to 5 s for the port, so the next step (open YouTube...) can use it
        if _cdp_up():
            return
        time.sleep(0.25)


def _proc_exe(p) -> str:
    try:
        return (p.exe() or "").lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return ""


# --- what each open_app started, so close_app can close exactly that ---------------------------------
# Closing by guessing a process name fails for many apps (Calculator runs as CalculatorApp.exe, a Steam
# game as "TEKKEN 7.exe"...). Instead: the WINDOWS that appeared right after the open, and their new
# processes. Shared Windows hosts are never ended - a Store app's window belongs to ApplicationFrameHost,
# which hosts every Store app (found in the app sweep: closing Calculator left it running, as it should).
_OPENED: dict[str, dict] = {}   # app -> {"hwnds": set, "pids": set}
_TRACK_SECONDS = 8.0   # how long after an open to look for its new windows
_CLOSE_WAIT = 5.0      # how long a polite close (like clicking X) gets before the next step
_SHARED_HOSTS = {"applicationframehost.exe", "runtimebroker.exe", "dllhost.exe", "svchost.exe", "explorer.exe",
                 "conhost.exe", "sihost.exe", "taskhostw.exe", "searchhost.exe", "startmenuexperiencehost.exe",
                 "shellexperiencehost.exe", "textinputhost.exe", "backgroundtaskhost.exe"}


def _pids() -> set[int]:
    return set(psutil.pids())


def _windows() -> dict[int, int]:
    """Visible, titled top-level windows: hwnd -> pid."""
    import win32gui
    import win32process
    found = {}

    def add(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            found[hwnd] = win32process.GetWindowThreadProcessId(hwnd)[1]
    win32gui.EnumWindows(add, None)
    return found


def _window_pids() -> set[int]:
    return set(_windows().values())


def _proc_name(pid: int) -> str:
    try:
        return psutil.Process(pid).name().lower()
    except psutil.Error:
        return ""


def _track_opened(app_name: str, before: set[int]) -> None:
    before_windows = set(_windows())

    def run():
        deadline, key = time.time() + _TRACK_SECONDS, app_name.strip().lower()
        hwnds, pids = set(), set()
        while time.time() < deadline:
            time.sleep(0.5)
            new_pids = _pids() - before
            for hwnd, pid in _windows().items():
                # a new window - of a new process, or of a shared host (a Store app's frame)
                if hwnd not in before_windows and (pid in new_pids or _proc_name(pid) in _SHARED_HOSTS):
                    hwnds.add(hwnd)
                    if _proc_name(pid) not in _SHARED_HOSTS:
                        pids.add(pid)
            for pid in new_pids:  # the app's own windowless processes (a Store app's CalculatorApp.exe...)
                if _proc_name(pid) not in _SHARED_HOSTS and (pid in pids or hwnds):
                    try:
                        if psutil.Process(pid).create_time() >= time.time() - _TRACK_SECONDS - 2:
                            pids.add(pid)
                    except psutil.Error:
                        pass
            if hwnds:
                # Recorded as found, not at the end - a "close it" right after the open must see it (found in
                # the app sweep: 7-Zip closed 4 s after opening and close_app didn't know it yet).
                _OPENED[key] = {"hwnds": set(hwnds), "pids": set(pids)}
    import threading
    threading.Thread(target=run, daemon=True, name="track-open").start()


def _opened_pids(app_name: str) -> set[int]:
    """The live processes and windows Vaayu started for this app (any id; empty if none)."""
    entry = _opened_entry(app_name)
    return (entry["pids"] | entry["hwnds"]) if entry else set()


def _opened_entry(app_name: str) -> dict | None:
    import win32gui
    key = app_name.strip().lower()
    if key not in _OPENED:
        from assistant.tools.names import sound_key
        import difflib
        match = max(_OPENED, key=lambda k: difflib.SequenceMatcher(None, sound_key(k), sound_key(key)).ratio(),
                    default=None)
        if not match or difflib.SequenceMatcher(None, sound_key(match), sound_key(key)).ratio() < 0.8:
            return None
        key = match
    entry = _OPENED[key]
    entry = {"hwnds": {h for h in entry["hwnds"] if win32gui.IsWindow(h)},
             "pids": {p for p in entry["pids"] if psutil.pid_exists(p)}, "key": key}
    if not entry["hwnds"] and not entry["pids"]:
        _OPENED.pop(key, None)
        return None
    return entry


def _close_tracked(app_name: str, _ids=None) -> str:
    """Close politely (WM_CLOSE to its own windows, like clicking X); a window still asking about unsaved
    changes is left for the user (their rule: ask before saving); its own windowless leftovers are ended."""
    import win32con
    import win32gui
    entry = _opened_entry(app_name)
    if entry is None:
        return f"Closed {app_name}."
    for hwnd in entry["hwnds"]:
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        except Exception:
            pass
    deadline = time.time() + _CLOSE_WAIT
    while time.time() < deadline and any(win32gui.IsWindow(h) and win32gui.IsWindowVisible(h) for h in entry["hwnds"]):
        time.sleep(0.25)
    if any(win32gui.IsWindow(h) and win32gui.IsWindowVisible(h) for h in entry["hwnds"]):
        raise RuntimeError(f"{app_name} is still open - it may be asking you to save something.")
    time.sleep(0.5)
    for pid in entry["pids"]:  # its own processes, windowless now (tray, background, a Store app's process)
        if psutil.pid_exists(pid) and _proc_name(pid) not in _SHARED_HOSTS:
            try:
                psutil.Process(pid).terminate()
            except psutil.Error:
                pass
    _OPENED.pop(entry["key"], None)
    return f"Closed {app_name}."


_START_APPS_CACHE: tuple[float, dict[str, str]] | None = None
_START_APPS_MAX_AGE = 600  # re-read after 10 minutes, so a freshly installed app shows up


def _start_menu_apps() -> dict[str, str]:
    """Windows' live Start-menu list: name -> AppID (an app path, a store id, or a
    URL like steam://rungameid/389730 for Steam games)."""
    global _START_APPS_CACHE
    if _START_APPS_CACHE and time.time() - _START_APPS_CACHE[0] < _START_APPS_MAX_AGE:
        return _START_APPS_CACHE[1]
    import json
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-StartApps | ConvertTo-Json -Compress"], capture_output=True, text=True,
                         timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    rows = json.loads(out or "[]")
    apps = {r["Name"]: r["AppID"] for r in (rows if isinstance(rows, list) else [rows]) if r.get("Name")}
    _START_APPS_CACHE = (time.time(), apps)
    return apps


def _open_from_start_menu(app_name: str) -> str | None:
    """Launch the Start-menu app that sounds like app_name; its name, or None if none is close."""
    import os
    from assistant.tools.names import sound_key
    apps = _start_menu_apps()
    # "open tekken game which is present in my laptop" (real log) - the generic words aren't its name
    name = re.sub(r"\b(?:game|app|application|software|program|the|my)\b", " ", app_name, flags=re.I)
    want = sound_key(name)
    best = max(apps, key=lambda n: __import__("difflib").SequenceMatcher(None, want, sound_key(n)).ratio(), default=None)
    if not best or __import__("difflib").SequenceMatcher(None, want, sound_key(best)).ratio() < 0.75:
        return None
    app_id = apps[best]
    if "://" in app_id:  # steam://rungameid/..., http://... - the URL's own handler launches it
        os.startfile(app_id)
    else:
        subprocess.Popen(["explorer", f"shell:AppsFolder\\{app_id}"])
    return best


@registry.register(
    name="open_app",
    description=(
        "Launch an installed application on this Windows computer by name, e.g. 'notepad', "
        "'spotify', 'calculator', 'armoury crate' (a real ASUS system utility for laptop settings - "
        "despite the name, it is NOT a game or a game feature; call this tool with it exactly like "
        "any other app name, do not ask the user what game they mean). Use this (not open_website) "
        "for launching a web browser itself by name - 'chrome', 'brave', 'firefox', 'edge' are "
        "applications, not websites. Brought to the foreground by default - only pass "
        "foreground=false if the user explicitly says to open it in the background. Hinglish for "
        "THIS action (open/start): 'khol do', 'kholo', 'chalu karo'. Do NOT use this for 'band "
        "karo'/'band kar do'/'band karde' - that means close, the opposite action - use close_app "
        "for that instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Name of the application to open, e.g. 'chrome' or 'notepad'.",
            },
            "foreground": {
                "type": "boolean",
                "description": (
                    "Whether to bring the app to the foreground (default true). Set to false only "
                    "if the user explicitly asked for it to open in the background."
                ),
            },
        },
        "required": ["app_name"],
    },
)
def open_app(app_name: str, foreground: bool = True) -> str:
    check_not_payment(app_name)
    browser_reply = _open_browser_with_automation(app_name, foreground)
    if browser_reply:
        return browser_reply
    resolved_name = _resolve_open_app_name(app_name)
    before = _pids()
    try:
        with _quiet_stdout():
            app_open(resolved_name, match_closest=True, output=True, throw_error=True)
    except Exception as exc:
        # AppOpener's app list is a snapshot from when it was built - found live, Tekken 7
        # (installed later) failed with its "TEKKEN 7 is not running" (= not found).
        if not _open_from_start_menu(app_name):
            raise RuntimeError(f"I couldn't find an app called {app_name} on this laptop.") from exc
    _track_opened(app_name, before)
    browser_state.note_app_opened(app_name)
    if foreground:
        foreground_by_title(app_name)
    return f"Opened {app_name}" + ("." if foreground else " in the background.")


@registry.register(
    name="close_app",
    description=(
        "Close a currently running application on this Windows computer by name - this closes the "
        "WHOLE application, including a browser and every tab in it. For closing just one browser "
        "tab, use close_browser_tab instead. This applies to any app by name, including one that "
        "sounds like it could be gaming slang but is a real installed program (e.g. 'armoury "
        "crate', ASUS's real system utility, not a game feature) - call this tool with the name "
        "exactly as given rather than asking what game is meant. Hinglish for THIS action "
        "(close/shut/stop), in EITHER "
        "word order - the app name can come before OR after the verb: 'band karo', 'band kar do', "
        "'band karde', '<app> band karo', '<app> band kar do' (e.g. 'brave band kar do' means close "
        "brave, same as '<app> ko band karo'). Do NOT use open_app for any of these phrases - "
        "'band' always means close, never open, even though it names the same app you might have "
        "just opened and even without the word 'ko' in the sentence."
    ),
    parameters={
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "Name of the application to close, e.g. 'chrome' or 'notepad'.",
            }
        },
        "required": ["app_name"],
    },
)
def close_app(app_name: str) -> str:
    if _opened_entry(app_name):  # an app Vaayu opened: close exactly the windows/programs it started
        return _close_tracked(app_name)
    # No alias resolution here, deliberately - see _OPEN_APP_NAME_ALIASES'
    # docstring, close_things() matches real running process names, not
    # the registered-app-name database open_things() uses, so "vlc" alone
    # already matches "vlc.exe" correctly without help.
    with _quiet_stdout():
        app_close(app_name, match_closest=True, output=True, throw_error=True)

    # Found live: "close notepad" reported success while notepad.exe kept
    # running - AppOpener's throw_error is real now (see output=True fix
    # above) but real process termination isn't always instant, so verify
    # independently rather than trusting a non-exception alone.
    still_running = _find_matching_process_name(app_name)
    retries = 3
    while still_running and retries > 0:
        time.sleep(0.3)
        still_running = _find_matching_process_name(app_name)
        retries -= 1
    if still_running:
        raise RuntimeError(
            f"Tried to close {app_name} but it still appears to be running - "
            "it may have an unsaved-changes prompt or another dialog blocking it."
        )

    return f"Closed {app_name}."
