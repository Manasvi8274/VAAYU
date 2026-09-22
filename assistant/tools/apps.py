import contextlib
import io
import time

import psutil
from AppOpener import close as app_close
from AppOpener import open as app_open

from assistant.tools import browser_state
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
    resolved_name = _resolve_open_app_name(app_name)
    with _quiet_stdout():
        app_open(resolved_name, match_closest=True, output=True, throw_error=True)
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
