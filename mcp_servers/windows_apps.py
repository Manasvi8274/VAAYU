"""
"windows-apps": Vaayu's own MCP server for working INSIDE any app - click its buttons and menus, type
into it, press its shortcuts, read what it shows.

One server for every app, not one per app: Windows exposes every app's buttons, menus and text boxes the
same way (UI Automation, the accessibility layer screen readers use), so "click Save in Notepad" and
"click Equals in Calculator" are the same tool. Controls are found by NAME (what the user says), not by
screen position, so Vaayu's small brain can drive it in one call.

Built rather than downloaded (user's rule: a free MCP first, else his own): the free Windows-MCP needs
Python 3.14 (a second Python install), sends usage telemetry, and clicks by screen coordinates.

Never: password fields, payment apps (guardian/rules.py), Windows' admin prompts.

    .venv/Scripts/python.exe -m mcp_servers.windows_apps       # stdio server (Vaayu starts it)
"""

import difflib
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil  # noqa: E402
import uiautomation as auto  # noqa: E402
from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp_types import ToolAnnotations  # noqa: E402

from guardian.rules import check_not_payment  # noqa: E402

server = MCPServer("windows-apps", instructions="Work inside any open Windows app by the names of its controls.")


def _said(fn):
    """The SDK hides an unexpected error's message ("Error executing tool app_click") - Vaayu should say
    what went wrong ("I can't see notepad open"), so every failure goes back as a ToolError."""
    import functools
    from mcp.server.mcpserver.exceptions import ToolError

    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(str(exc) or type(exc).__name__) from exc
    return run

_INTERACTIVE = {"ButtonControl", "MenuItemControl", "TabItemControl", "ListItemControl", "CheckBoxControl",
                "RadioButtonControl", "HyperlinkControl", "ComboBoxControl", "EditControl", "DocumentControl",
                "SplitButtonControl", "TreeItemControl", "MenuBarControl"}
_MAX_NODES = 600   # calibration knob: how much of a big app's tree one look walks
_MAX_DEPTH = 14
_NEVER = {"consent.exe", "credentialuibroker.exe", "lockapp.exe"}  # admin prompt, password prompt, lock screen
_KEY_NAMES = {"escape": "esc", "return": "enter", "del": "delete", "control": "ctrl", "windows": "win",
              "page up": "page up", "pgup": "page up", "pgdn": "page down"}


def _similar(a: str, b: str) -> float:
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if re.search(rf"\b{re.escape(a)}\b", b):
        return 0.95 - min(len(b) - len(a), 40) / 200  # "save" in "save as" < "save" exactly
    return difflib.SequenceMatcher(None, a, b).ratio()


def _proc(pid: int) -> str:
    try:
        return psutil.Process(pid).name().lower()
    except psutil.Error:
        return ""


def _top_windows() -> list[tuple[int, str, int]]:
    """Visible titled top-level windows: (hwnd, title, pid) - Windows' own list, near-instant (asking UI
    Automation for the same took 0.5 s)."""
    import win32gui
    import win32process
    out = []

    def add(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            out.append((hwnd, win32gui.GetWindowText(hwnd), win32process.GetWindowThreadProcessId(hwnd)[1]))
    win32gui.EnumWindows(add, None)
    return out


def _window(app: str) -> auto.Control:
    """The app's top-level window, by its title ("Untitled - Notepad") or its program's name."""
    check_not_payment(app)
    best, score = None, 0.0
    for hwnd, title, pid in _top_windows():
        proc = _proc(pid)
        if proc in _NEVER:
            continue
        parts = [title] + re.split(r"\s+[-|—]\s+", title) + [proc.removesuffix(".exe")]
        s = max(_similar(app, p) for p in parts)
        if s > score:
            best, score = hwnd, s
    if best is None or score < 0.6:
        raise ValueError(f"I can't see {app} open on the screen.")
    return auto.ControlFromHandle(best)


class _Seen:
    """One control as fetched in the batch: its name and type read once, the live control made on use."""
    def __init__(self, element, name: str, type_name: str):
        self.element, self.Name, self.ControlTypeName = element, name, type_name

    @property
    def live(self) -> auto.Control:
        return auto.Control.CreateControlFromElement(self.element)


def _all(window: auto.Control, types: set[str]) -> list[_Seen]:
    """Every named, on-screen control of these types - in ONE cross-process call (UIA's FindAllBuildCache:
    names and types fetched together). Walking the tree a property at a time took 1.8 s per click."""
    from uiautomation import uiautomation as core
    client = core._AutomationClient.instance()
    uia = client.IUIAutomation
    cache = uia.CreateCacheRequest()
    for prop in (auto.PropertyId.NameProperty, auto.PropertyId.ControlTypeProperty,
                 auto.PropertyId.IsOffscreenProperty):
        cache.AddProperty(prop)
    found = window.Element.FindAllBuildCache(client.UIAutomationCore.TreeScope_Descendants,
                                             uia.ControlViewCondition, cache)
    out = []
    for i in range(min(found.Length, _MAX_NODES * 4)):
        e = found.GetElement(i)
        type_name = auto.ControlTypeNames.get(e.CachedControlType, "")
        if type_name in types and e.CachedName and not e.CachedIsOffscreen:
            out.append(_Seen(e, _clean(e.CachedName), type_name))
    return out


def _clean(text: str) -> str:
    """Icon-font glyphs (Calculator's buttons carry U+E81C...) aren't words."""
    return re.sub(r"[-]+", "", text or "").strip()


def _controls(window: auto.Control) -> list[_Seen]:
    return _all(window, _INTERACTIVE)


def _popups(window: auto.Control) -> list[auto.Control]:
    """An open menu is its own top-level window (classic apps) - same program as the app."""
    import win32gui
    return [auto.ControlFromHandle(h) for h, _title, pid in _top_windows() + _menus()
            if pid == window.ProcessId and h != window.NativeWindowHandle and win32gui.GetClassName(h) == "#32768"]


def _menus() -> list[tuple[int, str, int]]:
    """Open classic menus: class #32768, untitled - so not in _top_windows."""
    import win32gui
    import win32process
    out = []

    def add(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetClassName(hwnd) == "#32768":
            out.append((hwnd, "", win32process.GetWindowThreadProcessId(hwnd)[1]))
    win32gui.EnumWindows(add, None)
    return out


def _find(window: auto.Control, target: str, types=None) -> auto.Control:
    pool = [c for w in [window] + _popups(window) for c in _controls(w)]
    if types:
        pool = [c for c in pool if c.ControlTypeName in types] or pool
    best = max(pool, key=lambda c: _similar(target, c.Name), default=None)
    if best is None or _similar(target, best.Name) < 0.6:
        near = ", ".join(dict.fromkeys(c.Name for c in pool if c.Name))[:300]
        raise ValueError(f"I can't find '{target}' there. I can see: {near}")
    live = best.live
    live.Name_ = best.Name  # the cleaned name, for replies
    return live


def _press(c: auto.Control) -> None:
    """Like a click, without moving the user's mouse: the control's own action."""
    for get, act in (("GetInvokePattern", "Invoke"), ("GetTogglePattern", "Toggle"),
                     ("GetSelectionItemPattern", "Select"), ("GetExpandCollapsePattern", "Expand")):
        try:
            pattern = getattr(c, get)()
            if pattern:
                getattr(pattern, act)(waitTime=0)  # uiautomation sleeps 0.5 s after each action by default
                return
        except Exception:
            continue
    c.Click(simulateMove=False)


def _front(window: auto.Control) -> None:
    from assistant.tools.window_utils import force_foreground
    hwnd = window.NativeWindowHandle
    if hwnd:
        force_foreground(hwnd)
        time.sleep(0.2)


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
@_said
def app_controls(app: str) -> str:
    """List the buttons, menus, tabs and text boxes of an open app - what can be clicked or typed into."""
    with auto.UIAutomationInitializerInThread():
        window = _window(app)
        names = list(dict.fromkeys(f"{c.Name} ({c.ControlTypeName.removesuffix('Control').lower()})"
                                   for c in _controls(window)))
        return f"{window.Name}: " + "; ".join(names[:80]) if names else f"{window.Name} shows no named controls."


@server.tool()
@_said
def app_click(app: str, target: str) -> str:
    """Click a button, menu item, tab or list item in an open app, by the name shown on it."""
    with auto.UIAutomationInitializerInThread():
        window = _window(app)
        c = _find(window, target)
        _press(c)
        return f"Clicked {c.Name_} in {window.Name}."


@server.tool()
@_said
def app_menu(app: str, path: str) -> str:
    """Open a menu path in an app, like 'File > Save As' or 'View > Zoom > Zoom in'."""
    with auto.UIAutomationInitializerInThread():
        window = _window(app)
        steps = [s.strip() for s in re.split(r"\s*(?:>|->|/|,|\bthen\b)\s*", path) if s.strip()]
        for step in steps:
            c = _find(window, step, types={"MenuItemControl", "MenuBarControl", "ButtonControl", "SplitButtonControl"})
            _press(c)
            time.sleep(0.4)  # the next menu opens
        return f"Opened {' > '.join(steps)} in {window.Name}."


def _text_box(window: auto.Control, field: str = "") -> auto.Control | None:
    if field:
        return _find(window, field, types={"EditControl", "ComboBoxControl", "DocumentControl"})
    seen = next(iter(_all(window, {"DocumentControl"}) or _all(window, {"EditControl"})), None)
    return seen.live if seen else None


@server.tool()
@_said
def app_type(app: str, text: str, field: str = "") -> str:
    """Type text into an open app - into the named box (field), or where its cursor is (a document)."""
    # Pasted, the user's clipboard put back after (type_text's way): key-by-key typing came out as
    # "Hello yyyyyu" in Windows 11 Notepad, and Hindi text can't be typed key by key at all.
    from assistant.tools.dictation import _paste_text_via_clipboard
    with auto.UIAutomationInitializerInThread():
        window = _window(app)
        box = _text_box(window, field)
        if box is not None and box.IsPassword:
            raise PermissionError("That's a password box - I never type passwords. Please type it yourself.")
        _front(window)
        if box is not None:
            try:
                box.SetFocus()
            except Exception:
                pass
        _paste_text_via_clipboard(text)
        return f"Typed it into {window.Name}" + (f" ({box.Name_})." if field and box is not None else ".")


@server.tool()
@_said
def app_keys(app: str, keys: str) -> str:
    """Press a shortcut in an open app, like 'ctrl+s', 'ctrl+b' or 'f5'."""
    import keyboard
    with auto.UIAutomationInitializerInThread():
        window = _window(app)
        combo = "+".join(_KEY_NAMES.get(k.strip().lower(), k.strip().lower()) for k in re.split(r"\s*\+\s*", keys))
        _front(window)
        keyboard.send(combo)
        return f"Pressed {combo} in {window.Name}."


@server.tool(annotations=ToolAnnotations(read_only_hint=True))
@_said
def app_read(app: str) -> str:
    """Read what an open app shows: its document's text, or the text on its screen."""
    with auto.UIAutomationInitializerInThread():
        window = _window(app)
        box = _text_box(window)
        if box is not None and not box.IsPassword:
            text = ""
            try:
                text = box.GetTextPattern().DocumentRange.GetText(4000)
            except Exception:
                try:
                    text = box.GetValuePattern().Value
                except Exception:
                    pass
            if text and text.strip():
                return f"{window.Name} says: {_clean(text)[:4000]}"
        texts = list(dict.fromkeys(c.Name for c in _all(window, {"TextControl"}) if c.Name))
        return (f"{window.Name} shows: " + " | ".join(texts)[:4000]) if texts else f"{window.Name} shows no text."


if __name__ == "__main__":
    auto.SetGlobalSearchTimeout(2)
    server.run("stdio")
