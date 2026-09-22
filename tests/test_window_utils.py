"""
window_utils.py's force_foreground had a real fix behind it (UWP apps like
Calculator can deny AttachThreadInput with "Access is denied", needing an
Alt-key-tap fallback) but no direct test coverage before this.
"""

from unittest.mock import patch

from assistant.tools import window_utils


def test_list_windows_filters_to_visible_titled_windows():
    def _fake_enum(callback, _extra):
        # simulate three windows: one visible+titled, one visible+untitled, one invisible
        for hwnd, visible, title in [(1, True, "Notepad"), (2, True, ""), (3, False, "Hidden")]:
            with patch.object(window_utils.win32gui, "IsWindowVisible", return_value=visible), patch.object(
                window_utils.win32gui, "GetWindowText", return_value=title
            ):
                callback(hwnd, None)

    with patch.object(window_utils.win32gui, "EnumWindows", side_effect=_fake_enum):
        result = window_utils.list_windows()

    assert result == [(1, "Notepad")]


def test_force_foreground_restores_minimized_window_first():
    with patch.object(window_utils.win32gui, "IsIconic", return_value=True), patch.object(
        window_utils.win32gui, "ShowWindow"
    ) as mock_show, patch.object(
        window_utils.win32process, "GetWindowThreadProcessId", return_value=(123, 0)
    ), patch.object(
        window_utils.win32api, "GetCurrentThreadId", return_value=123
    ), patch.object(window_utils.win32gui, "SetForegroundWindow") as mock_set_fg:
        window_utils.force_foreground(999)

    mock_show.assert_called_once_with(999, window_utils.win32con.SW_RESTORE)
    mock_set_fg.assert_called_once_with(999)


def test_force_foreground_same_thread_sets_directly_without_attach():
    with patch.object(window_utils.win32gui, "IsIconic", return_value=False), patch.object(
        window_utils.win32process, "GetWindowThreadProcessId", return_value=(123, 0)
    ), patch.object(window_utils.win32api, "GetCurrentThreadId", return_value=123), patch.object(
        window_utils.win32process, "AttachThreadInput"
    ) as mock_attach, patch.object(window_utils.win32gui, "SetForegroundWindow") as mock_set_fg:
        window_utils.force_foreground(999)

    mock_attach.assert_not_called()
    mock_set_fg.assert_called_once_with(999)


def test_force_foreground_different_thread_attaches_and_detaches():
    with patch.object(window_utils.win32gui, "IsIconic", return_value=False), patch.object(
        window_utils.win32process, "GetWindowThreadProcessId", return_value=(456, 0)
    ), patch.object(window_utils.win32api, "GetCurrentThreadId", return_value=123), patch.object(
        window_utils.win32process, "AttachThreadInput"
    ) as mock_attach, patch.object(window_utils.win32gui, "SetForegroundWindow") as mock_set_fg:
        window_utils.force_foreground(999)

    assert mock_attach.call_args_list == [
        ((123, 456, True),),
        ((123, 456, False),),
    ]
    mock_set_fg.assert_called_once_with(999)


def test_force_foreground_falls_back_to_alt_tap_when_attach_denied():
    # The real bug this guards: UWP/AppContainer apps (e.g. Windows 11
    # Calculator) can deny AttachThreadInput with "Access is denied" -
    # must fall back to the alt-key-tap trick, not just crash/give up.
    with patch.object(window_utils.win32gui, "IsIconic", return_value=False), patch.object(
        window_utils.win32process, "GetWindowThreadProcessId", return_value=(456, 0)
    ), patch.object(window_utils.win32api, "GetCurrentThreadId", return_value=123), patch.object(
        window_utils.win32process, "AttachThreadInput", side_effect=Exception("Access is denied")
    ), patch.object(window_utils, "_alt_key_tap") as mock_alt_tap, patch.object(
        window_utils.win32gui, "SetForegroundWindow"
    ) as mock_set_fg:
        window_utils.force_foreground(999)

    mock_alt_tap.assert_called_once()
    mock_set_fg.assert_called_with(999)


def test_foreground_by_title_finds_and_focuses_matching_window():
    with patch.object(
        window_utils, "list_windows", return_value=[(111, "Notepad"), (222, "Google Chrome")]
    ), patch.object(window_utils, "force_foreground") as mock_fg:
        result = window_utils.foreground_by_title("chrome")

    assert result is True
    mock_fg.assert_called_once_with(222)


def test_foreground_by_title_retries_then_gives_up(monkeypatch):
    monkeypatch.setattr(window_utils.time, "sleep", lambda *_: None)
    with patch.object(window_utils, "list_windows", return_value=[]):
        result = window_utils.foreground_by_title("nonexistent", retries=2, retry_delay=0)
    assert result is False
