from unittest.mock import patch

import pytest
import win32con

from assistant.tools import windows


def test_list_open_windows_returns_titles_only():
    with patch.object(windows, "list_windows", return_value=[(111, "Notepad"), (222, "Calculator")]):
        result = windows.list_open_windows()
    assert result == ["Notepad", "Calculator"]


def test_switch_to_window_by_title():
    with patch.object(
        windows, "list_windows", return_value=[(111, "Notepad"), (222, "Calculator")]
    ), patch.object(windows, "force_foreground") as mock_fg:
        result = windows.switch_to_window("calc")
    mock_fg.assert_called_once_with(222)
    assert result == "Switched to window: Calculator"


def test_switch_to_window_raises_when_no_match():
    with patch.object(windows, "list_windows", return_value=[(111, "Notepad")]):
        with pytest.raises(ValueError):
            windows.switch_to_window("nonexistent")


def test_minimize_window_by_title():
    with patch.object(
        windows, "list_windows", return_value=[(111, "Notepad"), (222, "Calculator")]
    ), patch.object(windows.win32gui, "ShowWindow") as mock_show:
        result = windows.minimize_window("notepad")
    mock_show.assert_called_once_with(111, win32con.SW_MINIMIZE)
    assert result == "Minimized: Notepad"


def test_minimize_window_no_title_uses_foreground_window():
    with patch.object(windows.win32gui, "GetForegroundWindow", return_value=999), patch.object(
        windows.win32gui, "GetWindowText", return_value="Brave"
    ), patch.object(windows.win32gui, "ShowWindow") as mock_show:
        result = windows.minimize_window()
    mock_show.assert_called_once_with(999, win32con.SW_MINIMIZE)
    assert result == "Minimized: Brave"


def test_minimize_window_raises_when_no_match():
    with patch.object(windows, "list_windows", return_value=[(111, "Notepad")]):
        with pytest.raises(ValueError):
            windows.minimize_window("nonexistent")


def test_restore_window_by_title():
    with patch.object(
        windows, "list_windows", return_value=[(111, "Notepad")]
    ), patch.object(windows, "force_foreground") as mock_fg:
        result = windows.restore_window("notepad")
    mock_fg.assert_called_once_with(111)
    assert result == "Restored: Notepad"


def test_restore_window_raises_when_no_match():
    with patch.object(windows, "list_windows", return_value=[]):
        with pytest.raises(ValueError):
            windows.restore_window("nonexistent")
