"""
browser_state.py resolves which browser open_website should target - it
fixed a real reported bug ("open brave" then "open youtube" silently opened
Chrome instead) but had zero direct tests of its resolution order before
this, only indirect coverage through web.py's mocked-at-the-boundary tests.
"""

from unittest.mock import patch

from assistant.tools import browser_state


_REGISTERED = {
    "brave": "C:\\Brave\\brave.exe",
    "google chrome": "C:\\Chrome\\chrome.exe",
}


def test_open_url_prefers_last_opened_app_when_it_matches_a_registered_browser():
    browser_state._last_browser_name = "brave"
    with patch.object(
        browser_state, "_enumerate_registered_browsers", return_value=_REGISTERED
    ), patch.object(browser_state, "subprocess") as mock_subprocess, patch.object(
        browser_state, "foreground_by_title"
    ):
        result = browser_state.open_url_in_last_browser("https://youtube.com")

    assert result is True
    mock_subprocess.Popen.assert_called_once_with(["C:\\Brave\\brave.exe", "https://youtube.com"])


def test_open_url_falls_back_to_foreground_browser_when_no_last_opened_app():
    browser_state._last_browser_name = None
    with patch.object(
        browser_state, "_enumerate_registered_browsers", return_value=_REGISTERED
    ), patch.object(
        browser_state, "_foreground_browser_exe", return_value="C:\\Chrome\\chrome.exe"
    ), patch.object(browser_state, "subprocess") as mock_subprocess, patch.object(
        browser_state, "foreground_by_title"
    ):
        result = browser_state.open_url_in_last_browser("https://example.com")

    assert result is True
    mock_subprocess.Popen.assert_called_once_with(["C:\\Chrome\\chrome.exe", "https://example.com"])


def test_open_url_falls_back_to_any_running_registered_browser():
    browser_state._last_browser_name = None
    with patch.object(
        browser_state, "_enumerate_registered_browsers", return_value=_REGISTERED
    ), patch.object(browser_state, "_foreground_browser_exe", return_value=None), patch.object(
        browser_state, "_running_browser_exe", return_value="C:\\Brave\\brave.exe"
    ), patch.object(browser_state, "subprocess") as mock_subprocess, patch.object(
        browser_state, "foreground_by_title"
    ):
        result = browser_state.open_url_in_last_browser("https://example.com")

    assert result is True
    mock_subprocess.Popen.assert_called_once_with(["C:\\Brave\\brave.exe", "https://example.com"])


def test_open_url_returns_false_when_nothing_registered():
    browser_state._last_browser_name = None
    with patch.object(browser_state, "_enumerate_registered_browsers", return_value={}):
        result = browser_state.open_url_in_last_browser("https://example.com")
    assert result is False


def test_open_url_returns_false_when_no_browser_resolvable():
    browser_state._last_browser_name = None
    with patch.object(
        browser_state, "_enumerate_registered_browsers", return_value=_REGISTERED
    ), patch.object(browser_state, "_foreground_browser_exe", return_value=None), patch.object(
        browser_state, "_running_browser_exe", return_value=None
    ):
        result = browser_state.open_url_in_last_browser("https://example.com")
    assert result is False


def test_note_app_opened_records_lowercased_name():
    browser_state.note_app_opened("  Brave  ")
    assert browser_state._last_browser_name == "brave"


def test_open_url_last_opened_app_not_a_browser_falls_through():
    # e.g. user just opened Notepad - "notepad" isn't a registered browser,
    # so resolution must fall through to foreground/running instead of
    # silently failing or matching something wrong.
    browser_state._last_browser_name = "notepad"
    with patch.object(
        browser_state, "_enumerate_registered_browsers", return_value=_REGISTERED
    ), patch.object(
        browser_state, "_foreground_browser_exe", return_value="C:\\Chrome\\chrome.exe"
    ), patch.object(browser_state, "subprocess") as mock_subprocess, patch.object(
        browser_state, "foreground_by_title"
    ):
        result = browser_state.open_url_in_last_browser("https://example.com")

    assert result is True
    mock_subprocess.Popen.assert_called_once_with(["C:\\Chrome\\chrome.exe", "https://example.com"])
