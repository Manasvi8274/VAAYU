"""
browser_automation.py (browse_to/read_page/click_on_page) had zero direct
test coverage before this, despite being the generic fallback the system
prompt tells the model to use for "anything else that needs clicking through
a website" - exactly the kind of open-ended browser task this whole project
is meant to handle.
"""

import json
from unittest.mock import patch

import pytest

from assistant.tools import browser_automation


def _tab():
    return {"webSocketDebuggerUrl": "ws://fake"}


def _cdp_value(value):
    return {"result": {"result": {"value": value}}}


# --- _target_tab ---


def test_target_tab_picks_first_listed_tab():
    tabs = [{"id": "a"}, {"id": "b"}]
    with patch.object(browser_automation, "_list_tabs", return_value=tabs):
        assert browser_automation._target_tab() == {"id": "a"}


def test_target_tab_raises_when_no_tabs_open():
    with patch.object(browser_automation, "_list_tabs", return_value=[]):
        with pytest.raises(ConnectionError, match="remote-debugging-port"):
            browser_automation._target_tab()


# --- browse_to ---


def test_browse_to_navigates_current_tab():
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command"
    ) as mock_send:
        result = browser_automation.browse_to("example.com")
    mock_send.assert_called_once_with("ws://fake", "Page.navigate", {"url": "https://example.com"})
    assert result == "Navigated to https://example.com."


def test_browse_to_leaves_full_urls_unmodified():
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command"
    ) as mock_send:
        browser_automation.browse_to("http://example.com/page")
    assert mock_send.call_args.args[2]["url"] == "http://example.com/page"


# --- read_page ---


def test_read_page_parses_text_clickable_and_input_elements():
    payload = json.dumps({
        "text": "Welcome to the site",
        "clickable": ["Sign in", "Learn more"],
        "inputs": ["Mobile number [text field]"],
    })
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(payload)
    ):
        result = browser_automation.read_page()
    assert result == {
        "visible_text": "Welcome to the site",
        "clickable_elements": ["Sign in", "Learn more"],
        "input_fields": ["Mobile number [text field]"],
    }


def test_read_page_raises_when_nothing_returned():
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(None)
    ):
        with pytest.raises(RuntimeError, match="Could not read"):
            browser_automation.read_page()


# --- click_on_page ---


def test_click_on_page_returns_success_message_when_found():
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(True)
    ):
        result = browser_automation.click_on_page("Sign in")
    assert result == "Clicked: Sign in"


def test_click_on_page_raises_when_nothing_matches():
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(False)
    ):
        with pytest.raises(ValueError, match="No clickable element"):
            browser_automation.click_on_page("Nonexistent Button")


# --- focus_input_field ---
# Built directly from a real live-testing finding: a real Hotstar login
# field had no placeholder/name/id text at all (only a `title` attribute
# and matchable parent text), and click_on_page's a/button-only selector
# couldn't reach it regardless - confirming this needed to be a genuinely
# separate tool/selector, not an extension of click_on_page.


def test_focus_input_field_returns_success_message_when_found():
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(True)
    ):
        result = browser_automation.focus_input_field("mobile number")
    assert result == "Focused: mobile number"


def test_focus_input_field_raises_when_nothing_matches():
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(False)
    ):
        with pytest.raises(ValueError, match="No input field found"):
            browser_automation.focus_input_field("nonexistent field")


def test_focus_input_field_embeds_search_text_safely_via_json_dumps():
    tricky_text = 'field with "quotes" and \\backslash\\'
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(True)
    ) as mock_send:
        browser_automation.focus_input_field(tricky_text)
    expression = mock_send.call_args.args[2]["expression"]
    assert json.dumps(tricky_text) in expression
