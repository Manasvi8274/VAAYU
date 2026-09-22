import json
from unittest.mock import MagicMock, patch

import pytest

from assistant.tools import browser

_SAMPLE_TABS = [
    {"type": "page", "id": "abc", "title": "Wikipedia", "url": "https://www.wikipedia.org/"},
    {"type": "page", "id": "def", "title": "Example Domain", "url": "https://example.com/"},
    {"type": "background_page", "id": "ghi", "title": "Some Extension", "url": "chrome-extension://x/"},
]


def _mock_response(payload: bytes):
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value.read.return_value = payload
    return mock_cm


def test_list_browser_tabs_filters_to_pages_only():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        result = browser.list_browser_tabs()
    assert result == [
        {"position": 1, "title": "Wikipedia", "url": "https://www.wikipedia.org/"},
        {"position": 2, "title": "Example Domain", "url": "https://example.com/"},
    ]


def test_switch_to_browser_tab_matches_by_title():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        result = browser.switch_to_browser_tab(query="wikipedia")
    assert result == "Switched to tab: Wikipedia"


def test_switch_to_browser_tab_raises_when_no_match():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        with pytest.raises(ValueError):
            browser.switch_to_browser_tab(query="nonexistent")


def test_switch_to_browser_tab_by_position():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        result = browser.switch_to_browser_tab(position=2)
    assert result == "Switched to tab: Example Domain"


def test_switch_to_browser_tab_position_out_of_range():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        with pytest.raises(ValueError, match="only 2 tab"):
            browser.switch_to_browser_tab(position=5)


def test_switch_to_browser_tab_no_args_raises():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        with pytest.raises(ValueError, match="Need either"):
            browser.switch_to_browser_tab()


def test_close_browser_tab_by_position():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        result = browser.close_browser_tab(position=1)
    assert result == "Closed tab: Wikipedia"


def test_close_browser_tab_by_query():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        result = browser.close_browser_tab(query="example")
    assert result == "Closed tab: Example Domain"


def test_close_browser_tab_raises_when_no_match():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        with pytest.raises(ValueError):
            browser.close_browser_tab(query="nonexistent")


def test_close_browser_tab_position_zero_is_out_of_range():
    with patch("urllib.request.urlopen", return_value=_mock_response(json.dumps(_SAMPLE_TABS).encode())):
        with pytest.raises(ValueError):
            browser.close_browser_tab(position=0)


def test_connection_error_raises_helpful_message():
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(ConnectionError, match="remote debugging port"):
            browser.list_browser_tabs()
