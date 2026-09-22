from unittest.mock import patch

from assistant.tools import web


def _no_cdp_browser():
    """Patches _new_tab to behave like browser.py's real _new_tab does when
    no CDP-debugged browser is reachable at all - raises ConnectionError -
    so tests can exercise the fallback-to-browser_state path without a real
    (slow) network attempt against a closed port 9222."""
    return patch.object(web, "_new_tab", side_effect=ConnectionError("no debugged browser"))


def test_open_website_url_opens_in_last_browser():
    with _no_cdp_browser(), patch.object(
        web.browser_state, "open_url_in_last_browser", return_value=True
    ) as mock_open:
        result = web.open_website("youtube.com")
    mock_open.assert_called_once_with("https://youtube.com", foreground=True)
    assert result == "Opened https://youtube.com in the browser."


def test_open_website_plain_query_runs_as_search():
    with _no_cdp_browser(), patch.object(
        web.browser_state, "open_url_in_last_browser", return_value=True
    ) as mock_open:
        result = web.open_website("best pizza near me")
    called_url = mock_open.call_args.args[0]
    assert called_url.startswith("https://www.google.com/search?q=")
    assert result == "Searched the web for: best pizza near me."


def test_open_website_falls_back_to_default_browser():
    with _no_cdp_browser(), patch.object(
        web.browser_state, "open_url_in_last_browser", return_value=False
    ), patch.object(web.webbrowser, "open") as mock_webopen:
        web.open_website("example.com")
    mock_webopen.assert_called_once_with("https://example.com")


def test_open_website_moviesmod_style_site_suffix_becomes_direct_domain():
    with _no_cdp_browser(), patch.object(
        web.browser_state, "open_url_in_last_browser", return_value=True
    ) as mock_open:
        result = web.open_website("moviesmod website")
    mock_open.assert_called_once_with("https://moviesmod.com", foreground=True)
    assert result == "Opened https://moviesmod.com in the browser."


def test_open_website_site_suffix_with_existing_domain_unaffected():
    with _no_cdp_browser(), patch.object(
        web.browser_state, "open_url_in_last_browser", return_value=True
    ) as mock_open:
        web.open_website("moviesmod.org website")
    mock_open.assert_called_once_with("https://moviesmod.org", foreground=True)


def test_open_website_site_suffix_with_multi_word_name_falls_back_to_search():
    # "the movies mod website" -> stripped candidate "the movies mod" has
    # spaces, so it's ambiguous - must not guess a domain, fall back to search.
    with _no_cdp_browser(), patch.object(
        web.browser_state, "open_url_in_last_browser", return_value=True
    ) as mock_open:
        web.open_website("the movies mod website")
    called_url = mock_open.call_args.args[0]
    assert called_url.startswith("https://www.google.com/search?q=")


def test_open_website_new_tab_prefers_cdp_over_spawning_a_second_process():
    # The core fix from the real-usage bug hunt: when a CDP-debugged browser
    # is already reachable, open_website must create the new tab through
    # CDP (browser._new_tab), never through browser_state's Popen-based
    # approach - spawning a second process against the same profile is what
    # crashed the whole browser / silently lost tabs, found live.
    with patch.object(web, "_new_tab", return_value={"id": "tab-123"}) as mock_new_tab, patch.object(
        web, "_activate_tab"
    ) as mock_activate, patch.object(web.browser_state, "open_url_in_last_browser") as mock_popen_path:
        result = web.open_website("youtube.com")
    mock_new_tab.assert_called_once_with("https://youtube.com")
    mock_activate.assert_called_once_with("tab-123")
    mock_popen_path.assert_not_called()
    assert result == "Opened https://youtube.com in the browser."


def test_open_website_new_tab_background_skips_activation():
    with patch.object(web, "_new_tab", return_value={"id": "tab-123"}), patch.object(
        web, "_activate_tab"
    ) as mock_activate:
        result = web.open_website("youtube.com", foreground=False)
    mock_activate.assert_not_called()
    assert result.endswith("in the background.")


def test_open_website_new_tab_falls_back_when_cdp_unreachable():
    # No CDP-debugged browser at all -> must fall back to the old
    # browser_state path rather than raising ConnectionError to the caller.
    with _no_cdp_browser(), patch.object(
        web.browser_state, "open_url_in_last_browser", return_value=True
    ) as mock_open:
        result = web.open_website("youtube.com")
    mock_open.assert_called_once_with("https://youtube.com", foreground=True)
    assert result == "Opened https://youtube.com in the browser."


def test_open_website_same_tab_navigates_current_tab_via_cdp():
    fake_tab = {"webSocketDebuggerUrl": "ws://fake"}
    with patch.object(web, "_target_tab", return_value=fake_tab), patch.object(
        web, "_send_cdp_command"
    ) as mock_send, patch.object(web.browser_state, "open_url_in_last_browser") as mock_open_new:
        result = web.open_website("youtube.com", same_tab=True)
    mock_send.assert_called_once_with(
        "ws://fake", "Page.navigate", {"url": "https://youtube.com"}
    )
    mock_open_new.assert_not_called()  # must not also spawn a new tab/process
    assert result == "Opened https://youtube.com in the browser (same tab)."


def test_open_website_same_tab_with_search_query():
    fake_tab = {"webSocketDebuggerUrl": "ws://fake"}
    with patch.object(web, "_target_tab", return_value=fake_tab), patch.object(
        web, "_send_cdp_command"
    ) as mock_send:
        result = web.open_website("fairytale music lyrics", same_tab=True)
    navigated_url = mock_send.call_args.args[2]["url"]
    assert navigated_url.startswith("https://www.google.com/search?q=")
    assert result.endswith("(same tab).")
