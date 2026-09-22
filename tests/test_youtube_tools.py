from unittest.mock import patch

import pytest

from assistant.tools import youtube


def _cdp_value(value):
    """Shape a fake CDP Runtime.evaluate response around `value`."""
    return {"result": {"result": {"value": value}}}


def _tab():
    return {"webSocketDebuggerUrl": "ws://fake"}


def _sequence(*values):
    """Returns a side_effect function yielding one _cdp_value(...) per
    Runtime.evaluate call, in order. Page.navigate/Input.dispatchKeyEvent
    calls are no-ops (return {}), matching that youtube.py never reads
    their return value."""
    it = iter(values)

    def _fake(ws_url, method, params=None):
        if method != "Runtime.evaluate":
            return {}
        return _cdp_value(next(it))

    return _fake


# Volume/fullscreen used to be tested here (they used to be youtube-only
# tools) - moved to test_video_player.py along with the module they now
# live in (tools/video_player.py), since they're general video-tab controls
# now, not YouTube-specific.


# --- _eval_js_retry_if_empty: cold-start page-render timing fix ---
# Found live: an empty result right after navigating to a freshly
# cold-started browser's first page load doesn't always mean "not found" -
# it can mean "not rendered yet". Reproduced directly (see youtube.py's
# comment on this function for the exact repro).


def test_find_channel_url_retries_once_then_succeeds():
    with patch.object(
        youtube, "_send_cdp_command", side_effect=_sequence(None, "https://youtube.com/@ogcrew")
    ), patch("time.sleep") as mock_sleep:
        result = youtube._find_channel_url("OG Crew", _tab())
    assert result == "https://youtube.com/@ogcrew"
    # _navigate's own wait (1.5s) plus the retry wait (2.0s) - the retry
    # call specifically is the signal that the retry path actually ran.
    assert mock_sleep.call_args_list == [((1.5,),), ((2.0,),)]


def test_find_channel_url_returns_none_when_still_empty_after_retry():
    # retries defaults to 2 now (3 total attempts) - all three must come
    # back empty to reach the real "still not found" outcome.
    with patch.object(
        youtube, "_send_cdp_command", side_effect=_sequence(None, None, None)
    ), patch("time.sleep"):
        assert youtube._find_channel_url("Nonexistent Channel XYZ", _tab()) is None


def test_find_channel_url_no_retry_needed_when_found_immediately():
    with patch.object(
        youtube, "_send_cdp_command", side_effect=_sequence("https://youtube.com/@ogcrew")
    ), patch("time.sleep") as mock_sleep:
        result = youtube._find_channel_url("OG Crew", _tab())
    assert result == "https://youtube.com/@ogcrew"
    # only _navigate's own wait - no extra retry-wait call
    mock_sleep.assert_called_once_with(1.5)


def test_find_first_video_url_retries_once_then_succeeds():
    with patch.object(
        youtube, "_send_cdp_command", side_effect=_sequence(None, "https://youtube.com/watch?v=abc")
    ), patch("time.sleep"):
        result = youtube._find_first_video_url(_tab())
    assert result == "https://youtube.com/watch?v=abc"


# --- play_video_on_youtube reuses the current tab (never opens a new one) ---


def test_play_video_on_youtube_only_uses_target_tab_navigation():
    calls = []

    def _fake_send(ws_url, method, params=None):
        calls.append(method)
        if method == "Runtime.evaluate":
            return _cdp_value("https://www.youtube.com/watch?v=abc123")
        return {}

    with patch.object(youtube, "_target_tab", return_value=_tab()), patch.object(
        youtube, "_send_cdp_command", side_effect=_fake_send
    ), patch("time.sleep"):
        result = youtube.play_video_on_youtube("fairytale music")

    assert result == "Playing on YouTube: fairytale music"
    # every navigation went through Page.navigate on the existing tab -
    # nothing here ever opens a subprocess/new browser tab.
    assert "Page.navigate" in calls


def test_play_video_on_youtube_raises_when_nothing_found():
    # _find_first_video_url retries (with a longer wait) before giving up on
    # an empty result - see _eval_js_retry_if_empty, default retries=2 (3
    # total attempts) - so this needs three None values, not one, to reach
    # the real "not found" outcome.
    with patch.object(youtube, "_target_tab", return_value=_tab()), patch.object(
        youtube, "_send_cdp_command", side_effect=_sequence(None, None, None)
    ), patch("time.sleep"):
        with pytest.raises(ValueError, match="No video found"):
            youtube.play_video_on_youtube("some obscure query")


# --- new_tab=True: real capability, not a workaround (real-usage bug hunt
# found no way to play a video in a genuine new tab at all - open_website's
# new-tab path was being used as a silent, dishonest-success fallback
# instead). Uses browser._new_tab (the real CDP endpoint), not a second
# browser process, and stays on that SAME tab for every subsequent
# navigation in the same call - not just the first one.


def test_play_video_on_youtube_new_tab_uses_cdp_new_tab_for_every_navigation():
    created_tab = {"id": "new-tab-1", "webSocketDebuggerUrl": "ws://new-tab"}
    navigated_ws_urls = []

    def _fake_send(ws_url, method, params=None):
        if method == "Page.navigate":
            navigated_ws_urls.append(ws_url)
            return {}
        if method == "Runtime.evaluate":
            return _cdp_value("https://www.youtube.com/watch?v=abc123")
        return {}

    with patch.object(youtube, "_new_tab", return_value=created_tab) as mock_new_tab, patch.object(
        youtube, "_activate_tab"
    ) as mock_activate, patch.object(youtube, "_target_tab") as mock_target_tab, patch.object(
        youtube, "_send_cdp_command", side_effect=_fake_send
    ), patch("time.sleep"):
        result = youtube.play_video_on_youtube("fairytale music", new_tab=True)

    mock_new_tab.assert_called_once_with("about:blank")
    mock_activate.assert_called_once_with("new-tab-1")
    mock_target_tab.assert_not_called()  # must not fall back to "current tab" logic at all
    assert navigated_ws_urls == ["ws://new-tab", "ws://new-tab"]  # search + the video, same new tab
    assert result == "Playing on YouTube: fairytale music (new tab)."


def test_play_video_on_youtube_default_reuses_current_tab():
    with patch.object(youtube, "_new_tab") as mock_new_tab, patch.object(
        youtube, "_target_tab", return_value=_tab()
    ), patch.object(
        youtube, "_send_cdp_command", side_effect=_sequence("https://www.youtube.com/watch?v=abc123")
    ), patch("time.sleep"):
        result = youtube.play_video_on_youtube("fairytale music")

    mock_new_tab.assert_not_called()
    assert result == "Playing on YouTube: fairytale music"


def test_play_latest_video_from_channel_new_tab_stays_on_same_tab_across_steps():
    # Channel search -> /videos page -> the video itself is 3 navigations;
    # all 3 must land on the one new tab created at the start, not drift.
    created_tab = {"id": "new-tab-2", "webSocketDebuggerUrl": "ws://new-tab-2"}
    navigated_ws_urls = []

    def _fake_send(ws_url, method, params=None):
        if method == "Page.navigate":
            navigated_ws_urls.append(ws_url)
            return {}
        if method == "Runtime.evaluate":
            return _cdp_value("https://youtube.com/@ogcrew" if "search_query" in str(params) else "https://youtube.com/watch?v=xyz")
        return {}

    with patch.object(youtube, "_new_tab", return_value=created_tab), patch.object(
        youtube, "_activate_tab"
    ), patch.object(youtube, "_target_tab") as mock_target_tab, patch.object(
        youtube, "_send_cdp_command", side_effect=_fake_send
    ), patch.object(youtube.cache, "get", return_value=None), patch.object(
        youtube.cache, "set"
    ), patch("time.sleep"):
        result = youtube.play_latest_video_from_channel("OG Crew", new_tab=True)

    mock_target_tab.assert_not_called()
    assert all(ws == "ws://new-tab-2" for ws in navigated_ws_urls)
    assert len(navigated_ws_urls) == 3
    assert result == "Playing the latest video from OG Crew in a new tab."


# --- play_latest_video_from_channel caches the channel URL lookup ---


def test_play_latest_video_from_channel_caches_on_first_lookup():
    with patch.object(youtube, "_target_tab", return_value=_tab()), patch.object(
        youtube, "_send_cdp_command",
        side_effect=_sequence("https://youtube.com/@ogcrew", "https://youtube.com/watch?v=xyz"),
    ), patch.object(youtube.cache, "get", return_value=None) as mock_get, patch.object(
        youtube.cache, "set"
    ) as mock_set, patch("time.sleep"):
        result = youtube.play_latest_video_from_channel("OG Crew")

    assert result == "Playing the latest video from OG Crew."
    mock_get.assert_called_once_with("youtube_channel", "OG Crew", youtube._CHANNEL_CACHE_MAX_AGE_SECONDS)
    mock_set.assert_called_once_with("youtube_channel", "OG Crew", "https://youtube.com/@ogcrew")


def test_play_latest_video_from_channel_skips_search_on_cache_hit():
    methods_called = []

    def _fake_send(ws_url, method, params=None):
        methods_called.append((method, (params or {}).get("expression", (params or {}).get("url"))))
        if method == "Runtime.evaluate":
            return _cdp_value("https://youtube.com/watch?v=xyz")
        return {}

    with patch.object(youtube, "_target_tab", return_value=_tab()), patch.object(
        youtube, "_send_cdp_command", side_effect=_fake_send
    ), patch.object(
        youtube.cache, "get", return_value="https://youtube.com/@ogcrew"
    ), patch.object(youtube.cache, "set") as mock_set, patch("time.sleep"):
        result = youtube.play_latest_video_from_channel("OG Crew")

    assert result == "Playing the latest video from OG Crew."
    # only one Runtime.evaluate (finding the video) - no search-page lookup at all
    eval_calls = [c for c in methods_called if c[0] == "Runtime.evaluate"]
    assert len(eval_calls) == 1
    navigated_urls = [c[1] for c in methods_called if c[0] == "Page.navigate"]
    assert any("@ogcrew" in u for u in navigated_urls)
    assert "results?search_query" not in "".join(navigated_urls)
    mock_set.assert_not_called()  # nothing new to cache - it was already a hit


def test_play_latest_video_from_channel_falls_back_to_fresh_search_when_cache_is_stale():
    with patch.object(youtube, "_target_tab", return_value=_tab()), patch.object(
        youtube, "_send_cdp_command",
        side_effect=_sequence(
            None, None, None,  # cached URL's /videos page has no video (stale) - all 3 attempts empty
            "https://youtube.com/@ogcrew-new",  # fresh search finds the real channel
            "https://youtube.com/watch?v=xyz",  # video found this time
        ),
    ), patch.object(
        youtube.cache, "get", return_value="https://youtube.com/@ogcrew-old"
    ), patch.object(youtube.cache, "set") as mock_set, patch("time.sleep"):
        result = youtube.play_latest_video_from_channel("OG Crew")

    assert result == "Playing the latest video from OG Crew."
    mock_set.assert_called_once_with("youtube_channel", "OG Crew", "https://youtube.com/@ogcrew-new")
