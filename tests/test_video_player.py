"""
tools/video_player.py: site-agnostic video controls (volume, fullscreen,
seek, play/pause) that work on any browser tab's <video> element, plus the
multi-video disambiguation - if more than one tab has a video actively
playing, these must ask which one instead of silently guessing.

Mocking note: unlike the old YouTube-only tools (which always targeted
"whichever tab is first in the list"), _resolve_video_tab checks every open
tab's play state first when no tab_hint is given - so every test here mocks
_list_tabs (not just _target_tab) and accounts for that extra "is this tab
playing" probe in the eval sequence.
"""

from unittest.mock import patch

import pytest

from assistant.tools import video_player


def _tab(tab_id="tab1", title="Video Tab", url="https://example.com/watch"):
    return {
        "id": tab_id,
        "type": "page",
        "title": title,
        "url": url,
        "webSocketDebuggerUrl": f"ws://fake/{tab_id}",
    }


def _cdp_value(value):
    return {"result": {"result": {"value": value}}}


def _eval_sequence(*values):
    it = iter(values)

    def _fake(ws_url, method, params=None):
        if method != "Runtime.evaluate":
            return {}
        return _cdp_value(next(it))

    return _fake


def _single_playing_tab_mocks(*eval_values, tab=None):
    """Standard setup: exactly one tab, it's playing (so disambiguation
    resolves to it immediately), followed by the tool's own eval sequence."""
    tab = tab or _tab()
    return patch.object(video_player, "_list_tabs", return_value=[tab]), patch.object(
        video_player, "_send_cdp_command", side_effect=_eval_sequence(True, *eval_values)
    )


# ============================================================================
# set_video_volume
# ============================================================================


def test_set_video_volume_up_default_amount():
    p1, p2 = _single_playing_tab_mocks(0.5, None, 0.7)
    with p1, p2:
        result = video_player.set_video_volume(direction="up")
    assert result == "Turned the video volume up to 70% (Video Tab)."


def test_set_video_volume_down_custom_amount():
    p1, p2 = _single_playing_tab_mocks(0.5, None, 0.4)
    with p1, p2:
        result = video_player.set_video_volume(direction="down", amount=10)
    assert result == "Turned the video volume down to 40% (Video Tab)."


def test_set_video_volume_absolute_level():
    p1, p2 = _single_playing_tab_mocks(0.3, None, 0.6)
    with p1, p2:
        result = video_player.set_video_volume(level=60)
    assert result == "Set the video volume to 60% (Video Tab)."


def test_set_video_volume_mute():
    p1, p2 = _single_playing_tab_mocks(0.5, None, True)
    with p1, p2:
        result = video_player.set_video_volume(direction="mute")
    assert result == "Muted the video (Video Tab)."


def test_set_video_volume_unmute():
    p1, p2 = _single_playing_tab_mocks(0.5, None, False)
    with p1, p2:
        result = video_player.set_video_volume(direction="unmute")
    assert result == "Unmuted the video (Video Tab)."


def test_set_video_volume_no_video_on_tab_raises():
    p1, p2 = _single_playing_tab_mocks(None)
    with p1, p2:
        with pytest.raises(RuntimeError, match="No playing video"):
            video_player.set_video_volume(direction="up")


def test_set_video_volume_no_args_raises():
    p1, p2 = _single_playing_tab_mocks(0.5)
    with p1, p2:
        with pytest.raises(ValueError, match="Give either"):
            video_player.set_video_volume()


def test_set_video_volume_verifies_and_raises_if_unchanged():
    p1, p2 = _single_playing_tab_mocks(0.5, None, 0.5)
    with p1, p2:
        with pytest.raises(RuntimeError, match="didn't take effect"):
            video_player.set_video_volume(direction="up")


def test_set_video_volume_with_tab_hint_skips_disambiguation():
    tabs = [_tab("t1", "YouTube - Music"), _tab("t2", "Hotstar - Series")]
    with patch.object(video_player, "_list_tabs", return_value=tabs), patch.object(
        video_player, "_send_cdp_command", side_effect=_eval_sequence(0.5, None, 0.7)
    ) as mock_send:
        result = video_player.set_video_volume(direction="up", tab_hint="hotstar")
    assert "Hotstar - Series" in result
    # only tab2's websocket was ever used - the hint resolved directly, no
    # "which tab is playing" probe against either tab.
    urls_used = {c.args[0] for c in mock_send.call_args_list}
    assert urls_used == {"ws://fake/t2"}


def test_set_video_volume_unknown_tab_hint_raises():
    with patch.object(video_player, "_list_tabs", return_value=[_tab()]):
        with pytest.raises(ValueError, match="No open tab found"):
            video_player.set_video_volume(direction="up", tab_hint="nonexistent site")


# ============================================================================
# set_video_fullscreen
# ============================================================================


def test_set_video_fullscreen_enters_when_not_fullscreen():
    p1, p2 = _single_playing_tab_mocks(False, True)
    with p1, p2:
        result = video_player.set_video_fullscreen(True)
    assert result == "Entered fullscreen."


def test_set_video_fullscreen_exits_when_fullscreen():
    p1, p2 = _single_playing_tab_mocks(True, False)
    with p1, p2:
        result = video_player.set_video_fullscreen(False)
    assert result == "Exited fullscreen."


def test_set_video_fullscreen_already_in_state_is_a_noop():
    p1, p2 = _single_playing_tab_mocks(True)
    with p1, p2:
        result = video_player.set_video_fullscreen(True)
    assert result == "Already in fullscreen."


def test_set_video_fullscreen_raises_if_toggle_had_no_effect():
    # Three values: initial check, first post-dispatch check, and the one
    # retry check added after a live one-off failure - all three still
    # showing no change is what a genuine (not just slow) failure looks like.
    p1, p2 = _single_playing_tab_mocks(False, False, False)
    with p1, p2, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="didn't change"):
            video_player.set_video_fullscreen(True)


def test_set_video_fullscreen_succeeds_on_retry_after_slow_transition():
    # The retry itself succeeding - not just the final failure path - needs
    # its own coverage: initial check False, first post-dispatch check still
    # False (looked like a failure), retry check now True (it just needed
    # more time).
    p1, p2 = _single_playing_tab_mocks(False, False, True)
    with p1, p2, patch("time.sleep"):
        result = video_player.set_video_fullscreen(True)
    assert result == "Entered fullscreen."


# ============================================================================
# seek_video
# ============================================================================


def test_seek_video_forward_default_seconds():
    p1, p2 = _single_playing_tab_mocks(30.0, None, 40.0)
    with p1, p2, patch("time.sleep"):
        result = video_player.seek_video(direction="forward")
    assert result == "Skipped forward 10 seconds (Video Tab)."


def test_seek_video_backward_custom_seconds():
    p1, p2 = _single_playing_tab_mocks(30.0, None, 5.0)
    with p1, p2, patch("time.sleep"):
        result = video_player.seek_video(direction="backward", seconds=25)
    assert result == "Skipped backward 25 seconds (Video Tab)."


def test_seek_video_backward_clamps_at_zero():
    p1, p2 = _single_playing_tab_mocks(5.0, None, 0.0)
    with p1, p2, patch("time.sleep"):
        result = video_player.seek_video(direction="backward", seconds=30)
    assert result == "Skipped backward 30 seconds (Video Tab)."


def test_seek_video_no_video_raises():
    p1, p2 = _single_playing_tab_mocks(None)
    with p1, p2, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="No playing video"):
            video_player.seek_video(direction="forward")


def test_seek_video_raises_if_no_movement_detected():
    # currentTime read back unchanged (or only reflecting normal playback
    # drift, not a real seek) - forward seek must show actual > current.
    p1, p2 = _single_playing_tab_mocks(30.0, None, 30.0)
    with p1, p2, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="didn't take effect"):
            video_player.seek_video(direction="forward")


def test_seek_video_backward_raises_if_it_actually_moved_forward():
    # Simulates a failed backward seek where only normal playback drift
    # occurred (time moved slightly forward, not backward).
    p1, p2 = _single_playing_tab_mocks(30.0, None, 30.2)
    with p1, p2, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="didn't take effect"):
            video_player.seek_video(direction="backward")


# ============================================================================
# control_video_playback
# ============================================================================


def test_control_video_playback_pause():
    p1, p2 = _single_playing_tab_mocks(False, None, True)  # not paused -> pause() -> now paused
    with p1, p2, patch("time.sleep"):
        result = video_player.control_video_playback("pause")
    assert result == "Paused the video (Video Tab)."


def test_control_video_playback_play():
    p1, p2 = _single_playing_tab_mocks(True, None, False)  # paused -> play() -> now not paused
    with p1, p2, patch("time.sleep"):
        result = video_player.control_video_playback("play")
    assert result == "Playing the video (Video Tab)."


def test_control_video_playback_toggle_from_playing_pauses():
    p1, p2 = _single_playing_tab_mocks(False, None, True)
    with p1, p2, patch("time.sleep"):
        result = video_player.control_video_playback("toggle")
    assert result == "Paused the video (Video Tab)."


def test_control_video_playback_no_video_raises():
    p1, p2 = _single_playing_tab_mocks(None)
    with p1, p2, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="No video found"):
            video_player.control_video_playback("play")


def test_control_video_playback_raises_if_action_had_no_effect():
    p1, p2 = _single_playing_tab_mocks(False, None, False)  # asked to pause, still shows not paused
    with p1, p2, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="didn't take effect"):
            video_player.control_video_playback("pause")


# ============================================================================
# Multi-video disambiguation (the core of this feature)
# ============================================================================


def test_resolve_video_tab_raises_when_two_tabs_playing_at_once():
    tabs = [_tab("t1", "YouTube - Some Music"), _tab("t2", "Hotstar - Some Series")]
    with patch.object(video_player, "_list_tabs", return_value=tabs), patch.object(
        video_player, "_send_cdp_command", side_effect=_eval_sequence(True, True)
    ):
        with pytest.raises(ValueError, match="More than one video is playing"):
            video_player.set_video_volume(direction="up")


def test_resolve_video_tab_error_names_every_playing_tab():
    tabs = [_tab("t1", "YouTube - Some Music"), _tab("t2", "Hotstar - Some Series"), _tab("t3", "Netflix - A Movie")]
    with patch.object(video_player, "_list_tabs", return_value=tabs), patch.object(
        video_player, "_send_cdp_command", side_effect=_eval_sequence(True, True, True)
    ):
        with pytest.raises(ValueError) as exc_info:
            video_player.control_video_playback("pause")
    message = str(exc_info.value)
    assert "YouTube - Some Music" in message
    assert "Hotstar - Some Series" in message
    assert "Netflix - A Movie" in message


def test_resolve_video_tab_only_one_playing_among_several_open_resolves_to_it():
    tabs = [_tab("t1", "Just a docs page"), _tab("t2", "Hotstar - Some Series")]
    # t1 not playing (False), t2 playing (True), then the volume-tool's own calls
    with patch.object(video_player, "_list_tabs", return_value=tabs), patch.object(
        video_player, "_send_cdp_command", side_effect=_eval_sequence(False, True, 0.5, None, 0.7)
    ):
        result = video_player.set_video_volume(direction="up")
    assert "Hotstar - Some Series" in result


def test_resolve_video_tab_none_playing_falls_back_to_most_recent_tab():
    tabs = [_tab("t1", "Most Recent Tab"), _tab("t2", "Other Tab")]
    with patch.object(video_player, "_list_tabs", return_value=tabs), patch.object(
        video_player, "_send_cdp_command", side_effect=_eval_sequence(False, False, 0.5, None, 0.7)
    ):
        result = video_player.set_video_volume(direction="up")
    assert "Most Recent Tab" in result


def test_resolve_video_tab_no_tabs_at_all_raises_connection_error():
    with patch.object(video_player, "_list_tabs", return_value=[]):
        with pytest.raises(ConnectionError, match="remote-debugging-port"):
            video_player.set_video_volume(direction="up")


def test_tabs_with_playing_video_skips_tabs_that_error_on_eval():
    # A tab that can't be evaluated (e.g. an internal chrome:// page) must
    # not crash the whole disambiguation scan - just isn't a candidate.
    tabs = [_tab("t1", "Broken Tab"), _tab("t2", "Working Tab - Playing")]

    def _fake(ws_url, method, params=None):
        if "t1" in ws_url:
            raise ConnectionError("can't evaluate this tab")
        if method != "Runtime.evaluate":
            return {}
        return _cdp_value(True)

    with patch.object(video_player, "_list_tabs", return_value=tabs), patch.object(
        video_player, "_send_cdp_command", side_effect=_fake
    ):
        result = video_player._tabs_with_playing_video()
    assert [t["id"] for t in result] == ["t2"]
