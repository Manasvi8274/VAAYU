from unittest.mock import patch

import pywintypes
import pytest

from assistant.tools import dictation, media


# --- dictation.py: clipboard retry (real, live-found Windows contention bug) ---


def _clipboard_error():
    return pywintypes.error(6, "SetClipboardData", "The handle is invalid.")


def test_set_clipboard_text_retries_and_succeeds_after_transient_failures():
    # Fails twice (matching what live testing actually showed - intermittent,
    # not permanent) then succeeds on the third attempt.
    call_count = {"n": 0}

    def _flaky_set(fmt, data):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise _clipboard_error()

    with patch.object(dictation.win32clipboard, "OpenClipboard"), patch.object(
        dictation.win32clipboard, "CloseClipboard"
    ), patch.object(dictation.win32clipboard, "EmptyClipboard"), patch.object(
        dictation.win32clipboard, "SetClipboardData", side_effect=_flaky_set
    ), patch("time.sleep"):
        dictation._set_clipboard_text("hello")

    assert call_count["n"] == 3


def test_set_clipboard_text_raises_after_exhausting_all_retries():
    with patch.object(dictation.win32clipboard, "OpenClipboard"), patch.object(
        dictation.win32clipboard, "CloseClipboard"
    ), patch.object(dictation.win32clipboard, "EmptyClipboard"), patch.object(
        dictation.win32clipboard, "SetClipboardData", side_effect=_clipboard_error()
    ), patch("time.sleep") as mock_sleep:
        with pytest.raises(pywintypes.error):
            dictation._set_clipboard_text("hello")

    assert mock_sleep.call_count == dictation._CLIPBOARD_RETRIES


def test_set_clipboard_text_still_closes_clipboard_on_every_failed_attempt():
    # A leaked OpenClipboard (never closed) would make every subsequent
    # attempt - by this call or any other process - fail too, which is
    # exactly the kind of thing that turns one transient failure into a
    # persistent one.
    with patch.object(dictation.win32clipboard, "OpenClipboard") as mock_open, patch.object(
        dictation.win32clipboard, "CloseClipboard"
    ) as mock_close, patch.object(dictation.win32clipboard, "EmptyClipboard"), patch.object(
        dictation.win32clipboard, "SetClipboardData", side_effect=_clipboard_error()
    ), patch("time.sleep"):
        with pytest.raises(pywintypes.error):
            dictation._set_clipboard_text("hello")

    assert mock_open.call_count == mock_close.call_count == dictation._CLIPBOARD_RETRIES


def test_get_clipboard_text_retries_on_transient_failure():
    call_count = {"n": 0}

    def _flaky_is_available(fmt):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise _clipboard_error()
        return True

    with patch.object(dictation.win32clipboard, "OpenClipboard"), patch.object(
        dictation.win32clipboard, "CloseClipboard"
    ), patch.object(
        dictation.win32clipboard, "IsClipboardFormatAvailable", side_effect=_flaky_is_available
    ), patch.object(dictation.win32clipboard, "GetClipboardData", return_value="clipboard content"), patch(
        "time.sleep"
    ):
        result = dictation._get_clipboard_text()

    assert result == "clipboard content"
    assert call_count["n"] == 2


# --- dictation.py ---


def test_type_text_sets_clipboard_and_sends_paste():
    with patch.object(dictation, "_get_clipboard_text", return_value="old content"), patch.object(
        dictation, "_set_clipboard_text"
    ) as mock_set, patch.object(dictation.keyboard, "send") as mock_send, patch("time.sleep"):
        result = dictation.type_text("hello world")

    mock_send.assert_called_once_with("ctrl+v")
    assert result == "Typed: hello world"
    # sets the new text first, then restores the original clipboard content
    assert mock_set.call_args_list[0].args == ("hello world",)
    assert mock_set.call_args_list[-1].args == ("old content",)


def test_type_text_restores_clipboard_even_if_paste_fails():
    with patch.object(dictation, "_get_clipboard_text", return_value="old content"), patch.object(
        dictation, "_set_clipboard_text"
    ) as mock_set, patch.object(dictation.keyboard, "send", side_effect=RuntimeError("boom")), patch(
        "time.sleep"
    ):
        with pytest.raises(RuntimeError):
            dictation.type_text("hello")

    assert mock_set.call_args_list[-1].args == ("old content",)


def test_type_sensitive_field_types_the_given_value():
    with patch.object(dictation, "_get_clipboard_text", return_value="old content"), patch.object(
        dictation, "_set_clipboard_text"
    ) as mock_set, patch.object(dictation.keyboard, "send") as mock_send, patch("time.sleep"):
        result = dictation.type_sensitive_field("9876543210", "phone number field")

    mock_send.assert_called_once_with("ctrl+v")
    assert result == "Entered the phone number field."
    assert mock_set.call_args_list[0].args == ("9876543210",)
    assert mock_set.call_args_list[-1].args == ("old content",)  # clipboard restored


def test_type_sensitive_field_restores_clipboard_even_if_paste_fails():
    with patch.object(dictation, "_get_clipboard_text", return_value="old content"), patch.object(
        dictation, "_set_clipboard_text"
    ) as mock_set, patch.object(dictation.keyboard, "send", side_effect=RuntimeError("boom")), patch(
        "time.sleep"
    ):
        with pytest.raises(RuntimeError):
            dictation.type_sensitive_field("123456", "OTP field")

    assert mock_set.call_args_list[-1].args == ("old content",)


def test_type_sensitive_field_requires_confirmation():
    from assistant.tools.registry import registry as real_registry

    assert real_registry.requires_confirmation("type_sensitive_field") is True


def test_type_sensitive_field_confirmation_prompt_quotes_value_and_field():
    from assistant.tools.registry import registry as real_registry

    prompt = real_registry.get_confirmation_prompt(
        "type_sensitive_field", {"value": "123456", "field_description": "OTP field"}
    )
    assert "123456" in prompt
    assert "OTP field" in prompt


def test_type_text_does_not_require_confirmation():
    # type_text is the generic/non-sensitive dictation tool - must stay
    # ungated, or ordinary dictation would prompt for confirmation constantly.
    from assistant.tools.registry import registry as real_registry

    assert real_registry.requires_confirmation("type_text") is False


# --- media.py ---


@pytest.mark.parametrize("action, key, reply", [
    ("play_pause", "play/pause media", "Sent play/pause."), ("play", "play/pause media", "Sent play/pause."),
    ("pause", "play/pause media", "Sent play/pause."), ("next", "next track", "Sent next track."),
    ("previous", "previous track", "Sent previous track."),
])
def test_control_media_uses_media_keys_when_no_youtube_video(action, key, reply):
    # A media key can't be read back, so the reply says what was SENT - never claims it worked.
    with patch.object(media, "_youtube_video", return_value=None), patch.object(media.keyboard, "send") as send:
        assert media.control_media(action) == reply
    send.assert_called_once_with(key)


def test_control_media_rejects_unknown_action():
    with pytest.raises(ValueError, match="Unknown media action"):
        media.control_media("skip_to_the_end")


@pytest.mark.parametrize("alias", ["resume", "stop", "toggle"])
def test_control_media_accepts_spoken_aliases(alias):
    with patch.object(media, "_youtube_video", return_value=None), patch("assistant.tools.media.keyboard") as kb:
        media.control_media(alias)
    kb.send.assert_called_once_with("play/pause media")


class _FakeEndpoint:
    def __init__(self, level=0.5, muted=0):
        self.level, self.muted = level, muted

    def GetMasterVolumeLevelScalar(self):
        return self.level

    def SetMasterVolumeLevelScalar(self, v, ctx):
        self.level = v

    def GetMute(self):
        return self.muted

    def SetMute(self, m, ctx):
        self.muted = m


@pytest.mark.parametrize("action, level, start, expected", [
    ("volume_set", 40, 0.9, 40), ("volume_up", None, 0.5, 60), ("volume_down", None, 0.5, 40),
    ("volume_down", None, 0.05, 0), ("volume_up", None, 0.95, 100),
])
def test_system_volume_is_set_and_read_back(action, level, start, expected):
    ep = _FakeEndpoint(start)
    with patch.object(media, "_youtube_video", return_value=None), patch.object(media, "_endpoint", return_value=ep):
        assert media.control_media(action, level) == f"Laptop volume is now {expected}%."
    assert round(ep.level * 100) == expected


def test_system_volume_that_does_not_change_is_reported_as_a_failure():
    class Stuck(_FakeEndpoint):
        def SetMasterVolumeLevelScalar(self, v, ctx):
            pass  # Windows ignored it
    with patch.object(media, "_youtube_video", return_value=None), patch.object(media, "_endpoint", return_value=Stuck(0.9)):
        with pytest.raises(RuntimeError, match="still|but it's at 90%"):
            media.control_media("volume_set", 75)


def _yt(playing):
    from unittest.mock import MagicMock
    yp = MagicMock()
    yp.pause.return_value, yp.play.return_value = "Paused.", "Playing."
    yp.set_volume.return_value = "YouTube volume is now 75%."
    return yp, {"url": "https://www.youtube.com/watch?v=x"}, playing


def test_pause_goes_to_a_playing_youtube_video():
    yp, tab, playing = _yt(True)
    with patch.object(media, "_youtube_video", return_value=(yp, tab, playing)), patch.object(media.keyboard, "send") as send:
        assert media.control_media("pause") == "Paused."
    send.assert_not_called()


def test_volume_goes_to_youtube_only_while_its_video_is_playing():
    yp, tab, _ = _yt(True)
    with patch.object(media, "_youtube_video", return_value=(yp, tab, True)):
        assert media.control_media("volume_set", 75) == "YouTube volume is now 75%."
    yp.set_volume.assert_called_once_with(level=75, tab=tab)
    ep = _FakeEndpoint(0.9)
    with patch.object(media, "_youtube_video", return_value=(yp, tab, False)), patch.object(media, "_endpoint", return_value=ep):
        assert media.control_media("volume_set", 75) == "Laptop volume is now 75%."


def test_play_resumes_a_paused_youtube_video():
    yp, tab, _ = _yt(False)
    with patch.object(media, "_youtube_video", return_value=(yp, tab, False)):
        assert media.control_media("play") == "Playing."
