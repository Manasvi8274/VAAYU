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


def test_control_media_sends_correct_key_for_each_action():
    for action, expected_key in media._KEY_MAP.items():
        with patch.object(media.keyboard, "send") as mock_send:
            result = media.control_media(action)
        mock_send.assert_called_once_with(expected_key)
        assert result == f"Sent media command: {action}"


def test_control_media_rejects_unknown_action():
    with pytest.raises(ValueError, match="Unknown media action"):
        media.control_media("skip_to_the_end")
