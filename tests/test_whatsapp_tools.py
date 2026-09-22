from unittest.mock import patch

import pytest

from assistant.tools import whatsapp


def _tab(url="https://web.whatsapp.com/"):
    return {"webSocketDebuggerUrl": "ws://fake", "url": url}


def _cdp_value(value):
    return {"result": {"result": {"value": value}}}


def _sequence(*values):
    it = iter(values)

    def _fake(ws_url, method, params=None):
        if method != "Runtime.evaluate":
            return {}
        return _cdp_value(next(it))

    return _fake


def test_type_into_clears_existing_content_before_inserting():
    # Guards a real fix: without clearing first, execCommand('insertText')
    # appends to any leftover draft/search text instead of replacing it, so
    # the message actually sent could differ from what the user confirmed.
    captured = {}

    def _fake(ws_url, method, params=None):
        if method == "Runtime.evaluate":
            captured["expression"] = params["expression"]
            return _cdp_value(True)
        return {}

    with patch.object(whatsapp, "_target_tab", return_value=_tab()), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_fake
    ):
        whatsapp._type_into(whatsapp._MESSAGE_BOX_SELECTOR, "hello")

    js = captured["expression"]
    assert js.index("selectAll") < js.index("delete") < js.index("insertText")


def test_send_whatsapp_message_full_happy_path():
    # order: type search -> click result -> type message -> click send -> read last bubble
    with patch.object(whatsapp, "_target_tab", return_value=_tab()), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_sequence(True, True, True, True, "On my way!")
    ), patch("time.sleep"):
        result = whatsapp.send_whatsapp_message("Mom", "On my way!")
    assert result == "Sent to Mom: On my way!"


def test_send_whatsapp_message_navigates_when_not_already_on_whatsapp_web():
    methods_called = []
    # Runtime.evaluate results in order: search box found, contact found,
    # message box found, send button found, last-bubble readback.
    eval_results = iter([True, True, True, True, "hello there"])

    def _fake_send(ws_url, method, params=None):
        methods_called.append(method)
        if method == "Runtime.evaluate":
            return _cdp_value(next(eval_results))
        return {}

    # _target_tab called multiple times - first call (in _ensure_whatsapp_web_open)
    # reports NOT on whatsapp, subsequent calls report already there (post-navigate).
    tabs = iter([_tab(url="https://example.com/"), *([_tab()] * 10)])
    with patch.object(whatsapp, "_target_tab", side_effect=lambda: next(tabs)), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_fake_send
    ), patch("time.sleep"):
        result = whatsapp.send_whatsapp_message("Dad", "hello there")
    assert "Page.navigate" in methods_called
    assert result == "Sent to Dad: hello there"


def test_send_whatsapp_message_raises_when_search_box_not_found():
    with patch.object(whatsapp, "_target_tab", return_value=_tab()), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_sequence(False)
    ), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="search box"):
            whatsapp.send_whatsapp_message("Mom", "hi")


def test_send_whatsapp_message_raises_when_contact_not_found():
    with patch.object(whatsapp, "_target_tab", return_value=_tab()), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_sequence(True, False)
    ), patch("time.sleep"):
        with pytest.raises(ValueError, match="No WhatsApp contact"):
            whatsapp.send_whatsapp_message("Nonexistent Person", "hi")


def test_send_whatsapp_message_raises_when_message_box_not_found():
    with patch.object(whatsapp, "_target_tab", return_value=_tab()), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_sequence(True, True, False)
    ), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="message box"):
            whatsapp.send_whatsapp_message("Mom", "hi")


def test_send_whatsapp_message_raises_when_send_button_not_found():
    with patch.object(whatsapp, "_target_tab", return_value=_tab()), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_sequence(True, True, True, False)
    ), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="send button"):
            whatsapp.send_whatsapp_message("Mom", "hi")


def test_send_whatsapp_message_raises_when_send_not_verified_in_chat():
    # Everything reports success, but the message never actually shows up in
    # the chat's last outgoing bubble - must not claim success anyway
    # (this is the "confirmed but didn't happen" bug class, applied here).
    with patch.object(whatsapp, "_target_tab", return_value=_tab()), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_sequence(True, True, True, True, "totally unrelated text")
    ), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="couldn't verify"):
            whatsapp.send_whatsapp_message("Mom", "On my way!")


def test_send_whatsapp_message_raises_when_no_message_bubble_found():
    with patch.object(whatsapp, "_target_tab", return_value=_tab()), patch.object(
        whatsapp, "_send_cdp_command", side_effect=_sequence(True, True, True, True, None)
    ), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="couldn't verify"):
            whatsapp.send_whatsapp_message("Mom", "hi")


# --- registry wiring: confirmation is required and reads back the real text ---


def test_send_whatsapp_message_requires_confirmation():
    from assistant.tools.registry import registry as real_registry

    assert real_registry.requires_confirmation("send_whatsapp_message") is True


def test_send_whatsapp_message_confirmation_prompt_quotes_real_text():
    from assistant.tools.registry import registry as real_registry

    prompt = real_registry.get_confirmation_prompt(
        "send_whatsapp_message", {"contact_name": "Mom", "message": "On my way!"}
    )
    assert "On my way!" in prompt
    assert "Mom" in prompt
