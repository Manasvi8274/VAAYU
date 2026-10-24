from unittest.mock import MagicMock, patch

import pytest

import assistant.tools  # noqa: F401 - registers every tool
from assistant.llm import router
from assistant.llm.ollama_client import OllamaBrain
from assistant.tools.registry import registry  # noqa: E402


@pytest.mark.parametrize(
    "text, expected",
    [
        ("open notepad", ("open_app", {"app_name": "notepad"})),
        ("Vaayu, open the brave browser.", ("open_app", {"app_name": "brave"})),
        ("whatsapp open kro", ("open_app", {"app_name": "whatsapp"})),
        ("chrome band karo", ("close_app", {"app_name": "chrome"})),
        ("close spotify", ("close_app", {"app_name": "spotify"})),
        ("pause", ("control_media", {"action": "play_pause"})),
        ("next song", ("control_media", {"action": "next"})),
        ("volume badhao", ("control_media", {"action": "volume_up"})),
        ("turn down the volume", ("control_media", {"action": "volume_down"})),
        ("battery kitni hai", ("get_system_status", {})),
        ("take a screenshot", ("take_screenshot", {})),
        ("lock my laptop", ("lock_computer", {})),
        ("minimize this window", ("minimize_window", {})),
    ],
)
def test_fast_route_hits(text, expected):
    assert router.fast_route(text) == ("tools", [expected])


@pytest.mark.parametrize(
    "text",
    [
        "play something nice",  # no target to search for
        "open my notes file in documents",  # a file
        "close vaayu",  # the assistant itself, not an app
        "send it",  # no recipient/message to parse
        "what is the capital of france",
        "",
    ],
)
def test_fast_route_leaves_ambiguous_requests_to_the_llm(text):
    assert router.fast_route(text) is None


def test_time_question_is_answered_without_llm():
    kind, reply = router.fast_route("what time is it")
    assert kind == "reply" and reply.startswith("It's ")


@pytest.mark.parametrize(
    "text, tool",
    [
        ("send hello to mom on whatsapp", "send_whatsapp_message"),
        ("skip forward 30 seconds", "seek_video"),
        ("is bluetooth on", "list_connected_devices"),
        ("note that I have a meeting at 5", "add_note"),
        ("switch to the youtube tab", "switch_to_browser_tab"),
    ],
)
def test_select_tools_keeps_the_right_tool_in_the_top_5(text, tool):
    assert tool in router.select_tools(text, registry, k=5)._tools


def test_fast_route_dispatches_without_calling_the_llm():
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="m")
    brain.client = MagicMock()
    fake = MagicMock()
    fake._tools = {"open_app": None}
    fake.requires_confirmation.return_value = False
    fake.dispatch.return_value = {"ok": True, "result": "Opened notepad."}

    reply, _ = brain.respond([{"role": "user", "content": "open notepad"}], fake)

    assert reply == "Opened notepad."
    fake.dispatch.assert_called_once_with("open_app", {"app_name": "notepad"})
    brain.client.chat.assert_not_called()


def test_successful_action_tool_skips_the_second_llm_call():
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="m")
    brain.client = MagicMock()
    brain.client.chat.return_value = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "play_video_on_youtube", "arguments": {"query": "lofi"}}}]}}
    with patch.object(type(registry), "dispatch", return_value={"ok": True, "result": "Playing lofi on YouTube."}):
        reply, _ = brain.respond([{"role": "user", "content": "put on some lofi"}], registry)
    assert reply == "Playing lofi on YouTube."
    assert brain.client.chat.call_count == 1


def test_model_is_never_allowed_to_pick_a_whatsapp_recipient():
    # Benchmark: the model sent a "SAVED MESSAGES" request to "mom".
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="m")
    brain.client = MagicMock()
    brain.client.chat.return_value = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "send_whatsapp_message", "arguments": {"contact_name": "mom", "message": "buy milk"}}}]}}
    with patch.object(type(registry), "dispatch") as dispatch:
        reply, _ = brain.respond([{"role": "user", "content": "message SAVED MESSAGES buy milk on whatsapp"}],
                                 registry, confirm=lambda n, a: True)
    dispatch.assert_not_called()
    assert reply == router.WHATSAPP_HOWTO


@pytest.mark.parametrize(
    "text, expected",
    [
        ('send message "By Bye" to saved messages in whatsapp', ("saved messages", "By Bye")),
        ("send going to office to SAVED MESSAGES on whatsapp", ("SAVED MESSAGES", "going to office")),
        ("whatsapp pe SAVED MESSAGES ko dinner ready bhejo", ("SAVED MESSAGES", "dinner ready")),
        ("whatsapp message to SAVED MESSAGES: lunch at 2", ("SAVED MESSAGES", "lunch at 2")),
        ("send ok to Raman Garg on whatsapp", ("Raman Garg", "ok")),
    ],
)
def test_whatsapp_recipient_and_message_are_taken_verbatim(text, expected):
    _kind, [(name, args)] = router.fast_route(text)
    assert name == "send_whatsapp_message"
    assert (args["contact_name"], args["message"]) == expected


def test_two_whatsapp_recipients_are_refused_not_guessed():
    kind, reply = router.fast_route("send thanks to SAVED MESSAGES and Raman Garg on whatsapp")
    assert kind == "reply" and "one WhatsApp chat" in reply


def test_whatsapp_context_carries_into_the_next_step():
    assert router.fast_route("whatsapp kholo aur SAVED MESSAGES ko hello bhejo") == ("tools", [
        ("open_app", {"app_name": "whatsapp"}),
        ("send_whatsapp_message", {"contact_name": "SAVED MESSAGES", "message": "hello"})])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("turn off wifi", {"kind": "wifi", "state": "off"}), ("wifi band karo", {"kind": "wifi", "state": "off"}),
        ("bluetooth chalu karo", {"kind": "bluetooth", "state": "on"}), ("enable bluetooth", {"kind": "bluetooth", "state": "on"}),
    ],
)
def test_radio_on_off(text, expected):
    assert router.fast_route(text) == ("tools", [("set_radio", expected)])


@pytest.mark.parametrize(
    "text",
    ["shut down the laptop", "restart my computer", "put the laptop to sleep", "delete everything in downloads",
     "format the d drive", "install vlc", "open phonepe and send money", "pay 200 on paytm"],
)
def test_dangerous_requests_are_refused_without_any_tool(text):
    assert router.fast_route(text)[0] == "reply"


def test_reminder_mentioning_pay_is_still_a_note():
    assert router.fast_route("remember to pay rent on the 5th") == ("tools", [("add_note", {"text": "pay rent on the 5th"})])


def test_mixed_multi_step_is_split_into_separate_commands():
    assert router.split_steps("open spotify and play some music") == ["open spotify", "play some music"]
    assert router.split_steps("send bread and milk to mom") is None  # one message, not two commands


@pytest.mark.parametrize(
    "text, expected",
    [
        ("send buy milk to SAVED MESSAGES on whatsapp", [("send_whatsapp_message", {"contact_name": "SAVED MESSAGES", "message": "buy milk"})]),
        ("SAVED MESSAGES ko whatsapp pe meeting at 5 bhejo", [("send_whatsapp_message", {"contact_name": "SAVED MESSAGES", "message": "meeting at 5"})]),
        ("take a screenshot and then lock the laptop", [("take_screenshot", {}), ("lock_computer", {})]),
        ("Volume. Up.", [("control_media", {"action": "volume_up"})]),
        ("um turn up the volume please", [("control_media", {"action": "volume_up"})]),
        ("open word in the background", [("open_app", {"app_name": "word", "foreground": False})]),
        ("kill file explorer", [("close_app", {"app_name": "file explorer"})]),
        ("close the second tab", [("close_browser_tab", {"position": 2})]),
        ("close this tab", [("close_browser_tab", {})]),
        ("mute the video", [("set_video_volume", {"direction": "mute"})]),
        ("skip forward a minute", [("seek_video", {"direction": "forward", "seconds": 60})]),
        ("open google.com", [("open_website", {"url_or_query": "google.com"})]),
        ("search for weather in delhi", [("open_website", {"url_or_query": "weather in delhi"})]),
        ("is bluetooth on", [("list_connected_devices", {"category": "bluetooth"})]),
    ],
)
def test_fast_route_round_two_patterns(text, expected):
    assert router.fast_route(text) == ("tools", expected)


@pytest.mark.parametrize("text", ["thank you", "ok", "kya haal hai", "never mind"])
def test_small_talk_is_answered_without_tools(text):
    assert router.fast_route(text)[0] == "reply"


def test_search_mentioning_local_files_is_not_sent_to_the_web():
    assert router.fast_route("search for invoice in my documents") is None


def test_repair_arguments_renames_a_single_misnamed_required_argument():
    from assistant.tools.registry import repair_arguments
    spec = registry._tools["open_website"]
    assert repair_arguments(spec, {"url": "github.com"}) == {"url_or_query": "github.com"}
