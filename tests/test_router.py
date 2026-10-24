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
        "play perfect by ed sheeran on youtube",  # needs a real search tool
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
        {"function": {"name": "send_whatsapp_message", "arguments": {"contact_name": "x", "message": "y"}}}]}}
    with patch.object(type(registry), "dispatch", return_value={"ok": True, "result": "Sent 'y' to x."}):
        reply, _ = brain.respond([{"role": "user", "content": "let x know y via whatsapp"}], registry,
                                 confirm=lambda n, a: True)
    assert reply == "Sent 'y' to x."
    assert brain.client.chat.call_count == 1


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
