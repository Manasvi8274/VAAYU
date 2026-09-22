from unittest.mock import MagicMock, patch

import pytest

from assistant.llm.ollama_client import (
    OllamaBrain,
    _GROUNDING_NUDGE,
    _KEEP_ALIVE,
    _NO_TOOL_FALLBACK,
    _NUM_CTX,
    _looks_like_action_request,
    _sanitize_reply,
)
from assistant.tools.registry import ToolRegistry


@pytest.fixture
def brain():
    with patch("assistant.llm.ollama_client.ollama.Client"):
        b = OllamaBrain(host="http://fake", model="fake-model")
    b.client = MagicMock()
    return b


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(
        name="close_app",
        description="close an app",
        parameters={"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]},
    )(lambda app_name: f"Closed {app_name}.")
    return reg


def _msg(content="", tool_calls=None):
    return {"role": "assistant", "content": content, "tool_calls": tool_calls or []}


# --- _looks_like_action_request ---


@pytest.mark.parametrize(
    "text",
    [
        "close the brave browser",
        "play fairytale music",
        "increase the youtube volume",
        "send a whatsapp message to mom",
        "minimize the window",
        "open chrome",
    ],
)
def test_looks_like_action_request_true_for_real_commands(text):
    assert _looks_like_action_request(text) is True


@pytest.mark.parametrize("text", ["how are you", "what's the weather like", "tell me a joke"])
def test_looks_like_action_request_false_for_conversation(text):
    assert _looks_like_action_request(text) is False


# --- OllamaBrain.respond grounding behavior ---


def test_conversational_reply_with_no_tool_call_is_accepted_immediately(brain, registry):
    brain.client.chat.return_value = {"message": _msg("I'm doing well, thanks!")}
    messages = [{"role": "user", "content": "how are you"}]

    reply, _ = brain.respond(messages, registry)

    assert reply == "I'm doing well, thanks!"
    assert brain.client.chat.call_count == 1  # no nudge/retry for non-action requests


def test_action_request_with_immediate_tool_call_needs_no_nudge(brain, registry):
    brain.client.chat.side_effect = [
        {
            "message": _msg(
                tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}]
            )
        },
        {"message": _msg("Closed Brave.")},
    ]
    messages = [{"role": "user", "content": "close the brave browser"}]

    reply, final_messages = brain.respond(messages, registry)

    assert reply == "Closed Brave."
    assert brain.client.chat.call_count == 2
    assert not any(m.get("content") == _GROUNDING_NUDGE for m in final_messages)


def test_hallucinated_success_with_no_tool_call_triggers_grounding_retry(brain, registry):
    # First pass: model claims success in plain text with NO tool call at all -
    # this is exactly the reported "said brave was closed but it wasn't" bug.
    # Second pass (after the nudge): model actually calls the tool.
    brain.client.chat.side_effect = [
        {"message": _msg("Brave is now closed.")},
        {
            "message": _msg(
                tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}]
            )
        },
        {"message": _msg("Closed Brave.")},
    ]
    messages = [{"role": "user", "content": "close the brave browser"}]

    reply, final_messages = brain.respond(messages, registry)

    assert reply == "Closed Brave."
    assert brain.client.chat.call_count == 3
    assert any(m.get("content") == _GROUNDING_NUDGE for m in final_messages)
    # the tool must have genuinely been dispatched, not just talked about
    assert any(m.get("role") == "tool" and m.get("name") == "close_app" for m in final_messages)


def test_nudge_only_fires_once_even_if_model_still_gives_no_tool_call(brain, registry):
    brain.client.chat.side_effect = [
        {"message": _msg("Brave is now closed.")},
        {"message": _msg("It's closed, I promise.")},
    ]
    messages = [{"role": "user", "content": "close the brave browser"}]

    reply, _ = brain.respond(messages, registry)

    # Accepted after exactly one retry - no infinite loop, no third call.
    assert reply == "It's closed, I promise."
    assert brain.client.chat.call_count == 2


def test_every_chat_call_sets_keep_alive_and_num_ctx(brain, registry):
    # Regression guard for two real measured bugs: Ollama unloading the model
    # after 5 minutes idle (~20s reload penalty on the next command - very
    # likely the actual cause of "responses take too long") and the default
    # 4096-token context window being nearly exhausted by tool schemas alone
    # (3214 tokens measured for this project's tool set), silently truncating
    # conversation history as it grows. Every call must set both.
    brain.client.chat.return_value = {"message": _msg("done")}
    messages = [{"role": "user", "content": "what time is it"}]

    brain.respond(messages, registry)

    kwargs = brain.client.chat.call_args.kwargs
    assert kwargs["keep_alive"] == _KEEP_ALIVE
    assert kwargs["options"]["num_ctx"] == _NUM_CTX


def test_step_cap_final_summary_call_also_sets_keep_alive_and_num_ctx(brain, registry):
    # The step-cap fallback path (tools=[...] withheld) uses a separate
    # client.chat() call - must not forget these on that path either.
    tool_call_msg = _msg(
        tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}]
    )
    brain.client.chat.side_effect = [{"message": tool_call_msg}] * 5 + [{"message": _msg("summary")}]
    messages = [{"role": "user", "content": "close brave"}]

    brain.respond(messages, registry)

    final_kwargs = brain.client.chat.call_args.kwargs
    assert final_kwargs["keep_alive"] == _KEEP_ALIVE
    assert final_kwargs["options"]["num_ctx"] == _NUM_CTX


def test_no_nudge_for_non_action_requests_even_with_no_tool_call(brain, registry):
    brain.client.chat.return_value = {"message": _msg("Sure, ask away.")}
    messages = [{"role": "user", "content": "can I ask you something"}]

    reply, _ = brain.respond(messages, registry)

    assert reply == "Sure, ask away."
    assert brain.client.chat.call_count == 1


# --- issue logging (feeds the daily check-in - see self_review/) ---


@pytest.fixture
def failing_registry():
    reg = ToolRegistry()
    reg.register(
        name="close_app",
        description="close an app",
        parameters={"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]},
    )(lambda app_name: (_ for _ in ()).throw(RuntimeError(f"{app_name} not found")))
    return reg


def test_real_tool_failure_is_logged_as_an_issue(brain, failing_registry):
    brain.client.chat.side_effect = [
        {"message": _msg(tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}])},
        {"message": _msg("Sorry, I couldn't close it.")},
    ]
    messages = [{"role": "user", "content": "close the brave browser"}]

    with patch("assistant.llm.ollama_client.issue_log.log_issue") as mock_log:
        brain.respond(messages, failing_registry)

    mock_log.assert_called_once()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["tool_name"] == "close_app"
    assert kwargs["user_said"] == "close the brave browser"
    assert "not found" in kwargs["error"]


def test_respond_returns_deterministic_failure_not_models_text_after_real_failure(brain, failing_registry):
    # This is the same scenario as test_real_tool_failure_is_logged_as_an_issue
    # above, but checking the actual returned reply - found live
    # (qwen2.5:7b-instruct, "send hey there to vedant on whatsapp" with no
    # browser running): the model's own follow-up text falsely claimed
    # success after a genuine tool failure, and nothing caught it. The
    # ground-truth error must be what's returned, never the model's text.
    brain.client.chat.side_effect = [
        {"message": _msg(tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}])},
        {"message": _msg("Closing it now, all done!")},  # false success claim - must be ignored
    ]
    messages = [{"role": "user", "content": "close the brave browser"}]

    reply, _ = brain.respond(messages, failing_registry)

    assert reply == "That didn't work - brave not found"


def test_respond_reproduces_and_fixes_the_real_whatsapp_false_success_scenario(brain):
    reg = ToolRegistry()
    reg.register(
        name="send_whatsapp_message",
        description="send",
        parameters={
            "type": "object",
            "properties": {"contact_name": {"type": "string"}, "message": {"type": "string"}},
            "required": ["contact_name", "message"],
        },
        requires_confirmation=True,
    )(lambda contact_name, message: (_ for _ in ()).throw(ConnectionError("Can't reach Chrome's remote debugging port (9222).")))
    brain.client.chat.side_effect = [
        {
            "message": _msg(
                tool_calls=[
                    {
                        "function": {
                            "name": "send_whatsapp_message",
                            "arguments": {"contact_name": "vedant", "message": "hey there"},
                        }
                    }
                ]
            )
        },
        {"message": _msg('Sending "hey there" to Vedant on WhatsApp.')},  # the exact false claim observed live
    ]
    messages = [{"role": "user", "content": "send hey there to vedant on whatsapp"}]

    reply, _ = brain.respond(messages, reg, confirm=lambda name, args: True)

    assert reply == "That didn't work - Can't reach Chrome's remote debugging port (9222)."
    assert "Sending" not in reply  # the model's false claim must not leak through


def test_respond_allows_models_own_reply_when_confirmation_was_declined(brain):
    # Declining is a normal, wanted outcome - the model's natural "okay, not
    # sending that" must NOT be overridden by the deterministic-failure path,
    # which is reserved for genuine tool execution failures only.
    reg = ToolRegistry()
    reg.register(
        name="send_whatsapp_message",
        description="send",
        parameters={
            "type": "object",
            "properties": {"contact_name": {"type": "string"}, "message": {"type": "string"}},
            "required": ["contact_name", "message"],
        },
        requires_confirmation=True,
    )(lambda contact_name, message: "sent")
    brain.client.chat.side_effect = [
        {
            "message": _msg(
                tool_calls=[
                    {"function": {"name": "send_whatsapp_message", "arguments": {"contact_name": "Mom", "message": "hi"}}}
                ]
            )
        },
        {"message": _msg("Okay, I won't send that.")},
    ]
    messages = [{"role": "user", "content": "message mom on whatsapp"}]

    reply, _ = brain.respond(messages, reg, confirm=lambda name, args: False)

    assert reply == "Okay, I won't send that."


def test_respond_recovers_normally_when_retry_tool_call_succeeds_after_a_failure(brain):
    # A failed tool call followed by the model correctly retrying with a
    # real (successful) tool call must proceed normally - the deterministic
    # override is only for when the model does NOT retry.
    reg = ToolRegistry()
    calls = {"n": 0}

    def _flaky(app_name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return f"Closed {app_name}."

    reg.register(
        name="close_app",
        description="close an app",
        parameters={"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]},
    )(_flaky)

    brain.client.chat.side_effect = [
        {"message": _msg(tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}])},
        {"message": _msg(tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}])},
        {"message": _msg("Closed Brave.")},
    ]
    messages = [{"role": "user", "content": "close the brave browser"}]

    reply, _ = brain.respond(messages, reg)

    assert reply == "Closed Brave."


def test_declined_confirmation_is_not_logged_as_an_issue(brain, registry):
    # Registry here has no requires_confirmation tools, so build one inline.
    reg = ToolRegistry()
    reg.register(
        name="send_whatsapp_message",
        description="send",
        parameters={
            "type": "object",
            "properties": {"contact_name": {"type": "string"}, "message": {"type": "string"}},
            "required": ["contact_name", "message"],
        },
        requires_confirmation=True,
    )(lambda contact_name, message: "sent")
    brain.client.chat.side_effect = [
        {
            "message": _msg(
                tool_calls=[
                    {
                        "function": {
                            "name": "send_whatsapp_message",
                            "arguments": {"contact_name": "Mom", "message": "hi"},
                        }
                    }
                ]
            )
        },
        {"message": _msg("Okay, not sending that.")},
    ]
    messages = [{"role": "user", "content": "message mom on whatsapp"}]

    with patch("assistant.llm.ollama_client.issue_log.log_issue") as mock_log:
        brain.respond(messages, reg, confirm=lambda name, args: False)

    mock_log.assert_not_called()


def test_successful_tool_call_is_not_logged_as_an_issue(brain, registry):
    brain.client.chat.side_effect = [
        {"message": _msg(tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}])},
        {"message": _msg("Closed Brave.")},
    ]
    messages = [{"role": "user", "content": "close brave"}]

    with patch("assistant.llm.ollama_client.issue_log.log_issue") as mock_log:
        brain.respond(messages, registry)

    mock_log.assert_not_called()


def test_grounding_nudge_firing_is_logged_as_an_issue(brain, registry):
    brain.client.chat.side_effect = [
        {"message": _msg("Brave is now closed.")},  # no tool call - triggers the nudge
        {"message": _msg(tool_calls=[{"function": {"name": "close_app", "arguments": {"app_name": "brave"}}}])},
        {"message": _msg("Closed Brave.")},
    ]
    messages = [{"role": "user", "content": "close the brave browser"}]

    with patch("assistant.llm.ollama_client.issue_log.log_issue") as mock_log:
        brain.respond(messages, registry)

    mock_log.assert_called_once()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["kind"] == "grounding_nudge_fired"
    assert kwargs["user_said"] == "close the brave browser"


# --- _sanitize_reply: hallucinated fake tool-call text ---
# Found live comparing qwen2.5:3b against qwen2.5:7b-instruct on this
# project's own real bug-case commands: after the grounding nudge, 3b
# sometimes generates Qwen's raw <tool_call>{...}</tool_call> text format
# itself instead of using the real tool-calling API, and when that JSON is
# malformed, Ollama can't parse it into a real tool_calls entry and returns
# it as plain content - which would have been spoken to the user verbatim
# without this check. Reproduced with a clean, single-model GPU state, so
# it's a real model behavior, not a symptom of testing two models at once.

_REAL_CAPTURED_GARBAGE = '{"name": "close_browser_tab", "arguments": {"position: 3}}\n</tool_call>'


def test_sanitize_reply_catches_the_real_captured_garbage_string():
    assert _sanitize_reply(_REAL_CAPTURED_GARBAGE) == _NO_TOOL_FALLBACK


@pytest.mark.parametrize("garbage", [
    '{"name": "open_app", "arguments": {"app_name": "brave"}}',
    "<tool_call>\n{\"name\": \"close_app\", \"arguments\": {}}\n</tool_call>",
    "</tool_call>",
    "<tool_call>",
])
def test_sanitize_reply_catches_various_fake_tool_call_shapes(garbage):
    assert _sanitize_reply(garbage) == _NO_TOOL_FALLBACK


@pytest.mark.parametrize("normal_reply", [
    "That sounds great, happy to help!",
    "Your battery is at 82%.",
    "I don't have a tool for ordering food.",
    "Sure, I can name that for you.",  # contains "name" but not JSON-shaped
])
def test_sanitize_reply_does_not_false_positive_on_normal_replies(normal_reply):
    assert _sanitize_reply(normal_reply) == normal_reply


def test_respond_returns_fallback_when_model_hallucinates_fake_tool_call_after_nudge(brain, registry):
    brain.client.chat.side_effect = [
        {"message": _msg("Brave is now closed.")},  # no tool call - triggers the nudge
        {"message": _msg(_REAL_CAPTURED_GARBAGE)},  # nudge "recovers" into garbage, not a real call
    ]
    messages = [{"role": "user", "content": "close the brave browser"}]

    reply, _ = brain.respond(messages, registry)

    assert reply == _NO_TOOL_FALLBACK
    assert brain.client.chat.call_count == 2  # bounded - no infinite retry loop
