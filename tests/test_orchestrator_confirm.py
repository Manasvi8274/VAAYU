"""
Tests for Orchestrator._voice_confirm - the voice-based confirmation gate for
sensitive tools (e.g. sending a WhatsApp message). Built because the previous
default confirm callback (cli_confirm) blocks on terminal input() and would
just hang forever in voice mode, silently breaking any requires_confirmation
tool the moment one was ever added.

Constructs the Orchestrator without running its real __init__ (which would
load Whisper/pyttsx3/Ollama and open real audio devices) - only the specific
attributes/methods _voice_confirm actually touches are stubbed.
"""

from unittest.mock import MagicMock

import numpy as np

from assistant.orchestrator import Orchestrator
from assistant.tools.registry import ToolRegistry


def _make_orchestrator(transcript: str, capture_returns_audio: bool = True):
    orch = Orchestrator.__new__(Orchestrator)  # skip real __init__
    orch._speak = MagicMock()
    orch._capture_utterance = MagicMock(
        return_value=(np.zeros(10, dtype=np.float32) if capture_returns_audio else np.array([]))
    )
    orch.stt = MagicMock()
    orch.stt.transcribe.return_value = transcript
    return orch


def _registry_with_whatsapp():
    reg = ToolRegistry()
    reg.register(
        name="send_whatsapp_message",
        description="send a whatsapp message",
        parameters={
            "type": "object",
            "properties": {"contact_name": {"type": "string"}, "message": {"type": "string"}},
            "required": ["contact_name", "message"],
        },
        requires_confirmation=True,
        confirmation_prompt=lambda args: f"Send '{args['message']}' to {args['contact_name']} on WhatsApp?",
    )(lambda contact_name, message: f"Sent to {contact_name}.")
    return reg


def test_voice_confirm_speaks_the_actual_confirmation_prompt(monkeypatch):
    reg = _registry_with_whatsapp()
    monkeypatch.setattr("assistant.orchestrator.registry", reg)
    orch = _make_orchestrator(transcript="yes")

    orch._voice_confirm("send_whatsapp_message", {"contact_name": "Mom", "message": "On my way"})

    orch._speak.assert_called_once_with("Send 'On my way' to Mom on WhatsApp?")


def test_voice_confirm_true_on_affirmative_reply(monkeypatch):
    reg = _registry_with_whatsapp()
    monkeypatch.setattr("assistant.orchestrator.registry", reg)
    orch = _make_orchestrator(transcript="yeah go ahead")

    result = orch._voice_confirm("send_whatsapp_message", {"contact_name": "Mom", "message": "hi"})

    assert result is True


def test_voice_confirm_false_on_negative_reply(monkeypatch):
    reg = _registry_with_whatsapp()
    monkeypatch.setattr("assistant.orchestrator.registry", reg)
    orch = _make_orchestrator(transcript="no don't send it")

    result = orch._voice_confirm("send_whatsapp_message", {"contact_name": "Mom", "message": "hi"})

    assert result is False


def test_voice_confirm_false_when_nothing_heard(monkeypatch):
    reg = _registry_with_whatsapp()
    monkeypatch.setattr("assistant.orchestrator.registry", reg)
    orch = _make_orchestrator(transcript="", capture_returns_audio=False)

    result = orch._voice_confirm("send_whatsapp_message", {"contact_name": "Mom", "message": "hi"})

    assert result is False
    orch.stt.transcribe.assert_not_called()  # no audio -> must not even attempt to transcribe


def test_voice_confirm_false_on_unrelated_reply(monkeypatch):
    reg = _registry_with_whatsapp()
    monkeypatch.setattr("assistant.orchestrator.registry", reg)
    orch = _make_orchestrator(transcript="what's the weather")

    result = orch._voice_confirm("send_whatsapp_message", {"contact_name": "Mom", "message": "hi"})

    assert result is False


def test_a_crashing_or_timed_out_model_call_is_spoken_as_an_apology_not_fatal(monkeypatch):
    # A benchmark run hung on one model call; with the client timeout that now
    # raises - which must not end Vaayu's listening loop.
    from assistant.orchestrator import State

    orch = _make_orchestrator(transcript="open the thing")
    orch._maybe_run_daily_checkin = MagicMock()
    orch.state = State.AWAKE
    orch.messages = []
    orch.brain = MagicMock()
    orch.brain.respond.side_effect = TimeoutError("timed out")
    monkeypatch.setattr("assistant.orchestrator.issue_log.log_issue", MagicMock())

    orch.handle_one_turn()

    orch._speak.assert_called_once_with("Sorry, that took too long or went wrong - please try again.")


def _read_log():
    import json
    from assistant import interaction_log
    return [json.loads(line) for line in interaction_log.LOG_PATH.read_text(encoding="utf-8").splitlines()]


def test_every_turn_is_logged_with_what_was_heard_the_reply_and_the_tool_trace(monkeypatch):
    from assistant import interaction_log
    from assistant.orchestrator import State

    interaction_log.LOG_PATH.unlink(missing_ok=True)
    orch = _make_orchestrator(transcript="open notepad")
    orch._maybe_run_daily_checkin = MagicMock()
    orch.state = State.AWAKE
    orch.messages = []
    orch.brain = MagicMock(trace=[])

    def fake_respond(messages, reg, confirm):
        orch.brain.trace.append({"path": "fast", "tool": "open_app", "args": {"app_name": "notepad"}, "ok": True})
        return "Opened notepad.", messages

    orch.brain.respond.side_effect = fake_respond
    orch.handle_one_turn()

    events = _read_log()
    assert [e["kind"] for e in events] == ["heard", "turn"]
    assert events[0]["text"] == "open notepad" and events[0]["state"] == "awake"
    turn = events[1]
    assert turn["reply"] == "Opened notepad."
    assert turn["trace"] == [{"path": "fast", "tool": "open_app", "args": {"app_name": "notepad"}, "ok": True}]
    assert {"brain_ms", "speak_ms", "stt_ms"} <= turn.keys()


def test_speech_ignored_while_asleep_is_still_logged():
    from assistant import interaction_log

    interaction_log.LOG_PATH.unlink(missing_ok=True)
    orch = _make_orchestrator(transcript="what a nice day")
    orch._maybe_run_daily_checkin = MagicMock()
    from assistant.orchestrator import State
    orch.state = State.SLEEPING
    orch.handle_one_turn()
    assert [e["kind"] for e in _read_log()] == ["heard", "ignored_sleeping"]


def test_history_keeps_only_the_last_few_user_turns():
    from assistant.orchestrator import _HISTORY_USER_TURNS, _trim_history

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(10):
        msgs += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    trimmed = _trim_history(msgs)
    assert trimmed[0]["content"] == "sys"
    assert [m["content"] for m in trimmed if m["role"] == "user"] == [f"u{i}" for i in range(10 - _HISTORY_USER_TURNS, 10)]
