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
