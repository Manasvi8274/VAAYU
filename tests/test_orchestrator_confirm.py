import pytest
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


def _awake(transcript, monkeypatch, media=True):
    from assistant import orchestrator
    from assistant.orchestrator import State
    monkeypatch.setattr(orchestrator, "_media_playing", lambda: media)
    orch = _make_orchestrator(transcript=transcript)
    orch._maybe_run_daily_checkin = MagicMock()
    orch.state = State.AWAKE
    orch.messages = []
    orch._name_window_until = 0.0
    orch.brain = MagicMock(trace=[])
    orch.brain.respond.side_effect = lambda m, r, confirm: ("ok", m)
    return orch


def test_video_speech_without_the_name_is_ignored_while_media_plays(monkeypatch):
    # Real log: "Today's episode is sponsored by the Great Courses Plus" was acted on.
    orch = _awake("Today's episode is sponsored by the Great Courses Plus", monkeypatch)
    orch.handle_one_turn()
    orch.brain.respond.assert_not_called()


def test_while_media_plays_only_the_words_after_the_name_are_used(monkeypatch):
    orch = _awake("sitting in the shower in the dark. Vaayu, pause the video", monkeypatch)
    orch.handle_one_turn()
    sent = orch.brain.respond.call_args.args[0]
    assert sent[-1]["content"] == "pause the video"


def test_bare_name_during_media_opens_a_window_for_the_next_sentence(monkeypatch):
    orch = _awake("Vaayu", monkeypatch)
    orch.handle_one_turn()
    orch._speak.assert_called_once_with("Yes?")
    orch.stt.transcribe.return_value = "volume 40"
    orch.handle_one_turn()
    assert orch.brain.respond.call_args.args[0][-1]["content"] == "volume 40"


def test_no_name_needed_when_nothing_is_playing(monkeypatch):
    orch = _awake("pause", monkeypatch, media=False)
    orch.handle_one_turn()
    orch.brain.respond.assert_called_once()


def _monitor():
    from assistant.orchestrator import _MediaMonitor
    m = _MediaMonitor.__new__(_MediaMonitor)  # no background thread
    m.last_sound, m.suppress_until = 0.0, 0.0
    return m


def test_media_counts_as_playing_through_a_quiet_gap():
    # Real log: an utterance ends on a pause - exactly when the video can be quiet too.
    m = _monitor()
    m.observe(0.3, now=100.0)
    m.observe(0.0, now=101.0)  # quiet gap in the video
    assert m.playing(now=102.5)
    assert not m.playing(now=110.0)


def test_vaayus_own_voice_is_not_media():
    m = _monitor()
    m.suppress_until = 200.0
    m.observe(0.5, now=199.0)
    assert not m.playing(now=199.5)



def test_saved_audio_is_a_playable_wav_and_old_files_are_pruned(tmp_path, monkeypatch):
    import wave
    import numpy as np
    from assistant import orchestrator
    monkeypatch.setattr(orchestrator, "_AUDIO_DIR", tmp_path)
    monkeypatch.setattr(orchestrator, "_AUDIO_KEEP", 2)
    save = orchestrator._real_save_audio  # the suite-wide stub replaces _save_audio itself
    names = [save(np.full(1600, 0.5, dtype=np.float32)) for _ in range(3)]
    assert all(names)
    left = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert len(left) == 2 and names[-1] in left
    with wave.open(str(tmp_path / names[-1])) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 1600)


@pytest.mark.parametrize("text, passes", [
    ("Pause the video.", True),  # real log: heard perfectly, ignored for lacking "Vaayu"
    ("Change volume to 20 percent for youtube.", True),
    ("skip ad", True),
    ("Did you hear your opinion of me just confirms my opinion of you yeah oh I smell something burning.", False),
    ("To 20 percent again to make all this I have sense.", False),
    ("Go ahead hi.", False),
    ("open brave", False),  # not a player control: still needs the name while media plays
    ("send hi to SAVED MESSAGES on whatsapp", False),
])
def test_player_commands_need_no_name_while_media_plays(text, passes):
    from assistant.orchestrator import _is_player_command
    assert _is_player_command(text) is passes
