import threading
import time
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from assistant.audio import barge_in
from assistant.audio.barge_in import BargeIn, is_stop


@pytest.mark.parametrize("heard, speaking, stops", [
    ("Stop.", "Here are three videos for you.", True),
    ("Vaayu stop", "...", True),
    ("bas karo", "...", True),
    ("Bus karo.", "...", True),  # how Whisper wrote "bas karo" in the live check
    ("ruko", "...", True),
    ("chup", "...", True),
    ("okay stop", "...", True),
    ("Stop, stop.", "...", True),
    ("Stop.", "To pause the video, say stop.", False),  # its own words coming back through the mic
    ("play kesariya", "...", False),
    ("", "...", False),
])
def test_stop_phrases(heard, speaking, stops):
    assert is_stop(heard, speaking) is stops


class _Listener:
    _noise_floor_rms = 100.0
    energy_threshold_multiplier = 1.0
    tap = None

    def __init__(self):
        self.vad = MagicMock()
        self.vad.is_speech.side_effect = lambda b, sr: np.abs(np.frombuffer(b, np.int16)).mean() > 300

    @staticmethod
    def _frame_rms(frame):
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


def _feed(listener, loud_frames=12, quiet_frames=12):
    for _ in range(loud_frames):
        listener.tap(np.full(480, 3000, np.int16))
    for _ in range(quiet_frames):
        listener.tap(np.zeros(480, np.int16))


@pytest.mark.parametrize("transcript, speaking, stops", [("Stop.", "Here you go.", True),
                                                        ("Stop.", "Say stop to pause.", False),
                                                        ("open brave", "Here you go.", True),  # any real sentence now (user rule)
                                                        ("here you go", "Here you go.", False)])  # his own echo
def test_detector_sets_stop_only_for_a_real_stop(transcript, speaking, stops):
    listener, stt, stop = _Listener(), MagicMock(), threading.Event()
    stt.transcribe.return_value = transcript
    b = BargeIn(listener, stt, stop, speaking)
    _feed(listener)
    stop.wait(1.0)
    b.close()
    assert stop.is_set() is stops and listener.tap is None
    assert stt.transcribe.called  # it did listen to the burst


def test_tts_playback_stops_when_told(tmp_path):
    from assistant.tts.pyttsx3_engine import Pyttsx3Engine

    def fake_save(text, path):
        with wave.open(path, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
            w.writeframes(np.zeros(16000 * 5, np.int16).tobytes())  # 5 s of "speech"
    stop = threading.Event()
    engine = Pyttsx3Engine()
    with patch.object(engine, "save_to_file", side_effect=fake_save), \
            patch("assistant.tts.pyttsx3_engine.sd") as sd:
        threading.Timer(0.2, stop.set).start()
        t = time.time()
        engine.speak("a long reply", stop_event=stop)
    assert time.time() - t < 2 and sd.stop.called  # cut off, not the full 5 s


def test_speak_reports_a_barge_in_and_invites_the_next_sentence(monkeypatch):
    from assistant import orchestrator
    o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    o.continuous, o.gui_queue, o.stt, o._name_window_until = True, None, MagicMock(), 0.0
    o.listener = MagicMock()
    o.tts = MagicMock()
    o.tts.speak.side_effect = lambda text, stop_event=None: stop_event.set()  # "stop" heard mid-reply

    class FakeBarge:
        heard = "stop"

        def __init__(self, *a):
            pass

        def close(self):
            pass
    monkeypatch.setattr(orchestrator, "BargeIn", FakeBarge)
    assert o._speak("a long reply") is True
    assert o._name_window_until > time.time()
    o.listener.unmute.assert_called()


@pytest.mark.parametrize("heard, interrupts", [
    ("open chrome", True), ("no no wait", True), ("what is the weather", True),
    ("based on what you watch", False),  # his own words coming back
    ("Okay.", False), ("hmm", False),  # one word that isn't "stop"
])
def test_any_real_sentence_cuts_vaayu_off(heard, interrupts):
    from assistant.audio.barge_in import should_interrupt
    assert should_interrupt(heard, "Here are three videos based on what you watch.") is interrupts


def test_a_command_said_over_vaayu_runs_next_turn(monkeypatch):
    from assistant import orchestrator
    o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    o.continuous, o.gui_queue, o.stt, o._name_window_until = True, None, MagicMock(), 0.0
    o.listener, o.tts = MagicMock(), MagicMock()
    o.tts.speak.side_effect = lambda text, stop_event=None: stop_event.set()

    class FakeBarge:
        heard = "open chrome"

        def __init__(self, *a):
            pass

        def close(self):
            pass
    monkeypatch.setattr(orchestrator, "BargeIn", FakeBarge)
    o._speak("a long reply")
    assert o._pending_text == "open chrome"
    handled = []
    monkeypatch.setattr(o, "_handle_utterance", lambda audio, heard=None: handled.append(heard), raising=False)
    o._maybe_run_daily_checkin = lambda: None
    o.handle_one_turn()
    assert handled == ["open chrome"] and o._pending_text is None


@pytest.mark.parametrize("heard", ["Merrick by c.", "You're a hero.", "Yes code.", "By code deco.", "Brought it by.",
                                   "Lobby on kand-.", "Right open.", "Cutting down."])
def test_his_own_garbled_voice_never_cuts_him_off(heard):
    # Real logs 10-26..10-31: all 13 cut-offs were these - and the user couldn't hear him "in the middle".
    from assistant.audio.barge_in import should_interrupt
    assert not should_interrupt(heard, "Bhai, it sounds like your speakers are having a hiccup.")


def test_his_name_always_cuts_him_off():
    from assistant.audio.barge_in import should_interrupt
    assert should_interrupt("Vaayu listen", "Here is a long answer about the weather.")
