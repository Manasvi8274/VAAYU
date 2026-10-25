import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from assistant.tts import kokoro_engine
from assistant.tts.kokoro_engine import KokoroEngine, _sentences


def test_sentences_split_and_short_bits_ride_along():
    assert _sentences("Okay. Opened Brave. It's 25 degrees and clear!") == ["Okay. Opened Brave.", "It's 25 degrees and clear!"]
    assert _sentences("One long sentence without a stop") == ["One long sentence without a stop"]


def _engine(seconds_per_sentence=0.3):
    e = KokoroEngine.__new__(KokoroEngine)
    e.voice, e.speed, e.device = "hm_psi", 1.0, "cpu"
    e.synth_calls = []

    def synth(text):
        e.synth_calls.append(text)
        return np.zeros(int(24000 * seconds_per_sentence), np.float32), 24000
    e._synth = synth
    return e


def test_plays_every_sentence_in_order():
    e = _engine()
    with patch.object(kokoro_engine, "sd") as sd:
        e.speak("First sentence here. Second sentence here. Third one here.")
    assert e.synth_calls == ["First sentence here.", "Second sentence here.", "Third one here."]
    assert sd.play.call_count == 3


def test_stop_cuts_it_off_and_stops_synthesising():
    e = _engine(seconds_per_sentence=2.0)
    stop = threading.Event()
    with patch.object(kokoro_engine, "sd") as sd:
        threading.Timer(0.3, stop.set).start()
        t = time.time()
        e.speak("First sentence here. Second sentence here. Third one here. Fourth one here.", stop_event=stop)
    assert time.time() - t < 1.5 and sd.stop.called and sd.play.call_count == 1


def test_missing_kokoro_files_fall_back_to_the_windows_voice(tmp_path, monkeypatch):
    from assistant import orchestrator
    from assistant.config_schema import TTSConfig
    from assistant.tts.pyttsx3_engine import Pyttsx3Engine
    monkeypatch.setattr(kokoro_engine, "MODEL_DIR", tmp_path)  # empty: no model files
    assert isinstance(orchestrator._real_make_tts(TTSConfig()), Pyttsx3Engine)


def test_short_replies_are_cached_long_ones_are_not():
    e = KokoroEngine.__new__(KokoroEngine)
    e.voice, e.speed, e._cache = "hm_psi", 1.0, {}
    e.kokoro = MagicMock()
    e.kokoro.create.return_value = (np.zeros(10, np.float32), 24000)
    e._synth("Opened Brave.")
    e._synth("Opened Brave.")
    e._synth("x" * 200)
    e._synth("x" * 200)
    assert e.kokoro.create.call_count == 3


def test_change_voice_by_voice_and_it_sticks(tmp_path, monkeypatch):
    from assistant.llm import router
    from assistant.tools import voice
    monkeypatch.setattr(kokoro_engine, "VOICE_FILE", tmp_path / "voice.txt")
    e = KokoroEngine.__new__(KokoroEngine)
    e.voice, e._cache = "am_echo", {"x": 1}
    e.kokoro = MagicMock()
    e.kokoro.get_voices.return_value = list(kokoro_engine.VOICES.values())
    monkeypatch.setattr(kokoro_engine, "ACTIVE", e)
    assert router.fast_route("change your voice") == ("tools", [("set_voice", {"voice": "next"})])
    assert router.fast_route("use the george voice") == ("tools", [("set_voice", {"voice": "george"})])
    assert "Lewis" in voice.set_voice("next")  # echo -> lewis
    assert e.voice == "bm_lewis" and e._cache == {}
    assert kokoro_engine._saved_voice() == "bm_lewis"  # remembered for the next start
    voice.set_voice("george")
    assert e.voice == "bm_george"
