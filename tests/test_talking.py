"""Talking like a person (user, 2026-11-05): echo cancelling, going quiet when talked over, "Yes?" to his name,
"Okay, checking the weather." the moment a slow command ends."""
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from assistant.audio import echo_cancel as ec


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000)
    env = np.repeat(rng.uniform(0.2, 1.0, n // 1600 + 1), 1600)[:n]  # syllable-ish loudness changes
    return (np.convolve(rng.normal(0, 1, n), np.ones(4) / 4, "same") * env * 0.3).astype(np.float32)


def _run(mic: np.ndarray, t0: float) -> np.ndarray:
    return np.concatenate([ec.clean(mic[k:k + 480], now=t0 + (k + 480) / 16000)
                           for k in range(0, len(mic) - 479, 480)]).astype(np.float64)


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


@pytest.fixture
def fresh():
    ec.reset()
    yield
    ec.reset()


def test_his_own_voice_is_taken_out_of_the_mic(fresh):
    pytest.importorskip("livekit")
    ref = _speechlike(5, 1)
    delay = int(0.2 * 16000)  # speaker -> mic
    echo = np.concatenate([np.zeros(delay), ref])[:len(ref)] * 0.3 * 32768
    ec.playing(ref, 16000, now=100.0)
    out = _run(echo.astype(np.int16), 100.0)
    late = slice(int(2 * 16000), len(out))  # once it has adapted
    assert 20 * np.log10(_rms(echo[late]) / max(_rms(out[late]), 1)) > 10  # measured 17-35 dB


def test_the_mic_is_untouched_while_he_is_quiet(fresh):
    user = (_speechlike(1, 2) * 32768).astype(np.int16)
    assert np.array_equal(_run(user, 50.0), user[:len(user) // 480 * 480])


def test_resampled_reference_lines_up_with_the_clock(fresh):
    ec.playing(np.ones(24000, np.float32) * 0.5, 24000, now=10.0)  # 1 s of a 24 kHz voice
    assert ec._reference(480, now=10.5).min() > 0  # half a second in: playing
    assert ec._reference(480, now=12.0).max() == 0  # it ended
    ec.stopped(now=12.0)
    assert ec._ref is None


# --- going quiet when talked over ------------------------------------------------------------------------------

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


@pytest.mark.parametrize("heard, interrupts", [("give me a new one", True), ("another one", True),
                                               ("hmm", False), ("Merrick by c.", False)])
def test_he_pauses_the_moment_the_user_talks_then_decides(heard, interrupts):
    from assistant.audio.barge_in import BargeIn
    listener, stop = _Listener(), threading.Event()
    stt = MagicMock()
    paused_when_heard = []
    stt.transcribe.side_effect = lambda audio: paused_when_heard.append(b.pause.is_set()) or heard
    b = BargeIn(listener, stt, stop, "Why did the scarecrow win an award? Because he was outstanding in his field.")
    for _ in range(6):
        listener.tap(np.full(480, 3000, np.int16))
    deadline = time.time() + 1
    while not b.pause.is_set() and time.time() < deadline:
        time.sleep(0.01)
    assert b.pause.is_set()  # quiet while the user is still talking
    for _ in range(25):
        listener.tap(np.zeros(480, np.int16))
    stop.wait(1.0)
    time.sleep(0.05)
    b.close()
    assert paused_when_heard == [True]
    assert stop.is_set() is interrupts
    assert (b.heard == heard) is interrupts
    assert not b.pause.is_set()  # not a real interruption: he carries on


def test_kokoro_goes_quiet_and_picks_up_again(monkeypatch):
    from assistant.tts import kokoro_engine
    engine = kokoro_engine.KokoroEngine.__new__(kokoro_engine.KokoroEngine)
    engine._synth = lambda text: (np.ones(24000 * 1, np.float32) * 0.1, 24000)  # 1 s of sound
    played = []
    sd = MagicMock()
    sd.play.side_effect = lambda audio, samplerate: played.append(len(audio))
    monkeypatch.setattr(kokoro_engine, "sd", sd)
    pause = threading.Event()
    threading.Timer(0.7, pause.set).start()
    threading.Timer(1.0, pause.clear).start()
    t = time.time()
    engine.speak("A joke.", stop_event=threading.Event(), pause_event=pause)
    assert len(played) == 2  # started, then picked up again after the pause
    assert 24000 * 0.5 < played[1] < 24000 * 0.8  # from just before where it stopped, not the start
    assert time.time() - t >= 1.5  # the pause was real silence


def test_a_new_command_said_over_a_joke_replaces_it(monkeypatch):
    from assistant import orchestrator
    o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    o.continuous, o.gui_queue, o.stt, o._name_window_until = True, None, MagicMock(), 0.0
    o.listener, o.tts = MagicMock(), MagicMock()
    o.tts.speak.side_effect = lambda text, stop_event=None: stop_event.set()

    class FakeBarge:
        heard, pause = "give me a new one", None

        def __init__(self, *a):
            pass

        def close(self):
            pass
    monkeypatch.setattr(orchestrator, "BargeIn", FakeBarge)
    o._speak("Why did the chicken cross the road?")
    assert o._pending_text == "give me a new one"  # not a router command - still taken, the brain has the joke


# --- "Yes?" to his name -------------------------------------------------------------------------------------------

def test_his_name_gets_a_spoken_yes_that_stops_when_the_user_talks():
    from assistant import orchestrator
    o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    o.continuous, o.listener = True, MagicMock()
    o.listener.on_speech_start = None
    started = threading.Event()
    stops = []

    def speak(text, stop_event=None):
        started.set()
        o.listener.on_speech_start()  # the user goes straight on with the command
        stops.append((text, stop_event.is_set()))
    o.tts = MagicMock()
    o.tts.speak.side_effect = speak
    orchestrator.REAL_SAY_YES(o)
    assert started.wait(2)
    time.sleep(0.05)
    assert stops == [("Yes?", True)]
    o.listener.mute.assert_not_called()  # the mic was never taken away
    assert o.listener.on_speech_start is None


@pytest.mark.parametrize("heard, said, left", [("Yes? Play Bang Bang.", "Yes?", "Play Bang Bang"),
                                               ("Yes.", "Yes?", ""),
                                               ("Yesterday was fun", "Yes?", "Yesterday was fun"),
                                               ("Go ahead and open chrome", "Go ahead.", "and open chrome")])
def test_his_own_yes_coming_back_is_dropped(heard, said, left):
    from assistant.orchestrator import _without_echo
    assert _without_echo(heard, said) == left


# --- "Okay, checking." -----------------------------------------------------------------------------------------

def test_slow_work_is_acknowledged_at_once_in_its_own_words():
    from assistant.orchestrator import ack_for, ack_now
    assert ack_now("what's the weather in delhi") and ack_for("what's the weather in delhi") == "Okay, checking the weather."
    assert not ack_now("open notepad")  # instant anyway
    assert not ack_now("tell me a joke")  # the brain - acknowledged only if it's slow (ACK_AFTER)
