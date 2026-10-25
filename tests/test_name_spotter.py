import time
from unittest.mock import MagicMock

import numpy as np

from assistant.audio import name_spotter
from assistant.audio.name_spotter import NameSpotter, heard_name
from assistant.audio.vad_listener import VADListener


def test_only_the_real_name_counts_not_lyric_sound_alikes():
    assert heard_name(" Vaayu.") and heard_name("hey vayu") and not heard_name("why you love me")


def _spotter(heard, monkeypatch):
    monkeypatch.setattr(name_spotter, "_EVERY", 0.02)
    listener = MagicMock()
    listener.recent_audio.return_value = np.full(25600, 0.1, np.float32)
    fired = []
    s = NameSpotter.__new__(NameSpotter)
    s.listener, s.stt, s.should_listen, s.on_name = listener, None, lambda: True, fired.append
    s._last = s._first = 0.0
    texts = iter(heard)
    s._transcribe = lambda audio: next(texts, "")
    return s, fired


def test_needs_two_looks_in_a_row(monkeypatch):
    import threading
    s, fired = _spotter(["la la", " Vaayu.", "lyrics", " Vaayu.", "more lyrics"], monkeypatch)  # lone hits only
    threading.Thread(target=s._run, daemon=True).start()
    time.sleep(0.4)
    assert fired == []
    s2, fired2 = _spotter(["la la", " Vaayu.", " Vaayu, volume"], monkeypatch)
    threading.Thread(target=s2._run, daemon=True).start()
    time.sleep(0.4)
    assert fired2 == [" Vaayu, volume"]


def test_listener_keeps_the_last_two_seconds_and_can_cut_a_recording_short():
    v = VADListener()
    for i in range(100):
        v._callback(np.full((480, 1), i, np.int16), 480, None, None)
    assert len(v.recent_audio(1.0)) == 16000 * 1 // 480 * 480 or len(v.recent_audio(1.0)) > 15000
    v.interrupt()
    assert len(v._segment(iter([np.full(480, 5000, np.int16)] * 10))) == 0  # nothing: start fresh


def test_name_over_the_song_pauses_it_and_waits_for_the_command(monkeypatch):
    from assistant import orchestrator
    from assistant.tools import music
    held = []
    monkeypatch.setattr(music, "hold_for_listening", lambda: held.append(True))
    o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    o.listener, o.state, o.gui_queue, o._name_window_until = MagicMock(), orchestrator.State.SLEEPING, None, 0.0
    o._speak = MagicMock()
    o._on_name_over_music(" Vaayu.")
    assert held == [True] and o.listener.interrupt.called and o.state is orchestrator.State.AWAKE
    o._speak.assert_called_once_with("Yes?")
    assert o._name_window_until > time.time() + 5
