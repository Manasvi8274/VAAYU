"""
"Stop" while Vaayu is talking: cut the reply off and go back to listening.

While Vaayu speaks, the listener is muted (so it doesn't hear itself) and its
mic frames are tapped here instead. A short burst of speech is transcribed;
if it's a stop phrase - and not just Vaayu's own words coming back through
the mic - the reply is stopped. The laptop's mic array cancels its own
speaker output (measured earlier for the guardian), and the energy gate is
set above the normal one, so Vaayu's voice rarely gets this far anyway.
"""

import queue
import re
import threading

import numpy as np

STOP_PHRASE = re.compile(
    r"^(?:(?:hey|ok|okay|arre|arey|vaa?yu|vayu|bhai|please)\s+)*"
    r"(?:stop(?: it| that| talking| speaking| now| please)?|stop stop|b[au]s(?: karo| kar| kar do| bas)?|ruko|ruk jao|"
    r"chup(?: karo| ho jao| raho)?|shut up|enough|wait|quiet|be quiet|cancel|thik hai|theek hai|okay stop|ok stop|"
    r"hold on|never ?mind)(?:\s+(?:vaa?yu|vayu|bhai|please|yaar))*$", re.I)
_GATE_MULTIPLIER = 1.3  # calibration knob: x the listener's normal speech gate while Vaayu talks
_SPEECH_FRAMES = 4      # 120 ms of speech starts a capture (a short "stop." missed at 180)
_END_SILENCE_FRAMES = 10  # 300 ms of quiet ends it
_MAX_FRAMES = 60        # 1.8 s: stop phrases are short


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", (text or "").lower()))


def is_stop(heard: str, speaking: str) -> bool:
    """A stop phrase that isn't simply an echo of what Vaayu is saying."""
    said = _words(heard)
    if not said or not STOP_PHRASE.match(said):
        return False
    return said not in _words(speaking)  # "...say stop to pause" coming back through the mic


class BargeIn:
    """Runs while one reply is spoken; sets `stop` when the user says a stop phrase."""

    def __init__(self, listener, stt, stop: threading.Event, speaking: str):
        self.listener, self.stt, self.stop, self.speaking = listener, stt, stop, speaking
        self.heard: str | None = None
        self._frames: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="barge-in")
        listener.tap = self._frames.put
        self._thread.start()

    def close(self) -> None:
        self.listener.tap = None
        self._frames.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        gate = self.listener._noise_floor_rms * self.listener.energy_threshold_multiplier * _GATE_MULTIPLIER
        buf, speech, silence = [], 0, 0
        while not self.stop.is_set():
            frame = self._frames.get()
            if frame is None:
                return
            loud = self.listener._frame_rms(frame) >= gate
            voiced = loud and self.listener.vad.is_speech(frame.tobytes(), 16000)
            if speech < _SPEECH_FRAMES:
                speech = speech + 1 if voiced else 0
                buf = buf + [frame] if voiced else []
                continue
            buf.append(frame)
            silence = 0 if voiced else silence + 1
            if silence >= _END_SILENCE_FRAMES or len(buf) >= _MAX_FRAMES:
                audio = np.concatenate(buf).astype(np.float32) / 32768.0
                try:
                    text = self.stt.transcribe(audio)
                except Exception:
                    text = ""
                if is_stop(text, self.speaking):
                    self.heard = text
                    self.stop.set()
                    return
                buf, speech, silence = [], 0, 0
