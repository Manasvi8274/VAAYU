"""
"Vaayu" over a song: pause the music the moment the name is said, so the command after it is heard clean.

While Vaayu's own song plays (tools/music.py), the music drowns out much of the user's voice and the
listener can't find where a sentence ends. So a second, tiny loop looks only for the name: every ~0.3 s
it transcribes the last 1.6 s of mic audio with the same fine-tuned speech model (short, no VAD, a few
tokens - ~0.2 s on the GPU) and, on "Vaayu", pauses the song, cuts off the music-filled recording and
lets the orchestrator ask "Yes?". The song resumes once the conversation goes quiet (music.py).

Only the real name counts here, not the "why you" sound-alike: song lyrics say "why you" all the time.
"""

import re
import threading
import time

import numpy as np

_NAME = re.compile(r"\b(?:vaa?yu|vaayoo|vayoo|vaayou)\b", re.I)
_EVERY = 0.3          # seconds between looks
_WINDOW = 1.6         # seconds of audio per look - "Vaayu" plus a little either side
_MIN_RMS = 0.01       # calibration knob: skip near-silent windows
_COOLDOWN = 4.0       # seconds before it can fire again
_CONFIRM_WITHIN = 1.8  # the second sighting must come this soon after the first


def heard_name(text: str) -> bool:
    return bool(_NAME.search(text or ""))


class NameSpotter:
    def __init__(self, listener, stt, on_name, should_listen):
        self.listener, self.stt, self.on_name, self.should_listen = listener, stt, on_name, should_listen
        self._last = self._first = 0.0
        threading.Thread(target=self._run, daemon=True, name="name-spotter").start()

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.stt.model.transcribe(
            audio, language="en", vad_filter=False, beam_size=1, without_timestamps=True,
            max_new_tokens=12, condition_on_previous_text=False, initial_prompt="Vaayu.")
        return " ".join(s.text for s in segments)

    def _run(self) -> None:
        while True:
            time.sleep(_EVERY)
            try:
                if not self.should_listen() or time.time() - self._last < _COOLDOWN:
                    continue
                audio = self.listener.recent_audio(_WINDOW)
                if len(audio) < 8000 or np.sqrt(np.mean(audio ** 2)) < _MIN_RMS:
                    continue
                text = self._transcribe(audio)
                if not heard_name(text):
                    self._first = 0.0
                    continue
                # Measured on 4 real songs: one look alone heard "Vaayu" in the lyrics in 6 of 396 looks
                # (a false "Yes?" every ~20 s). The real name, said once, lands in two overlapping looks
                # in a row - so it takes two.
                if self._first and time.time() - self._first <= _CONFIRM_WITHIN:
                    self._first = 0.0
                    self._last = time.time()
                    self.on_name(text)
                else:
                    self._first = time.time()
            except Exception:
                time.sleep(1)  # a failed look must never take the listener down
