import os
import tempfile
import time
import wave
from typing import Callable

import numpy as np
import pyttsx3
import sounddevice as sd

from assistant.tts.base import TTSEngine


class Pyttsx3Engine(TTSEngine):
    """
    Creates a fresh pyttsx3 engine per call instead of reusing one instance.
    Tested and confirmed this matters: reusing a single engine across
    multiple say()+runAndWait() calls plays real audio on the first call only
    - later calls complete with no error but produce no audible output at all
    (RMS ~3-5, versus thousands for real speech). This is a known pyttsx3/
    SAPI5 quirk on Windows; re-initializing per call is the standard fix.
    """

    def __init__(self, rate: int = 175):
        self.rate = rate

    def _new_engine(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        return engine

    def speak(self, text: str) -> None:
        if not text:
            return
        engine = self._new_engine()
        engine.say(text)
        engine.runAndWait()

    def save_to_file(self, text: str, path: str) -> None:
        engine = self._new_engine()
        engine.save_to_file(text, path)
        engine.runAndWait()

    def speak_with_amplitude(
        self,
        text: str,
        on_amplitude: Callable[[float], None] | None = None,
        chunk_seconds: float = 0.03,
    ) -> None:
        """Like speak(), but synthesizes to a buffer and plays it via
        sounddevice instead of pyttsx3's own playback, reporting real RMS
        amplitude every ~30ms during playback via on_amplitude - used to
        drive a voice-reactive UI (an actual waveform-driven glow, not a
        generic canned pulse animation)."""
        if not text:
            return

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self.save_to_file(text, tmp_path)
            with wave.open(tmp_path, "rb") as wf:
                channels = wf.getnchannels()
                framerate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())

            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            sd.play(audio, samplerate=framerate)
            chunk_size = max(1, int(framerate * chunk_seconds))
            pos = 0
            while pos < len(audio):
                chunk = audio[pos : pos + chunk_size]
                if on_amplitude is not None and len(chunk) > 0:
                    rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                    on_amplitude(rms)
                pos += chunk_size
                time.sleep(chunk_seconds)
            sd.wait()
        finally:
            if on_amplitude is not None:
                on_amplitude(0.0)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
