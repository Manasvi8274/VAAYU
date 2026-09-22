import numpy as np
import sounddevice as sd

from assistant.audio.capture import SAMPLE_RATE


def play(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Plays a raw audio buffer. Not used by pyttsx3 (it plays its own audio
    via SAPI5) - this exists for a future TTS engine (e.g. Piper, M6) that
    returns raw samples instead of playing them itself."""
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
