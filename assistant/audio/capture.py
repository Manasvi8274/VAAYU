import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # what faster-whisper expects


def record(duration_seconds: float) -> np.ndarray:
    """Records a fixed-length mono clip from the default input device."""
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()
