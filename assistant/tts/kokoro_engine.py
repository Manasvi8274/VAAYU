"""
Kokoro TTS (82M, ONNX): a natural voice instead of Windows' SAPI one.

Measured on this laptop: on the GTX 1650, ~0.12 s of synthesis per second of
speech ("Opened Brave." in 0.25 s); on the CPU ~0.5 s per second. Long
replies are spoken sentence by sentence - the next sentence is synthesised
while the current one plays - so the first words come out almost at once.

Default voice hm_psi: an Indian male voice, the clearest of the six tried
(6.6% word errors when the speech model transcribed what it said).

Files: models/kokoro/kokoro-v1.0.onnx + voices-v1.0.bin (from the
kokoro-onnx model-files-v1.0 release; gitignored with models/).
"""

import queue
import re
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

from assistant.tts.base import TTSEngine

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "kokoro"
_GPU_MIB = 1100  # measured: +1.1 GB on the GPU after a few replies
_GPU_ARENA_LIMIT = 1200 * 1024 * 1024  # hard cap so the 4 GB card never overflows into system RAM
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE.split(text.strip()) if p.strip()]
    out: list[str] = []
    for p in parts:  # very short bits ("Okay.") ride along with the next sentence - fewer gaps
        if out and len(out[-1]) < 12:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


class KokoroEngine(TTSEngine):
    def __init__(self, voice: str = "hm_psi", speed: float = 1.05, device: str = "auto"):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        model, voices = MODEL_DIR / "kokoro-v1.0.onnx", MODEL_DIR / "voices-v1.0.bin"
        if not model.exists() or not voices.exists():
            raise FileNotFoundError(f"Kokoro model files not found in {MODEL_DIR}")
        providers = ["CPUExecutionProvider"]
        self.device = "cpu"
        if device in ("auto", "cuda"):
            from assistant.stt.whisper_stt import _GPU_MARGIN_MIB, _gpu_free_mib
            if _gpu_free_mib() >= _GPU_MIB + _GPU_MARGIN_MIB:
                try:
                    ort.preload_dlls()
                except Exception:
                    pass
                providers = [("CUDAExecutionProvider", {"gpu_mem_limit": _GPU_ARENA_LIMIT,
                                                        "arena_extend_strategy": "kSameAsRequested"}),
                             "CPUExecutionProvider"]
        session = ort.InferenceSession(str(model), providers=providers)
        if "CUDAExecutionProvider" in session.get_providers():
            self.device = "cuda"
        self.kokoro = Kokoro.from_session(session, str(voices))
        self.voice, self.speed = voice, speed
        self._synth("Ready.")  # first call is slow (kernel setup) - pay it at startup, not on the first reply

    def _synth(self, text: str) -> tuple[np.ndarray, int]:
        samples, rate = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return np.asarray(samples, dtype=np.float32), rate

    def speak(self, text: str, stop_event=None) -> None:
        self.speak_with_amplitude(text, stop_event=stop_event)

    def save_to_file(self, text: str, path: str) -> None:
        import soundfile as sf
        samples, rate = self._synth(text)
        sf.write(path, samples, rate)

    def speak_with_amplitude(self, text: str, on_amplitude: Callable[[float], None] | None = None,
                             chunk_seconds: float = 0.03, stop_event=None) -> None:
        if not text or not text.strip():
            return
        # Unbounded: a bounded queue left the producer blocked forever after a "stop" (replies are short).
        ready: "queue.Queue[tuple[np.ndarray, int] | None]" = queue.Queue()

        def produce():  # synthesise ahead while the previous sentence plays
            try:
                for sentence in _sentences(text):
                    if stop_event is not None and stop_event.is_set():
                        break
                    ready.put(self._synth(sentence))
            finally:
                ready.put(None)
        threading.Thread(target=produce, daemon=True, name="kokoro-synth").start()
        try:
            while (item := ready.get()) is not None:
                audio, rate = item
                sd.play(audio, samplerate=rate)
                step = max(1, int(rate * chunk_seconds))
                for pos in range(0, len(audio), step):
                    if stop_event is not None and stop_event.is_set():  # the user said "stop"
                        sd.stop()
                        return
                    if on_amplitude is not None:
                        chunk = audio[pos:pos + step]
                        on_amplitude(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) if len(chunk) else 0.0)
                    time.sleep(chunk_seconds)
                sd.wait()
        finally:
            if on_amplitude is not None:
                on_amplitude(0.0)
