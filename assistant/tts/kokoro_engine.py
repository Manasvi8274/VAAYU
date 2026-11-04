"""
Kokoro TTS (82M, ONNX): a natural voice instead of Windows' SAPI one.

Runs on the CPU (~0.5 s of synthesis per second of speech here). The GPU was
tried and dropped: onnxruntime's CUDA build crashed Vaayu at startup
(0xC0000409, "cudnnGetLibConfig" - CTranslate2, onnxruntime and torch each
bring their own cuDNN into the one process), and DirectML can't run
Kokoro's ConvTranspose layers. To stay quick: replies are spoken sentence
by sentence (the next is synthesised while the current one plays), and the
audio of replies already said is cached ("Opened Brave." is instant the
second time).

Default voice am_echo: a US male voice like the old Windows one (Microsoft David) - the
user found hm_psi "very heavy". "Vaayu, change your voice" switches (tools/voice.py).

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
_SENTENCE = re.compile(r"(?<=[.!?।])\s+")  # "।" is Hindi's full stop
_CLAUSE = re.compile(r"(?<=[,;:])\s+|\s+(?=- )")
_CACHE_SIZE = 300
_CACHE_MAX_CHARS = 80


def _sentences(text: str) -> list[str]:
    parts = []
    for sentence in (p.strip() for p in _SENTENCE.split(text.strip()) if p.strip()):
        # Long sentences go by clause: on the CPU the first sound comes sooner (live: a 4 s weather
        # sentence took 2 s before anything played).
        parts += [c.strip() for c in _CLAUSE.split(sentence) if c.strip()] if len(sentence) > 50 else [sentence]
    out: list[str] = []
    for p in parts:  # very short bits ("Okay.") ride along with the next sentence - fewer gaps
        if out and len(out[-1]) < 12:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


# The voices worth offering, lightest/closest to the old Windows voice (Microsoft David) first -
# measured: pitch/pace vs David, and word errors when the speech model heard each one.
VOICES = {"echo": "am_echo", "lewis": "bm_lewis", "george": "bm_george", "daniel": "bm_daniel",
          "fable": "bm_fable", "adam": "am_adam", "omega": "hm_omega", "psi": "hm_psi", "michael": "am_michael",
          "puck": "am_puck", "heart": "af_heart", "emma": "bf_emma"}
# Hindi (user, 2026-11-04: "I want Vaayu to speak in Hindi also"): a sentence in Devanagari is spoken with Kokoro's
# own Hindi voice - measured: Whisper large-v3 read hm_omega/hm_psi/hf_alpha back almost word for word.
HINDI_VOICE = "hm_omega"  # Kokoro's Hindi - used only if the native Hindi voice below can't load
# User (2026-11-05): "I do not like his Hindi accent, I want the real Hindi accent". Kokoro's Hindi comes from
# pronunciation rules; Piper's Hindi voices are trained on native Hindi speakers (models/piper, rhasspy/piper-voices)
# and are 4x faster (0.33 s for 6 s of speech, CPU). "Vaayu, Hindi voice Rohan" switches (data/voice_hi.txt).
PIPER_DIR = MODEL_DIR.parent / "piper"
HINDI_VOICES = {"pratham": "male", "rohan": "male", "priyamvada": "female"}
HINDI_VOICE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "voice_hi.txt"
# The male voices said "वाई" for his name inside a long sentence; written with a hyphen they say it right.
_HINDI_SAY = {"वायु": "वा-यु", "वायू": "वा-यु"}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
VOICE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "voice.txt"
ACTIVE: "KokoroEngine | None" = None  # the running engine, for the set_voice tool


def _saved_voice() -> str | None:
    try:
        return VOICE_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _saved_hindi_voice() -> str:
    try:
        name = HINDI_VOICE_FILE.read_text(encoding="utf-8").strip()
        return name if name in HINDI_VOICES else "pratham"
    except OSError:
        return "pratham"  # calibration knob: the default native Hindi voice


class KokoroEngine(TTSEngine):
    def __init__(self, voice: str = "hm_psi", speed: float = 1.05, device: str = "cpu"):
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        model, voices = MODEL_DIR / "kokoro-v1.0.onnx", MODEL_DIR / "voices-v1.0.bin"
        if not model.exists() or not voices.exists():
            raise FileNotFoundError(f"Kokoro model files not found in {MODEL_DIR}")
        self.device = "cpu"  # see the module docstring for why not the GPU
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self.kokoro = Kokoro.from_session(session, str(voices))
        saved = _saved_voice()
        self.voice = saved if saved in self.kokoro.get_voices() else voice  # "Vaayu, change your voice" sticks
        self.speed = speed
        self._synth("Ready.")  # first call is slow (kernel setup) - pay it at startup, not on the first reply
        threading.Thread(target=self._piper, daemon=True, name="hindi-voice-load").start()  # 3 s, not on a reply
        global ACTIVE
        ACTIVE = self

    def _piper(self):
        """The native Hindi voice (loaded on first Hindi sentence); None if it isn't there - Kokoro's then."""
        name = _saved_hindi_voice()
        if getattr(self, "_piper_voice", (None, None))[0] != name:
            try:
                from piper import PiperVoice
                self._piper_voice = (name, PiperVoice.load(str(PIPER_DIR / f"hi_IN-{name}-medium.onnx")))
            except Exception:
                self._piper_voice = (name, None)
        return self._piper_voice[1]

    def set_hindi_voice(self, name: str) -> None:
        if name not in HINDI_VOICES:
            raise ValueError(f"No Hindi voice called {name} - there's {', '.join(HINDI_VOICES)}.")
        HINDI_VOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        HINDI_VOICE_FILE.write_text(name, encoding="utf-8")
        self._cache = {k: v for k, v in self._cache.items() if not _DEVANAGARI.search(k)}

    def set_voice(self, voice: str) -> None:
        if voice not in self.kokoro.get_voices():
            raise ValueError(f"No voice called {voice}.")
        self.voice, self._cache = voice, {}
        VOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        VOICE_FILE.write_text(voice, encoding="utf-8")

    def _synth(self, text: str) -> tuple[np.ndarray, int]:
        if text in self._cache:
            return self._cache[text]
        hindi = bool(_DEVANAGARI.search(text))
        if hindi and (piper := self._piper()):
            said = text
            for word, spoken in _HINDI_SAY.items():
                said = said.replace(word, spoken)
            samples = np.concatenate([c.audio_float_array for c in piper.synthesize(said)] or [np.zeros(1)])
            rate = piper.config.sample_rate
        else:
            samples, rate = self.kokoro.create(text, voice=HINDI_VOICE if hindi else self.voice, speed=self.speed,
                                               lang="hi" if hindi else "en-us")
        out = np.asarray(samples, dtype=np.float32), rate
        if len(text) <= _CACHE_MAX_CHARS:  # short, repeatable replies ("Playing.", "Opened Brave.")
            if len(self._cache) >= _CACHE_SIZE:
                self._cache.pop(next(iter(self._cache)))  # oldest first
            self._cache[text] = out
        return out

    def prewarm(self, replies: list[str]) -> None:
        """Synthesise frequent replies in the background, so they play instantly the first time too."""
        def run():
            for text in replies:
                for part in _sentences(text):
                    try:
                        self._synth(part)
                    except Exception:
                        return
        threading.Thread(target=run, daemon=True, name="kokoro-prewarm").start()

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
