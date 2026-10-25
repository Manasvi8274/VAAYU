import numpy as np
from faster_whisper import WhisperModel

# Whisper-family models are trained to always produce fluent text and can
# hallucinate plausible-sounding sentences from pure noise/silence (a known,
# unsolved limitation - confirmed in testing: ambient room noise alone
# produced full invented sentences with moderate confidence scores, so
# no_speech_prob/avg_logprob thresholds alone weren't reliable). This is a
# pragmatic, not perfect, mitigation - a wake word (M5) would cut this down
# far more, since STT would then only run after a deliberate trigger.
_MIN_WORDS = 2

# Whisper's initial_prompt biases transcription toward expected vocabulary.
# Tested and confirmed decisive for this project: "open whatsapp" was
# misheard as "Open what's up?"/"Open what's that?" with no prompt, and
# correctly transcribed as "Open WhatsApp." once given this hint. Keep this
# list roughly in sync with what the assistant's tools actually reference
# (app names especially) - add new ones as they come up.
#
# Hinglish (Hindi-English code-mixed) command words are included too, since
# the model was switched to a multilingual one (see WhisperSTT below)
# specifically to support commands like "whatsapp open kro". IMPORTANT
# CAVEAT: this part is NOT verified the way the rest of this file's design
# choices are - there is no Hindi voice or language pack on the dev machine
# this was tested on, so realistic Hindi/Hinglish pronunciation could not be
# tested end-to-end. If specific Hindi words keep getting misheard in real
# use, add them here the same way app names were added.
# Whisper keeps only the LAST ~223 tokens of initial_prompt. Found by
# measuring: the old prompt was 258 tokens, so its opening - the line that
# teaches the name "Vaayu" - was silently cut off every time. Keep this under
# ~200 tokens (tests/test_whisper_prompt.py checks it) and the name LAST.
# Payment apps were removed: Vaayu refuses them anyway.
_VOCAB_PROMPT = (
    "Apps: WhatsApp, Telegram, Discord, Zoom, Gmail, Outlook, Chrome, Brave, Edge, YouTube, Spotify, "
    "Netflix, Hotstar, Notepad, Calculator, VS Code, Steam, Tekken. "
    "Commands: pause the video, resume the video, play, volume 50 percent, mute, next video, skip ad, "
    "captions, subtitles, full screen, theater mode, speed, like, subscribe, Bluetooth, Wi-Fi, battery, "
    "screenshot, tab, window. "
    "Hinglish: whatsapp open kro, chrome band karo, battery kitni hai, volume kam karo, gaana chalao, "
    "video roko, kya haal hai. "
    "The assistant is named Vaayu: wake up Vaayu, sleep Vaayu, shutdown Vaayu, Vaayu die. Vaayu."
)


# One-word commands are real ("pause", "mute") - the 2-word minimum above
# silently dropped every one of them. These pass; other single words (the
# usual noise hallucinations: "you", "Thank you", "Bye") still don't.
_ONE_WORD_COMMANDS = {"pause", "play", "resume", "stop", "mute", "unmute", "next", "previous", "louder", "quieter",
                      "skip", "vaayu", "vayu", "replay", "restart", "fullscreen", "screenshot", "minimize",
                      "ruko", "bas", "chup", "wait", "enough", "quiet", "cancel"}  # "stop" while Vaayu talks (audio/barge_in.py)

# Approximate extra VRAM per model on the GPU (int8_float16, measured on the
# GTX 1650 with the 3B brain loaded: small +348 MiB, large-v3-turbo +964 MiB).
_GPU_MIB = {"tiny": 200, "base": 250, "small": 400, "medium": 900, "large-v3-turbo": 1000, "turbo": 1000}
_MAX_TOKENS = 120  # per 30 s window (see transcribe)
_GPU_MARGIN_MIB = 300  # left free for Windows/browser so the 4 GB card never overflows


def _load_cuda_libraries() -> None:
    """NVIDIA's cuBLAS/cuDNN come as pip packages (nvidia-cublas-cu12,
    nvidia-cudnn-cu12); Windows only finds their DLLs if told where."""
    import os
    try:
        import nvidia.cublas
        import nvidia.cudnn
    except ImportError:
        return
    for mod in (nvidia.cublas, nvidia.cudnn):
        d = os.path.join(list(mod.__path__)[0], "bin")
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _gpu_free_mib() -> int:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        total, used = (int(x) for x in out.strip().splitlines()[0].split(","))
        return total - used
    except Exception:
        return 0


def _resolve_model(model_size: str) -> str:
    """A folder path (the fine-tuned model, scripts/stt_finetune.py) resolves
    from the repo root; if it's missing, the stock model it was trained from."""
    import os
    if "/" not in model_size and "\\" not in model_size:
        return model_size
    from pathlib import Path
    path = Path(model_size)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path
    if os.path.isfile(path / "model.bin"):
        return str(path)
    fallback = next((k for k in ("small", "base", "medium", "tiny") if k in path.name.lower()), "small")
    print(f"Vaayu: speech model folder {path} not found - using the stock '{fallback}' model.")
    return fallback


class WhisperSTT:
    def __init__(
        self,
        model_size: str = "base",
        compute_type: str | None = None,
        device: str = "auto",
        language: str = "en",
    ):
        # model_size is multilingual (no ".en" models) - English-only models
        # are architecturally incapable of producing Hindi/Hinglish tokens.
        #
        # device "auto": the GPU when the CUDA libraries load AND there's room
        # for the model plus a safety margin (so the 4 GB card shared with the
        # 3B brain never overflows), otherwise the CPU. faster-whisper's own
        # "auto" can't be used - it picks the GPU and then crashes if the CUDA
        # runtime is missing, instead of falling back.
        self.language = language
        self.device = "cpu"
        model_size = _resolve_model(model_size)
        if device in ("auto", "cuda"):
            size = next((k for k in sorted(_GPU_MIB, key=len, reverse=True) if k in model_size.lower()), None)
            need = _GPU_MIB.get(size, 1000) + _GPU_MARGIN_MIB
            if _gpu_free_mib() >= need:
                try:
                    _load_cuda_libraries()
                    self.model = WhisperModel(model_size, device="cuda", compute_type=compute_type or "int8_float16")
                    self.device = "cuda"
                    return
                except Exception as exc:  # no usable CUDA runtime - fall back, don't crash
                    print(f"Vaayu: speech model can't use the GPU ({exc}) - using the CPU.")
            else:
                print(f"Vaayu: not enough free GPU memory for the '{model_size}' speech model - using the CPU.")
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type or "int8")

    def transcribe(self, audio: np.ndarray) -> str:
        # vad_filter=True runs faster-whisper's built-in (Silero) VAD as a
        # second, more accurate pass before transcription - it materially
        # reduces (but does not eliminate) hallucination on non-speech audio.
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            initial_prompt=_VOCAB_PROMPT,
            # Real log: two utterances over loud music took 9 s each - Whisper looping on song lyrics.
            # A spoken command or chat sentence is well under this many tokens per 30 s window.
            max_new_tokens=_MAX_TOKENS,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if text.isupper() and len(text) > 12:  # the fine-tune's EdAcc labels are ALL CAPS - it picked that up
            text = text[0] + text[1:].lower()

        words = text.split()
        if len(words) < _MIN_WORDS and not (words and words[0].strip(".,!?").lower() in _ONE_WORD_COMMANDS):
            return ""
        return text
