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
_VOCAB_PROMPT = (
    "The assistant is named Vaayu, also spelled Vayu. Commands may start with 'wake up Vaayu', "
    "'sleep Vaayu', or 'shutdown Vaayu'. Vaayu. Vaayu. "
    "Commands may mention apps: WhatsApp, Instagram, Telegram, Discord, Zoom, Gmail, Outlook, "
    "Chrome, Brave, Edge, Firefox, YouTube, Spotify, Netflix, Notepad, Calculator, VS Code, "
    "Steam, PhonePe, Paytm, Google Pay, Zomato, Swiggy, Amazon, Flipkart, Hotstar, Jio. "
    "Commands may mention: Bluetooth, Wi-Fi, battery, volume, microphone, speaker, notes, "
    "tab, window, USB, printer, display, notification, screenshot, folder, file, password, "
    "settings, brightness, reminder, calendar, email, message, search, download, upload. "
    "Commands are sometimes in Hinglish (Hindi and English mixed in Roman script), for example: "
    "whatsapp open kro, chrome band karo, battery kitni hai, volume kam karo, ye kholo, "
    "youtube pe search karo, notepad mein likho, kya haal hai."
)


class WhisperSTT:
    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
        device: str = "cpu",
        language: str = "en",
    ):
        # faster-whisper's "auto" device detection can find a GPU but then
        # crash if the matching CUDA/cuBLAS runtime isn't installed, instead
        # of falling back to CPU - force CPU explicitly since that's this
        # project's target (see PLAN.md), rather than relying on autodetection.
        #
        # model_size defaults to "base" (multilingual), NOT "base.en" -
        # English-only models are architecturally incapable of producing
        # Hindi/Hinglish tokens at all, regardless of prompting.
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio: np.ndarray) -> str:
        # vad_filter=True runs faster-whisper's built-in (Silero) VAD as a
        # second, more accurate pass before transcription - it materially
        # reduces (but does not eliminate) hallucination on non-speech audio.
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            initial_prompt=_VOCAB_PROMPT,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()

        if len(text.split()) < _MIN_WORDS:
            return ""
        return text
