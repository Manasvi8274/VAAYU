import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "vaayu-finetuned"


class ActivationConfig(BaseModel):
    # "continuous" (default): hands-free, mic always listening, utterances are
    # auto-segmented by voice activity (see ListeningConfig). "hotkey": the
    # original M3 push-to-talk mode, kept available but not used by default.
    mode: Literal["continuous", "hotkey", "wakeword"] = "continuous"
    hotkey: str = "ctrl+alt+j"
    wake_phrase: str = "hey_jarvis"
    record_seconds: float = 5.0  # only used in "hotkey" mode


class ListeningConfig(BaseModel):
    vad_aggressiveness: int = 2  # 0 (least strict) - 3 (most strict)
    silence_duration_ms: int = 900  # silence needed after speech to end an utterance
    max_utterance_seconds: float = 20.0
    preroll_ms: int = 300  # rolling lookback so soft-onset words aren't clipped at utterance start


class TTSConfig(BaseModel):
    rate: int = 210  # pyttsx3 words-per-minute; 210 = 1.2x the original 175 default
    engine: str = "kokoro"  # "kokoro" (natural voice, models/kokoro) or "pyttsx3" (Windows' built-in voice)
    voice: str = "hm_psi"  # Kokoro voice - hm_psi: Indian male; bm_george: British male; af_heart: US female
    speed: float = 1.05
    device: str = "cpu"  # Kokoro runs on the CPU (see tts/kokoro_engine.py for why not the GPU)


class STTConfig(BaseModel):
    # "base" (multilingual) not "base.en" - required for Hinglish/Hindi words
    # at all, since the .en models are English-only and cannot produce
    # non-English tokens no matter how they're spoken. See PLAN.md for the
    # testing behind this (and its limits - could not be fully verified with
    # real Hindi pronunciation on this machine, no Hindi voice is installed).
    whisper_model_size: str = "base"
    # "en" keeps output in Latin/Roman script (matches how Hinglish is
    # normally typed, e.g. "whatsapp open kro") rather than switching to
    # Devanagari for the Hindi portions - "hi" or leaving this unset are
    # available to try if "en" doesn't work well for you in practice.
    language: str = "en"
    # "auto": GPU when it fits next to the 3B brain with a safety margin (the
    # 4 GB limit), else CPU. "cpu"/"cuda" to force one.
    device: str = "auto"


class SpeakerVerificationConfig(BaseModel):
    enabled: bool = False
    similarity_threshold: float = 0.75


class FilesConfig(BaseModel):
    safe_roots: list[str] = ["~/Desktop", "~/Documents", "~/Downloads"]

    def resolved_roots(self) -> list[Path]:
        return [Path(p).expanduser().resolve() for p in self.safe_roots]


class LoggingConfig(BaseModel):
    level: str = "INFO"


class SelfReviewConfig(BaseModel):
    # Daily log review (assistant/daily_review.py): at this time each day,
    # in the background, Vaayu digests the day's logs, learns from them, runs
    # a Claude fix session if claude_fix (the user's Claude Pro login via the
    # `claude` CLI - guardian changes are always rejected), then deletes the
    # raw logs. Times are HH:MM 24-hour, IST by default per explicit
    # instruction ("timings will be in IST until I mention any timezone format").
    enabled: bool = True
    check_in_time: str = "21:00"
    timezone: str = "Asia/Kolkata"
    claude_fix: bool = True


class ResearchConfig(BaseModel):
    # Learning about the world while idle (assistant/research.py): topics from the user's own
    # questions and their YouTube taste, summarised into knowledge notes. Budgeted, and only when
    # nobody has spoken for idle_minutes and nothing is playing.
    enabled: bool = True
    per_hour: int = 6
    per_day: int = 40
    idle_minutes: int = 10


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    activation: ActivationConfig = ActivationConfig()
    listening: ListeningConfig = ListeningConfig()
    tts: TTSConfig = TTSConfig()
    stt: STTConfig = STTConfig()
    speaker_verification: SpeakerVerificationConfig = SpeakerVerificationConfig()
    files: FilesConfig = FilesConfig()
    logging: LoggingConfig = LoggingConfig()
    self_review: SelfReviewConfig = SelfReviewConfig()
    research: ResearchConfig = ResearchConfig()


# Repo root, computed from this file's own location rather than the current
# working directory - matters for auto-start (Task Scheduler / Startup
# folder launch a process with an unpredictable working directory, unlike a
# manual "cd to the repo, then run" session).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path = _REPO_ROOT / "config" / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Copy config/config.example.yaml to "
            f"config/config.yaml first."
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)


def set_check_in_time(hhmm: str, path: str | Path | None = None) -> None:
    """Persists a new self_review.check_in_time into config.yaml - a
    targeted text edit (find/replace the existing line, or append a new
    self_review section) rather than a full re-serialize of the whole
    config, specifically to avoid destroying the user's own comments and
    formatting in their hand-edited config.yaml (a full
    yaml.dump(config.model_dump()) would flatten all of that)."""
    if path is None:
        path = _REPO_ROOT / "config" / "config.yaml"
    path = Path(path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    if re.search(r"^self_review:\s*$", text, re.MULTILINE):
        if re.search(r"^\s+check_in_time:.*$", text, re.MULTILINE):
            new_text = re.sub(
                r"^(\s+check_in_time:).*$",
                lambda m: f"{m.group(1)} \"{hhmm}\"",
                text,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            new_text = re.sub(
                r"^(self_review:\s*)$",
                lambda m: f'{m.group(1)}\n  check_in_time: "{hhmm}"',
                text,
                count=1,
                flags=re.MULTILINE,
            )
    else:
        separator = "\n" if text and not text.endswith("\n") else ""
        new_text = text + separator + f'\nself_review:\n  check_in_time: "{hhmm}"\n'

    path.write_text(new_text, encoding="utf-8")
