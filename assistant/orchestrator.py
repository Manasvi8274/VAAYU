"""
Vaayu's runtime pipeline. Default ("continuous") mode: the mic stays open,
utterances are auto-segmented by voice activity -> transcribe -> check for a
wake/sleep/shutdown command -> otherwise LLM (with tools) -> speak the reply
-> immediately back to listening, hands-free. "hotkey" mode (push-to-talk,
the original M3 design) is kept available via config but is no longer the
default.

State machine (see control_phrases.py for command detection):
  SLEEPING (the boot state) - everything is transcribed (has to be, to catch
    "wake up Vaayu"), but only wake/shutdown commands are acted on; every
    other utterance, including real commands, is silently ignored.
  AWAKE - normal operation; sleep/shutdown commands are also checked here.
  Shutdown works from either state and exits the process.

Note: this is NOT a low-power wake word in the openWakeWord/M5 sense - full
Whisper transcription still runs continuously even while "sleeping" to check
for the wake phrase, since a proper lightweight wake-word model would need
custom training data for the name "Vaayu" that wasn't available. Acceptable
for a laptop that's plugged in; a real concern for battery life on unplugged
long-running use.

Speaker verification (M4) is not wired in yet - every transcript is trusted.

Known limitation (confirmed in testing, not just theoretical): without a
true lightweight wake word, the assistant can't tell "the user talking to
it" apart from any other audio in the room while AWAKE - background
video/music audio bleeding into the mic gets transcribed and treated as a
command just like real speech does. STT-side filtering (see
stt/whisper_stt.py) only helps against silence/noise being misread as
speech; it can't distinguish real-but-irrelevant audio. The SLEEPING state
mitigates this somewhat, since irrelevant audio only matters if it happens
to contain "Vaayu" alongside a control word.
"""

import threading
from pathlib import Path
import time
from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

import assistant.tools  # noqa: F401  (registers every tool)
from assistant.activation.hotkey_listener import HotkeyActivator
from assistant.audio.barge_in import BargeIn
from assistant.audio.capture import record
from assistant.audio.vad_listener import VADListener
from assistant.config_schema import AppConfig
from assistant.control_phrases import (
    ControlCommand,
    detect_control_command,
    has_command_content,
    is_affirmative,
    is_name_mentioned,
    text_after_name,
)
from assistant.llm.ollama_client import OllamaBrain
from assistant.llm.prompts import SYSTEM_PROMPT
from assistant import interaction_log, memory
from assistant.interaction_log import Timer
from assistant.self_review import issue_log
from assistant.stt.whisper_stt import WhisperSTT
from assistant.tools.registry import registry
from assistant.tts.pyttsx3_engine import Pyttsx3Engine


# Conversation memory sent to the model: the last few turns only. Unbounded
# history made every LLM call slower over a session, and stale turns ("why
# did you not open youtube?") confused fresh commands.
_HISTORY_USER_TURNS = 2  # this turn + the one before (enough to answer "which one?")
_SAMPLE_RATE = 16000
# Each heard command's audio is kept next to its log entry (data/audio/), so a
# later fine-tune of the speech model on the user's own voice has real
# recordings + what Vaayu heard. Local only; oldest deleted past the cap.
_AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"
_AUDIO_KEEP = 3000  # ~3 s each at 16 kHz mono int16 -> roughly 300 MB at most


def _save_audio(audio) -> str | None:
    try:
        import wave
        import numpy as np
        _AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path, n = _AUDIO_DIR / f"{stamp}.wav", 1
        while path.exists():  # Windows' clock can repeat a timestamp - never overwrite a clip
            path, n = _AUDIO_DIR / f"{stamp}_{n}.wav", n + 1
        pcm = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SAMPLE_RATE)
            w.writeframes((pcm * 32767).astype(np.int16).tobytes())
        old = sorted(_AUDIO_DIR.glob("*.wav"))
        for f in old[:-_AUDIO_KEEP]:
            f.unlink(missing_ok=True)
        return path.name
    except Exception:
        return None  # a full disk etc. must never stop Vaayu listening


# While the laptop is playing sound, the mic hears it: found live, Vaayu
# acted on a YouTube video's own speech ("Today's episode is sponsored by
# the Great Courses Plus" -> seek backward). So while sound is playing, only
# sentences containing Vaayu's name count, and only the words after it. A
# bare "Vaayu" opens a short window where the next sentence counts without it.
_MEDIA_PEAK_THRESHOLD = 0.01  # calibration knob: Windows output meter peak (0-1) that counts as "playing"
_MEDIA_RECENT_SECONDS = 4.0  # sound this recently = media is playing
_NAME_WINDOW_SECONDS = 8


class _MediaMonitor:
    """Samples the Windows output meter every 100 ms in the background.

    A one-off check right after an utterance was not enough (found live): an
    utterance ends on a pause, which is exactly when the video can be in a
    quiet gap too, so 18 s of video speech got through as a command. Vaayu's
    own voice is excluded, otherwise every reply would demand the name next."""

    def __init__(self):
        self.last_sound = 0.0
        self.suppress_until = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import comtypes
        from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
        comtypes.CoInitialize()
        meter = None
        while True:
            try:
                if meter is None:
                    dev = AudioUtilities.GetSpeakers()._dev
                    meter = dev.Activate(IAudioMeterInformation._iid_, comtypes.CLSCTX_ALL, None) \
                        .QueryInterface(IAudioMeterInformation)
                self.observe(meter.GetPeakValue())
            except Exception:
                meter = None  # device changed (headphones plugged in...) - reacquire
                time.sleep(1)
            time.sleep(0.1)

    def observe(self, peak: float, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if peak > _MEDIA_PEAK_THRESHOLD and now > self.suppress_until:
            self.last_sound = now

    def playing(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) - self.last_sound < _MEDIA_RECENT_SECONDS


_monitor: "_MediaMonitor | None" = None


def _media_monitor() -> "_MediaMonitor":
    global _monitor
    if _monitor is None:
        _monitor = _MediaMonitor()
    return _monitor


# While media plays, these need no "Vaayu" first: a short sentence that routes
# straight to a player control (pause, volume, skip...). Video dialogue rarely
# is one, and if it ever is, the worst case is a pause - everything else still
# needs the name.
_PLAYER_TOOLS = {"control_media", "youtube_control", "control_video_playback", "set_video_volume", "seek_video", "music_control",
                 "set_video_fullscreen"}
_PLAYER_MAX_WORDS = 8
_MUSIC_DUCK = 0.25  # music volume while Vaayu speaks over it


def _is_player_command(text: str) -> bool:
    from assistant.llm import router
    if len(text.split()) > _PLAYER_MAX_WORDS:
        return False
    routed = router.fast_route(text)
    return bool(routed and routed[0] == "tools" and all(name in _PLAYER_TOOLS for name, _a in routed[1]))


def _media_playing() -> bool:
    try:
        return _media_monitor().playing()
    except Exception:
        return False


_COMMON_REPLIES = ["Yes? I'm listening.", "Yes?", "Okay.", "Playing.", "Paused.", "Okay, no problem.", "You're welcome!",
                   "Opened Brave.", "Opened WhatsApp.", "Sorry, I didn't get that - could you say it again?"]


def _frequent_replies(n: int = 60) -> list[str]:
    """The replies Vaayu says most (from the interaction log), short ones only - worth having ready."""
    from collections import Counter
    counts = Counter()
    try:
        for line in interaction_log.LOG_PATH.read_text(encoding="utf-8").splitlines()[-3000:]:
            if '"kind": "turn"' in line:
                reply = __import__("json").loads(line).get("reply") or ""
                if 0 < len(reply) <= 80:
                    counts[reply] += 1
    except (OSError, ValueError):
        pass
    return list(dict.fromkeys(_COMMON_REPLIES + [r for r, _ in counts.most_common(n)]))


def _make_tts(cfg):
    """Kokoro if configured and its files load, else Windows' built-in voice - never no voice."""
    if cfg.engine == "kokoro":
        try:
            from assistant.tts.kokoro_engine import KokoroEngine
            engine = KokoroEngine(voice=cfg.voice, speed=cfg.speed, device=cfg.device)
            print(f"Vaayu: Kokoro voice '{cfg.voice}' on the {engine.device}.")
            engine.prewarm(_frequent_replies())
            return engine
        except Exception as exc:
            print(f"Vaayu: Kokoro voice unavailable ({exc}) - using the Windows voice.")
            interaction_log.log_error("tts_init", exc)
    return Pyttsx3Engine(rate=cfg.rate)


class State(Enum):
    SLEEPING = "sleeping"
    AWAKE = "awake"


class Orchestrator:
    def __init__(self, config: AppConfig, gui_queue=None):
        self.config = config
        self.continuous = config.activation.mode == "continuous"
        self.state = State.SLEEPING
        self.should_exit = False
        # Optional queue.Queue to a status_orb.StatusOrb running on the main
        # thread (see gui/status_orb.py) - when set, replies are spoken via
        # speak_with_amplitude so the orb can glow in sync with real speech.
        self.gui_queue = gui_queue

        if self.continuous:
            self.listener = VADListener(
                aggressiveness=config.listening.vad_aggressiveness,
                silence_duration_ms=config.listening.silence_duration_ms,
                max_utterance_seconds=config.listening.max_utterance_seconds,
                preroll_frames=max(1, config.listening.preroll_ms // 30),
            )
        else:
            self.activator = HotkeyActivator(config.activation.hotkey)

        self.stt = WhisperSTT(model_size=config.stt.whisper_model_size, language=config.stt.language,
                              device=config.stt.device)
        self.tts = _make_tts(config.tts)
        self.brain = OllamaBrain(host=config.llm.host, model=config.llm.model)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._name_window_until = 0.0

    def _capture_utterance(self):
        if self.continuous:
            return self.listener.listen_for_utterance()
        self.activator.wait_for_trigger()
        return record(self.config.activation.record_seconds)

    def _set_state(self, new_state: "State") -> None:
        self.state = new_state
        if self.gui_queue is not None:
            self.gui_queue.put({"kind": "state", "value": new_state.value})

    def _speak(self, text: str) -> bool:
        """Speaks; True if the user cut it off with "stop" (audio/barge_in.py) - Vaayu then
        just goes back to listening."""
        stop, barge = threading.Event(), None
        if self.continuous:
            self.listener.mute()
            barge = BargeIn(self.listener, self.stt, stop, text)
        if _monitor is not None:
            _monitor.suppress_until = float("inf")  # Vaayu's own voice isn't "media"
        from assistant.tools import music
        music.player.duck = _MUSIC_DUCK  # a song Vaayu is playing goes quiet while he talks
        try:
            if self.gui_queue is not None:
                self.tts.speak_with_amplitude(
                    text,
                    on_amplitude=lambda v: self.gui_queue.put({"kind": "amplitude", "value": v}),
                    **({"stop_event": stop} if barge else {}),
                )
            elif barge:
                self.tts.speak(text, stop_event=stop)
            else:
                self.tts.speak(text)
        finally:
            music.player.duck = 1.0
            if barge:
                barge.close()
            if _monitor is not None:
                _monitor.suppress_until = time.time() + 0.6  # the speaker's tail after playback ends
            if self.continuous:
                self.listener.unmute()
        if stop.is_set():
            print(f"Vaayu: (stopped - heard {barge.heard!r})")
            interaction_log.log_event("barge_in", heard=barge.heard, was_saying=text)
            self._name_window_until = time.time() + _NAME_WINDOW_SECONDS  # the next sentence is for Vaayu
        return stop.is_set()

    def _voice_confirm(self, name: str, arguments: dict) -> bool:
        """Confirmation gate for sensitive tools (requires_confirmation=True),
        e.g. sending a WhatsApp message - speaks the actual question (reading
        back real argument values like the message text, not a vague "are you
        sure") and listens for a real spoken yes/no in the same turn, instead
        of the old cli_confirm default which blocked on terminal input() and
        would just hang forever in voice mode."""
        question = registry.get_confirmation_prompt(name, arguments)
        print(f"Vaayu (confirming): {question}")
        self._speak(question)

        audio = self._capture_utterance()
        if len(audio) == 0:
            print("Vaayu: No response heard - not going ahead with that.")
            return False

        answer_text = self.stt.transcribe(audio)
        print(f"You said: {answer_text}")
        confirmed = is_affirmative(answer_text)
        interaction_log.log_event("confirm", tool=name, args=arguments, question=question, heard=answer_text,
                                  confirmed=confirmed)
        if not confirmed:
            print("Vaayu: Okay, not doing that.")
        return confirmed

    def _maybe_run_daily_checkin(self) -> None:
        """Proactive, Vaayu-initiated check-in about issues logged during
        real use (see self_review/issue_log.py) - NOT autonomous code
        changes, just asking permission to write a report for a real fix
        session. Runs once per calendar day (in the configured timezone),
        at or after the configured time, regardless of sleep/awake state -
        this is Vaayu speaking up on its own, not gated behind the user
        addressing it first. If today's check-in already happened (said yes
        or no), it stays quiet the rest of the day; if the process was
        asleep/off straight through the scheduled time, it still asks the
        next time this runs after that time, even if that's hours later -
        it does not skip a day just because the exact moment passed."""
        if not self.config.self_review.enabled:
            return
        try:
            tz = ZoneInfo(self.config.self_review.timezone)
            hour, minute = (int(x) for x in self.config.self_review.check_in_time.split(":"))
        except (ValueError, KeyError):
            return  # misconfigured - fail quiet rather than crash the main loop over this

        now = datetime.now(tz)
        if (now.hour, now.minute) < (hour, minute):
            return

        today_str = now.date().isoformat()
        if issue_log.read_last_checkin_date() == today_str:
            return
        issue_log.write_last_checkin_date(today_str)  # offered today, regardless of outcome below

        issues = issue_log.read_unacknowledged()
        if not issues:
            return

        summary = issue_log.summarize_for_speech(issues)
        print(f"Vaayu (daily check-in): {summary}")
        self._speak(f"Quick check-in - I ran into {summary}. Want me to write these up for review?")

        audio = self._capture_utterance()
        if len(audio) == 0:
            return
        answer_text = self.stt.transcribe(audio)
        print(f"You said: {answer_text}")

        if is_affirmative(answer_text):
            report_path = issue_log.write_daily_report(issues)
            issue_log.acknowledge_all()
            print(f"Vaayu: wrote {report_path}")
            self._speak("Update's done - I've written up today's issues for review.")
        else:
            self._speak("Okay, I'll bring it up again tomorrow.")

    def handle_one_turn(self) -> None:
        self._maybe_run_daily_checkin()
        audio = self._capture_utterance()
        if len(audio) == 0:
            return

        with Timer() as stt_time:
            text = self.stt.transcribe(audio)
        audio_seconds = round(len(audio) / _SAMPLE_RATE, 1)
        if not text.strip():
            interaction_log.log_event("heard_nothing", audio_seconds=audio_seconds, stt_ms=stt_time.ms,
                                      state=self.state.value)
            return

        print(f"You said: {text}")
        interaction_log.log_event("heard", text=text, state=self.state.value, audio_seconds=audio_seconds,
                                  stt_ms=stt_time.ms, audio_file=_save_audio(audio))
        command = detect_control_command(text)
        if command is not ControlCommand.NONE:
            interaction_log.log_event("control", command=command.value, text=text, state=self.state.value)

        if command is ControlCommand.SHUTDOWN:
            print("Vaayu: Shutting down. Goodbye.")
            self._speak("Shutting down. Goodbye.")
            if self.gui_queue is not None:
                self.gui_queue.put({"kind": "shutdown"})
            self.should_exit = True
            return

        if command is ControlCommand.SLEEP:
            if self.state is State.AWAKE:
                self._set_state(State.SLEEPING)
                print("Vaayu: Going to sleep. Say Vaayu when you need me.")
                self._speak("Going to sleep. Say Vaayu when you need me.")
            return

        woke_with_command = False
        if self.state is State.SLEEPING:
            if not is_name_mentioned(text):
                interaction_log.log_event("ignored_sleeping", text=text)
                return  # not addressed to Vaayu at all - ignore silently
            self._set_state(State.AWAKE)
            if not has_command_content(text):
                # bare "wake up Vaayu" / "Vaayu" - nothing else to act on
                print("Vaayu: Yes? I'm listening.")
                self._speak("Yes? I'm listening.")
                self._name_window_until = time.time() + _NAME_WINDOW_SECONDS
                return
            # else: "Vaayu, open brave" style - wake AND handle the attached
            # command in this same turn, falling through below.
            woke_with_command = True
            text = text_after_name(text) or text

        if self.state is State.AWAKE and not woke_with_command and _media_playing():
            if time.time() < self._name_window_until:
                self._name_window_until = 0.0  # this sentence was invited by a bare "Vaayu"
            elif not is_name_mentioned(text):
                if not _is_player_command(text):
                    interaction_log.log_event("ignored_media", text=text)
                    return
                # Real log: "Pause the video." heard perfectly, twice, and ignored for lacking "Vaayu".
                interaction_log.log_event("player_command", text=text)
            else:
                rest = text_after_name(text)
                if not has_command_content(rest):
                    self._speak("Yes?")
                    self._name_window_until = time.time() + _NAME_WINDOW_SECONDS
                    interaction_log.log_event("name_only_media", text=text)
                    return
                interaction_log.log_event("media_command", heard=text, used=rest)
                text = rest

        self.messages.append({"role": "user", "content": text})
        self.brain.trace = []
        error = None
        with Timer() as brain_time:
            try:
                reply, self.messages = self.brain.respond(self.messages, registry, confirm=self._voice_confirm)
            except Exception as exc:  # e.g. a model call timing out - must not end the listening loop
                error = exc
                issue_log.log_issue(user_said=text, tool_name=None, error=f"brain.respond failed: {exc!r}", kind="respond_crash")
                interaction_log.log_error("brain.respond", exc, text=text)
                reply = "Sorry, that took too long or went wrong - please try again."
        self.messages = _trim_history(self.messages)
        # Remember the talk (data/conversations.jsonl), and learn lasting facts
        # from personal conversation - here, not in brain.respond, so only real
        # turns are remembered, never benchmark/test runs.
        memory.log_turn(text, reply)
        if error is None and any(step.get("path") == "chat" for step in self.brain.trace):
            memory.learn_async(self.brain.client, self.brain.model, text, reply)
        print(f"Assistant: {reply}")
        with Timer() as speak_time:
            self._speak(reply)
        interaction_log.log_event("turn", text=text, reply=reply, trace=self.brain.trace, brain_ms=brain_time.ms,
                                  speak_ms=speak_time.ms, stt_ms=stt_time.ms, failed=error is not None)

    def run_forever(self) -> None:
        _media_monitor()  # start watching the sound level before the first command
        from assistant.tools import youtube_taste
        youtube_taste.start_watch_logger()  # what plays on YouTube, whoever started it -> taste
        from assistant import research
        research.start(self.config, self.brain.client)  # reads about their topics while idle
        if self.continuous:
            self.listener.start()
            print(
                "Vaayu is running (sleeping). Say 'Vaayu' to wake it, e.g. 'Vaayu, open brave' "
                "wakes it and runs that command in one go (Ctrl+C to force quit)."
            )
        else:
            print(
                f"Vaayu is running (sleeping, hotkey mode). Press {self.config.activation.hotkey}, "
                "then say 'Vaayu' (e.g. 'Vaayu, open brave') to wake it and run a command in one go "
                "(Ctrl+C to force quit)."
            )
        interaction_log.log_event("start", model=self.config.llm.model, mode=self.config.activation.mode,
                                  stt_model=self.config.stt.whisper_model_size,
                                  stt_device=getattr(self.stt, "device", "?"),
                                  tts=type(self.tts).__name__, tts_device=getattr(self.tts, "device", "cpu"))
        try:
            while not self.should_exit:
                try:
                    self.handle_one_turn()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    # Any other error used to end this thread while the orb
                    # stayed up - Vaayu looked alive but was deaf. Log, go on.
                    interaction_log.log_error("handle_one_turn", exc, state=self.state.value)
                    print(f"Vaayu: internal error, still listening - {exc!r}")
        except KeyboardInterrupt:
            print("\nGoodbye.")
        finally:
            interaction_log.log_event("stop", should_exit=self.should_exit)
            if self.continuous:
                self.listener.stop()


def _trim_history(messages: list[dict]) -> list[dict]:
    """System prompt + the last _HISTORY_USER_TURNS user turns (each with
    whatever assistant/tool messages followed it)."""
    system = [m for m in messages[:1] if m.get("role") == "system"]
    user_idx = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_idx) <= _HISTORY_USER_TURNS:
        return messages
    return system + messages[user_idx[-_HISTORY_USER_TURNS]:]
