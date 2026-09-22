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

from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

import assistant.tools  # noqa: F401  (registers every tool)
from assistant.activation.hotkey_listener import HotkeyActivator
from assistant.audio.capture import record
from assistant.audio.vad_listener import VADListener
from assistant.config_schema import AppConfig
from assistant.control_phrases import (
    ControlCommand,
    detect_control_command,
    has_command_content,
    is_affirmative,
    is_name_mentioned,
)
from assistant.llm.ollama_client import OllamaBrain
from assistant.llm.prompts import SYSTEM_PROMPT
from assistant.self_review import issue_log
from assistant.stt.whisper_stt import WhisperSTT
from assistant.tools.registry import registry
from assistant.tts.pyttsx3_engine import Pyttsx3Engine


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

        self.stt = WhisperSTT(model_size=config.stt.whisper_model_size, language=config.stt.language)
        self.tts = Pyttsx3Engine(rate=config.tts.rate)
        self.brain = OllamaBrain(host=config.llm.host, model=config.llm.model)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _capture_utterance(self):
        if self.continuous:
            return self.listener.listen_for_utterance()
        self.activator.wait_for_trigger()
        return record(self.config.activation.record_seconds)

    def _set_state(self, new_state: "State") -> None:
        self.state = new_state
        if self.gui_queue is not None:
            self.gui_queue.put({"kind": "state", "value": new_state.value})

    def _speak(self, text: str) -> None:
        if self.continuous:
            self.listener.mute()
        try:
            if self.gui_queue is not None:
                self.tts.speak_with_amplitude(
                    text,
                    on_amplitude=lambda v: self.gui_queue.put({"kind": "amplitude", "value": v}),
                )
            else:
                self.tts.speak(text)
        finally:
            if self.continuous:
                self.listener.unmute()

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

        text = self.stt.transcribe(audio)
        if not text.strip():
            return

        print(f"You said: {text}")
        command = detect_control_command(text)

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

        if self.state is State.SLEEPING:
            if not is_name_mentioned(text):
                return  # not addressed to Vaayu at all - ignore silently
            self._set_state(State.AWAKE)
            if not has_command_content(text):
                # bare "wake up Vaayu" / "Vaayu" - nothing else to act on
                print("Vaayu: Yes? I'm listening.")
                self._speak("Yes? I'm listening.")
                return
            # else: "Vaayu, open brave" style - wake AND handle the attached
            # command in this same turn, falling through below.

        self.messages.append({"role": "user", "content": text})
        reply, self.messages = self.brain.respond(self.messages, registry, confirm=self._voice_confirm)
        print(f"Assistant: {reply}")
        self._speak(reply)

    def run_forever(self) -> None:
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
        try:
            while not self.should_exit:
                self.handle_one_turn()
        except KeyboardInterrupt:
            print("\nGoodbye.")
        finally:
            if self.continuous:
                self.listener.stop()
