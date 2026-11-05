"""
The user talking while Vaayu talks: he goes quiet at once, hears the whole sentence, then decides.

User (2026-11-05): "when I speak I want Vaayu to stop saying anything like in pause mode and then hear my voice
first ... I said give me a joke, then while Vaayu is telling me a joke I tell him give me a new one - he
understands it and accepts that command instead of the previous one".

While Vaayu speaks, the listener is muted and its mic frames - with his own voice already taken out
(audio/echo_cancel.py) - are tapped here instead. Speech starting sets `pause` (the TTS goes quiet, mid-word);
the sentence is captured to its end and transcribed:
  - a stop phrase, or a real sentence of the user's -> `stop` is set and `heard` kept (the orchestrator runs it
    as the next command, in place of the rest of the reply)
  - his own echo / a cough / "hmm" -> `pause` clears and he carries on from just before where he stopped.
"""

import queue
import re
import threading

import numpy as np

from assistant.audio import echo_cancel

STOP_PHRASE = re.compile(
    r"^(?:(?:hey|ok|okay|arre|arey|vaa?yu|vayu|bhai|please|no)\s+)*"
    r"(?:stop(?: it| that| talking| speaking| now| please)?|stop stop|b[au]s(?: karo| kar| kar do| bas)?|ruko|ruk jao|"
    r"chup(?: karo| ho jao| raho)?|shut up|enough|wait|quiet|be quiet|cancel|thik hai|theek hai|okay stop|ok stop|"
    r"hold on|never ?mind)(?:\s+(?:vaa?yu|vayu|bhai|please|yaar))*$", re.I)
_GATE_MULTIPLIER = 1.3  # calibration knob: x the listener's normal speech gate while Vaayu talks
# 180 ms of speech starts a capture and pauses him. Voice test 2026-11-05: at 120 ms he paused for nobody in almost
# every reply (his own voice leaking through the canceller as it adapts reached exactly 4 frames, never 6) - each
# false pause cost ~1 s, replies took twice their length. (A short "stop." was once missed at 180 ms - before the
# echo canceller; a real voice now comes through it whole.)
_SPEECH_FRAMES = 6
_END_SILENCE_FRAMES = 20  # 600 ms of quiet ends the sentence (he is quiet now - a whole command may follow)
_MAX_FRAMES = 300       # 9 s
# calibration knob: words of the user's own needed to cut Vaayu off (besides "stop" or his name). Real logs
# (10-26..10-31): 13 cut-offs, every one his own voice garbled by the mic - "Merrick by c.", "You're a hero.",
# "Yes code.", "By code deco." (2-3 words) - and the user said "I'm not able to listen to you sometimes in the
# middle". 4 words keeps those out (kept with the echo canceller too - those garbles were 3 words; short
# follow-ups go by _FOLLOW_UP, short commands by the router).
_MIN_WORDS = 4
# short follow-ups that only make sense said to him mid-reply: "another one", "next one", "one more"
_FOLLOW_UP = re.compile(r"^(?:(?:no|ok|okay|arre|bhai)\s+)*(?:(?:tell |give )?(?:me )?another(?: one)?|next(?: one)?|one more|new one|"
                        r"something else|dusra|dusri|aur ek|ek aur|skip(?: it)?|louder|slower|faster|repeat(?: that)?)"
                        r"(?:\s+(?:please|bhai|yaar|vaa?yu))*$", re.I)
_NAME = re.compile(r"\b(?:vaa?yu|vayu|waayu|baayu)\b", re.I)
_ECHO_SHARE = 0.6       # this share of the words found in what Vaayu is saying = his own echo


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z]+", (text or "").lower()))


def is_stop(heard: str, speaking: str) -> bool:
    """A stop phrase that isn't simply an echo of what Vaayu is saying."""
    said = _words(heard)
    if not said or not STOP_PHRASE.match(said):
        return False
    return said not in _words(speaking)  # "...say stop to pause" coming back through the mic


def is_echo(heard: str, speaking: str) -> bool:
    """Mostly Vaayu's own words, picked up by the mic."""
    said, spoken = _words(heard).split(), set(_words(speaking).split())
    return bool(said) and sum(w in spoken for w in said) / len(said) >= _ECHO_SHARE


def should_interrupt(heard: str, speaking: str) -> bool:
    """User (2026-10-25): "if Vaayu is saying something and then I say something, he should stop reciting
    and start listening to me instead" - a stop phrase, or any real sentence of theirs (not his own echo)."""
    if is_stop(heard, speaking):
        return True
    words = _words(heard).split()
    if _NAME.search(heard or "") and len(words) >= 2:  # "Vaayu, listen" - his name is never in his own replies
        return True
    if _FOLLOW_UP.match(" ".join(words)):
        return True
    if is_echo(heard, speaking) or len(words) < 2:
        return False
    if len((heard or '').split()) >= _MIN_WORDS:  # spoken words ("you're" is one)
        return True
    from assistant.llm import router  # a short one only if it's a real command: "open chrome", "pause the song"
    return bool(router.fast_route(heard))


class BargeIn:
    """Runs while one reply is spoken: `pause` while the user talks, `stop` if what they said is for Vaayu."""

    def __init__(self, listener, stt, stop: threading.Event, speaking: str):
        self.listener, self.stt, self.stop, self.speaking = listener, stt, stop, speaking
        self.pause = threading.Event()
        self.heard: str | None = None
        self.pauses: list[str] = []  # what was heard each time he went quiet (logged by the orchestrator)
        self._frames: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="barge-in")
        listener.tap = self._frames.put
        self._thread.start()

    def close(self) -> None:
        self.listener.tap = None
        self.pause.clear()
        self._frames.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        gate = self.listener._noise_floor_rms * self.listener.energy_threshold_multiplier * _GATE_MULTIPLIER
        buf, speech, silence = [], 0, 0
        while not self.stop.is_set():
            frame = self._frames.get()
            if frame is None:
                return
            loud = self.listener._frame_rms(frame) >= gate
            voiced = loud and self.listener.vad.is_speech(frame.tobytes(), 16000)
            if speech < _SPEECH_FRAMES and echo_cancel.settling():  # a sentence just started: not the user yet
                voiced = False
            if speech < _SPEECH_FRAMES:
                speech = speech + 1 if voiced else 0
                buf = buf + [frame] if voiced else []
                if speech >= _SPEECH_FRAMES:
                    self.pause.set()  # the user is talking: quiet, listen to all of it
                continue
            buf.append(frame)
            silence = 0 if voiced else silence + 1
            if silence >= _END_SILENCE_FRAMES or len(buf) >= _MAX_FRAMES:
                audio = np.concatenate(buf).astype(np.float32) / 32768.0
                try:
                    text = self.stt.transcribe(audio)
                except Exception:
                    text = ""
                if should_interrupt(text, self.speaking):
                    self.heard = text
                    self.stop.set()
                    return
                self.pauses.append(text)
                self.pause.clear()  # a cough, "hmm", his own echo - carry on
                buf, speech, silence = [], 0, 0
