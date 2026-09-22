"""
Continuous, hands-free listening: the microphone stays open, and an
utterance is automatically segmented by voice activity (speech starts ->
speech ends after a run of silence) instead of a fixed recording window or a
push-to-talk hotkey.

Uses a callback-based sounddevice InputStream, not a blocking read loop.
Tested and confirmed this matters: with a blocking read loop, nothing drains
the stream while the main thread is blocked inside tts.speak() (which can
take several seconds), so PortAudio's buffer backs up - by the time
listening resumes, it processes a stale backlog instead of live audio,
truncating the next real utterance. A callback runs on its own thread
managed by PortAudio, so it keeps draining audio continuously regardless of
what the main thread is doing; frames captured while muted are dropped
immediately in the callback rather than queued.

The core state machine (_segment) is separated from the live stream on
purpose: it consumes any iterable of int16 audio frames, which lets it be
unit tested with synthetic frame sequences (real recorded silence + real
synthesized speech) without needing someone to physically talk into a
microphone during automated testing.
"""

import queue
from collections import deque
from typing import Iterable, Iterator

import numpy as np
import sounddevice as sd
import webrtcvad

from assistant.audio.capture import SAMPLE_RATE

FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


class VADListener:
    def __init__(
        self,
        aggressiveness: int = 2,
        silence_duration_ms: int = 900,
        speech_start_frames: int = 3,
        max_utterance_seconds: float = 20.0,
        energy_threshold_multiplier: float = 2.5,
        min_absolute_rms: float = 150.0,
        preroll_frames: int = 10,
    ):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.silence_frames_threshold = silence_duration_ms // FRAME_MS
        self.speech_start_frames = speech_start_frames
        self.max_frames = int(max_utterance_seconds * 1000 / FRAME_MS)
        # Tested and found necessary: a soft-onset trigger word at the very
        # start of an utterance (e.g. "type", with its quiet "t" sound) can
        # fall below the speech threshold for its first ~100-200ms, before
        # the louder vowel that follows clearly registers - without a
        # pre-roll, that word gets silently dropped from the transcript.
        # preroll_frames=10 * FRAME_MS(30ms) = 300ms of rolling lookback.
        self.preroll_frames = preroll_frames
        self.energy_threshold_multiplier = energy_threshold_multiplier
        self.min_absolute_rms = min_absolute_rms
        self._stream: sd.InputStream | None = None
        self._muted = False
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        # Webrtcvad alone was tested and found to false-trigger on this room's
        # ambient noise (fan/background sound) even at max aggressiveness -
        # requiring frames to also be meaningfully louder than a calibrated
        # baseline (see calibrate_noise_floor) is a second, independent signal
        # that catches what VAD alone missed.
        self._noise_floor_rms = min_absolute_rms

    def _callback(self, indata, _frames, _time_info, _status) -> None:
        if not self._muted:
            self._queue.put(indata.copy().flatten())

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._callback,
        )
        self._stream.start()
        self.calibrate_noise_floor()

    def calibrate_noise_floor(self, duration_seconds: float = 1.0) -> None:
        """Samples ambient audio briefly to measure this room's baseline
        noise level, so quiet/moderate background noise doesn't get treated
        as a candidate utterance. Call after start()."""
        n_frames = max(1, int(duration_seconds * 1000 / FRAME_MS))
        samples = [self._queue.get() for _ in range(n_frames)]
        combined = np.concatenate(samples).astype(np.float64)
        measured_rms = float(np.sqrt(np.mean(combined**2)))
        self._noise_floor_rms = max(measured_rms, self.min_absolute_rms)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def mute(self) -> None:
        self._muted = True
        self._drain_queue()

    def unmute(self) -> None:
        self._drain_queue()
        self._muted = False

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _frame_rms(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

    def _segment(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        buffer: list[np.ndarray] = []
        preroll: "deque[np.ndarray]" = deque(maxlen=self.preroll_frames)
        speech_frames_seen = 0
        silence_run = 0
        in_speech = False
        frame_count = 0
        energy_gate = self._noise_floor_rms * self.energy_threshold_multiplier

        for frame in frames:
            is_loud_enough = self._frame_rms(frame) >= energy_gate
            is_speech = is_loud_enough and self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)

            if not in_speech:
                if is_speech:
                    speech_frames_seen += 1
                    buffer.append(frame)
                    if speech_frames_seen >= self.speech_start_frames:
                        in_speech = True
                        silence_run = 0
                        # Prepend the rolling pre-speech context, which was
                        # only accumulated while genuinely silent (see the
                        # else branch below), so there's no overlap/duplicate
                        # frames with what's already in `buffer`.
                        buffer = list(preroll) + buffer
                else:
                    speech_frames_seen = 0
                    buffer.clear()
                    preroll.append(frame)
            else:
                buffer.append(frame)
                frame_count += 1
                if is_speech:
                    silence_run = 0
                else:
                    silence_run += 1
                    if silence_run >= self.silence_frames_threshold:
                        break
                if frame_count >= self.max_frames:
                    break

        if not buffer:
            return np.zeros(0, dtype=np.float32)

        audio_i16 = np.concatenate(buffer).flatten()
        return audio_i16.astype(np.float32) / 32768.0

    def _live_frames(self) -> Iterator[np.ndarray]:
        if self._stream is None:
            raise RuntimeError("VADListener.start() must be called before listening.")
        while True:
            yield self._queue.get()

    def listen_for_utterance(self) -> np.ndarray:
        """Blocks until it detects speech followed by a run of silence, and
        returns the utterance as a float32 16kHz mono array."""
        self._drain_queue()  # defensive: discard anything queued just before this call
        return self._segment(self._live_frames())
