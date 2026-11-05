"""
Echo cancellation: Vaayu's own voice taken back out of the mic, so he can hear the user while he talks.

WebRTC's AEC3 (the canceller in Chrome/Meet calls; via the `livekit` package's AudioProcessingModule,
free, runs on the CPU in ~0.05 ms per 10 ms frame). The reference is exactly what Vaayu is playing: the TTS
engine calls `playing(audio, rate)` when a piece starts and `stopped()` when it ends or is cut off; the
mic callback (audio/vad_listener.py) passes every frame through `clean()`, which lines the reference up
by the clock - AEC3 finds the remaining speaker->mic delay itself.

The laptop's Realtek mic array already cancels some of its own speaker output; this takes out what gets
past it (real logs: "Merrick by c.", "You're a hero." - his own words garbled back as the user's).
If livekit can't load, frames pass through unchanged.
"""

import threading
import time

import numpy as np

RATE = 16000
_SUB = RATE // 100  # AEC3 works on 10 ms frames
_lock = threading.Lock()
_ref: np.ndarray | None = None  # what is playing now, 16 kHz int16
_ref_start = 0.0
_ref_end = 0.0  # when the last piece stopped: its echo still rings for TAIL seconds
TAIL = 0.6  # calibration knob (seconds)
_apm = None
_failed = False
_clock: float | None = None  # the mic's own sample clock (callbacks arrive in bursts; counting samples doesn't jitter)
OUTPUT_DELAY_MS = 180  # calibration knob: speaker output latency here (sounddevice reported 0.18 s)


def _to16k(audio: np.ndarray, rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if rate != RATE and len(audio):
        n = int(len(audio) * RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    if audio.dtype != np.int16:
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        audio = audio * (32767.0 if peak <= 1.0 else 1.0)  # float TTS samples are -1..1
    return np.clip(audio, -32768, 32767).astype(np.int16)


def playing(audio: np.ndarray, rate: int, now: float | None = None) -> None:
    """Vaayu starts playing `audio` (any rate, float -1..1) now."""
    global _ref, _ref_start
    with _lock:
        _ref, _ref_start = _to16k(audio, rate), time.perf_counter() if now is None else now


def stopped(now: float | None = None) -> None:
    global _ref, _ref_end
    with _lock:
        if _ref is not None:
            _ref_end = time.perf_counter() if now is None else now
        _ref = None


def _module():
    global _apm, _failed
    if _apm is None and not _failed:
        try:
            from livekit import rtc
            _apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
            _apm.set_stream_delay_ms(OUTPUT_DELAY_MS)
        except Exception:
            _failed = True  # no canceller - the mic is used as is
    return _apm


def _reference(n: int, now: float) -> np.ndarray:
    """The n reference samples that were playing over the n mic samples ending at `now`."""
    with _lock:
        ref, start = _ref, _ref_start
    out = np.zeros(n, dtype=np.int16)
    if ref is None:
        return out
    end = int((now - start) * RATE)
    lo, hi = max(0, end - n), min(len(ref), end)
    if hi > lo:
        out[n - (end - lo): n - (end - hi)] = ref[lo:hi]
    return out


def clean(frame: np.ndarray, now: float | None = None) -> np.ndarray:
    """One int16 16 kHz mic frame (a multiple of 10 ms) with Vaayu's voice taken out."""
    if now is None:
        global _clock
        t, step = time.perf_counter(), len(frame) / RATE
        # re-anchored when it drifts from the real clock (a sound card's clock is off by ~0.01%)
        _clock = t if _clock is None or abs(_clock + step - t) > 0.1 else _clock + step
        now = _clock
    if _ref is None and now > _ref_end + TAIL:
        return frame  # Vaayu is quiet: the mic as is (measured: AEC3 can swallow speech with no echo to cancel)
    apm = _module()
    if apm is None or len(frame) % _SUB:
        return frame
    from livekit import rtc
    ref = _reference(len(frame), now)
    out = np.empty_like(frame)
    for i in range(0, len(frame), _SUB):
        apm.process_reverse_stream(rtc.AudioFrame(ref[i:i + _SUB].tobytes(), RATE, 1, _SUB))
        mic = rtc.AudioFrame(frame[i:i + _SUB].astype(np.int16).tobytes(), RATE, 1, _SUB)
        apm.process_stream(mic)
        out[i:i + _SUB] = np.frombuffer(bytes(mic.data), dtype=np.int16)
    return out


def reset() -> None:
    """A fresh canceller (tests)."""
    global _apm, _failed, _ref, _ref_end
    _apm, _failed, _ref, _ref_end = None, False, None, 0.0
