"""
Voice tests of the running Vaayu, the way a user talks to him: a spoken sentence (Microsoft's Indian English / Hindi
neural voices, edge-tts - cached in data/voice_test/) is played on the laptop speakers, Vaayu's own microphone hears
it, and what happened is read back from his interaction log; the mic is also recorded here, so the time until Vaayu's
voice is actually heard is measured, not guessed.

    python scripts/voice_test.py            # the default scenario list
    python scripts/voice_test.py joke time  # only scenarios whose name contains one of these words

Results: data/voice_test/results_<time>.jsonl and a table on screen. Never says the kill phrase; never sends,
pays, joins meetings or changes the volume.
"""

import asyncio
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "voice_test"
LOG = ROOT / "data" / "interactions.jsonl"
VOICES = {"en": "en-IN-PrabhatNeural", "hi": "hi-IN-MadhurNeural"}
LEVEL = 0.5  # calibration knob: peak loudness of the played voice (0..1)
RATE = 24000

# (name, [(said, language, seconds to wait after it - or "turn" to wait for Vaayu's turn to finish)], checks)
SCENARIOS = [
    ("name_only", [("Vaayu", "en", 4)], {"says": True}),
    ("time", [("Vaayu, what's the time?", "en", "turn")], {"tool_or_reply": "time"}),
    ("weather", [("Vaayu, what's the weather today?", "en", "turn")], {"tool": "get_weather"}),
    ("battery", [("Vaayu, how much battery do I have?", "en", "turn")], {"tool": "get_system_status"}),
    ("joke_then_new_one", [("Vaayu, tell me a joke.", "en", 3.0), ("Give me a new one.", "en", "turn")],
     {"barge": True}),
    ("notepad", [("Vaayu, open notepad.", "en", "turn"), ("Vaayu, close notepad.", "en", "turn")],
     {"tool": "open_app"}),
    ("song_then_pause", [("Vaayu, play the song kesariya.", "en", 9), ("Vaayu, pause the song.", "en", "turn")],
     {"tool": "play_music"}),
    ("memory", [("Vaayu, what did I tell you about my exam?", "en", "turn")], {"says": True}),
    ("settings", [("Vaayu, open bluetooth settings.", "en", "turn")], {"tool": "open_settings_page"}),
    ("hindi_weather", [("वायु, आज मौसम कैसा है?", "hi", "turn")], {"tool": "get_weather"}),
    ("hindi_joke", [("वायु, एक चुटकुला सुनाओ।", "hi", "turn")], {"says": True}),
    ("stop_mid_reply", [("Vaayu, tell me a long story about a brave little sparrow.", "en", 4.0),
                        ("Stop.", "en", 4)], {"barge": True}),
    ("think", [("Vaayu, think about whether I should learn Rust or Go first.", "en", "turn")], {"tool": "think_hard"}),
    ("screen", [("Vaayu, what's on my screen?", "en", "turn")], {"tool": "look_at_screen"}),
    ("routine", [("Vaayu, when I say test mode, open notepad and close notepad.", "en", "turn"),
                 ("Vaayu, delete the test mode routine.", "en", "turn")], {"says": True}),
    ("sleep", [("Vaayu, go to sleep.", "en", 4)], {"says": True}),
]


def _voice(text: str, lang: str) -> np.ndarray:
    import edge_tts
    from faster_whisper import decode_audio
    OUT.mkdir(parents=True, exist_ok=True)
    import hashlib
    path = OUT / f"{hashlib.md5((lang + text).encode()).hexdigest()[:12]}_{lang}.mp3"
    if not path.exists():
        asyncio.run(edge_tts.Communicate(text, VOICES[lang]).save(str(path)))
    audio = decode_audio(str(path), sampling_rate=RATE)
    return (audio / (np.abs(audio).max() or 1) * LEVEL).astype(np.float32)


class MicTap:
    """The mic's loudness every 30 ms, with the time - to see when Vaayu's voice comes."""

    def __init__(self):
        self.frames: list[tuple[float, float]] = []
        self._s = sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=480, callback=self._cb)

    def _cb(self, indata, *_):
        self.frames.append((time.time(), float(np.sqrt(np.mean(indata.astype(np.float64) ** 2)))))

    def __enter__(self):
        self._s.start()
        return self

    def __exit__(self, *a):
        self._s.stop()
        self._s.close()

    def first_sound_after(self, t0: float, until: float, floor: float) -> float | None:
        loud = [t for t, r in self.frames if t0 <= t <= until and r > floor]
        # 3 loud frames within 150 ms = speech, not a click
        for i, t in enumerate(loud):
            if sum(1 for u in loud[i:i + 6] if u - t <= 0.15) >= 3:
                return t
        return None


def _events_since(ts: float) -> list[dict]:
    out = []
    try:
        with LOG.open(encoding="utf-8") as f:
            f.seek(max(0, LOG.stat().st_size - 400_000))
            for line in f:
                try:
                    e = json.loads(line)
                    if datetime.fromisoformat(e["ts"]).timestamp() >= ts:
                        out.append(e)
                except (ValueError, KeyError):
                    continue
    except OSError:
        pass
    return out


def _wait_turn(since: float, timeout: float = 45) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(e.get("kind") in ("turn", "ignored_sleeping", "ignored_media", "heard_nothing", "name_only")
               for e in _events_since(since)):
            time.sleep(1.5)  # his reply finishes / the next thing settles
            return
        time.sleep(0.3)


def _ensure_quiet(mic: "MicTap", floor: float) -> None:
    """A song left playing would drown every later test: stop it the way a user would, louder."""
    global LEVEL
    for _ in range(2):
        recent = [r for t, r in mic.frames if t > time.time() - 2]
        if not recent or float(np.median(recent)) < floor:
            return
        print("   (music still playing - asking him to stop it)", flush=True)
        level, LEVEL = LEVEL, 1.0
        try:
            for said, wait in (("Vaayu!", 3.0), ("Stop the music.", 5.0)):
                sd.play(_voice(said, "en"), RATE)
                sd.wait()
                time.sleep(wait)
        finally:
            LEVEL = level


def run(selected=None) -> list[dict]:
    results = []
    with MicTap() as mic:
        time.sleep(1.0)
        floor = max(150.0, 6 * float(np.median([r for _, r in mic.frames]) or 25))
        for name, steps, checks in SCENARIOS:
            if selected and not any(w in name for w in selected):
                continue
            start = time.time()
            step_rows = []
            for said, lang, wait in steps:
                audio = _voice(said, lang)
                t_play = time.time()
                sd.play(audio, RATE)
                sd.wait()
                t_end = time.time()
                if wait == "turn":
                    _wait_turn(t_play)
                else:
                    time.sleep(wait)
                first = mic.first_sound_after(t_end + 0.25, time.time(), floor)
                step_rows.append({"said": said, "first_sound_s": round(first - t_end, 2) if first else None})
            time.sleep(1.0)
            events = _events_since(start)
            turns = [e for e in events if e.get("kind") == "turn"]
            row = {"scenario": name, "steps": step_rows,
                   "heard": [e.get("text") for e in events if e.get("kind") == "heard"],
                   "replies": [e.get("reply") for e in turns],
                   "tools": [t.get("tool") for e in turns for t in e.get("trace", []) if t.get("tool")],
                   "paths": [t.get("path") for e in turns for t in e.get("trace", [])],
                   "stt_ms": [e.get("stt_ms") for e in events if e.get("kind") == "heard"],
                   "brain_ms": [e.get("brain_ms") for e in turns],
                   "acks": [e.get("text") for e in events if e.get("kind") == "ack"],
                   "barge": [e.get("heard") for e in events if e.get("kind") == "barge_in"],
                   "other": sorted({e.get("kind") for e in events} - {"heard", "turn", "ack", "barge_in"}),
                   "checks": checks}
            ok = True
            if checks.get("tool") and checks["tool"] not in row["tools"]:
                ok = False
            if checks.get("barge") and not row["barge"]:
                ok = False
            if checks.get("says") and not any(s["first_sound_s"] for s in step_rows):
                ok = False
            row["ok"] = ok
            results.append(row)
            print(f"{'OK ' if ok else 'BAD'} {name:18} first sound {[s['first_sound_s'] for s in step_rows]} "
                  f"heard {row['heard']} -> {row['tools'] or row['paths']} | {[(r or '')[:70] for r in row['replies']]}",
                  flush=True)
            _ensure_quiet(mic, floor)
            time.sleep(2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"results_{datetime.now():%Y%m%d_%H%M}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")
    print("saved", out)
    return results


if __name__ == "__main__":
    run(sys.argv[1:] or None)
