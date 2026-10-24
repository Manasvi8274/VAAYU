"""
Record the user reading Vaayu commands - labelled audio for training the
speech model on their voice/accent (data/my_voice/, gitignored with data/).

    .venv/Scripts/python.exe scripts/record_my_voice.py        # resumes where you stopped

Each line: press Enter, read it aloud, it stops by itself. r = redo the last one, q = quit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data" / "my_voice"
RATE = 16000


def phrases():
    import stt_eval
    return list(dict.fromkeys(stt_eval._commands(n=160, seed=11)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / "index.jsonl"
    done = [json.loads(l) for l in index.read_text(encoding="utf-8").splitlines()] if index.exists() else []
    todo = phrases()
    i = len(done)
    print(f"{len(todo)} lines, {i} already recorded. Speak normally, like you talk to Vaayu.\n")
    while i < len(todo):
        text = todo[i]
        key = input(f"[{i + 1}/{len(todo)}]  {text}\n   Enter = record, r = redo previous, q = quit: ").strip().lower()
        if key == "q":
            break
        if key == "r" and done:
            done.pop()
            i -= 1
            index.write_text("".join(json.dumps(d) + "\n" for d in done), encoding="utf-8")
            continue
        seconds = 1.5 + 0.45 * len(text.split())  # calibration knob: lengthen if you get cut off
        audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float32")
        sd.wait()
        audio = audio[:, 0]
        if np.abs(audio).max() < 0.01:
            print("   didn't hear anything - try again (check the mic)\n")
            continue
        path = OUT / f"{i:04d}.wav"
        sf.write(path, audio, RATE)
        done.append({"path": str(path), "text": text})
        index.write_text("".join(json.dumps(d) + "\n" for d in done), encoding="utf-8")
        i += 1
    print(f"\n{len(done)} recordings saved in {OUT}")


if __name__ == "__main__":
    main()
