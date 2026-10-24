"""
Which speech model understands an Indian accent best, within the 4 GB GPU?

Test audio:
  * commands Vaayu really gets (YouTube, apps, WhatsApp, names like OG Crew /
    SAVED MESSAGES / Raman Garg, Hinglish) spoken by two Indian-English neural
    voices (Microsoft en-IN Neerja + Prabhat, via edge-tts), clean and mixed
    with real noise recorded from this laptop's mic
  * real human Indian English speech (EdAcc corpus, University of Edinburgh)

Per model: "understood the same" (the transcript routes to the same command
as the correct text), word error rate, and seconds per clip - all with
Vaayu's own recognition settings (vocabulary prompt, VAD filter).

    python scripts/stt_eval.py prepare              # make/cached test audio (needs internet once)
    python scripts/stt_eval.py run base:cpu small:cuda large-v3-turbo:cuda
"""

import asyncio
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data" / "datasets" / "stt_eval"
VOICES = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"]

NAME_COMMANDS = [
    "play the latest video from OG Crew", "OG Crew ka latest video chalao", "open the OG Crew channel",
    "play the latest video from Carryminati", "play the latest video from Dhruv Rathee",
    "Technical Guruji ka naya video chalao", "play the latest video from Samay Raina",
    "send hello to SAVED MESSAGES on whatsapp", "SAVED MESSAGES ko whatsapp pe meeting at 5 bhejo",
    "send call me later to Raman Garg on whatsapp", "whatsapp pe SAVED MESSAGES ko dinner ready bhejo",
    "open Tekken 7", "close Tekken 7", "open Armoury Crate", "open WhatsApp", "whatsapp kholo",
    "play kesariya on youtube", "play arijit singh songs on youtube", "youtube pe naatu naatu chalao",
    "open brave", "chrome band karo", "battery kitni hai", "volume kam karo", "awaz 40 par karo",
    "video roko", "video chalao", "pause the video", "resume the video", "skip ad", "next video",
    "turn on captions", "set the volume to 75 percent", "go to 2:30", "play at 1.5x speed",
    "what is playing", "take a screenshot", "lock my laptop", "turn off wifi", "bluetooth chalu karo",
]


def _commands(n=160, seed=5):
    import benchmark_commands as bench
    import youtube_command_suite as ytsuite
    rng = random.Random(seed)
    yt = [c["text"] for c in ytsuite.build() if not c["noisy"]]
    other = [c["text"] for c in bench.build_cases() + bench.build_holdout() + bench.build_holdout2()
             if c["tool"] and not c["cat"].startswith(("stt_noise", "negative"))]
    picked = rng.sample(sorted(set(yt)), n // 2) + rng.sample(sorted(set(other)), n // 2)
    return NAME_COMMANDS + picked


async def _tts(text, voice, path):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(str(path))


def prepare():
    import numpy as np
    import sounddevice as sd
    from faster_whisper.audio import decode_audio
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    noise_path = OUT / "room_noise.npy"
    if not noise_path.exists():  # real background noise from this laptop's own mic
        print("recording 8 s of room noise from the mic...")
        noise = sd.rec(8 * 16000, samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        np.save(noise_path, noise[:, 0])
    items = []
    for i, text in enumerate(_commands()):
        for voice in VOICES:
            mp3 = OUT / "clips" / f"{i:03d}_{voice}.mp3"
            if not mp3.exists():
                for attempt in range(3):
                    try:
                        asyncio.run(_tts(text, voice, mp3))
                        break
                    except Exception as e:
                        print("tts retry", attempt, e)
                        time.sleep(2)
            items.append({"kind": "command", "text": text, "voice": voice, "path": str(mp3)})
    # EdAcc: real Indian English speech (one file of the public test split)
    import pyarrow.parquet as pq
    t = pq.read_table(ROOT / "data" / "datasets" / "edacc" / "test-00001.parquet").to_pylist()
    real = [r for r in t if r["accent"] == "Indian English" and "<" not in r["text"] and len(r["text"].split()) >= 4]
    rng = random.Random(3)
    for j, r in enumerate(rng.sample(real, min(120, len(real)))):
        wav = OUT / "clips" / f"edacc_{j:03d}.wav"
        if not wav.exists():
            audio = decode_audio(io.BytesIO(r["audio"]["bytes"]), sampling_rate=16000)
            import soundfile as sf
            sf.write(wav, audio, 16000)
        items.append({"kind": "edacc", "text": r["text"], "voice": r["speaker"], "path": str(wav)})
    (OUT / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"prepared {len(items)} clips in {OUT}")


def _words(s):
    s = s.lower().replace("%", " percent")
    return re.findall(r"[a-z0-9']+", s)


def _wer(ref, hyp):
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def _route_key(text):
    from assistant.llm import router
    r = router.fast_route(text)
    if not r:
        return None
    if r[0] == "reply":
        return ("reply", r[1][:12])
    return tuple((n, tuple(sorted((k, str(v).lower()) for k, v in a.items()))) for n, a in r[1])


def _load_cuda_dlls():
    import nvidia.cublas
    import nvidia.cudnn
    for mod in (nvidia.cublas, nvidia.cudnn):
        d = os.path.join(list(mod.__path__)[0], "bin")
        os.add_dll_directory(d)
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def run(specs):
    import numpy as np
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from assistant.stt.whisper_stt import _VOCAB_PROMPT
    _load_cuda_dlls()
    items = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    noise = np.load(OUT / "room_noise.npy")
    rng = np.random.default_rng(0)
    audios = {}
    for it in items:
        audios[it["path"]] = decode_audio(it["path"], sampling_rate=16000)
    report = {}
    for spec in specs:
        name, device = spec.split(":")
        model = WhisperModel(name, device=device, compute_type="int8_float16" if device == "cuda" else "int8")
        rows = []
        for it in items:
            variants = [("clean", audios[it["path"]])]
            if it["kind"] == "command":  # + real room noise at roughly conversational SNR
                a = audios[it["path"]]
                start = rng.integers(0, max(1, len(noise) - len(a)))
                n = np.resize(noise[start:], len(a))
                n = n / (np.sqrt(np.mean(n ** 2)) or 1) * np.sqrt(np.mean(a ** 2)) * 0.25
                variants.append(("noisy", (a + n).astype(np.float32)))
            for cond, audio in variants:
                t0 = time.perf_counter()
                segs, _ = model.transcribe(audio, language="en", vad_filter=True, initial_prompt=_VOCAB_PROMPT,
                                           beam_size=5)
                hyp = " ".join(s.text.strip() for s in segs).strip()
                secs = time.perf_counter() - t0
                row = {"kind": it["kind"], "cond": cond, "voice": it["voice"], "ref": it["text"], "hyp": hyp,
                       "wer": _wer(it["text"], hyp), "secs": secs}
                if it["kind"] == "command":
                    ref_key = _route_key(it["text"])
                    row["understood"] = ref_key is not None and _route_key(hyp) == ref_key
                    row["routable"] = ref_key is not None
                rows.append(row)
        del model
        report[spec] = rows
        print(summarize(spec, rows), flush=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


def summarize(spec, rows):
    def pct(xs):
        return f"{100 * sum(xs) / max(1, len(xs)):.1f}%"
    out = [f"\n== {spec}"]
    for cond in ("clean", "noisy"):
        cmd = [r for r in rows if r["kind"] == "command" and r["cond"] == cond and r["routable"]]
        names = [r for r in cmd if r["ref"] in NAME_COMMANDS]
        out.append(f"  commands {cond:5}: understood {pct([r['understood'] for r in cmd])} of {len(cmd)} | "
                   f"name-heavy {pct([r['understood'] for r in names])} | WER {sum(r['wer'] for r in cmd) / max(1, len(cmd)):.3f}")
    real = [r for r in rows if r["kind"] == "edacc"]
    out.append(f"  real Indian speech (EdAcc): WER {sum(r['wer'] for r in real) / max(1, len(real)):.3f} on {len(real)} clips")
    secs = sorted(r["secs"] for r in rows if r["kind"] == "command")
    out.append(f"  seconds per command: median {secs[len(secs) // 2]:.2f}, p90 {secs[int(len(secs) * .9)]:.2f}")
    return "\n".join(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "prepare":
        prepare()
    else:
        run(sys.argv[2:])
