"""
Fine-tune Whisper small for Indian-accented Vaayu commands (LoRA, fits the 4 GB GTX 1650).

Training audio - kept apart from scripts/stt_eval.py so its numbers stay honest:
  * Vaayu commands + name-heavy commands spoken by the hi-IN Swara/Madhur voices
    (the eval uses the en-IN Neerja/Prabhat voices, and different sentences)
  * real Indian English speech from EdAcc's validation split (the eval uses test)
  * the user's own recordings (scripts/record_my_voice.py), when there are any
Noise: a fresh room-noise recording, not the eval's.

    python scripts/stt_finetune.py prepare
    python scripts/stt_finetune.py train          # stop Ollama first - needs the whole GPU
    python scripts/stt_finetune.py prepare rounds=2 && python scripts/stt_finetune.py train round=2 epochs=1
    python scripts/stt_eval.py run small:cuda models/whisper-small-vaayu:cuda
"""

import asyncio
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data" / "datasets" / "stt_train"
MODEL_OUT = ROOT / "models" / "whisper-small-vaayu"
VOICES = ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"]
RATES = ["-10%", "+0%", "+0%", "+15%"]
PITCHES = ["-12Hz", "-6Hz", "+6Hz", "+12Hz"]

NAME_TEMPLATES = [
    "open {c} on youtube", "{c} ki nayi video lagao", "play {c}'s newest video", "search {c} on youtube",
    "{c} ka channel kholo", "show me the latest upload from {c}", "play something from {c}",
]
CONTACT_TEMPLATES = [
    "message {p} on whatsapp saying {m}", "whatsapp {p} {m}", "{p} ko {m} bhej do",
    "send {m} to {p}", "tell {p} on whatsapp that {m}",
]
MESSAGES = ["I am on my way", "running late", "call me back", "good morning", "send the report",
            "kal milte hain", "done for today", "check your mail"]
APP_TEMPLATES = ["open {a}", "close {a}", "{a} kholo", "{a} band karo", "start {a}", "launch {a}"]


def _texts(n=1100, seed=99):
    import benchmark_commands as bench
    import stt_eval
    import youtube_command_suite as ytsuite
    from assistant.tools import names
    held_out = set(stt_eval._commands())  # never train on an eval sentence
    rng = random.Random(seed)
    yt = sorted({c["text"] for c in ytsuite.build() if not c["noisy"]} - held_out)
    other = sorted({c["text"] for c in bench.build_cases() + bench.build_holdout() + bench.build_holdout2()
                    if c["tool"] and not c["cat"].startswith(("stt_noise", "negative"))} - held_out)
    named = [t.format(c=c) for t in NAME_TEMPLATES for c in names.DEFAULTS["channel"]]
    named += [t.format(p=p, m=rng.choice(MESSAGES)) for t in CONTACT_TEMPLATES for p in names.DEFAULTS["contact"]]
    named += [t.format(a=a) for t in APP_TEMPLATES for a in names.DEFAULTS["app"]]
    named = [t for t in named if t not in held_out]
    rest = n - len(named)
    return named + rng.sample(yt, rest // 2) + rng.sample(other, rest - rest // 2)


async def _tts_all(jobs, parallel=8):
    import edge_tts
    sem = asyncio.Semaphore(parallel)

    async def one(text, voice, rate, pitch, path):
        async with sem:
            for _ in range(3):
                try:
                    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(path))
                    return
                except Exception:
                    await asyncio.sleep(2)
    await asyncio.gather(*(one(*j) for j in jobs if not j[4].exists()))


def prepare(rounds=1):
    import io
    import numpy as np
    import pyarrow.parquet as pq
    import sounddevice as sd
    import soundfile as sf
    from faster_whisper.audio import decode_audio
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    noise_path = OUT / "room_noise.npy"
    if not noise_path.exists():
        print("recording 10 s of room noise for training...")
        noise = sd.rec(10 * 16000, samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        np.save(noise_path, noise[:, 0])
    rng = random.Random(7)
    items, jobs = [], []
    for r in range(1, rounds + 1):  # each extra round: other sentences, voices pitched as new pseudo-speakers
        for i, text in enumerate(_texts(seed=99 + 1000 * (r - 1))):
            for voice in VOICES:
                path = OUT / "clips" / (f"{i:04d}_{voice}.mp3" if r == 1 else f"r{r}_{i:04d}_{voice}.mp3")
                pitch = "+0Hz" if r == 1 else rng.choice(PITCHES)
                jobs.append((text, voice, rng.choice(RATES), pitch, path))
                items.append({"text": text, "path": str(path), "src": "tts"})
    asyncio.run(_tts_all(jobs))
    eval_speakers = {it["voice"] for it in json.loads((ROOT / "data/datasets/stt_eval/items.json").read_text())
                     if it["kind"] == "edacc"}
    for f in sorted((ROOT / "data/datasets/edacc").glob("*.parquet")):  # every split; eval speakers excluded
        rows = pq.read_table(f, filters=[("accent", "=", "Indian English")]).to_pylist()
        for j, r in enumerate(rows):
            if r["speaker"] in eval_speakers or "<" in r["text"]:
                continue
            audio = decode_audio(io.BytesIO(r["audio"]["bytes"]), sampling_rate=16000)
            if not 1 < len(audio) / 16000 < 29 or len(r["text"].split()) < 3:
                continue
            wav = OUT / "clips" / f"edacc_{f.stem[:16]}_{r['speaker']}_{j:05d}.wav"
            sf.write(wav, audio, 16000)
            items.append({"text": _clean_edacc(r["text"]), "path": str(wav), "src": "edacc"})
    mine = ROOT / "data" / "my_voice" / "index.jsonl"
    if mine.exists():  # the user's own voice, weighted x3
        for line in mine.read_text(encoding="utf-8").splitlines():
            items += [{**json.loads(line), "src": "mine"}] * 3
    items = [it for it in items if Path(it["path"]).exists()]
    (OUT / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"prepared {len(items)} clips:", {s: sum(it['src'] == s for it in items) for s in ("tts", "edacc", "mine")})


def _clean_edacc(text):
    """EdAcc labels are ALL CAPS with fillers ("UH YEAH SO...") - round 1 learned to
    answer in capitals. Plain case, no fillers, like Whisper's own style."""
    t = re.sub(r"\b(?:uh+|um+|uhm|hmm+|er+|ah+|mm+)\b", " ", text.lower())
    t = re.sub(r"\bi\b", "I", re.sub(r"\s+", " ", t)).strip()
    return t or text


def _target(text):
    text = text.strip()
    return text[0].upper() + text[1:] + ("" if text[-1] in ".?!" else ".")


def train(epochs=2, batch=4, accum=4, lr=1e-4, round=1):
    """round 1: stock small on everything. round N: continues from round N-1's merged
    model, on that round's new sentences plus the real speech."""
    import numpy as np
    import psutil
    import torch
    from faster_whisper.audio import decode_audio
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from assistant.stt.whisper_stt import _VOCAB_PROMPT

    proc = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")
    tok = proc.tokenizer
    prompt = list(tok.get_prompt_ids(_VOCAB_PROMPT))
    items = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    if round > 1:
        items = [it for it in items if it["src"] != "tts" or Path(it["path"]).name.startswith(f"r{round}_")]
    base = "openai/whisper-small" if round == 1 else str(OUT / f"hf_model_r{round - 1}")
    model_out = MODEL_OUT if round == 1 else MODEL_OUT.with_name(f"{MODEL_OUT.name}-r{round}")
    print(f"round {round}: {len(items)} clips from {base}", flush=True)
    noise = np.load(OUT / "room_noise.npy")
    rng = np.random.default_rng(1)

    def example(it):
        a = decode_audio(it["path"], sampling_rate=16000)
        if rng.random() < 0.6:  # room noise at a random level
            start = rng.integers(0, max(1, len(noise) - len(a)))
            n = np.resize(noise[start:], len(a))
            n = n / (np.sqrt(np.mean(n ** 2)) or 1) * np.sqrt(np.mean(a ** 2)) * rng.uniform(0.1, 0.5)
            a = a + n
        a = (a * rng.uniform(0.5, 1.5)).astype(np.float32)
        feats = proc.feature_extractor(a, sampling_rate=16000).input_features[0]
        full = prompt + tok(" " + _target(it["text"])).input_ids  # Vaayu always transcribes with the prompt
        return feats, full, len(prompt)

    model = WhisperForConditionalGeneration.from_pretrained(base)
    model.config.forced_decoder_ids = None
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                              target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    model.print_trainable_parameters()
    model.cuda().train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps = epochs * len(items) // (batch * accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps + 1, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    eot = tok.eos_token_id
    step, ckpt = 0, OUT / f"checkpoint_r{round}.pt"
    if ckpt.exists():  # resume after a stop - progress is saved every 20 steps
        state = torch.load(ckpt, weights_only=False)
        set_peft_model_state_dict(model, state["lora"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"resumed at step {step}", flush=True)
    micro = 0
    for epoch in range(epochs):
        order = np.random.default_rng(100 + epoch).permutation(len(items))  # same order on resume
        for b in range(0, len(order) - batch + 1, batch):
            micro += 1
            if micro <= step * accum:
                continue
            exs = [example(items[i]) for i in order[b:b + batch]]
            width = -(-(max(len(f) for _, f, _ in exs) - 1) // 64) * 64  # few distinct shapes: less allocator/kernel-cache growth
            dec = torch.full((batch, width), eot)
            lab = torch.full((batch, width), -100)
            for k, (_, full, np_) in enumerate(exs):
                dec[k, :len(full) - 1] = torch.tensor(full[:-1])
                lab[k, np_ - 1:len(full) - 1] = torch.tensor(full[np_:])  # loss only on the transcript
            feats = torch.tensor(np.stack([f for f, _, _ in exs])).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=feats, decoder_input_ids=dec.cuda(), labels=lab.cuda()).loss / accum
            scaler.scale(loss).backward()
            if micro % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {loss.item() * accum:.3f} ram {psutil.Process().memory_info().rss >> 20} MiB", flush=True)
                    torch.save({"lora": get_peft_model_state_dict(model), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step}, ckpt)
    merged = model.merge_and_unload()
    hf_dir = OUT / f"hf_model_r{round}"
    merged.save_pretrained(hf_dir)
    proc.save_pretrained(hf_dir)
    proc.feature_extractor.save_pretrained(hf_dir)  # transformers 5 writes processor_config.json; ct2 wants preprocessor_config.json
    import ctranslate2.converters
    ctranslate2.converters.TransformersConverter(
        str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]
    ).convert(str(model_out), quantization="int8_float16", force=True)
    ckpt.unlink(missing_ok=True)
    print("saved", model_out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import ast
    kwargs = {k: ast.literal_eval(v) for k, v in (a.split("=") for a in sys.argv[2:])}  # e.g. train round=2 epochs=1
    {"prepare": prepare, "train": train}[sys.argv[1]](**kwargs)
