"""
QLoRA fine-tune of Vaayu's brain (Qwen2.5-3B-Instruct) on scripts/build_distill_data.py's Claude-written data,
sized for the 4 GB GTX 1650 (Turing: fp16 only), then exported as a GGUF LoRA for Ollama.

Memory: a tool turn is ~3,900 tokens, ~3,700 of them the fixed system prompt + tool schemas - too long to train
whole in 4 GB. That fixed part runs once per example with no gradients and is frozen as a KV cache; only the
conversation after it (the user's words onward, <= --max-len tokens) is trained, with gradient checkpointing.
Loss only on the assistant's tokens, and logits are computed only where there's a label.

    python scripts/train_llm_lora.py check              # CPU only: tokenization + label masking on 3 examples
    python scripts/train_llm_lora.py train [--epochs 1] # GPU (stop Ollama first); resumes from its checkpoint
    python scripts/train_llm_lora.py export             # adapter -> GGUF LoRA + models/Modelfile.distilled
"""

import argparse
import io
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = Path(*HERE.parts[:HERE.parts.index(".claude")]) if ".claude" in HERE.parts else HERE  # main checkout
DATA = ROOT / "data" / "datasets" / "llm_distill"
RUN = ROOT / "data" / "llm_distill_run"  # checkpoint + adapter
MODELS = ROOT / "models"
BASE = "Qwen/Qwen2.5-3B-Instruct"  # the same model models/Qwen2.5-3B-Instruct.Q4_K_M.gguf was quantized from
LORA_GGUF = "vaayu-distill-lora.gguf"
_ASSISTANT = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE)


def encode(ex, tok):
    """-> (ids, labels, prefix_len). prefix = system prompt + tools (frozen); labels = assistant tokens only."""
    tools = ex.get("tools") or None
    text = tok.apply_chat_template(ex["messages"], tools=tools, tokenize=False)
    prefix = tok.apply_chat_template(ex["messages"][:1], tools=tools, tokenize=False)
    assert text.startswith(prefix), "system block renders differently alone"
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    spans = [m.span(1) for m in _ASSISTANT.finditer(text)]
    labels = [t if any(a <= s and e <= b for a, b in spans) else -100 for t, (s, e) in zip(ids, offsets)]
    prefix_len = len(tok(prefix, add_special_tokens=False)["input_ids"])
    assert ids[:prefix_len] == tok(prefix, add_special_tokens=False)["input_ids"]
    return ids, labels, prefix_len


def _load(name):
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def check():
    tok = _tokenizer()
    rows = _load("train")
    picks = [next(r for r in rows if r["cat"].startswith("chat")), next(r for r in rows if r["cat"] == "tool_chain"),
             next(r for r in rows if r["cat"].startswith("web"))]
    for ex in picks:
        ids, labels, p = encode(ex, tok)
        trained = tok.decode([t for t in labels if t != -100])
        want = "".join(m.group(1) for m in _ASSISTANT.finditer(tok.apply_chat_template(
            ex["messages"], tools=ex.get("tools") or None, tokenize=False)))
        assert trained == want, (trained, want)
        assert all(lab == -100 for lab in labels[:p]), "a label fell inside the frozen prefix"
        print(f"[{ex['cat']}] {len(ids)} tokens: frozen prefix {p}, trained part {len(ids) - p}, "
              f"loss on {sum(lab != -100 for lab in labels)}")
        print("   trained text:", trained.replace("\n", "\\n")[:300])
    lens = [len(ids) - p for ids, _l, p in (encode(ex, tok) for ex in rows)]
    print(f"trained-part length over {len(lens)} train examples: max {max(lens)}, mean {sum(lens) / len(lens):.0f}")
    _check_frozen_prefix()
    print("check ok")


def _ram_mib():
    import psutil
    return psutil.Process().memory_info().rss >> 20


def _frozen_prefix(model, ids):
    """KV cache of the prefix, computed without gradients and never appended to - so gradient checkpointing can
    replay the trained part against it (HF's own cache would grow on every replay)."""
    import torch
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        cache = model(input_ids=ids, use_cache=True, logits_to_keep=1).past_key_values
    model.train()
    for layer in cache.layers:
        layer.update = (lambda ks, vs, *a, _k=layer.keys, _v=layer.values, **kw:
                        (torch.cat([_k, ks], dim=-2), torch.cat([_v, vs], dim=-2)))
    return cache


def _loss(model, ids, labels, p, grad=True):
    import torch
    import torch.nn.functional as F
    ids_t = torch.tensor([ids], device=model.device)
    lab = torch.tensor(labels[p:], device=model.device)
    keep = (lab[1:] != -100).nonzero().squeeze(-1)  # positions whose NEXT token is a label
    cache = _frozen_prefix(model, ids_t[:, :p])
    if not grad:
        model.eval()
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids_t[:, p:], past_key_values=cache, use_cache=False, logits_to_keep=keep)
    if not grad:
        model.train()
    return F.cross_entropy(out.logits[0].float(), ids_t[0, p:][keep + 1])


def _with_lora(model, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    from transformers.modeling_layers import GradientCheckpointingLayer
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for m in model.modules():  # safe with _frozen_prefix's cache (replays don't change it)
        if isinstance(m, GradientCheckpointingLayer):
            m._can_checkpoint_with_cache = True
    return get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))


def _model():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    return _with_lora(AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)))


def _check_frozen_prefix():
    """CPU, tiny random Qwen2: frozen-prefix loss == plain full-sequence loss, and backward runs through the
    checkpointed replay."""
    import torch
    import torch.nn.functional as F
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    model = _with_lora(Qwen2ForCausalLM(Qwen2Config(
        vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, attn_implementation="sdpa")), dropout=0.0)
    for _name, prm in model.named_parameters():  # LoRA B starts at zero; make the adapter matter
        if "lora_B" in _name:
            torch.nn.init.normal_(prm, std=0.05)
    ids = torch.randint(0, 500, (60,)).tolist()
    p = 40
    labels = [-100] * 45 + ids[45:]
    model.train()
    loss = _loss(model, ids, labels, p)
    loss.backward()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
    want = F.cross_entropy(logits[44:59].float(), torch.tensor(ids[45:60]))
    assert abs(loss.item() - want.item()) < 1e-4, (loss.item(), want.item())
    assert any(prm.grad is not None and prm.grad.abs().sum() > 0 for n, prm in model.named_parameters() if "lora" in n)
    print(f"frozen-prefix loss {loss.item():.5f} == full-sequence loss {want.item():.5f}; LoRA grads flow")


def train(epochs=1, accum=8, lr=1e-4, max_len=1024, save_every=10):
    import bitsandbytes as bnb
    import numpy as np
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    tok = _tokenizer()
    data = [e for e in (encode(ex, tok) for ex in _load("train")) if len(e[0]) - e[2] <= max_len]
    model = _model()
    model.print_trainable_parameters()
    opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = epochs * len(data) // accum
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warmup, cosine decay
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    scaler = torch.amp.GradScaler("cuda")
    RUN.mkdir(parents=True, exist_ok=True)
    ckpt, step = RUN / "checkpoint.pt", 0
    if ckpt.exists():  # resume - the memory reaper may stop long jobs
        state = torch.load(ckpt, weights_only=False)
        set_peft_model_state_dict(model, state["lora"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step = state["step"]
        print(f"resumed at step {step}", flush=True)
    print(f"{len(data)} examples, {epochs} epoch(s), {steps} optimizer steps of {accum}", flush=True)
    micro, run_loss, t0, skipped = 0, [], time.time(), 0
    for epoch in range(epochs):
        for i in np.random.default_rng(100 + epoch).permutation(len(data)):  # same order on resume
            micro += 1
            if micro <= step * accum:
                continue
            try:
                loss = _loss(model, *data[i]) / accum
                scaler.scale(loss).backward()
                run_loss.append(loss.item() * accum)
            except torch.cuda.OutOfMemoryError:  # one oversized example shouldn't end the run
                skipped += 1
                torch.cuda.empty_cache()
                continue
            if micro % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                rate = (time.time() - t0) / max(1, len(run_loss))
                print(f"epoch {epoch} step {step}/{steps} loss {sum(run_loss[-accum:]) / accum:.3f} "
                      f"ram {_ram_mib()} MiB gpu-peak {torch.cuda.max_memory_allocated() >> 20} MiB "
                      f"{rate:.1f}s/example eta {rate * accum * (steps - step) / 60:.0f} min skipped-oom {skipped}",
                      flush=True)
                if step % save_every == 0:
                    torch.save({"lora": get_peft_model_state_dict(model), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step}, ckpt)
    model.save_pretrained(RUN / "adapter")
    val = [e for e in (encode(ex, tok) for ex in _load("val")) if len(e[0]) - e[2] <= max_len]
    with torch.no_grad():
        vl = [_loss(model, *e, grad=False).item() for e in val]
    print(f"val loss {sum(vl) / len(vl):.3f} on {len(vl)} held-out examples", flush=True)
    ckpt.unlink(missing_ok=True)
    print("saved adapter to", RUN / "adapter")


def _llama_cpp():
    """llama.cpp's converter scripts + its gguf-py (models/llama.cpp, gitignored with models/)."""
    dest = MODELS / "llama.cpp"
    if not (dest / "conversion").is_dir():
        print("downloading llama.cpp converter scripts...")
        blob = urllib.request.urlopen("https://codeload.github.com/ggml-org/llama.cpp/zip/refs/heads/master").read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                rel = name.split("/", 1)[1]
                top_py = "/" not in rel and rel.endswith(".py")
                if not name.endswith("/") and (top_py or rel.startswith(("conversion/", "gguf-py/"))):
                    (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                    (dest / rel).write_bytes(z.read(name))
    return dest


def export(adapter=None):
    from huggingface_hub import snapshot_download
    adapter = Path(adapter) if adapter else RUN / "adapter"
    base_dir = snapshot_download(BASE, allow_patterns=["*.json", "*.txt"])  # config + tokenizer are all it needs
    out = MODELS / LORA_GGUF
    subprocess.run([sys.executable, str(_llama_cpp() / "convert_lora_to_gguf.py"), "--base", base_dir,
                    "--outtype", "f16", "--outfile", str(out), str(adapter)], check=True)
    modelfile = (MODELS / "Modelfile").read_text(encoding="utf-8")
    lines = modelfile.splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("FROM "))
    (MODELS / "Modelfile.distilled").write_text("\n".join(lines[:i + 1] + [f"ADAPTER {LORA_GGUF}"] + lines[i + 1:]) + "\n",
                                                encoding="utf-8")
    print(f"wrote {out} and {MODELS / 'Modelfile.distilled'}\nnext:  ollama create vaayu-distilled -f "
          f"{MODELS / 'Modelfile.distilled'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "train", "export"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=1024, help="max trained tokens after the frozen prefix")
    ap.add_argument("--adapter", help="export: adapter folder (default data/llm_distill_run/adapter)")
    a = ap.parse_args()
    if a.cmd == "check":
        check()
    elif a.cmd == "train":
        train(epochs=a.epochs, max_len=a.max_len)
    else:
        export(a.adapter)
