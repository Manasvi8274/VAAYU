"""
Fine-tunes Vaayu's model on data/finetune/tool_calling_dataset.jsonl (built
by build_finetune_dataset.py) via QLoRA, and exports a GGUF file Ollama can
run directly.

RETARGETED FROM 7B TO 3B: real-usage testing this session measured
qwen2.5:7b-instruct (even at its best-fitting quantization on this
machine's 4GB GPU) at 10-300+ seconds per real turn across dozens of live
tests, never once under 5 seconds - a hard hardware/model-size wall (only
~42-51% of any 7B quantization fits in 4GB VRAM, the rest runs on CPU),
not something a config tweak fixes. qwen2.5:3b already meets a <5s bar
structurally (fits fully in VRAM, ~4.2s warm, measured earlier this
session) - the real gap for 3B is tool-selection accuracy, which is
exactly what fine-tuning targets. So: fine-tune the model that's already
fast enough, rather than chase an unreachable speed target on the model
that's already accurate enough. The dataset itself is unchanged - it's
built from this project's real tool registry and targets real
tool-selection confusions, independent of model size (if anything more
relevant for 3B, which needs more help on these exact mix-ups than 7B does).

WHY THIS RUNS ON COLAB, NOT THIS LAPTOP: checked this machine's actual GPU
before writing this - an NVIDIA GTX 1650 with 4GB total VRAM, ~3.75GB free
with nothing else loaded. QLoRA fine-tuning even a 3B model in 4-bit still
needs meaningfully more than that for optimizer states/activations on top
of the base weights - it would OOM, not "run slowly" (and 3B needs less
than 7B did, but this was still not verified to fit locally - Colab remains
the safe, recommended path). Google Colab's free tier gives a T4 GPU with
16GB VRAM at no cost, which is a real, commonly-used way to do this without
buying hardware. This script is written to run there; running it locally on
this laptop's GPU is unverified and likely to fail.

HOW TO RUN:
1. Upload BOTH data/finetune/tool_calling_dataset.jsonl AND
   data/finetune/tool_schema.json to the Colab session's root (drag both
   into the Files panel, or mount Google Drive) - not into a subfolder,
   this script reads them by bare filename from the session's working
   directory. tool_schema.json is a precomputed dump of
   registry.get_ollama_tools() from the real Windows codebase (see the
   note by TOOL_SCHEMA_PATH below for why it's a static file here rather
   than a live import) - re-generate it locally and re-upload if the tool
   set changes:
       Run from the repo root (this avoids embedding a Windows path with
       backslashes in this docstring, which broke Python's own parsing of
       this file the first time - confirmed live):
       python -c "import json, assistant.tools; from assistant.tools.registry import registry; json.dump(registry.get_ollama_tools(), open('data/finetune/tool_schema.json', 'w', encoding='utf-8'), indent=2)"
2. Runtime -> Change runtime type -> T4 GPU (free tier).
3. Paste this whole file into a Colab cell and run it (NOT `%run` - this
   script deliberately has no `__file__`/path dependency at all, needed
   because a plain pasted cell doesn't define `__file__` the way a script
   file does; `%run` would also work but paste-and-run is what this project
   itself hit and fixed live, so it's the confirmed path). No verified
   runtime estimate is given here on purpose - the real
   measured max sequence length for this dataset against the real
   Qwen2.5-3B-Instruct tokenizer (8541 tokens, driven by the ~37-tool
   schema block Qwen's chat template injects, not the training examples
   themselves - this grew from an earlier-measured 7192 as more tools and
   longer tool descriptions were added this session) is unusually long for
   a ~100-example dataset, and this was never actually run on a T4 to time
   it. Expect it to be slower per-step than a typical short-instruction
   fine-tune of this size, not necessarily faster just because the example
   count or model size is small.
4. Download the resulting .gguf file from the Colab Files panel when it
   finishes.
5. On this machine, build an Ollama model from it (see the Modelfile
   instructions printed at the end of this script) and point
   config/config.yaml's llm.model at the new model name.

This was NOT run as part of building/retargeting it - there is no GPU
available in the environment this was written in either, only this
laptop's 4GB card, which the script itself declines to use for training.
Verify it actually works end-to-end on your own Colab run before relying
on the result; report back exactly what broke if anything does, the same
way everything else in this project has been tested for real rather than
assumed.
"""

# ============================================================================
# Step 1: install dependencies (Colab only - skip if already installed)
# ============================================================================
# !pip install -q unsloth
# !pip install -q --no-deps trl peft accelerate bitsandbytes

# unsloth imported first, deliberately - it patches trl/transformers/peft on
# import to apply its speed/memory optimizations, and warns loudly (checked
# live) if anything else is imported before it.
from unsloth import FastLanguageModel  # noqa: E402

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from datasets import Dataset  # noqa: E402
from trl import SFTTrainer, SFTConfig  # noqa: E402

# The exact tool schema ollama_client.py sends at inference time
# (registry.get_ollama_tools()) - see the note by _format_example below for
# why this matters. NOT imported live from assistant.tools here (found
# live: pasting this into a Colab cell raises "NameError: name '__file__'
# is not defined" - a plain pasted cell has no __file__ the way a real .py
# script does - and even fixing that, assistant.tools pulls in Windows-only
# packages, pywin32/AppOpener among them, that plain don't install on
# Colab's Linux runtime). Precomputed once on the real Windows machine
# instead (`python -c "...json.dump(registry.get_ollama_tools(), ...)"`)
# into tool_schema.json, uploaded alongside the dataset - same real schema,
# no live import needed here at all.
TOOL_SCHEMA_PATH = Path("tool_schema.json")  # uploaded into the Colab session, same as the dataset

# ============================================================================
# Step 2: load the base model in 4-bit, add LoRA adapters
# ============================================================================

# Measured directly against this dataset with the real Qwen2.5-3B-Instruct
# tokenizer (not estimated, via scripts/validate_finetune_dataset.py): the
# <tools> schema block alone that Qwen's template injects for this
# project's ~37 tools runs to several thousand tokens, and the longest full
# example (system prompt + tools + user + assistant tool call) came to
# 8541 tokens. This number has already grown once this session (was 7192
# against the same dataset shape earlier, before this session's bug-fixing
# work lengthened many tool descriptions and added new tools) - a first
# pass at this used 4096 and would have silently truncated the assistant's
# tool call off the END of most examples (SFTTrainer truncates quietly, no
# error), training on broken/incomplete targets without any visible sign
# something was wrong. Set with real headroom above the measured 8541,
# not just matched to it - re-run validate_finetune_dataset.py and adjust
# both this and its own copy of MAX_SEQ_LENGTH if the tool set changes
# again; the schema block will keep climbing as tools are added.
MAX_SEQ_LENGTH = 10240
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # auto-detect (bfloat16 on a T4-class GPU)
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ============================================================================
# Step 3: load and format the dataset
# ============================================================================
# Uses the tokenizer's own chat template (with tool-calling support built
# into Qwen2.5's template) rather than hand-building prompt strings - this
# is what keeps the training format consistent with how Ollama actually
# formats prompts for this model family at inference time.

DATASET_PATH = Path("tool_calling_dataset.jsonl")  # uploaded into the Colab session


def _load_raw_examples(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


with open(TOOL_SCHEMA_PATH, "r", encoding="utf-8") as f:
    _TOOLS = json.load(f)


def _format_example(example: dict) -> dict:
    # Passing tools= here matters, not optional: Qwen2.5's chat template
    # injects a whole <tools>...</tools> JSON block into the system message
    # when tools= is given, which is NOT present if you only pass the plain
    # system prompt string - confirmed by rendering both ways and diffing
    # (29342 chars with tools= vs ~7900 without). ollama_client.py always
    # sends tools=registry.get_ollama_tools() on every real call, so
    # training without it would train on a materially different prompt
    # shape than the model actually sees at inference time.
    text = tokenizer.apply_chat_template(
        example["messages"],
        tools=_TOOLS,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


raw_examples = _load_raw_examples(DATASET_PATH)
print(f"Loaded {len(raw_examples)} training examples.")

formatted = [_format_example(ex) for ex in raw_examples]
dataset = Dataset.from_list(formatted)

# ============================================================================
# Step 4: train
# ============================================================================

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        # Kept to batch_size=1 (with more accumulation steps to reach the
        # same effective batch size of 8) specifically because the real
        # measured max sequence length here (8541 tokens) is much longer
        # than a typical short-instruction dataset - I could not test the
        # actual peak memory use on a real T4 from this environment (no GPU
        # access here), so this is a deliberately conservative default to
        # reduce OOM risk, not a value verified against real Colab hardware.
        # If you hit an OOM anyway, this is the first thing to reduce further.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=3,  # small, targeted dataset - a few epochs, not many
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer.train()

# ============================================================================
# Step 5: export to GGUF for Ollama
# ============================================================================

GGUF_OUT_DIR = "vaayu_qwen2.5_3b_finetuned"
model.save_pretrained_gguf(GGUF_OUT_DIR, tokenizer, quantization_method="q4_k_m")

print(
    f"""
Done. Download the .gguf file from {GGUF_OUT_DIR}/ in the Colab Files panel.

To use it with Ollama on your machine:
1. Put the downloaded .gguf file somewhere in this repo, e.g. models/vaayu-finetuned.gguf
2. Create a file named Modelfile next to it containing:

     FROM ./vaayu-finetuned.gguf

3. Run: ollama create vaayu-finetuned -f Modelfile
4. Set config/config.yaml's llm.model to "vaayu-finetuned"
5. Run scripts/setup_check.py to confirm Ollama sees it, then compare its
   real behavior against BOTH qwen2.5:3b (the un-fine-tuned base - did
   fine-tuning actually improve tool-selection accuracy?) and
   qwen2.5:7b-instruct (is it now competitive with 7B's accuracy while
   keeping 3B's speed?) on the same real commands - don't assume the
   fine-tune helped just because training completed.
"""
)

