"""
Validates data/finetune/tool_calling_dataset.jsonl against the REAL
Qwen2.5-3B-Instruct tokenizer before it's ever handed to a training run -
specifically checks that every example, once rendered with the actual
tools= schema (see finetune_on_colab.py's _format_example for why that
matters), fits within finetune_on_colab.py's MAX_SEQ_LENGTH.

Retargeted from Qwen2.5-7B-Instruct to Qwen2.5-3B-Instruct: real-usage
testing this session measured 7B (even at its best quantization fit on this
machine's 4GB GPU) at 10-300+ seconds per turn, dozens of live turns, never
once under 5 seconds - a hard hardware/model-size wall, not a tunable
setting (see README's "Model upgrade + fine-tuning" and Q3_K_M sections).
3B already meets the speed bar structurally (fits fully in VRAM, ~4.2s
warm, measured earlier this session) - the real gap is accuracy/tool-
selection reliability, which is exactly what fine-tuning targets, so 3B is
the more promising base to fine-tune, not 7B. The dataset itself doesn't
change - it's built from this project's real tool registry and targets
real tool-selection confusions, independent of which model size learns
from it (if anything, more relevant for 3B, which needs more help on
exactly these mix-ups than 7B does).

This exists because it already caught a real bug: an earlier version of
finetune_on_colab.py used MAX_SEQ_LENGTH=4096, but the real measured max
across this dataset (rendered with the actual ~36-tool schema block Qwen's
template injects) was 7192 tokens against the 7B tokenizer - training would
have silently truncated the assistant's tool call off the end of many
examples, with SFTTrainer giving no error, just quietly training on broken
targets. Re-run this any time the tool set, the dataset, or the target
model changes; the schema block grows as tools are added, so this number
will keep climbing regardless of which Qwen2.5 size is targeted.

Requires transformers + jinja2 (not in requirements.txt - these are only
needed for this one-off local check, not for Vaayu's actual runtime):
    pip install transformers jinja2

Usage: python scripts/validate_finetune_dataset.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assistant.tools  # noqa: E402,F401 - registers every tool
from assistant.tools.registry import registry  # noqa: E402

_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "finetune" / "tool_calling_dataset.jsonl"
_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
_MAX_SEQ_LENGTH = 10240  # keep in sync with finetune_on_colab.py


def main() -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Missing dependency - run: pip install transformers jinja2")
        sys.exit(1)

    if not _DATASET_PATH.exists():
        print(f"No dataset found at {_DATASET_PATH} - run scripts/build_finetune_dataset.py first.")
        sys.exit(1)

    print(f"Loading the real {_MODEL_NAME} tokenizer (downloads once, then caches)...")
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    tools = registry.get_ollama_tools()

    examples = []
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Checking {len(examples)} examples against MAX_SEQ_LENGTH={_MAX_SEQ_LENGTH}...")

    max_tokens = 0
    over_budget = []
    for ex in examples:
        text = tokenizer.apply_chat_template(ex["messages"], tools=tools, tokenize=False, add_generation_prompt=False)
        n_tokens = len(tokenizer(text)["input_ids"])
        max_tokens = max(max_tokens, n_tokens)
        if n_tokens > _MAX_SEQ_LENGTH:
            over_budget.append((ex["messages"][1]["content"], n_tokens))

    print(f"Max tokens across dataset: {max_tokens} (budget: {_MAX_SEQ_LENGTH})")

    if over_budget:
        print(f"\n[FAIL] {len(over_budget)} example(s) would be silently truncated during training:")
        for user_text, n_tokens in over_budget:
            print(f"  - {n_tokens} tokens: {user_text!r}")
        print("\nRaise MAX_SEQ_LENGTH in both this file and finetune_on_colab.py, or trim the dataset.")
        sys.exit(1)

    headroom = _MAX_SEQ_LENGTH - max_tokens
    print(f"[OK] Every example fits, with {headroom} tokens of headroom on the longest one.")


if __name__ == "__main__":
    main()
