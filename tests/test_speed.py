"""The brain's speed rules (measured 2026-11-04 on the GTX 1650)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "assistant"


def test_every_brain_call_uses_one_context_size():
    # A different num_ctx makes Ollama RELOAD the model: 3.7-4.6 s at every chat <-> tool switch.
    for f in ROOT.rglob("*.py"):
        for line in f.read_text(encoding="utf-8").splitlines():
            m = re.search(r'"num_ctx":\s*(\d+)', line)
            if m and m.group(1) != "8192":
                assert "HINDI" in f.read_text(encoding="utf-8"), f"{f.name}: num_ctx {m.group(1)} - use BRAIN_CTX"


def test_the_same_step_is_done_once():
    from unittest.mock import MagicMock, patch
    from assistant.llm import router
    from assistant.llm.ollama_client import OllamaBrain
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="m")
    fake = MagicMock()
    fake._tools = {"get_system_status": None}
    fake.requires_confirmation.return_value = False
    fake.refusal_before_asking.return_value = None
    fake.dispatch.return_value = {"ok": True, "result": "Battery is at 100%."}
    with patch.object(router, "fast_route", return_value=("tools", [("get_system_status", {}),
                                                                     ("get_system_status", {})])):
        brain.respond([{"role": "user", "content": "battery and is it charging"}], fake, lambda *a: True)
    assert fake.dispatch.call_count == 1
