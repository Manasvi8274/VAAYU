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


def test_a_slow_turn_is_acknowledged_once_and_a_fast_one_never(monkeypatch):
    # User (2026-11-04): output within 2-3 s, never more.
    import time
    from unittest.mock import MagicMock
    from assistant import orchestrator
    monkeypatch.setattr(orchestrator, "ACK_AFTER", 0.05)
    o = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    said = []
    o._speak_now = lambda text: said.append(text) or False
    o._turn_done, o._last_spoke = False, 0
    started = time.time()
    o._ack_if_slow("open notepad and type hello world", started)
    assert said == ["On it: opening notepad, then typing it."]
    o._ack_if_slow("open notepad", started)  # something was already said this turn: no second ack
    assert len(said) == 1
    o._turn_done = True
    o._ack_if_slow("tell me a joke", time.time() + 1)
    assert len(said) == 1


def test_ack_words():
    from assistant.orchestrator import ack_for
    assert ack_for("play kesariya and set volume to 40") == "On it: playing kesariya, then volume to 40."
    assert ack_for("kya haal hai bhai") == "एक सेकंड भाई।"
    assert ack_for("tell me a joke", 1) == "One second."
    assert ack_for("tell me about black holes") == "Okay, let me check."  # a web lookup: in its own words


def test_shortcuts_work_alone_and_as_steps():
    from assistant.llm import router
    assert router.fast_route("open notepad then type hello world and save it") == ("tools", [
        ("open_app", {"app_name": "notepad"}), ("type_text", {"text": "hello world"}), ("press_hotkey", {"keys": "ctrl+s"})])
    assert router.fast_route("undo") == ("tools", [("press_hotkey", {"keys": "ctrl+z"})])
    assert router.fast_route("save the trees campaign is great") is None


def test_thinking_goes_to_claude_and_falls_back_to_the_local_model(monkeypatch):
    import subprocess
    from assistant.llm import router
    from assistant.tools import think
    assert router.fast_route("think about whether i should buy a car or a bike") == \
        ("tools", [("think_hard", {"question": "whether i should buy a car or a bike"})])
    assert router.fast_route("I think it is raining") is None
    monkeypatch.setattr(think.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(think.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "**Buy the bike.**", ""))
    assert think.think_hard("car or bike") == "Buy the bike."  # markdown stripped for speaking
    monkeypatch.setattr(think.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "You've hit your usage limit", ""))
    assert think.think_hard("car or bike") == "Local answer."  # limit reached: the local model thinks instead


def test_a_brain_partly_on_the_cpu_is_reloaded_once_not_forever():
    # Voice test 2026-11-05: 19% of the brain on the CPU made chat 5-7 s; a reload that can't fit must not repeat.
    from unittest.mock import MagicMock, patch
    from assistant.llm.ollama_client import OllamaBrain
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="vaayu-finetuned")
    brain.client.ps.return_value = {"models": [{"model": "vaayu-finetuned:latest", "size": 100, "size_vram": 81}]}
    brain.warm_up = MagicMock()
    assert brain.ensure_on_gpu() is True and brain.client.generate.call_count == 1
    assert brain.ensure_on_gpu() is False and brain.client.generate.call_count == 1  # came back no better: leave it
    brain.client.ps.return_value = {"models": [{"model": "vaayu-finetuned:latest", "size": 100, "size_vram": 100}]}
    assert brain.ensure_on_gpu() is False


def test_the_chat_prompt_keeps_its_cached_start():
    # What changes each turn (clock, facts, earlier talks) must not be in the system prompt: the cached start breaks.
    from unittest.mock import patch
    from assistant.llm.ollama_client import OllamaBrain
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="m")
    brain.trace, brain._hindi_turn = [], False
    brain.client.chat.return_value = {"message": {"content": "Bhai, take a break."}}
    systems = []
    for said in ("i am tired", "tell me a joke"):
        brain._chat([{"role": "user", "content": said}], said)
        msgs = brain.client.chat.call_args.kwargs["messages"]
        systems.append([m["content"] for m in msgs if m["role"] == "system"])
    assert systems[0] == systems[1] and len(systems[0]) == 1
