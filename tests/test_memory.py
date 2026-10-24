import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from assistant import memory
from assistant.llm import ollama_client
from assistant.llm.ollama_client import OllamaBrain, _is_conversation
from assistant.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def fresh_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "CONVERSATIONS_PATH", tmp_path / "conversations.jsonl")
    monkeypatch.setattr(memory, "FACTS_PATH", tmp_path / "memory_facts.json")


@pytest.fixture
def brain():
    with patch("assistant.llm.ollama_client.ollama.Client"):
        b = OllamaBrain(host="http://fake", model="fake-model")
    b.client = MagicMock()
    b.client.chat.return_value = {"message": {"role": "assistant", "content": "Dekh bhai, thoda break le."}}
    return b


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(name="close_app", description="close an app",
                 parameters={"type": "object", "properties": {"app_name": {"type": "string"}}})(
        lambda app_name: f"Closed {app_name}.")
    return reg


# --- memory store ---

def test_turns_are_logged_and_recent_ones_returned():
    memory.log_turn("hi", "hello")
    memory.log_turn("I'm tired", "Take rest.")
    assert [t["user"] for t in memory.recent_turns(6)] == ["hi", "I'm tired"]
    assert [t["user"] for t in memory.recent_turns(1)] == ["I'm tired"]


def test_old_turns_are_a_different_conversation():
    old = (datetime.now() - timedelta(hours=3)).isoformat(timespec="seconds")
    memory.CONVERSATIONS_PATH.write_text(json.dumps({"ts": old, "user": "x", "reply": "y"}) + "\n", encoding="utf-8")
    assert memory.recent_turns() == []


def test_facts_dedupe_cap_and_forget(monkeypatch):
    assert memory.add_facts(["Has a sister named Riya.", "has a sister named riya.", "Works at TCS."]) == 2
    assert memory.facts() == ["Has a sister named Riya.", "Works at TCS."]
    assert memory.forget("my sister") == 1
    assert memory.facts() == ["Works at TCS."]
    assert memory.forget("bitcoin") == 0
    assert memory.forget() == 1 and memory.facts() == []
    monkeypatch.setattr(memory, "_MAX_FACTS", 3)
    memory.add_facts([f"Fact {i}." for i in range(5)])
    assert memory.facts() == ["Fact 2.", "Fact 3.", "Fact 4."]


def test_learn_parses_facts_and_none():
    client = MagicMock()
    client.chat.return_value = {"message": {"content": "- Name is Manasvi.\n2. Wants to get fit.\n"}}
    assert memory.learn(client, "m", "I'm Manasvi and I want to get fit", "Nice!") == 2
    assert "tools" not in client.chat.call_args.kwargs
    client.chat.return_value = {"message": {"content": "NONE"}}
    assert memory.learn(client, "m", "open brave", "Opened Brave.") == 0
    assert memory.facts() == ["Name is Manasvi.", "Wants to get fit."]


def test_learn_async_never_raises():
    client = MagicMock()
    client.chat.side_effect = TimeoutError("ollama down")
    memory._real_learn_async(client, "m", "hi", "hello").join(5)  # logged, not raised


# --- conversation vs command ---

@pytest.mark.parametrize("text", [
    "I feel stuck in my job, what should I do?", "bhai mujhe gym jaane ka mann nahi karta", "I want to quit my job",
    "I am upset about my exams", "what do you think about crypto", "mujhe samajh nahi aa raha kya karu",
    "tell me a joke", "should I learn python or java", "kuch motivational batao",
])
def test_conversation(text):
    assert _is_conversation(text)


@pytest.mark.parametrize("text", [
    "I am bored, play something", "I want to close chrome", "close it", "check this", "press ctrl s",
    "what's on this page", "memory kitni free hai", "I want to open notepad", "firefox chahiye", "auto",
])
def test_command_not_conversation(text):
    assert not _is_conversation(text)


# --- chat path ---

def test_chat_uses_persona_memory_and_no_tools(brain, registry):
    memory.add_facts(["Has an exam on Monday."])
    memory.log_turn("I'm nervous", "That's normal.")
    reply, msgs = brain.respond([{"role": "user", "content": "I feel stuck in my job, what should I do?"}], registry)
    assert reply == "Dekh bhai, thoda break le." and msgs[-1]["content"] == reply
    kwargs = brain.client.chat.call_args.kwargs
    assert "tools" not in kwargs and kwargs["options"]["num_predict"] == 200
    system = kwargs["messages"][0]["content"]
    assert "elder brother" in system and "Has an exam on Monday." in system
    assert kwargs["messages"][1 + len(ollama_client._CHAT_EXAMPLES):] == [{"role": "user", "content": "I'm nervous"},
                                      {"role": "assistant", "content": "That's normal."},
                                      {"role": "user", "content": "I feel stuck in my job, what should I do?"}]
    assert brain.trace == [{"path": "chat"}]


def test_chat_reply_may_be_four_sentences(brain, registry):
    long = " ".join(f"Sentence number {i} is here to make this reply long enough to be cut." for i in range(6))
    brain.client.chat.return_value = {"message": {"content": long}}
    reply, _ = brain.respond([{"role": "user", "content": "what do you think about my plan"}], registry)
    assert reply.count(".") == 4


def test_commands_still_go_to_tools(brain, registry):
    brain.client.chat.return_value = {"message": {"role": "assistant", "content": "",
                                                  "tool_calls": [{"function": {"name": "close_app",
                                                                               "arguments": {"app_name": "zoom"}}}]}}
    brain.respond([{"role": "user", "content": "could you shut zoom for me if it is running"}], registry)
    assert "tools" in brain.client.chat.call_args_list[0].kwargs
    assert not any(t.get("path") == "chat" for t in brain.trace)


@pytest.mark.parametrize("text,remaining,reply", [
    ("forget everything", [], "Done, I've forgotten everything you told me."),
    ("forget what I told you about my sister", ["Works at TCS."], "Done, I've forgotten that."),
    ("forget about bitcoin", ["Has a sister named Riya.", "Works at TCS."], "I didn't have anything saved about that."),
])
def test_forget(brain, registry, text, remaining, reply):
    memory.add_facts(["Has a sister named Riya.", "Works at TCS."])
    got, _ = brain.respond([{"role": "user", "content": text}], registry)
    assert got == reply and memory.facts() == remaining
    brain.client.chat.assert_not_called()
    assert brain.trace == [{"path": "chat_forget"}]


# --- orchestrator: every real turn remembered, facts learned from chat only ---

@pytest.mark.parametrize("path,learns", [("chat", True), ("fast", False), ("chat_forget", False)])
def test_orchestrator_logs_every_turn_and_learns_from_chat(monkeypatch, path, learns):
    import numpy as np
    from assistant.orchestrator import Orchestrator, State
    orch = Orchestrator.__new__(Orchestrator)
    orch._speak = MagicMock()
    orch._capture_utterance = MagicMock(return_value=np.zeros(10, dtype=np.float32))
    orch.stt = MagicMock()
    orch.stt.transcribe.return_value = "I had a rough day"
    orch._maybe_run_daily_checkin = MagicMock()
    orch.state = State.AWAKE
    orch.messages = []
    orch.brain = MagicMock(trace=[])

    def fake_respond(messages, reg, confirm):
        orch.brain.trace.append({"path": path})
        return "Kya hua bhai?", messages
    orch.brain.respond.side_effect = fake_respond
    learn = MagicMock()
    monkeypatch.setattr(memory, "learn_async", learn)
    orch.handle_one_turn()
    assert [(t["user"], t["reply"]) for t in memory.recent_turns()] == [("I had a rough day", "Kya hua bhai?")]
    assert learn.called is learns


def test_devanagari_chat_reply_is_retried_then_romanized():
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="m")
    brain.client = MagicMock()
    brain.client.chat.return_value = {"message": {"role": "assistant", "content": "ठीक है भाई, चिंता मत कर।"}}
    reply, _ = brain._chat([], "bhai tension ho rahi hai")
    assert brain.client.chat.call_count == 2  # one retry asking for Roman letters
    assert not ollama_client._DEVANAGARI.search(reply) and reply.startswith("thik hai bhai")
