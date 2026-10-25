import json
import os
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from assistant import interaction_log, knowledge, memory, research
from assistant.tools import web_answer


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "DIR", tmp_path / "knowledge")
    monkeypatch.setattr(memory, "CONVERSATIONS_PATH", tmp_path / "conversations.jsonl")
    monkeypatch.setattr(memory, "FACTS_PATH", tmp_path / "memory_facts.json")
    return tmp_path


def _client(reply):
    client = MagicMock()
    client.chat.return_value = {"message": {"content": reply}}
    return client


# --- the store --------------------------------------------------------------------------------------

def test_add_dedupes_by_meaning_and_survives_a_reload():
    row, updated = knowledge.add("notes", "Narendra Modi is the Prime Minister of India.", {"url": "a"})
    assert not updated
    again, updated = knowledge.add("notes", "Narendra Modi is the prime minister of India", {"url": "b"})
    assert updated and again["id"] == row["id"] and again["meta"]["url"] == "b"
    knowledge.add("notes", "Kesariya is a song from Brahmastra.")
    knowledge._cache.clear()  # as if Vaayu restarted
    assert [r["text"] for r in knowledge.all("notes")] == ["Narendra Modi is the prime minister of India",
                                                           "Kesariya is a song from Brahmastra."]
    hits = knowledge.search("notes", "who is the prime minister of india", k=1)
    assert hits and "Modi" in hits[0][1]["text"]
    assert knowledge.remove("notes", [row["id"]]) == 1 and len(knowledge.all("notes")) == 1


def test_search_on_an_empty_kind_is_empty():
    assert knowledge.search("lessons", "anything") == [] and knowledge.all("lessons") == []


# --- Mem0-style memory ------------------------------------------------------------------------------

def test_learn_updates_a_contradicted_fact_instead_of_adding_one(monkeypatch):
    monkeypatch.setattr(memory, "_SHOWN", 0.2)  # thresholds are tuned for the real model, not the test's word-hash
    memory.add_facts(["Lives in Gurugram.", "Works at TCS."])
    client = _client("UPDATE 1: Lives in Pune.")
    assert memory.learn(client, "m", "I moved, I live in Pune now, not in Gurugram", "Nice!") == 1
    prompt = client.chat.call_args.kwargs["messages"][1]["content"]
    assert "ALREADY REMEMBERED:" in prompt and "Lives in Gurugram." in prompt
    assert memory.facts() == ["Lives in Pune.", "Works at TCS."]


def test_learn_adds_with_or_without_the_add_prefix():
    assert memory.learn(_client("ADD: Has a dog named Bruno.\n- Plays cricket on Sundays."), "m", "x", "y") == 2
    assert memory.facts() == ["Has a dog named Bruno.", "Plays cricket on Sundays."]
    assert memory.learn(_client("UPDATE 7: out of range becomes an add."), "m", "x", "y") == 1


def test_facts_for_gives_the_relevant_ones_plus_the_newest():
    memory.add_facts(["Has a sister named Riya."] + [f"Likes topic number {i}." for i in range(10)])
    got = memory.facts_for("my sister named Riya", k=3, recent=2)
    assert got[0] == "Has a sister named Riya." and got[-2:] == ["Likes topic number 8.", "Likes topic number 9."]


def test_old_fact_file_is_migrated_once(fresh):
    memory.FACTS_PATH.write_text(json.dumps([{"fact": "Name is Manasvi.", "ts": "2026-10-24T10:00:00"}]))
    memory._migrated.discard(memory.FACTS_PATH)
    assert memory.facts() == ["Name is Manasvi."]
    assert not memory.FACTS_PATH.exists() and memory.FACTS_PATH.with_suffix(".json.migrated").exists()


# --- answer cache and offline answers ---------------------------------------------------------------

def test_a_repeated_question_is_answered_from_the_cache(monkeypatch):
    looked = []
    monkeypatch.setattr(web_answer, "_look_up", lambda q: looked.append(q) or {"results": []})
    web_answer.remember_answer("who is the president of india", "Droupadi Murmu is the President of India.")
    assert web_answer.answer_question("who is the president of india") == "Droupadi Murmu is the President of India."
    assert looked == []
    web_answer.answer_question("what is the capital of france")
    assert looked == ["what is the capital of france"]
    asked = [json.loads(l)["q"] for l in (knowledge.DIR / "asked.jsonl").read_text().splitlines()]
    assert asked == ["who is the president of india", "what is the capital of france"]


def test_an_old_cached_answer_is_looked_up_again(monkeypatch):
    monkeypatch.setattr(web_answer, "_look_up", lambda q: {"results": [{"title": "fresh", "snippet": ""}]})
    web_answer.remember_answer("who is the president of india", "Old answer.")
    rows, mat = knowledge._load("answers")
    rows[0]["ts"] = (datetime.now() - timedelta(hours=13)).isoformat(timespec="seconds")
    assert isinstance(web_answer.answer_question("who is the president of india"), dict)


def test_failed_replies_are_not_cached():
    web_answer.remember_answer("q", "Sorry, I didn't get that - could you say it again?")
    assert knowledge.all("answers") == []


def test_offline_answers_from_what_was_read_or_says_offline(monkeypatch):
    def offline(q):
        raise ConnectionError(web_answer._OFFLINE)
    monkeypatch.setattr(web_answer, "_look_up", offline)
    knowledge.add("notes", "The Indian cricket team won the 2024 T20 World Cup final against South Africa.")
    got = web_answer.answer_question("who won the 2024 T20 World Cup final against South Africa")
    assert got.startswith("I'm offline, but from what I read on") and "South Africa" in got
    with pytest.raises(ConnectionError):
        web_answer.answer_question("what is the population of mars colony")


def test_the_brain_caches_the_spoken_answer_after_phrasing_it():
    from assistant.llm.ollama_client import OllamaBrain
    from assistant.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register(name="answer_question", description="look it up",
                 parameters={"type": "object", "properties": {"question": {"type": "string"}}})(
        lambda question: {"question": question, "results": [{"title": "Murmu", "snippet": "President"}]})
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="m")
    brain.client = _client("Droupadi Murmu is the President of India.")
    reply, _ = brain.respond([{"role": "user", "content": "who is the president of india"}], reg)
    assert reply == "Droupadi Murmu is the President of India."
    assert knowledge.all("answers")[0]["meta"]["answer"] == reply


# --- research loop -----------------------------------------------------------------------------------

def test_html_to_text_keeps_the_words_not_the_code():
    html = "<html><head><style>p{}</style><script>var x=1</script></head><body><nav>Menu</nav>" \
           "<h1>Title</h1><p>Real   text.</p></body></html>"
    assert research.html_to_text(html) == "Title Real text."


def test_budget_per_hour_and_per_day():
    now = time.time()
    research._log({"ts": datetime.fromtimestamp(now - 60).isoformat(), "topic": "a"})
    assert research._budget_left(per_hour=2, per_day=5, now=now)
    research._log({"ts": datetime.fromtimestamp(now - 30).isoformat(), "topic": "b"})
    assert not research._budget_left(per_hour=2, per_day=5, now=now)
    assert not research._budget_left(per_hour=10, per_day=2, now=now)


def test_idle_needs_quiet_and_no_media(monkeypatch, tmp_path):
    from assistant import orchestrator
    log = tmp_path / "interactions.jsonl"
    log.write_text("{}")
    monkeypatch.setattr(interaction_log, "LOG_PATH", log)
    monkeypatch.setattr(orchestrator, "_media_playing", lambda: False)
    os.utime(log, (time.time() - 700, time.time() - 700))
    assert research.idle(10)
    os.utime(log, None)  # someone just spoke
    assert not research.idle(10)
    os.utime(log, (time.time() - 700, time.time() - 700))
    monkeypatch.setattr(orchestrator, "_media_playing", lambda: True)
    assert not research.idle(10)


def test_next_topic_newest_question_first_then_taste(monkeypatch):
    from collections import Counter
    from assistant.tools import youtube_taste
    knowledge.DIR.mkdir(parents=True)
    (knowledge.DIR / "asked.jsonl").write_text(json.dumps({"q": "old question"}) + "\n" + json.dumps({"q": "new one"}) + "\n")
    assert research.next_topic() == "new one"
    research._log({"ts": research._now(), "topic": "new one"})
    research._log({"ts": research._now(), "topic": "old question"})
    monkeypatch.setattr(youtube_taste, "profile", lambda: (Counter({"og crew": 2.0}), Counter()))
    assert research.next_topic() == "latest news about og crew"


def test_research_reads_summarises_and_remembers(monkeypatch):
    monkeypatch.setattr(web_answer, "_ddg_text", lambda q: [{"title": "T", "href": "https://example.com/a", "body": "b"}])
    monkeypatch.setattr(research, "_fetch", lambda url: "Real page text about OG Crew. " * 30)
    client = _client("OG Crew is an Indian gaming YouTube channel. They posted a new vlog this week.")
    row = research.research("latest news about og crew", client, "m")
    assert row and knowledge.search("notes", "og crew gaming channel", k=1)[0][1]["meta"]["url"] == "https://example.com/a"
    assert "tools" not in client.chat.call_args.kwargs
    logged = [json.loads(l) for l in research._log_path().read_text().splitlines()]
    assert logged[-1]["topic"] == "latest news about og crew" and logged[-1]["note"].startswith("OG Crew")


def test_research_stops_if_the_user_starts_talking_and_skips_empty_summaries(monkeypatch):
    monkeypatch.setattr(web_answer, "_ddg_text", lambda q: [{"title": "T", "href": "https://example.com/a", "body": "b"}])
    monkeypatch.setattr(research, "_fetch", lambda url: "words " * 200)
    client = _client("NOTHING")
    assert research.research("x", client, "m", still_idle=lambda: False) is None
    client.chat.assert_not_called()
    assert research.research("x", client, "m") is None and knowledge.all("notes") == []


def test_research_thread_is_off_when_disabled():
    from assistant.config_schema import AppConfig
    cfg = AppConfig()
    cfg.research.enabled = False
    assert research._real_start(cfg, MagicMock()) is None
