"""Background jobs of several steps (assistant/agent.py)."""
import json

from assistant import agent
from assistant.tools.registry import ToolRegistry


def _registry(calls):
    r = ToolRegistry()

    @r.register(name="answer_question", description="Look something up on the web.",
                parameters={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]})
    def answer_question(question):
        calls.append(("answer_question", question))
        return "Redmi Note 13 at 17,999; Moto G64 at 16,999."

    @r.register(name="add_note", description="Save a note.",
                parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    def add_note(text):
        calls.append(("add_note", text))
        return "Saved the note."

    @r.register(name="send_whatsapp_message", description="Send a WhatsApp message.",
                parameters={"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}}})
    def send(**kw):
        calls.append(("send", kw))
        return "sent"
    return r


def test_a_job_runs_step_by_step_and_ends_with_a_summary():
    calls, prompts = [], []
    answers = iter(['{"tool": "answer_question", "args": {"question": "best phones under 20000"}}',
                    'Sure! {"tool": "add_note", "args": {"text": "Redmi Note 13, Moto G64"}}',
                    '{"done": "I found two phones and saved them in your notes."}'])
    job = agent.run("research phones under 20000 and save them in a note", _registry(calls),
                    ask=lambda p: prompts.append(p) or next(answers))
    assert [c[0] for c in calls] == ["answer_question", "add_note"]
    assert job["summary"] == "I found two phones and saved them in your notes."
    assert "Redmi Note 13 at 17,999" in prompts[1]  # each step's result goes into the next choice


def test_a_job_can_never_send_or_use_unlisted_tools():
    calls = []
    answers = iter(['{"tool": "send_whatsapp_message", "args": {"to": "Raman", "text": "hi"}}',
                    '{"done": "Say send it to Raman if you want it sent."}'])
    job = agent.run("tell raman the plan", _registry(calls), ask=lambda p: next(answers))
    assert calls == [] and "not one of the tools" in job["steps"][0]["result"]
    assert "send_whatsapp_message" not in agent._menu(agent._tools(_registry([])))


def test_a_job_stops_after_max_steps():
    calls = []
    job = agent.run("loop forever", _registry(calls),
                    ask=lambda p: json.dumps({"tool": "answer_question", "args": {"question": "again"}}))
    assert len(calls) == agent.MAX_STEPS and job["summary"].startswith(f"I did {agent.MAX_STEPS} steps")


def test_start_answers_at_once_and_announces_when_done(monkeypatch):
    import threading
    said, done = [], threading.Event()
    monkeypatch.setattr(agent, "ANNOUNCE", lambda text: said.append(text) or done.set())
    monkeypatch.setattr(agent, "run", lambda task: {"summary": "All done."})
    assert agent.start("compare python and rust").startswith("On it")
    assert done.wait(2) and said == ["Done with your task. All done."]


def test_jobs_are_routed_and_the_rest_is_not():
    from assistant.llm import router
    assert router.fast_route("research the best budget phones and put it in a document")[1][0][0] == "do_task"
    assert router.fast_route("how is my task going") == ("tools", [("task_status", {})])
    assert router.fast_route("find my file") is None
    assert router.fast_route("open chrome and play music")[1][0][0] == "open_app"
