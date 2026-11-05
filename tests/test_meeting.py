"""Holding the user's place in a meeting (assistant/meeting.py) - against a fake page, never a real meeting."""
import json
import time

import pytest

from assistant import meeting


class FakeMeet:
    """The labels Meet shows, and what was clicked, in order."""

    def __init__(self):
        self.clicked, self.chat, self.captions = [], [], ""
        self.buttons = {"Turn off microphone", "Turn off camera", "Join now", "Turn on captions", "Chat with everyone",
                        "Close"}

    def eval(self, ws, js):
        if js.startswith("(labels"):
            labels = json.loads(js[js.rindex("(") + 1:-1])
            hit = next((b for b in labels if b in self.buttons), "")
            if hit:
                self.clicked.append(hit)
            return hit
        if js.startswith("(line"):
            self.chat.append(json.loads(js[js.rindex("(") + 1:-1]))
            return True
        return self.captions


@pytest.fixture
def meet(monkeypatch, tmp_path):
    fake = FakeMeet()
    monkeypatch.setattr(meeting, "_eval", fake.eval)
    monkeypatch.setattr(meeting, "DIR", tmp_path)
    from assistant.tools import browser
    monkeypatch.setattr(browser, "_new_tab", lambda url: {"webSocketDebuggerUrl": "ws://fake"})
    yield fake
    if meeting._state:
        meeting._state["stop"].set()
        meeting._state.clear()


def test_mic_and_camera_go_off_before_joining_and_the_late_line_is_posted(meet):
    said = meeting.join("https://meet.google.com/abc-defg-hij", sleep=lambda s: None)
    assert meet.clicked[:3] == ["Turn off microphone", "Turn off camera", "Join now"]
    assert meet.chat == [meeting.LATE_LINE] and "mic and camera off" in said and "on your way" in said


def test_notes_from_the_captions_and_a_summary_when_the_user_arrives(meet, monkeypatch):
    meeting.join("https://meet.google.com/abc-defg-hij", sleep=lambda s: None)
    meeting._state["stop"].set()  # drive the note-taker by hand
    import threading
    stop = threading.Event()
    t = threading.Thread(target=meeting._take_notes, args=(stop, 0.01))
    t.start()
    meet.captions = "Riya: the launch moves to Friday."
    time.sleep(0.1)
    meet.captions = "Riya: the launch moves to Friday. Manasvi, can you send the deck?"
    time.sleep(0.1)
    stop.set()
    t.join(1)
    assert meeting._state["lines"] == ["Riya: the launch moves to Friday.", "Manasvi, can you send the deck?"]
    from assistant.tools import think
    prompts = []
    monkeypatch.setattr(think, "ask_claude", lambda p: prompts.append(p) or "The launch moved to Friday; send the deck.")
    said = meeting.stop_and_summarise()
    assert said.startswith("Welcome. The launch moved to Friday") and "send the deck?" in prompts[0]
    assert meeting._state == {}


def test_it_asks_first_and_is_routed_from_a_link_or_the_clipboard():
    from assistant.llm import router
    from assistant.tools.registry import registry
    from guardian.action_rules import OUTWARD, tier
    assert tier("join_meeting") == OUTWARD and registry.requires_confirmation("join_meeting", {})
    assert "mic and camera off" in registry.get_confirmation_prompt("join_meeting", {"time": "3 pm"})
    assert router.fast_route("join https://meet.google.com/abc-defg-hij at 3 pm") == (
        "tools", [("join_meeting", {"link": "https://meet.google.com/abc-defg-hij", "time": "3 pm"})])
    assert router.fast_route("join the meeting i copied") == ("tools", [("join_meeting", {"link": "", "time": ""})])
    assert router.fast_route("i'm here") is None  # only while in a meeting for the user
