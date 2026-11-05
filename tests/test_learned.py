"""Commands learned from the user's own way of saying them (assistant/learned.py)."""
import pytest

from assistant import learned
from assistant.llm import router


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(learned, "PATH", tmp_path / "learned.json")
    monkeypatch.setattr(learned, "_last", None)


def _llm(tool, args, ok=True):
    return [{"path": "llm", "tool": tool, "args": args, "ok": ok}]


def test_a_sentence_the_brain_worked_out_is_instant_next_time():
    said = "get that browser of mine going"
    assert router.fast_route(said) is None
    assert learned.from_turn(said, _llm("open_app", {"app_name": "chrome"}))
    assert router.fast_route("Get that browser of mine going.") == ("tools", [("open_app", {"app_name": "chrome"})])


@pytest.mark.parametrize("trace", [
    _llm("open_app", {"app_name": "chrome"}, ok=False),                                   # it failed
    _llm("send_whatsapp_message", {"contact": "SAVED MESSAGES", "message": "hi"}),        # outward
    _llm("open_app", {"app_name": "chrome"}) + _llm("open_app", {"app_name": "brave"}),   # two steps
    [{"path": "chat"}],                                                                   # just talk
    [{"path": "fast", "tool": "open_app", "args": {}, "ok": True}],                       # the router knew it
    _llm("think_hard", {"question": "x"}),                                                # never kept
])
def test_only_one_safe_call_that_worked_is_learned(trace):
    assert not learned.from_turn("some new way of saying it", trace)
    assert learned.match("some new way of saying it") is None


def test_no_right_after_unlearns_it():
    learned.from_turn("bajao kuch", _llm("control_media", {"action": "play"}))
    assert not learned.correction("open notepad")  # not a correction
    assert learned.correction("No, not that.")
    assert learned.match("bajao kuch") is None
