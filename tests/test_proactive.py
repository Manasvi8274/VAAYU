"""Vaayu speaking up on his own (assistant/proactive.py)."""
from datetime import datetime, timedelta

import pytest

from assistant import proactive
from assistant.llm import router

ALL = {"welcome": True, "breaks": True, "night": True}


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(proactive, "PATH", tmp_path / "proactive.json")
    monkeypatch.setattr(proactive, "_due_today", lambda: ["submit the report"])


def test_welcome_back_after_an_hour_away():
    w, t = proactive.Watcher(), datetime(2026, 11, 5, 15, 0)
    assert w.tick(t, 0, ALL) is None
    assert w.tick(t + timedelta(minutes=70), 70 * 60, ALL) is None          # away
    said = w.tick(t + timedelta(minutes=71), 2, ALL)                        # back
    assert said.startswith("Welcome back. It's 4:11 PM.") and "submit the report" in said
    assert w.tick(t + timedelta(minutes=72), 2, ALL) is None                # once


def test_a_break_after_two_hours_without_a_pause():
    w, t = proactive.Watcher(), datetime(2026, 11, 5, 10, 0)
    for minute in range(0, 119, 10):
        assert w.tick(t + timedelta(minutes=minute), 10, ALL) is None
    assert "two hours" in w.tick(t + timedelta(minutes=121), 10, ALL)
    w2 = proactive.Watcher()
    w2.tick(t, 10, ALL)
    w2.tick(t + timedelta(minutes=60), 6 * 60, ALL)                         # a 6-minute pause resets it
    assert w2.tick(t + timedelta(minutes=121), 10, ALL) is None


def test_late_night_once_and_nothing_when_off():
    w = proactive.Watcher()
    assert "sleep" in w.tick(datetime(2026, 11, 6, 1, 20), 5, ALL)
    assert w.tick(datetime(2026, 11, 6, 1, 50), 5, ALL) is None
    assert proactive.Watcher().tick(datetime(2026, 11, 6, 1, 20), 5, {k: False for k in ALL}) is None


def test_switched_by_voice():
    assert router.fast_route("turn on proactive mode") == ("tools", [("set_proactive", {"kind": "", "on": True})])
    assert router.fast_route("turn off break reminders") == ("tools", [("set_proactive", {"kind": "breaks", "on": False})])
    assert router.fast_route("stop telling me to sleep") == ("tools", [("set_proactive", {"kind": "night", "on": False})])
    assert proactive.settings() == {"welcome": False, "breaks": False, "night": False}  # off until asked
    proactive.switch(None, True)
    proactive.switch("night", False)
    assert proactive.settings() == {"welcome": True, "breaks": True, "night": False}
