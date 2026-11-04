import json
from datetime import date

import pytest

from assistant.llm import router
from assistant.tools import briefing


@pytest.mark.parametrize("said", ["good morning", "Good morning vaayu", "what is my day like today", "brief me",
                                  "aaj ka kya plan hai"])
def test_briefing_phrases(said):
    assert router.fast_route(said) == ("tools", [("daily_briefing", {})])


def test_morning_walk_is_not_a_briefing():
    assert router.fast_route("morning walk was nice") is None


def test_briefing_has_todays_tasks_and_survives_no_internet(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"task": "Call mom", "due": date.today().isoformat(), "done": False},
                                 {"task": "Old done", "due": "2020-01-01", "done": True},
                                 {"task": "Later", "due": "2099-01-01", "done": False}]))
    monkeypatch.setattr(briefing, "TASKS", tmp_path / "tasks.json")
    said = briefing.daily_briefing()  # conftest: no internet - the weather is just left out
    assert "1 thing to do today: Call mom." in said and "Later" not in said


def test_battery_is_mentioned_once_per_level_and_again_after_charging():
    said = set()
    got = [briefing.battery_warning(p, False, said) for p in (50, 19, 18, 9, 8)]
    assert [bool(g) for g in got] == [False, True, False, True, False]
    assert briefing.battery_warning(30, True, said) is None and briefing.battery_warning(19, False, said)
