import time
from datetime import datetime

import pytest

from assistant import alarms
from assistant.llm import router


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(alarms, "PATH", tmp_path / "alarms.json")
    monkeypatch.setattr(alarms, "_ringing", None)


@pytest.mark.parametrize("said, hhmm", [("1:30 pm", "13:30"), ("13:30", "13:30"), ("1:30", "13:30"),
                                        ("7 am", "07:00"), ("half past 6 in the morning", "06:30"), ("7 baje", "19:00")])
def test_times_as_people_say_them(said, hhmm):
    assert alarms.parse_time(said, datetime(2026, 10, 25, 9, 0)).strftime("%H:%M") == hhmm


@pytest.mark.parametrize("text, tool", [
    ("set an alarm to wake me up at 1:30pm", ("set_alarm", {"time": "1:30pm"})),  # the user's example
    ("set an alarm for 13:30", ("set_alarm", {"time": "13:30"})),  # the user's example
    ("wake me up at 7 am", ("set_alarm", {"time": "7 am"})),
    ("set a timer for 10 minutes", ("set_timer", {"seconds": 600})),
    ("cancel the alarm for 7 am", ("cancel_alarm", {"which": "7 am"})),
    ("what alarms do I have", ("list_alarms", {})),
])
def test_voice_commands(text, tool):
    assert router.fast_route(text) == ("tools", [tool])


def test_set_list_cancel():
    assert alarms.set_alarm("7 am").startswith("Alarm set for 7:00 AM")
    assert "alarm at 7:00 AM" in alarms.list_alarms()
    assert alarms.cancel("all") == "Cancelled 1 alarm." and alarms.list_alarms().startswith("You have no")


def test_rings_until_the_user_is_awake_then_stops(monkeypatch):
    monkeypatch.setattr(alarms, "_RING_EVERY", 0.2)
    said = []
    alarms.set_timer(1)
    alarms._real_start_scheduler(speak=said.append)
    time.sleep(2.0)
    assert alarms.is_ringing() and len(said) >= 2  # calling out again and again
    assert router.fast_route("I'm awake") == ("tools", [("stop_alarm", {"said": "I'm awake"})])
    assert alarms.dismiss("I'm awake").startswith("Okay, timer stopped")
    n = len(said)
    time.sleep(0.6)
    assert not alarms.is_ringing() and len(said) == n  # quiet now


def test_five_more_minutes_snoozes():
    alarms._ringing = {"kind": "alarm", "at": "2026-10-25T07:00:00"}
    assert alarms.dismiss("5 more minutes") == "Okay, 5 more minutes."
    assert "timer at" in alarms.list_alarms()
