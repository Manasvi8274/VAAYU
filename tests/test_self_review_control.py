from unittest.mock import patch

import pytest

from assistant.tools import self_review_control


# --- time parsing ---


@pytest.mark.parametrize("text,expected", [
    ("9pm", "21:00"),
    ("9 pm", "21:00"),
    ("9:30pm", "21:30"),
    ("9:30 pm", "21:30"),
    ("21:00", "21:00"),
    ("21:30", "21:30"),
    ("9am", "09:00"),
    ("12pm", "12:00"),  # noon
    ("12am", "00:00"),  # midnight
    ("9", "09:00"),
    ("PM 9", None),
    ("25:00", None),
    ("13pm", None),
    ("9:75pm", None),
    ("noon", None),
    ("", None),
])
def test_parse_time_to_hhmm(text, expected):
    assert self_review_control._parse_time_to_hhmm(text) == expected


# --- set_daily_update_time tool ---


def test_set_daily_update_time_persists_and_confirms():
    fake_config = type("Cfg", (), {"self_review": type("SR", (), {"timezone": "Asia/Kolkata"})()})()
    with patch.object(self_review_control, "load_config", return_value=fake_config), patch.object(
        self_review_control, "set_check_in_time"
    ) as mock_set:
        result = self_review_control.set_daily_update_time("9pm")
    mock_set.assert_called_once_with("21:00")
    assert result == "Daily check-in time set to 21:00 IST."


def test_set_daily_update_time_raises_on_unparseable_time():
    with pytest.raises(ValueError, match="Couldn't understand"):
        self_review_control.set_daily_update_time("whenever")


def test_set_daily_update_time_uses_raw_timezone_label_when_not_ist():
    fake_config = type("Cfg", (), {"self_review": type("SR", (), {"timezone": "America/New_York"})()})()
    with patch.object(self_review_control, "load_config", return_value=fake_config), patch.object(
        self_review_control, "set_check_in_time"
    ):
        result = self_review_control.set_daily_update_time("9pm")
    assert "America/New_York" in result
