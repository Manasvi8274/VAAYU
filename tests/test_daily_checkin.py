"""
Orchestrator._maybe_run_daily_checkin - now the trigger for the daily log review
(assistant/daily_review.py): once a day at/after the configured time, in the
background, no speaking and no question. Same __init__-bypass construction
pattern as test_orchestrator_confirm.py.
"""

import datetime as real_datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from assistant import daily_review
from assistant.config_schema import AppConfig
from assistant.orchestrator import Orchestrator
from assistant.self_review import issue_log

_TZ = ZoneInfo("Asia/Kolkata")


def _make_orchestrator():
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = AppConfig()  # defaults: enabled, 21:00, Asia/Kolkata
    orch._speak = MagicMock()
    orch._capture_utterance = MagicMock()
    return orch


def _at(hour, minute, date=(2026, 9, 20)):
    return real_datetime.datetime(*date, hour, minute, tzinfo=_TZ)


def _patched_now(dt):
    mock_dt = MagicMock(wraps=real_datetime.datetime)
    mock_dt.now.return_value = dt
    return patch("assistant.orchestrator.datetime", mock_dt)


def _run(tmp_path, when, enabled=True):
    started = MagicMock(return_value=True)
    with patch.object(issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"), \
            patch.object(daily_review, "start_in_background", started), _patched_now(when):
        orch = _make_orchestrator()
        orch.config.self_review.enabled = enabled
        orch._maybe_run_daily_checkin()
        return orch, started, issue_log.read_last_checkin_date()


def test_does_nothing_before_the_time(tmp_path):
    orch, started, last = _run(tmp_path, _at(20, 59))
    started.assert_not_called()
    assert last is None


def test_does_nothing_when_disabled(tmp_path):
    _orch, started, _last = _run(tmp_path, _at(21, 5), enabled=False)
    started.assert_not_called()


def test_starts_the_review_in_the_background_without_a_word(tmp_path):
    orch, started, last = _run(tmp_path, _at(21, 5))
    started.assert_called_once_with("scheduled")
    orch._speak.assert_not_called()
    orch._capture_utterance.assert_not_called()  # no question, nothing blocks the turn
    assert last == "2026-09-20"


def test_runs_once_a_day_and_again_the_next_day(tmp_path):
    started = MagicMock(return_value=True)
    with patch.object(issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"), \
            patch.object(daily_review, "start_in_background", started):
        orch = _make_orchestrator()
        for when in (_at(21, 5), _at(23, 0), _at(21, 30, date=(2026, 9, 21))):
            with _patched_now(when):
                orch._maybe_run_daily_checkin()
    assert started.call_count == 2


def test_late_start_still_runs_hours_after_the_time(tmp_path):
    _orch, started, _last = _run(tmp_path, _at(23, 55))
    started.assert_called_once()
