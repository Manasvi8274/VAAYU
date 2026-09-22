"""
Orchestrator._maybe_run_daily_checkin - the proactive, permission-gated
check-in described explicitly by the user: logs issues persistently, asks
once a day at a configurable time, and on "yes" writes a report (NOT
autonomous code changes - see self_review/__init__.py for why that line is
held). Same __init__-bypass construction pattern as test_orchestrator_confirm.py.

Every assertion that reads issue_log state lives inside the same
patch.object(...) block that did the writes - reading after the block exits
would read through whatever path the autouse isolation fixture (see
conftest.py) set up instead, which is a different, never-written-to file.
"""

import datetime as real_datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np

from assistant.config_schema import AppConfig
from assistant.orchestrator import Orchestrator
from assistant.self_review import issue_log

_TZ = ZoneInfo("Asia/Kolkata")


def _make_orchestrator(transcript: str = "yes", capture_returns_audio: bool = True):
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = AppConfig()  # defaults: enabled, 21:00, Asia/Kolkata
    orch._speak = MagicMock()
    orch._capture_utterance = MagicMock(
        return_value=(np.zeros(10, dtype=np.float32) if capture_returns_audio else np.array([]))
    )
    orch.stt = MagicMock()
    orch.stt.transcribe.return_value = transcript
    return orch


def _at(hour, minute, date=(2026, 9, 20)):
    return real_datetime.datetime(*date, hour, minute, tzinfo=_TZ)


def _patched_now(dt):
    mock_dt = MagicMock(wraps=real_datetime.datetime)
    mock_dt.now.return_value = dt
    return patch("assistant.orchestrator.datetime", mock_dt)


def test_does_nothing_before_threshold_time(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ), _patched_now(_at(20, 59)):
        issue_log.log_issue("x", "close_app", "err")
        orch = _make_orchestrator()
        orch._maybe_run_daily_checkin()

        orch._speak.assert_not_called()
        assert issue_log.read_last_checkin_date() is None


def test_does_nothing_when_disabled(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ), _patched_now(_at(21, 5)):
        issue_log.log_issue("x", "close_app", "err")
        orch = _make_orchestrator()
        orch.config.self_review.enabled = False
        orch._maybe_run_daily_checkin()

        orch._speak.assert_not_called()


def test_does_nothing_when_no_issues_logged(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ), _patched_now(_at(21, 5)):
        orch = _make_orchestrator()
        orch._maybe_run_daily_checkin()

        orch._speak.assert_not_called()
        # still marks today as checked-in, so it doesn't re-scan every turn
        assert issue_log.read_last_checkin_date() == "2026-09-20"


def test_asks_and_writes_report_on_yes(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ), patch.object(issue_log, "REPORTS_DIR", tmp_path / "reports"), _patched_now(_at(21, 5)):
        issue_log.log_issue("close the brave browser", "close_app", "AppNotFound")
        orch = _make_orchestrator(transcript="yes")
        orch._maybe_run_daily_checkin()

        assert issue_log.read_unacknowledged() == []  # acknowledged after yes
        reports = list((tmp_path / "reports").glob("*.md"))
        assert len(reports) == 1
        assert "close the brave browser" in reports[0].read_text(encoding="utf-8")

        assert orch._speak.call_args_list[0].args[0].startswith("Quick check-in")
        assert (
            orch._speak.call_args_list[-1].args[0]
            == "Update's done - I've written up today's issues for review."
        )


def test_declines_leaves_issues_unacknowledged(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ), patch.object(issue_log, "REPORTS_DIR", tmp_path / "reports"), _patched_now(_at(21, 5)):
        issue_log.log_issue("close the brave browser", "close_app", "AppNotFound")
        orch = _make_orchestrator(transcript="no not now")
        orch._maybe_run_daily_checkin()

        assert len(issue_log.read_unacknowledged()) == 1  # still there for tomorrow
        assert orch._speak.call_args_list[-1].args[0] == "Okay, I'll bring it up again tomorrow."


def test_no_response_leaves_issues_unacknowledged(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ), _patched_now(_at(21, 5)):
        issue_log.log_issue("x", "close_app", "err")
        orch = _make_orchestrator(capture_returns_audio=False)
        orch._maybe_run_daily_checkin()

        assert len(issue_log.read_unacknowledged()) == 1
        orch.stt.transcribe.assert_not_called()


def test_only_asks_once_per_day(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ), patch.object(issue_log, "REPORTS_DIR", tmp_path / "reports"), _patched_now(_at(21, 5)):
        issue_log.log_issue("x", "close_app", "err")
        orch = _make_orchestrator(transcript="no")
        orch._maybe_run_daily_checkin()
        orch._speak.reset_mock()

        # same day, later - even with new unacknowledged issues, must stay quiet
        issue_log.log_issue("y", "seek_video", "err2")
        orch._maybe_run_daily_checkin()

        orch._speak.assert_not_called()


def test_missed_day_still_asks_when_next_online_even_hours_late(tmp_path):
    # Directly matches: "if he misses some updates time and shut off they
    # are not deleted, but instead he picks them up in next update" -
    # simulated here as "checked in on day 1, nothing happens on day 2
    # (process was off), then it's now day 3 at 23:00 (hours past the
    # 21:00 threshold) - it must still ask, not silently skip the day.
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ), patch.object(issue_log, "REPORTS_DIR", tmp_path / "reports"):
        issue_log.write_last_checkin_date("2026-09-18")
        issue_log.log_issue("missed day issue", "open_app", "err")

        with _patched_now(_at(23, 0, date=(2026, 9, 20))):
            orch = _make_orchestrator(transcript="yes")
            orch._maybe_run_daily_checkin()

        assert issue_log.read_last_checkin_date() == "2026-09-20"
        orch._speak.assert_any_call(
            "Quick check-in - I ran into 1 issue today, mostly around open_app. "
            "Want me to write these up for review?"
        )


def test_misconfigured_check_in_time_fails_quietly(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "issues.jsonl"), patch.object(
        issue_log, "LAST_CHECKIN_PATH", tmp_path / "last.txt"
    ):
        orch = _make_orchestrator()
        orch.config.self_review.check_in_time = "not a time"
        orch._maybe_run_daily_checkin()  # must not raise

        orch._speak.assert_not_called()
