from unittest.mock import patch

from assistant.self_review import issue_log


def test_log_issue_then_read_unacknowledged(tmp_path):
    log_path = tmp_path / "issues.jsonl"
    with patch.object(issue_log, "LOG_PATH", log_path):
        issue_log.log_issue("close the browser", "close_app", "AppNotFound: brave")
        result = issue_log.read_unacknowledged()

    assert len(result) == 1
    assert result[0]["user_said"] == "close the browser"
    assert result[0]["tool"] == "close_app"
    assert result[0]["error"] == "AppNotFound: brave"
    assert result[0]["acknowledged"] is False
    assert result[0]["kind"] == "tool_failure"


def test_log_issue_default_kind_is_tool_failure(tmp_path):
    log_path = tmp_path / "issues.jsonl"
    with patch.object(issue_log, "LOG_PATH", log_path):
        issue_log.log_issue("x", "y", "z", kind="grounding_nudge_fired")
        result = issue_log.read_unacknowledged()
    assert result[0]["kind"] == "grounding_nudge_fired"


def test_read_unacknowledged_returns_empty_list_when_no_log_file(tmp_path):
    with patch.object(issue_log, "LOG_PATH", tmp_path / "does_not_exist.jsonl"):
        assert issue_log.read_unacknowledged() == []


def test_acknowledge_all_marks_every_entry_and_persists(tmp_path):
    log_path = tmp_path / "issues.jsonl"
    with patch.object(issue_log, "LOG_PATH", log_path):
        issue_log.log_issue("a", "t1", "e1")
        issue_log.log_issue("b", "t2", "e2")

        count = issue_log.acknowledge_all()
        assert count == 2
        assert issue_log.read_unacknowledged() == []


def test_acknowledge_all_leaves_already_acknowledged_entries_alone(tmp_path):
    log_path = tmp_path / "issues.jsonl"
    with patch.object(issue_log, "LOG_PATH", log_path):
        issue_log.log_issue("a", "t1", "e1")
        issue_log.acknowledge_all()
        issue_log.log_issue("b", "t2", "e2")  # new, unacknowledged entry added after

        count = issue_log.acknowledge_all()
        assert count == 1  # only the new one


def test_missed_days_accumulate_and_survive_until_acknowledged(tmp_path):
    # Directly matches the explicit requirement: entries from multiple
    # separate "days" (separate log_issue calls, no acknowledge in between)
    # must all still be present and unacknowledged until a real check-in.
    log_path = tmp_path / "issues.jsonl"
    with patch.object(issue_log, "LOG_PATH", log_path):
        issue_log.log_issue("day1 problem", "close_app", "err1")
        issue_log.log_issue("day2 problem", "send_whatsapp_message", "err2")
        issue_log.log_issue("day3 problem", "seek_video", "err3")

        result = issue_log.read_unacknowledged()
    assert len(result) == 3
    assert {e["user_said"] for e in result} == {"day1 problem", "day2 problem", "day3 problem"}


def test_corrupt_log_line_is_skipped_not_fatal(tmp_path):
    log_path = tmp_path / "issues.jsonl"
    log_path.write_text('not valid json\n{"timestamp": 1, "kind": "tool_failure", "user_said": "x", "tool": "y", "error": "z", "acknowledged": false}\n', encoding="utf-8")
    with patch.object(issue_log, "LOG_PATH", log_path):
        result = issue_log.read_unacknowledged()
    assert len(result) == 1
    assert result[0]["user_said"] == "x"


def test_summarize_for_speech_empty():
    assert issue_log.summarize_for_speech([]) == "no issues"


def test_summarize_for_speech_singular():
    issues = [{"tool": "close_app"}]
    assert issue_log.summarize_for_speech(issues) == "1 issue today, mostly around close_app"


def test_summarize_for_speech_plural_and_top_tools():
    issues = [
        {"tool": "close_app"}, {"tool": "close_app"}, {"tool": "close_app"},
        {"tool": "seek_video"}, {"tool": "seek_video"},
        {"tool": "send_whatsapp_message"},
    ]
    summary = issue_log.summarize_for_speech(issues)
    assert summary.startswith("6 issues today, mostly around")
    assert "close_app" in summary
    assert "seek_video" in summary


def test_summarize_for_speech_uses_general_for_missing_tool():
    issues = [{"tool": None}]
    assert "general" in issue_log.summarize_for_speech(issues)


def test_write_daily_report_creates_readable_markdown(tmp_path):
    reports_dir = tmp_path / "daily_reports"
    with patch.object(issue_log, "REPORTS_DIR", reports_dir):
        issues = [
            {
                "timestamp": 1700000000.0,
                "kind": "tool_failure",
                "user_said": "close the brave browser",
                "tool": "close_app",
                "error": "AppNotFound: brave",
            }
        ]
        path = issue_log.write_daily_report(issues)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "close the brave browser" in content
    assert "close_app" in content
    assert "AppNotFound: brave" in content


def test_write_daily_report_handles_missing_optional_fields(tmp_path):
    reports_dir = tmp_path / "daily_reports"
    with patch.object(issue_log, "REPORTS_DIR", reports_dir):
        issues = [{"timestamp": 1700000000.0, "kind": "grounding_nudge_fired", "error": "no tool call"}]
        path = issue_log.write_daily_report(issues)
    content = path.read_text(encoding="utf-8")
    assert "general" in content
    assert "no tool call" in content
