import json
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from assistant import daily_review
from assistant.llm import router


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """A day of logs in tmp (never the real data/), the paths pointed at them."""
    for name, rel in (("INTERACTIONS", "interactions.jsonl"), ("ISSUES", "issues_log.jsonl"),
                      ("VAAYU_LOG", "vaayu.log"), ("AUDIO_DIR", "audio"), ("REVIEWS", "reviews")):
        monkeypatch.setattr(daily_review, name, tmp_path / rel)
    turns = [
        {"kind": "turn", "text": "open tekken", "reply": "That didn't work - TEKKEN 7 is not running", "brain_ms": 9500,
         "trace": [{"path": "llm", "tool": "open_app", "args": {"app_name": "TEKKEN 7"}, "ok": False}]},
        {"kind": "turn", "text": "open tekken game", "reply": "That didn't work - TEKKEN 7 is not running", "brain_ms": 900},
        {"kind": "turn", "text": "what is the date", "reply": "Today is Monday.", "brain_ms": 300,
         "trace": [{"path": "chat"}]},
        {"kind": "turn", "text": "play og crew", "reply": "Playing.", "brain_ms": 100,
         "trace": [{"path": "fast", "name_corrected": {"from": {"channel_name": "ogru"}, "to": {"channel_name": "OG Crew"}}}]},
        {"kind": "ignored_media", "text": "I smell something burning"},
        {"kind": "barge_in", "heard": "stop"},
        {"kind": "error", "where": "brain.respond"},
    ]
    (tmp_path / "interactions.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns), encoding="utf-8")
    (tmp_path / "issues_log.jsonl").write_text(json.dumps({"user_said": "open tekken", "tool_name": "open_app",
                                                           "error": "AppNotFound"}) + "\n", encoding="utf-8")
    (tmp_path / "vaayu.log").write_text("Traceback (most recent call last):\nValueError: boom\n", encoding="utf-8")
    (tmp_path / "audio").mkdir()
    for i in range(daily_review.AUDIO_KEEP + 5):
        (tmp_path / "audio" / f"20261025_{i:06d}.wav").write_bytes(b"x")
    return tmp_path


def test_digest_counts_what_happened(logs):
    events = daily_review._read_jsonl(logs / "interactions.jsonl")
    report, stats = daily_review.digest(events, daily_review._read_jsonl(logs / "issues_log.jsonl"),
                                        (logs / "vaayu.log").read_text())
    assert stats["turns"] == 4 and stats["failed"] == 2 and stats["slow"] == 1
    assert stats["ignored_media"] == 1 and stats["barge_in"] == 1 and stats["name_corrected"] == 1
    assert "2x \"That didn't work - TEKKEN 7 is not running\"" in report
    assert "ogru -> OG Crew" in report and "ValueError: boom" in report
    assert len(report.encode()) < 50_000


def test_review_writes_the_digest_then_deletes_the_logs(logs):
    seen = {}

    def fake_claude(review_dir, day):  # the logs must still be there (raw copy) while Claude reads them
        seen["raw"] = sorted(p.name for p in (review_dir / "raw").iterdir())
        seen["digest"] = (review_dir / "digest.md").exists()
        return "Claude found nothing to change."
    with patch.object(daily_review, "claude_session", fake_claude):
        where = daily_review.run_review("test", use_claude=True)
    assert seen == {"raw": ["interactions.jsonl", "issues_log.jsonl", "vaayu.log"], "digest": True}
    assert not (logs / "interactions.jsonl").exists() and not (logs / "issues_log.jsonl").exists()
    assert not (where / "raw").exists()
    assert len(list((logs / "audio").glob("*.wav"))) == daily_review.AUDIO_KEEP  # newest kept for voice training
    digest = (where / "digest.md").read_text(encoding="utf-8")
    assert "Claude found nothing to change." in digest and "Deleted: interactions.jsonl" in digest


def test_logs_are_kept_if_the_digest_cannot_be_written(logs):
    with patch.object(daily_review, "digest", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError):
            daily_review.run_review("test", use_claude=False)
    assert (logs / "interactions.jsonl").exists()  # nothing deleted without a digest


def test_a_second_review_does_not_start_while_one_runs(logs):
    release = threading.Event()
    with patch.object(daily_review, "claude_session", lambda d, day: release.wait(5) and "done"):
        t = threading.Thread(target=daily_review.run_review, args=("first",), kwargs={"use_claude": True})
        t.start()
        try:
            for _ in range(100):
                if daily_review._lock.locked():
                    break
                threading.Event().wait(0.02)
            assert daily_review.run_review("second") is None
            assert daily_review.start_in_background("third") is False
        finally:
            release.set()
            t.join(5)


def test_guardian_changes_are_rejected():
    assert daily_review.protected_changes(["assistant/llm/router.py", "guardian/core.py", "guardian\\rules.py"]) \
        == ["guardian/core.py", "guardian\\rules.py"]
    assert daily_review.protected_changes(["assistant/x.py", "tests/test_x.py"]) == []


def test_claude_session_discards_a_branch_that_touches_the_guardian(tmp_path, monkeypatch):
    calls = []

    def fake_git(*args, cwd=None, timeout=120):
        calls.append(args)
        out = {"rev-list": "2\n", "diff": "assistant/a.py\nguardian/core.py\n", "branch": "vaayu-v2\n"}.get(args[0], "")
        return subprocess.CompletedProcess(args, 1 if args[:2] == ("rev-parse", "--verify") else 0, out, "")
    monkeypatch.setattr(daily_review, "_git", fake_git)
    monkeypatch.setattr(daily_review.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(daily_review, "ROOT", tmp_path / "repo")
    with patch.object(daily_review.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ok", "")) as run:
        result = daily_review._real_claude_session(tmp_path / "review", "2026-10-25")
    assert "touched the guardian" in result and "guardian/core.py" in result
    assert ("branch", "-D", "vaayu-review-20261025") in calls
    assert not any(a[0] == "merge" for a in calls)
    claude_cmd = run.call_args_list[0].args[0]
    assert claude_cmd[:2] == ["claude", "-p"] and "Edit(guardian/**)" in claude_cmd


def test_claude_session_fast_forwards_only_when_the_tests_pass(tmp_path, monkeypatch):
    calls = []

    def fake_git(*args, cwd=None, timeout=120):
        calls.append(args)
        out = {"rev-list": "1\n", "diff": "assistant/a.py\ntests/test_a.py\n", "branch": "vaayu-v2\n"}.get(args[0], "")
        return subprocess.CompletedProcess(args, 1 if args[:2] == ("rev-parse", "--verify") else 0, out, "")
    monkeypatch.setattr(daily_review, "_git", fake_git)
    monkeypatch.setattr(daily_review.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(daily_review, "ROOT", tmp_path / "repo")
    for test_code, merged in ((1, False), (0, True)):
        calls.clear()
        results = [subprocess.CompletedProcess([], 0, "ok", ""), subprocess.CompletedProcess([], test_code, "5 passed", "")]
        with patch.object(daily_review.subprocess, "run", side_effect=results):
            daily_review._real_claude_session(tmp_path / "review", "2026-10-25")
        assert any(a[0] == "merge" for a in calls) is merged


def test_names_used_often_are_learned(logs, monkeypatch):
    from collections import Counter
    from assistant.tools import names
    remembered = []
    monkeypatch.setattr(names, "remember_name", lambda kind, name: remembered.append((kind, name)))
    daily_review._real_learn_locally([], logs, Counter({("channel", "Pranav Sharma", True): 3,
                                                        ("channel", "Once Only", True): 1,
                                                        ("app", "Broken App", False): 5}))
    assert remembered == [("channel", "Pranav Sharma")]


@pytest.mark.parametrize("text", ["review your logs", "Vaayu, scan your logs.", "logs check karo", "check the logs",
                                  "run the daily review"])
def test_voice_trigger(text):
    assert router.fast_route(text) == ("tools", [("review_logs", {})])


def test_voice_tool_starts_it_in_the_background(monkeypatch):
    from assistant.tools.self_review_control import review_logs
    monkeypatch.setattr(daily_review, "start_in_background", lambda trigger: True)
    assert review_logs().startswith("Started the log review")
    monkeypatch.setattr(daily_review, "start_in_background", lambda trigger: False)
    assert "already running" in review_logs()


def test_usage_limit_means_no_session_and_the_logs_are_kept(logs, tmp_path, monkeypatch):
    # Found live: "You've hit your session limit - resets 11am" (exit 1, no commits).
    def fake_git(*args, cwd=None, timeout=120):
        return subprocess.CompletedProcess(args, 1 if args[:2] == ("rev-parse", "--verify") else 0,
                                           "0\n" if args[0] == "rev-list" else "", "")
    monkeypatch.setattr(daily_review, "_git", fake_git)
    monkeypatch.setattr(daily_review.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(daily_review, "ROOT", tmp_path / "repo")
    limit = subprocess.CompletedProcess([], 1, "You've hit your session limit - resets 11am (Asia/Kolkata)", "")
    with patch.object(daily_review.subprocess, "run", return_value=limit):
        with pytest.raises(daily_review.ClaudeNotRun, match="usage limit"):
            daily_review._real_claude_session(tmp_path / "review", "2026-10-25")
    with patch.object(daily_review, "claude_session", side_effect=daily_review.ClaudeNotRun("usage limit")):
        where = daily_review.run_review("test", use_claude=True)
    assert (logs / "interactions.jsonl").exists() and (logs / "vaayu.log").exists()  # kept for the next review
    assert not (where / "raw").exists()
    assert "kept the logs for the next review" in (where / "digest.md").read_text(encoding="utf-8")


def test_digest_reports_the_latency_budget_for_simple_commands():
    from assistant import daily_review
    turns = [{"kind": "turn", "text": t, "reply": "ok", "stt_ms": stt, "brain_ms": brain, "trace": [{"path": p}]}
             for t, stt, brain, p in [("open notepad", 600, 40, "fast"), ("what time is it", 700, 5, "fast_reply"),
                                      ("close chrome", 2500, 30, "fast"), ("tell me a story", 900, 9000, "chat")]]
    report, stats = daily_review.digest(turns, [], "")
    assert (stats["simple_within_budget"], stats["simple"]) == (2, 3)  # the chat turn isn't a simple command
    assert "Latency budget" in report and "2/3" in report
