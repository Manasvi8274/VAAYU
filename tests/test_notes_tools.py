from unittest.mock import patch

from assistant.storage import db
from assistant.tools import notes


def test_add_note_then_list_notes_round_trip(tmp_path):
    with patch.object(db, "_DB_PATH", tmp_path / "assistant.db"):
        result = notes.add_note("buy milk")
        assert result == "Note saved."

        listed = notes.list_notes()
        assert len(listed) == 1
        assert listed[0]["text"] == "buy milk"
        assert listed[0]["created_at"]  # timestamp populated


def test_list_notes_returns_most_recent_first(tmp_path):
    with patch.object(db, "_DB_PATH", tmp_path / "assistant.db"):
        notes.add_note("first")
        notes.add_note("second")
        notes.add_note("third")

        listed = notes.list_notes()
        assert [n["text"] for n in listed] == ["third", "second", "first"]


def test_list_notes_respects_limit(tmp_path):
    with patch.object(db, "_DB_PATH", tmp_path / "assistant.db"):
        for i in range(10):
            notes.add_note(f"note {i}")

        listed = notes.list_notes(limit=3)
        assert len(listed) == 3


def test_list_notes_empty_db_returns_empty_list(tmp_path):
    with patch.object(db, "_DB_PATH", tmp_path / "assistant.db"):
        assert notes.list_notes() == []
