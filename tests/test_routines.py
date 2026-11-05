"""Routines: one phrase, several things (assistant/routines.py)."""
import pytest

from assistant import routines
from assistant.llm import router


@pytest.fixture(autouse=True)
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(routines, "PATH", tmp_path / "routines.json")
    monkeypatch.setattr(routines, "_DEFAULTS", dict(routines._DEFAULTS))


def test_made_by_voice_then_run_by_its_phrase():
    said = router.fast_route("when i say work mode, open vs code and open chrome and set volume to 20")
    assert said[0] == "reply" and "'work mode'" in said[1]
    assert router.fast_route("Work mode.") == ("tools", [("open_app", {"app_name": "vs code"}),
                                                         ("open_app", {"app_name": "chrome"}),
                                                         ("control_media", {"action": "volume_set", "level": 20})])


def test_without_a_comma_the_phrase_ends_where_the_commands_start():
    router.fast_route("create a routine called movie time pause the music and set volume to 80")
    assert router.fast_route("movie time")[1][0] == ("control_media", {"action": "pause"})


def test_steps_it_doesnt_know_are_not_saved():
    assert "couldn't follow" in router.fast_route("when i say blah do something weird")[1]
    assert router.fast_route("blah") is None


def test_listed_deleted_and_the_built_in_one():
    assert router.fast_route("I am leaving") == ("tools", [("control_media", {"action": "pause"}), ("lock_computer", {})])
    router.fast_route("when i say focus, open notepad")
    assert "'focus'" in router.fast_route("what are my routines")[1]
    assert router.fast_route("delete the focus routine") == ("reply", "Deleted the 'focus' routine.")
    assert router.fast_route("focus") != ("tools", [("open_app", {"app_name": "notepad"})])
