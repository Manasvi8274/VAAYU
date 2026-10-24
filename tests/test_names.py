import pytest

from assistant.llm import router
from assistant.tools import names


@pytest.mark.parametrize("heard", ["Ogru", "Ogkru", "OGCREO", "OG Creo", "O-G-C-R-E-W", "O G C R E W", "og crew"])
def test_real_mishearings_of_og_crew_are_corrected(heard):
    # Every one of these came out of Whisper in the real session log for "OG Crew".
    assert names.best_match(heard, "channel") == "OG Crew"


@pytest.mark.parametrize("heard, kind, want", [
    ("saved message", "contact", "SAVED MESSAGES"), ("Raman Gargh", "contact", "Raman Garg"),
    ("Tekken seven", "app", "Tekken 7"), ("Dhruv Rathi", "channel", "Dhruv Rathee"),
    ("Carry minati", "channel", "Carryminati"),
])
def test_other_sound_alikes(heard, kind, want):
    assert names.best_match(heard, kind) == want


@pytest.mark.parametrize("heard", ["some new channel", "Arijit Singh", "Channel Limits", "National Geographic"])
def test_unknown_names_are_left_alone(heard):
    assert names.best_match(heard, "channel") is None


def test_correct_arguments_only_touches_name_arguments():
    fixed = names.correct_arguments("play_latest_video_from_channel", {"channel_name": "Ogkru", "new_tab": True})
    assert fixed == {"channel_name": "OG Crew", "new_tab": True}
    assert names.correct_arguments("type_text", {"text": "Ogkru"}) == {"text": "Ogkru"}


def test_remembered_names_are_matched_from_then_on():
    assert names.best_match("Fukra Insan", "channel") is None
    assert names.remember_name("channel", "Fukra Insaan") == "Got it - I'll remember the channel Fukra Insaan."
    assert names.best_match("Fukra Insan", "channel") == "Fukra Insaan"


def test_remember_by_voice_is_routed():
    assert router.fast_route("remember the channel OG Crew") == (
        "tools", [("remember_name", {"kind": "channel", "name": "OG Crew"})])
