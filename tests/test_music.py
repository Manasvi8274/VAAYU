import threading
from unittest.mock import MagicMock, patch

import pytest

from assistant.llm import router
from assistant.tools import media, music


@pytest.mark.parametrize("text, query", [
    ("play kesariya", "kesariya"), ("play kesariya song", "kesariya"), ("kesariya gaana chalao", "kesariya"),
    ("arijit singh ke gaane bajao", "arijit singh"), ("play songs of arijit singh", "arijit singh"),
    ("play perfect by ed sheeran", "perfect by ed sheeran"), ("play some music", "some music"),
])
def test_songs_go_to_the_speaker_player(text, query):
    assert router.fast_route(text) == ("tools", [("play_music", {"query": query})])


@pytest.mark.parametrize("text, tool", [
    ("play kesariya on youtube", "play_video_on_youtube"), ("youtube pe kesariya chalao", "play_video_on_youtube"),
    ("play the video", "control_video_playback"), ("play latest video of og crew in youtube", "play_latest_video_from_channel"),
    ("play the first one", "youtube_control"), ("play", "control_media"),
])
def test_youtube_and_player_phrases_are_unchanged(text, tool):
    assert router.fast_route(text)[1][0][0] == tool


def test_play_tekken_is_not_a_song():
    assert router.fast_route("play tekken") is None


def test_spoken_title_drops_the_credits():
    assert music._spoken("Kesariya - Brahmastra | Ranbir Kapoor, Alia Bhatt | Pritam | 4K") == "Kesariya - Brahmastra"
    assert music._spoken("Tum Hi Ho (Full Video) Aashiqui 2") == "Tum Hi Ho"


@pytest.fixture
def playing(monkeypatch):
    p = music._Player()
    p._thread = MagicMock(is_alive=lambda: True)
    p.track = {"id": "a", "title": "Kesariya | 4K", "channel": "Sony Music India"}
    monkeypatch.setattr(music, "player", p)
    return p


def test_controls(playing):
    assert music.control("pause") == "Paused the song." and playing.paused.is_set()
    assert music.control("resume") == "Playing." and not playing.paused.is_set()
    assert music.control("volume_set", 30) == "Music volume is now 30%."
    assert "Kesariya by Sony Music India, playing" in music.control("now_playing")


def test_pause_and_volume_go_to_the_song_not_the_browser_while_it_plays(playing):
    with patch.object(media, "_on_youtube", side_effect=AssertionError("went to YouTube")):
        assert media.control_media("pause") == "Paused the song."
        assert media.control_media("volume_set", 40) == "Music volume is now 40%."


def test_player_output_is_silent_while_paused_and_follows_volume():
    import numpy as np
    p = music._Player()
    p._buf = np.ones((4096, 2), np.float32)
    out = np.zeros((1024, 2), np.float32)
    p.volume, p.duck = 0.5, 1.0
    p._callback(out, 1024, None, None)
    assert np.allclose(out, 0.5)
    p.paused.set()
    p._callback(out, 1024, None, None)
    assert not out.any()
