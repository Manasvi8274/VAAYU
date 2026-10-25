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


def test_pause_goes_to_the_song_but_volume_stays_the_speakers_while_it_plays(playing):
    # User: "here volume is the speaker volume not the song volume" - "song volume" is music_control.
    with patch.object(media, "_on_youtube", side_effect=AssertionError("went to YouTube")),             patch.object(media, "_system_volume", side_effect=lambda a, l: f"Laptop volume is now {l}%."):
        assert media.control_media("pause") == "Paused the song."
        assert media.control_media("volume_set", 40) == "Laptop volume is now 40%."
    assert playing.volume == 0.8  # the song's own level untouched


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


@pytest.mark.parametrize("text, want", [
    ("pause the audio", ("control_media", {"action": "pause"})),
    ("post the audio which you are playing", ("control_media", {"action": "pause"})),  # real log: Whisper's "pause"
    ("pause the song stop the song", ("control_media", {"action": "pause"})),  # real log: went to the LLM, 20 s
    ("start the music", ("control_media", {"action": "play"})),  # real log: LLM claimed "Resuming..." with no tool
    ("lower the reduce the audio volume to 10 percent", ("control_media", {"action": "volume_set", "level": 10})),
    ("play kesariya for 15 seconds", ("play_music", {"query": "kesariya", "seconds": 15})),
    ("play it for 10 seconds", ("music_control", {"action": "play_for", "level": 10})),
    ("which movie is this song from", ("ask_about_song", {"question": "which movie is this song from"})),
    ("ye gaana kis movie ka hai", ("ask_about_song", {"question": "ye gaana kis movie ka hai"})),
    ("It wadya sa gaana chala do ya bollywood.", ("play_music", {"query": "bollywood songs"})),  # real log
    ("switch your voice to a girl voice female voice", ("set_voice", {"voice": "heart"})),  # real log
    ("change your own voice", ("set_voice", {"voice": "next"})),  # real log
])
def test_todays_log_phrasings(text, want):
    assert router.fast_route(text) == ("tools", [want])


@pytest.mark.parametrize("said, query", [
    ("bang bang which is from the bollywood movie starring rithik roshan", "bang bang the hrithik roshan"),  # was a full movie
    ("a new song by Eminem", "Eminem latest song"),  # was Kid Rock
    ("the song named bang bang", "bang bang"),
    ("koi romantic", "romantic"),
    ("ek accha sa", ""),  # nothing left: their taste
    ("kesariya", "kesariya"),
])
def test_spoken_filler_is_dropped_from_searches(said, query):
    assert music._clean_query(said) == query


def test_titles_spoken_without_upload_words():
    assert music._spoken("Bang Bang Title Track Full Video | Hrithik") == "Bang Bang"


def test_seek_and_timed_play(playing, monkeypatch):
    seeks = []
    monkeypatch.setattr(playing, "seek", lambda s: seeks.append(s) or setattr(playing, "start_at", s))
    playing.track["duration"] = 247
    assert music.control("seek_to", 70) == "Playing from 1:10."
    music.control("seek_forward", 30)
    assert seeks == [70, 100]
    with pytest.raises(RuntimeError, match="only 4:07 long"):
        music.control("seek_to", 300)
    music.control("play_for", 15)
    assert playing.stop_at == pytest.approx(playing.position + 15)


def test_timed_play_pauses_at_the_mark():
    import numpy as np
    p = music._Player()
    p._buf = np.ones((48000 * 3, 2), np.float32)
    p.stop_at = 1.0
    out = np.zeros((48000, 2), np.float32)
    p._callback(out, 48000, None, None)
    assert p.paused.is_set() and p.stop_at is None


def test_video_commands_go_to_the_speaker_song_when_no_video_is_open(playing, monkeypatch):
    from assistant.tools import video_player, youtube_player
    monkeypatch.setattr(youtube_player, "youtube_tab", lambda require_watch=False: None)
    assert video_player.control_video_playback("pause") == "Paused the song."  # real log: "can't control tabs"
    monkeypatch.setattr(playing, "seek", lambda s: setattr(playing, "start_at", s))
    assert youtube_player.youtube_control("seek_to", "70") == "Playing from 1:10."


def test_this_song_question_uses_the_playing_title(playing, monkeypatch):
    asked = []
    monkeypatch.setattr("assistant.tools.web_answer.answer_question", lambda q: asked.append(q) or {"results": []})
    music.ask_about_song("which movie is this song from")
    assert asked == ['which movie is the song "Kesariya" from']


@pytest.mark.parametrize("text, tool", [
    ("play the song tu meri", "play_music"), ("play tu meri", "play_music"), ("tu meri gaana chalao", "play_music"),
    ("play kesariya song on youtube", "play_music"),  # said "song": the speakers, even with "youtube"
    ("play kesariya video", "play_video_on_youtube"), ("play the video of kesariya", "play_video_on_youtube"),
    ("kesariya ka video chalao", "play_video_on_youtube"), ("play kesariya on youtube", "play_video_on_youtube"),
])
def test_song_means_speakers_video_means_brave(text, tool):
    assert router.fast_route(text)[1][0][0] == tool


def test_the_model_choosing_youtube_for_a_song_is_rerouted_to_the_speakers(monkeypatch):
    # Real log: "...hollywood song starred by rintik roshan" -> the LLM chose play_video_on_youtube -> failed.
    from unittest.mock import MagicMock
    from assistant.llm.ollama_client import OllamaBrain
    from assistant.tools.registry import registry
    played = []
    monkeypatch.setattr(music, "play_music", lambda query, seconds=None: played.append(query) or "Playing it.")
    monkeypatch.setitem(registry._tools["play_music"].__dict__, "func", music.play_music)
    with patch("assistant.llm.ollama_client.ollama.Client"):
        brain = OllamaBrain(host="http://fake", model="m")
    brain.client = MagicMock()
    brain.client.chat.return_value = {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "play_video_on_youtube", "arguments": {"query": "bank bike movie song rintik roshan"}}}]}}
    reply, _ = brain.respond([{"role": "user", "content": "From bank bike movie hollywood song starred by rintik roshan."}],
                             registry, confirm=lambda n, a: True)
    assert played == ["bank bike movie song rintik roshan"]
    assert any("rerouted" in t for t in brain.trace)


def test_song_pauses_while_vaayu_talks_and_resumes_after_the_conversation(playing, monkeypatch):
    import time
    monkeypatch.setattr(music, "_CONVERSATION_GAP", 0.3)
    music.hold_for_speech()  # Vaayu starts a reply
    assert playing.paused.is_set()
    music.release_after_conversation()  # reply done...
    music.user_is_speaking()  # ...but the user says something else: still paused
    time.sleep(0.6)
    assert playing.paused.is_set()
    music.hold_for_speech()
    music.release_after_conversation()  # this time the conversation goes quiet
    time.sleep(0.8)
    assert not playing.paused.is_set()


def test_a_song_the_user_paused_stays_paused_after_vaayu_talks(playing, monkeypatch):
    import time
    monkeypatch.setattr(music, "_CONVERSATION_GAP", 0.2)
    music.control("pause")  # "pause the song"
    music.hold_for_speech()  # "Paused the song."
    music.release_after_conversation()
    time.sleep(0.6)
    assert playing.paused.is_set()
