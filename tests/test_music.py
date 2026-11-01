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


@pytest.mark.parametrize("text, want", [
    ("Lower down the system volume to 50 percent laptop volume not the song volume.",  # real log
     ("control_media", {"action": "volume_set", "level": 50})),
    ("lower the laptop volume", ("control_media", {"action": "volume_down"})),
    ("reduce the volume to 20 percent", ("control_media", {"action": "volume_set", "level": 20})),
    ("song volume 30", ("music_control", {"action": "volume_set", "level": 30})),
    ("gaane ki awaz kam karo", ("music_control", {"action": "volume_down"})),
])
def test_volume_means_the_speakers_unless_the_song_is_named(text, want):
    assert router.fast_route(text) == ("tools", [want])


def test_what_reached_the_speakers_is_logged_as_heard_on_stop(monkeypatch):
    # Recommendations count a song only if at least half of it was heard (user's rule).
    import numpy as np
    from assistant.tools import youtube_taste
    logged = []
    monkeypatch.setattr(youtube_taste, "heard", lambda vid, s, d=0: logged.append((vid, round(s), d)))
    p = music._Player()
    p.track = {"id": "k", "title": "Kesariya", "duration": 60}
    p.start_at = 40.0  # a seek ahead is not listening
    p._buf = np.ones((music._RATE * 3, 2), np.float32)
    out = np.zeros((music._RATE, 2), np.float32)
    for _ in range(3):
        p._callback(out, music._RATE, None, None)
    p.stop()
    assert logged == [("k", 3, 60)]


def test_the_best_matching_song_like_result_is_played_not_just_the_first():
    # Real log: a messy request played a Hindi news video (the first result); "Bollywood song" played a quiz.
    results = [{"title": "नही हुई कई जगह नमाज़े जुमा, एकतरफ़ा कार्रवाई", "channel": "News 24", "duration": 200},
               {"title": "Guess The Bollywood Song By Emoji", "channel": "Puzzle Time", "duration": 300},
               {"title": "Fakira - Student Of The Year 2 | Lyrical Video", "channel": "Zee Music Company", "duration": 230}]
    kept = [r for r in results if not music._NOT_MUSIC.search(r["title"])]
    assert "Guess" not in " ".join(r["title"] for r in kept)
    assert music._rank(kept, "de fakira to hona zaruri hai")[0]["title"].startswith("Fakira")


@pytest.mark.parametrize("said, action", [
    ("What is the name of the song.", "now_playing"),   # real log: 27 s to the model
    ("It's not a song.", "next"),                        # real log: a news video kept playing
    ("No stop the song entirely and then replay it from starting.", "restart"),  # real log: set a 15 s play
    ("play the song from the beginning", "restart"),
    ("play it again", "restart"),
])
def test_song_phrases_that_went_to_the_slow_model(said, action):
    from assistant.llm import router
    assert router.fast_route(said) == ("tools", [("youtube_control", {"action": action})])


@pytest.mark.parametrize("said, want", [
    ("Hey why you play band band bollywood music.", ("play_music", {"query": "band band bollywood"})),
    ("Hold the sound.", ("control_media", {"action": "pause"})),          # real log: said paused, wasn't
    ("Hey vaayu pause that.", ("control_media", {"action": "pause"})),
    ("I'm not able to listen to the song.", ("music_control", {"action": "cant_hear"})),
    ("Update yourself.", ("review_logs", {})),
    ("Have you checked your logs.", ("review_logs", {})),
    ("I want to switch to another song the song name is believer", ("play_music", {"query": "believer"})),
    ("Can you recommend me some song names.", ("music_control", {"action": "recommend"})),
])
def test_phrases_from_the_10_26_to_10_31_logs(said, want):
    from assistant.llm import router
    assert router.fast_route(said) == ("tools", [want])


def test_play_me_a_joke_is_not_a_song():
    from assistant.llm import router
    assert router.fast_route("Play me a joke.") is None


def test_a_cover_is_not_played_when_the_original_was_asked_for():
    results = [{"title": "Gulabi Aankhen Unplugged cover by Atif", "channel": "vishesh", "duration": 200},
               {"title": "Gulabi Aankhen Jo Teri Dekhi | Full Song", "channel": "Saregama Music", "duration": 230}]
    assert music._rank(results, "gulabi aankhen jo teri dekhi")[0]["channel"] == "Saregama Music"
    assert music._rank(results, "gulabi aankhen unplugged cover")[0]["channel"] == "vishesh"


def test_pause_after_his_own_hold_says_paused_and_stays_paused(playing):
    playing.paused.set()
    music._held.set()
    assert music.control("pause") == "Paused the song."
    assert not music._held.is_set() and playing.paused.is_set()
