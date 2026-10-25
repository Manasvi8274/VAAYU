import json
from datetime import datetime, timedelta

import pytest

from assistant.llm import router
from assistant.tools import youtube_player, youtube_taste

RECS = [{"id": "a", "title": "Sinking In | Stand-up Comedy by Pranav Sharma", "channel": "Pranav Sharma", "age": "12h ago"},
        {"id": "b", "title": "Arijit Singh Top 10 Songs", "channel": "T-Series", "age": "2y ago"},
        {"id": "c", "title": "OG Crew new vlog", "channel": "OG Crew", "age": "3d ago"}]


@pytest.fixture
def history(tmp_path, monkeypatch):
    path = tmp_path / "h.jsonl"
    monkeypatch.setattr(youtube_taste, "HISTORY_PATH", path)
    return path


@pytest.mark.parametrize("said, want", [
    ("syncing in stand-up comedy by Pranav Sharma", "a"),  # real log, misheard title
    ("sigging in stand up comedy by pranav sharma which is posted 12 hours ago", "a"),  # real log
    ("the one posted 12 hours ago", "a"),
    ("arijit singh songs", "b"),
    ("the og crew one", "c"),
])
def test_said_words_find_the_video(said, want):
    best, sure = youtube_taste.match(said, RECS)
    assert best["id"] == want and sure >= youtube_player._MATCH_SURE


def test_unrelated_words_match_nothing():
    assert youtube_taste.match("something random entirely", RECS)[1] < youtube_player._MATCH_SURE


@pytest.mark.parametrize("said, n", [("second video", 2), ("the third one", 3), ("2", 2), ("last", -1),
                                     ("pranav sharma", None), ("second pranav video", None)])
def test_position(said, n):
    assert youtube_taste.ordinal(said) == n


def test_taste_ranks_the_channel_they_watch_first_and_watched_videos_last(history):
    now = datetime.now()
    rows = [{"ts": (now - timedelta(days=d)).isoformat(), "id": f"x{d}", "title": f"OG Crew vlog {d}", "channel": "OG Crew"}
            for d in range(3)] + [{"ts": now.isoformat(), "id": "b", "title": "Arijit Singh Top 10 Songs", "channel": "T-Series"}]
    history.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ranked = [v["id"] for v in youtube_taste.rank(RECS)]
    assert ranked[0] == "c" and ranked[-1] == "b"  # b already watched


def test_record_skips_a_repeat_of_the_last_video(history):
    assert youtube_taste.record("a", "t", "ch") and not youtube_taste.record("a", "t", "ch")
    assert youtube_taste.record("b", "t2", "ch") and len(youtube_taste.history()) == 2


@pytest.mark.parametrize("text, want", [
    ("Play the video syncing in stand-up comedy by Pranav Sharma which is on the right side of the currently "
     "played videos.", ("play_recommended", "syncing in stand-up comedy by Pranav Sharma")),  # real log
    ("play the second video on the right", ("play_recommended", "second video")),
    ("recommendations me se samay raina wala video chalao", ("play_recommended", "samay raina")),
    ("recommend me something", ("recommend", None)),
    ("what should i watch", ("recommend", None)),
    ("kuch recommend karo", ("recommend", None)),
])
def test_router(text, want):
    action, value = want
    args = {"action": action} | ({"value": value} if value else {})
    assert router.fast_route(text) == ("tools", [("youtube_control", args)])


def test_router_leaves_plain_video_commands_alone():
    assert router.fast_route("play the video") == ("tools", [("control_video_playback", {"action": "play"})])
    assert router.fast_route("play the first one") == ("tools", [("youtube_control", {"action": "play_result", "value": "1"})])


def test_play_the_first_one_after_recommend_uses_the_suggestions(monkeypatch):
    played = []
    monkeypatch.setattr(youtube_player, "_last_list", (__import__("time").time(), RECS))
    monkeypatch.setattr(youtube_player, "_play_video", lambda tab, v: played.append(v["id"]) or "ok")
    youtube_player.play_result(2, tab={"id": "t"})
    assert played == ["b"]
