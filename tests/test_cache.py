import time
from unittest.mock import patch

from assistant.tools import cache


def test_get_returns_none_when_nothing_cached(tmp_path):
    with patch.object(cache, "_CACHE_FILE", tmp_path / "cache.json"), patch.object(
        cache, "_CACHE_DIR", tmp_path
    ):
        assert cache.get("youtube_channel", "OG Crew", max_age_seconds=999999) is None


def test_set_then_get_returns_the_value(tmp_path):
    with patch.object(cache, "_CACHE_FILE", tmp_path / "cache.json"), patch.object(
        cache, "_CACHE_DIR", tmp_path
    ):
        cache.set("youtube_channel", "OG Crew", "https://youtube.com/@ogcrew")
        result = cache.get("youtube_channel", "OG Crew", max_age_seconds=999999)
    assert result == "https://youtube.com/@ogcrew"


def test_get_is_case_insensitive_on_key(tmp_path):
    with patch.object(cache, "_CACHE_FILE", tmp_path / "cache.json"), patch.object(
        cache, "_CACHE_DIR", tmp_path
    ):
        cache.set("youtube_channel", "OG Crew", "https://youtube.com/@ogcrew")
        result = cache.get("youtube_channel", "og crew", max_age_seconds=999999)
    assert result == "https://youtube.com/@ogcrew"


def test_get_returns_none_when_entry_expired(tmp_path):
    cache_file = tmp_path / "cache.json"
    with patch.object(cache, "_CACHE_FILE", cache_file), patch.object(cache, "_CACHE_DIR", tmp_path):
        cache.set("youtube_channel", "OG Crew", "https://youtube.com/@ogcrew")
        # backdate the cached_at timestamp to simulate an old entry
        import json

        data = json.loads(cache_file.read_text())
        data["youtube_channel"]["og crew"]["cached_at"] = time.time() - 1000
        cache_file.write_text(json.dumps(data))

        result = cache.get("youtube_channel", "OG Crew", max_age_seconds=500)
    assert result is None


def test_namespaces_are_independent(tmp_path):
    with patch.object(cache, "_CACHE_FILE", tmp_path / "cache.json"), patch.object(
        cache, "_CACHE_DIR", tmp_path
    ):
        cache.set("youtube_channel", "same_key", "channel_value")
        cache.set("other_namespace", "same_key", "other_value")
        assert cache.get("youtube_channel", "same_key", 999999) == "channel_value"
        assert cache.get("other_namespace", "same_key", 999999) == "other_value"


def test_corrupt_cache_file_is_treated_as_empty(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text("not valid json{{{")
    with patch.object(cache, "_CACHE_FILE", cache_file), patch.object(cache, "_CACHE_DIR", tmp_path):
        assert cache.get("youtube_channel", "OG Crew", 999999) is None
        # must still be able to write a fresh entry over a corrupt file
        cache.set("youtube_channel", "OG Crew", "https://youtube.com/@ogcrew")
        assert cache.get("youtube_channel", "OG Crew", 999999) == "https://youtube.com/@ogcrew"
