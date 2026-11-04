"""Hindi and English both (user, 2026-11-04)."""
import pytest

from assistant.control_phrases import is_affirmative
from assistant.llm import router
from assistant.llm.ollama_client import is_hindi
from assistant.tts import kokoro_engine


@pytest.mark.parametrize("text, yes", [("हाँ भेजो", True), ("जी हाँ", True), ("नहीं मत भेजो", False),
                                       ("haan bhejo", True), ("nahi", False)])
def test_hindi_yes_and_no(text, yes):
    assert is_affirmative(text) is yes


@pytest.mark.parametrize("text, hindi", [
    # what the speech model wrote for Kokoro's Hindi voice - Roman Hindi
    ("Namaste bhai me vaayu hoon aaj mausam bohot achha hai aur me apka gana chala raha hoon.", True),
    ("aaj mera mood kharab hai kya karun", True), ("tum kaise ho bhai", True), ("मौसम कैसा है", True),
    ("Kya haal chao mere baayi.", True), ("Like terina hindi samaj niya ti.", True),  # real log 11-04, missed
    ("Vaayu play another song likhe jo khat tujhe bollywood song indian song.", False),
    ("Play a bollywood song song name is meher baan hua.", False),  # English with a Hindi song name
    ("Pause that song and can you recommend me some song names.", False),
    ("How can I find a new job I have been working as s-t-e-t so.", False),
])
def test_hindi_is_told_apart_from_english_and_hinglish_commands(text, hindi):
    assert is_hindi(text) is hindi


def test_hindi_commands_in_hindi_script_take_the_fast_path():
    assert router.fast_route("गाना रोको") == ("tools", [("control_media", {"action": "pause"})])
    assert router.fast_route("आवाज़ कम करो") == ("tools", [("control_media", {"action": "volume_down"})])


def test_hindi_sentences_get_the_hindi_voice():
    assert kokoro_engine._DEVANAGARI.search("नमस्ते भाई।") and not kokoro_engine._DEVANAGARI.search("Hello bhai.")
    said = "आज मौसम बहुत अच्छा है। मैं आपका गाना चला रहा हूँ।"  # "।" ends a Hindi sentence
    assert kokoro_engine._sentences(said) == ["आज मौसम बहुत अच्छा है।", "मैं आपका गाना चला रहा हूँ।"]


def test_hindi_voice_can_be_switched_by_voice(monkeypatch, tmp_path):
    from assistant.tools import voice
    monkeypatch.setattr(kokoro_engine, "HINDI_VOICE_FILE", tmp_path / "voice_hi.txt")
    engine = kokoro_engine.KokoroEngine.__new__(kokoro_engine.KokoroEngine)
    engine._cache = {"नमस्ते भाई।": 1, "Hello.": 2}
    monkeypatch.setattr(kokoro_engine, "ACTIVE", engine)
    assert router.fast_route("hindi voice rohan") == ("tools", [("set_voice", {"voice": "rohan"})])
    assert voice.set_voice("rohan") == "Okay, my Hindi voice is now Rohan."
    assert kokoro_engine._saved_hindi_voice() == "rohan" and engine._cache == {"Hello.": 2}  # old Hindi audio dropped
