import pytest

from assistant.stt.whisper_stt import _VOCAB_PROMPT


def test_vocab_prompt_fits_in_whispers_prompt_window():
    # Whisper keeps only the last ~223 prompt tokens; the old 258-token prompt
    # silently lost its opening line (the one teaching the name "Vaayu").
    faster_whisper = pytest.importorskip("faster_whisper")
    from tokenizers import Tokenizer
    try:
        tok = Tokenizer.from_pretrained("openai/whisper-base")
    except Exception:
        pytest.skip("whisper tokenizer not available offline")
    assert len(tok.encode(" " + _VOCAB_PROMPT).ids) <= 200
    assert faster_whisper  # imported for the skip above


def test_name_is_at_the_end_where_whisper_never_truncates():
    assert _VOCAB_PROMPT.rstrip().endswith("Vaayu.")


def test_one_word_commands_are_kept_but_noise_words_are_not(monkeypatch):
    # The 2-word minimum silently dropped "pause", "mute", "next" spoken alone.
    from types import SimpleNamespace
    from assistant.stt import whisper_stt
    stt = whisper_stt.WhisperSTT.__new__(whisper_stt.WhisperSTT)
    stt.language = "en"
    for heard, kept in [("Pause.", True), ("mute", True), ("Next!", True), ("you", False), ("Thank you.", True),
                        ("Bye.", False)]:
        stt.model = SimpleNamespace(transcribe=lambda audio, **kw: ([SimpleNamespace(text=heard)], None))
        assert (stt.transcribe(None) != "") == kept, heard
