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
