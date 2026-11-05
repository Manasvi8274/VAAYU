"""
English -> Hindi for Vaayu's replies when the user spoke Hindi (2026-11-05).

The user: "why is the Hindi version taking too long - about 30 s after 'ek sec bhai'". Measured: the 4B model that
wrote Hindi cost 13.7 s to swap into the 4 GB GPU, 7.3 s to read the prompt and 6.9 s to write. Now the fast 3B brain
answers (cached, 1-2 s) and Meta's NLLB-200 (distilled 600M, int8, CTranslate2 - models/nllb, ~0.8 s on the CPU)
translates it; the native Hindi voice speaks it.

Sentence by sentence, in one batch: given several at once NLLB dropped some ("Bhai, that happens to everyone." vanished).
"""

import re
import threading
from pathlib import Path

MODEL = Path(__file__).resolve().parent.parent.parent / "models" / "nllb"
_SENTENCES = re.compile(r"(?<=[.!?।])\s+")
_lock = threading.Lock()
_loaded: dict = {}


def _model():
    with _lock:
        if not _loaded:
            import ctranslate2
            import sentencepiece as spm
            _loaded["tr"] = ctranslate2.Translator(str(MODEL), device="cpu", compute_type="int8", intra_threads=4)
            _loaded["sp"] = spm.SentencePieceProcessor(model_file=str(MODEL / "sentencepiece.bpe.model"))
        return _loaded["tr"], _loaded["sp"]


def warm_up() -> None:
    """Load it at startup (1.5 s), not on the first Hindi reply."""
    try:
        _model()
    except Exception:
        pass


def to_hindi(text: str) -> str:
    """The reply in Hindi (Devanagari); the English as it was if the translator isn't there."""
    return _translate(text, "eng_Latn", "hin_Deva")


def to_english(text: str) -> str:
    """Hindi -> English: said in English when the Hindi voice can't load (kokoro_engine). As it was on failure."""
    return _translate(text, "hin_Deva", "eng_Latn")


def _translate(text: str, source: str, target: str) -> str:
    sentences = [s for s in _SENTENCES.split((text or "").strip()) if s]
    if not sentences:
        return text
    try:
        tr, sp = _model()
    except Exception:
        return text
    batch = [sp.encode(s, out_type=str) + ["</s>", source] for s in sentences]
    out = tr.translate_batch(batch, target_prefix=[[target]] * len(batch), beam_size=2, max_decoding_length=200)
    return " ".join(sp.decode(o.hypotheses[0][1:]) for o in out)
