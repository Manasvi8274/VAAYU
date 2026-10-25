"""
Vaayu's guardian: a small supervisor process that is NOT part of Vaayu's own
code, so Vaayu (including its future self-upgrades) can't edit or disable it.

    python -m guardian          start Vaayu under the guardian
    python -m guardian kill     stop Vaayu and the guardian completely
    python -m guardian status   is it running?

Kill paths - all handled here, none depend on Vaayu's code still working:
"Vaayu die" (the guardian's own mic listener), Ctrl+Alt+Shift+F12, and the
"Kill Vaayu" desktop shortcut. "Sleep Vaayu" is Vaayu's own sleep, not this.
"""

import os as _os
from pathlib import Path as _Path

# Everything Vaayu downloads stays in this folder (user's rule, 2026-10-25): the model caches (Hugging Face -
# Whisper, Qwen...), Ollama's models, pip's and yt-dlp's caches. Set here because the guardian starts Vaayu
# (children inherit it) and every Vaayu module imports this package first (assistant/__init__.py).
_REPO = _Path(__file__).resolve().parent.parent
STORAGE = {"HF_HOME": "models/hf", "TORCH_HOME": "models/torch", "OLLAMA_MODELS": "models/ollama",
           "XDG_CACHE_HOME": ".cache", "PIP_CACHE_DIR": ".cache/pip"}
for _name, _rel in STORAGE.items():
    _os.environ.setdefault(_name, str(_REPO / _rel))
