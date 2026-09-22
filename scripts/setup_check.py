"""
Environment "doctor": run this after installing dependencies to confirm
Ollama is reachable, the configured model is pulled, and (once M3's audio
deps are installed) a microphone is detected.

Usage: python scripts/setup_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.config_schema import load_config  # noqa: E402


def check_ollama(host: str, model: str) -> bool:
    try:
        import ollama
    except ImportError:
        print("[FAIL] 'ollama' package not installed. Run: pip install -r requirements.txt")
        return False

    try:
        client = ollama.Client(host=host)
        models = client.list().get("models", [])
        names = [m.get("model") or m.get("name") for m in models]
    except Exception as exc:
        print(f"[FAIL] Could not reach Ollama at {host}. Is it running? ({exc})")
        return False

    if not any(model in (n or "") for n in names):
        print(f"[FAIL] Model '{model}' not found. Run: ollama pull {model}")
        print(f"       Models currently available: {names}")
        return False

    print(f"[OK] Ollama reachable at {host}, model '{model}' is pulled.")
    return True


def check_microphone() -> bool:
    try:
        import sounddevice as sd
    except ImportError:
        print("[SKIP] 'sounddevice' not installed yet (needed from M3 onward).")
        return True

    devices = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
    if not devices:
        print("[FAIL] No input (microphone) devices detected.")
        return False

    print(f"[OK] Found {len(devices)} input device(s), default: {sd.query_devices(kind='input')['name']}")
    return True


def check_browser_debugging() -> bool:
    """Most of Vaayu's browser/tab/YouTube/WhatsApp tools need Chrome or
    Brave running with --remote-debugging-port=9222 (and --remote-allow-
    origins=* for anything that reads/clicks the page, not just tabs) - this
    is off by default and a very easy thing to forget, so check it here
    rather than letting it surface as a confusing tool error later."""
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:9222/json", timeout=2) as resp:
            tabs = json.loads(resp.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        print(
            "[SKIP] Browser remote debugging not detected on port 9222 - tab control, YouTube, "
            "WhatsApp, and page-reading tools won't work until the browser is (re)started with "
            "--remote-debugging-port=9222 --remote-allow-origins=* (see README)."
        )
        return True  # not a hard failure - plenty of Vaayu still works without this

    print(f"[OK] Browser remote debugging reachable on port 9222 ({len(tabs)} tab(s) open).")
    return True


def main() -> None:
    config = load_config()
    ok = check_ollama(config.llm.host, config.llm.model)
    ok = check_microphone() and ok
    ok = check_browser_debugging() and ok

    if ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed - fix the items above before running the assistant.")
        sys.exit(1)


if __name__ == "__main__":
    main()
