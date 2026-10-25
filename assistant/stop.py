"""
"Stop Vaayu" - the polite way: ask Vaayu to shut himself down, like saying
"shutdown Vaayu". He finishes what he's doing, says goodbye and exits
normally, and the guardian then stops too (an intended exit, not a crash).

"Kill Vaayu" (python -m guardian kill) is the emergency one: the guardian
force-kills everything, for when Vaayu isn't listening or this doesn't work.

    pythonw -m assistant.stop      (the "Stop Vaayu" desktop shortcut)
"""

import sys
import time
from pathlib import Path

STOP_FLAG = Path(__file__).resolve().parent.parent / "data" / "stop_request"
_WAIT_SECONDS = 30  # how long to wait for Vaayu to finish a turn and exit


def request_stop(wait: float = _WAIT_SECONDS) -> str:
    from guardian.core import _vaayu_pids  # read-only use of the guardian's process finder
    if not _vaayu_pids():
        STOP_FLAG.unlink(missing_ok=True)
        return "Vaayu isn't running."
    STOP_FLAG.parent.mkdir(parents=True, exist_ok=True)
    STOP_FLAG.write_text(str(time.time()), encoding="utf-8")
    deadline = time.time() + wait
    while time.time() < deadline:
        if not _vaayu_pids():
            return "Vaayu stopped."
        time.sleep(0.5)
    STOP_FLAG.unlink(missing_ok=True)
    return "Vaayu didn't stop by himself - use Kill Vaayu."


if __name__ == "__main__":
    result = request_stop()
    print(result)
    if result.startswith("Vaayu didn't"):  # the shortcut has no window: say it out loud
        try:
            from assistant.tts.pyttsx3_engine import Pyttsx3Engine
            Pyttsx3Engine().speak("Vaayu didn't stop by himself. Use Kill Vaayu.")
        except Exception:
            pass
    sys.exit(0)
