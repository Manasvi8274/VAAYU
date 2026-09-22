"""
Two independent problems, both handled here:

1. No real console attached (sys.stdout is None) - exactly the situation
   when launched via pythonw.exe with no parent console (auto-start does
   this, see PLAN.md). Without handling this, the very first print() call
   would crash with AttributeError on a None stdout, silently killing the
   assistant on every boot with no diagnostic. Fixed by redirecting to a log
   file, same as before.

2. A real console IS attached, but its encoding can't represent a character
   Vaayu actually needs to print - confirmed live, not theoretical: reading
   a real Hotstar page returned a rupee sign (₹) in its text, and printing
   that crashed with UnicodeEncodeError when launched from a Git Bash/MSYS
   terminal (sys.stdout.encoding reports cp1252 there), even though the
   exact same command from native PowerShell on the same machine reports
   utf-8 and works fine - this depends entirely on which terminal launched
   it, not anything about the machine itself. An uncaught UnicodeEncodeError
   inside a print() call in orchestrator.py's main loop is fatal (nothing
   catches it except KeyboardInterrupt), so this could crash Vaayu mid-
   conversation any time a reply or page-read result contains a character
   the launching terminal's codepage can't encode - a real risk, not just
   for currency symbols but em-dashes, curly quotes, emoji, etc.

   Fixed by forcing encoding="utf-8" outright (errors="replace" as a
   last-resort fallback for a terminal that genuinely can't render Unicode
   at all) rather than just adding errors="replace" to whatever Python
   auto-detected - verified live that forcing UTF-8 is strictly better, not
   just safer: the Git Bash terminal that reported cp1252 could actually
   render the rupee sign/em-dash/emoji correctly once told to use UTF-8, so
   Python's auto-detected encoding was simply wrong for that terminal, not a
   real limitation of it.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = _REPO_ROOT / "data" / "vaayu.log"


def ensure_output_stream() -> None:
    if sys.stdout is None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
        return

    # A real console is attached - keep printing to it (don't redirect to a
    # file), just force UTF-8 so it can't crash on a character the console's
    # own auto-detected codepage can't represent, with errors="replace" as a
    # last-resort fallback for the rare terminal that genuinely can't render
    # Unicode at all (rather than crashing there either).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
