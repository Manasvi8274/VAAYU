"""Every 10th case of scripts/youtube_command_suite.py (~2,000 of 20,000) - a
regression guard so router changes can't quietly break YouTube commands.
Run the full suite with: python scripts/youtube_command_suite.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import youtube_command_suite as suite  # noqa: E402


def test_youtube_commands_are_understood_instantly():
    cases = suite.build()[::10]
    misses = [c["text"] for c in cases if not suite.check(c)[0]]
    assert len(misses) / len(cases) <= 0.005, misses[:20]
