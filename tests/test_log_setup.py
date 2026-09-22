"""
log_setup.ensure_output_stream() - found via live testing (see log_setup.py's
docstring), not previously covered at all: a real console attached with an
encoding that can't represent a character Vaayu needs to print (confirmed
live with a rupee sign from a real Hotstar page, launched from a Git Bash
terminal that reported cp1252) used to crash the whole process with an
uncaught UnicodeEncodeError - nothing in orchestrator.py's main loop catches
that. These tests simulate both the "no console" and "wrong-encoding
console" cases without needing a second real terminal.
"""

import io

import pytest

from assistant import log_setup


def test_ensure_output_stream_redirects_to_log_file_when_no_console(monkeypatch, tmp_path):
    monkeypatch.setattr(log_setup, "LOG_PATH", tmp_path / "vaayu.log")
    monkeypatch.setattr("sys.stdout", None)
    monkeypatch.setattr("sys.stderr", None)

    log_setup.ensure_output_stream()

    import sys

    assert sys.stdout is not None
    assert sys.stdout is sys.stderr
    assert sys.stdout.encoding == "utf-8"
    print("test line with a rupee sign: ₹100")
    sys.stdout.flush()
    assert "₹100" in (tmp_path / "vaayu.log").read_text(encoding="utf-8")


def test_ensure_output_stream_forces_utf8_on_a_real_console(monkeypatch):
    # Simulate a real console whose reported encoding can't represent the
    # rupee sign (matches the real cp1252 Git Bash scenario) - after
    # ensure_output_stream(), it must be reconfigured to utf-8 rather than
    # crashing on that character.
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    fake_stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", fake_stdout)
    monkeypatch.setattr("sys.stderr", fake_stderr)

    # Sanity-check the crash this is guarding against actually reproduces
    # against the raw cp1252 stream before the fix is applied.
    with pytest.raises(UnicodeEncodeError):
        fake_stdout.write("₹100")

    log_setup.ensure_output_stream()

    import sys

    assert sys.stdout.encoding.lower().replace("_", "-") == "utf-8"
    sys.stdout.write("no crash: ₹100 — em dash \U0001F389 emoji")  # must not raise


def test_ensure_output_stream_does_not_redirect_a_real_console_to_a_file(monkeypatch):
    # A real console must keep printing to itself, not silently switch to
    # the log file the way the no-console case does.
    fake_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr("sys.stdout", fake_stdout)
    monkeypatch.setattr("sys.stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    log_setup.ensure_output_stream()

    import sys

    assert sys.stdout is fake_stdout


def test_ensure_output_stream_handles_streams_without_reconfigure_gracefully():
    # A stream that doesn't support reconfigure() (e.g. some test/CI
    # harness stand-ins) must not crash ensure_output_stream() itself.
    class _NoReconfigureStream:
        pass

    import sys as sys_module

    original_stdout, original_stderr = sys_module.stdout, sys_module.stderr
    try:
        sys_module.stdout = _NoReconfigureStream()
        sys_module.stderr = _NoReconfigureStream()
        log_setup.ensure_output_stream()  # must not raise
    finally:
        sys_module.stdout, sys_module.stderr = original_stdout, original_stderr
