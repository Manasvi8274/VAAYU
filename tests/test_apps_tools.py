"""
apps.py had zero test coverage before this - a real gap, given close_app is
the exact tool at the center of the reported "said it closed Brave but
didn't" bug (traced to the LLM skipping the tool call, not this code - see
llm/ollama_client.py's grounding fix - but that doesn't mean this code
shouldn't be covered too).
"""

from unittest.mock import patch

import pytest

from assistant.tools import apps


class _FakeAppNotFound(Exception):
    pass


def test_open_app_calls_app_opener_with_match_closest_and_throw_error():
    # output=True is deliberate, not a leftover default - see apps.py's
    # comment: AppOpener's own throw_error only ever fires when output is
    # truthy too, so output=False (the old value here) silently disabled
    # error detection completely. The prints it causes are captured/
    # discarded via _quiet_stdout, not actually shown to the user.
    with patch.object(apps, "app_open") as mock_open, patch.object(
        apps.browser_state, "note_app_opened"
    ) as mock_note, patch.object(apps, "foreground_by_title") as mock_fg:
        result = apps.open_app("notepad")

    mock_open.assert_called_once_with("notepad", match_closest=True, output=True, throw_error=True)
    mock_note.assert_called_once_with("notepad")
    mock_fg.assert_called_once_with("notepad")
    assert result == "Opened notepad."


def test_open_app_resolves_known_alias_before_calling_app_opener():
    # AppOpener's own fuzzy matching can't find "vlc" against its registered
    # "VLC Media Player" entry (too short a name for its similarity cutoff,
    # confirmed live) - the alias table works around that specific gap.
    with patch.object(apps, "app_open") as mock_open, patch.object(
        apps.browser_state, "note_app_opened"
    ), patch.object(apps, "foreground_by_title"):
        apps.open_app("vlc")

    mock_open.assert_called_once_with("vlc media player", match_closest=True, output=True, throw_error=True)


def test_open_app_background_does_not_force_foreground():
    with patch.object(apps, "app_open"), patch.object(apps.browser_state, "note_app_opened"), patch.object(
        apps, "foreground_by_title"
    ) as mock_fg:
        result = apps.open_app("notepad", foreground=False)

    mock_fg.assert_not_called()
    assert result == "Opened notepad in the background."


def test_open_app_propagates_not_found_as_a_real_failure():
    # If AppOpener can't find/match anything, that must surface as a real
    # exception (which registry.dispatch turns into ok: False) - never a
    # silent no-op that still returns a success-shaped string.
    with patch.object(apps, "app_open", side_effect=_FakeAppNotFound("NOPE")):
        with pytest.raises(_FakeAppNotFound):
            apps.open_app("some_nonexistent_app_xyz")


def test_close_app_calls_app_opener_with_match_closest_and_throw_error():
    with patch.object(apps, "app_close") as mock_close, patch.object(
        apps, "_find_matching_process_name", return_value=None
    ):
        result = apps.close_app("brave")

    mock_close.assert_called_once_with("brave", match_closest=True, output=True, throw_error=True)
    assert result == "Closed brave."


def test_close_app_raises_if_process_still_running_after_appopener_reports_success():
    # The core new-fix scenario: AppOpener's close_things() returns without
    # raising, but a real matching process is still alive - found live
    # (notepad.exe kept running after close_app claimed "Notepad has been
    # closed") - must not be trusted at face value.
    with patch.object(apps, "app_close"), patch.object(
        apps, "_find_matching_process_name", return_value="notepad.exe"
    ), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="still appears to be running"):
            apps.close_app("notepad")


def test_close_app_succeeds_if_process_disappears_within_retries():
    # Process termination isn't always instant - a still-running result on
    # the first check that clears up on retry must not be a false failure.
    calls = {"n": 0}

    def _fake_check(app_name):
        calls["n"] += 1
        return "notepad.exe" if calls["n"] < 2 else None

    with patch.object(apps, "app_close"), patch.object(
        apps, "_find_matching_process_name", side_effect=_fake_check
    ), patch("time.sleep"):
        result = apps.close_app("notepad")

    assert result == "Closed notepad."


def test_close_app_propagates_failure_instead_of_claiming_success():
    # The core grounding principle applied directly to this tool: if
    # AppOpener/taskkill genuinely fails, close_app must raise - not return a
    # "Closed X." string regardless of what actually happened.
    with patch.object(apps, "app_close", side_effect=_FakeAppNotFound("BRAVE")):
        with pytest.raises(_FakeAppNotFound):
            apps.close_app("brave")


def test_close_app_dispatch_through_real_registry_surfaces_failure_as_not_ok():
    # End-to-end through the real ToolRegistry.dispatch (not just the bare
    # function) - this is the exact path the orchestrator actually uses, and
    # exactly what needs to be true for the grounding-retry fix in
    # ollama_client.py to have real failure information to work with.
    from assistant.tools.registry import registry as real_registry

    with patch.object(apps, "app_close", side_effect=_FakeAppNotFound("BRAVE NOT RUNNING")):
        result = real_registry.dispatch("close_app", {"app_name": "brave"})

    assert result["ok"] is False
    assert "BRAVE NOT RUNNING" in result["error"]


def test_close_app_dispatch_through_real_registry_reports_success_only_when_real():
    from assistant.tools.registry import registry as real_registry

    with patch.object(apps, "app_close") as mock_close, patch.object(
        apps, "_find_matching_process_name", return_value=None
    ):
        result = real_registry.dispatch("close_app", {"app_name": "brave"})

    mock_close.assert_called_once()
    assert result["ok"] is True
    assert result["result"] == "Closed brave."
