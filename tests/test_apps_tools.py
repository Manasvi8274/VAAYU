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
        # (and not in the live Start menu either): a plain "couldn't find", not AppOpener's
        # misleading "X is not running"
        with pytest.raises(RuntimeError, match="couldn't find an app called some_nonexistent_app_xyz"):
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


_FAKE_BROWSERS = {"brave": r"C:\fake\brave.exe", "google chrome": r"C:\fake\chrome.exe"}


def _patched_browser_env(running_exes=()):
    from unittest.mock import MagicMock
    procs = [MagicMock(exe=MagicMock(return_value=e)) for e in running_exes]
    return (patch.object(apps.browser_state, "_enumerate_registered_browsers", return_value=_FAKE_BROWSERS),
            patch.object(apps.psutil, "process_iter", return_value=procs),
            patch.object(apps, "foreground_by_title"), patch.object(apps.browser_state, "note_app_opened"),
            patch.object(apps, "app_open"))


def test_open_brave_launches_it_with_the_devtools_port_not_via_appopener():
    a, b, c, d, e = _patched_browser_env()
    with a, b, c, d, e as app_open, patch.object(apps, "_launch_with_port") as launch:
        reply = apps.open_app("brave")
    launch.assert_called_once_with(r"C:\fake\brave.exe")
    app_open.assert_not_called()
    assert reply == "Opened Brave."


def test_brave_already_running_without_the_port_says_how_to_fix_it():
    a, b, c, d, e = _patched_browser_env(running_exes=[r"c:\fake\brave.exe"])
    with a, b, c, d, e, patch.object(apps, "_launch_with_port") as popen, patch.object(apps, "_cdp_up", return_value=False):
        reply = apps.open_app("brave browser")
    popen.assert_not_called()  # relaunching would just open another window
    assert "wasn't opened by me" in reply and "Close Brave" in reply


def test_brave_already_running_with_the_port_is_just_brought_forward():
    a, b, c, d, e = _patched_browser_env(running_exes=[r"c:\fake\brave.exe"])
    with a, b, c, d, e, patch.object(apps, "_launch_with_port") as popen, patch.object(apps, "_cdp_up", return_value=True):
        assert apps.open_app("brave") == "Brave is already open."
    popen.assert_not_called()


def test_non_browser_apps_still_use_appopener():
    a, b, c, d, e = _patched_browser_env()
    with a, b, c, d, e as app_open:
        apps.open_app("notepad")
    app_open.assert_called_once()



def test_app_missing_from_appopeners_old_list_opens_from_the_live_start_menu(monkeypatch):
    # Real log: Tekken 7 was installed after AppOpener's list was built -> "TEKKEN 7 is not running".
    from assistant.tools import apps
    launched = []
    monkeypatch.setattr(apps, "_start_menu_apps", lambda: {"TEKKEN 7": "steam://rungameid/389730", "Notepad": "x!App"})
    monkeypatch.setattr("os.startfile", lambda target: launched.append(target))
    assert apps._real_open_from_start_menu("tekken game") == "TEKKEN 7"
    assert launched == ["steam://rungameid/389730"]
    assert apps._real_open_from_start_menu("some made up app") is None


def test_only_the_apps_own_windowless_processes_are_counted_as_its(monkeypatch):
    # App sweep: wmiprvse/dashost started the same second as Camera and close_app tried to end them -
    # a new Brave tab of the user's could have been next.
    from types import SimpleNamespace
    from assistant.tools import apps
    procs = {
        1: SimpleNamespace(parents=lambda: [SimpleNamespace(pid=100)], exe=lambda: r"C:\Tools\helper.exe"),
        2: SimpleNamespace(parents=lambda: [SimpleNamespace(pid=7)], exe=lambda: r"C:\Windows\System32\wbem\WmiPrvSE.exe"),
        3: SimpleNamespace(parents=lambda: [SimpleNamespace(pid=7)],
                           exe=lambda: r"C:\Program Files\WindowsApps\Microsoft.WindowsCalculator\CalculatorApp.exe"),
    }
    monkeypatch.setattr(apps.psutil, "Process", lambda pid: procs[pid])
    assert apps._is_part_of(1, {100}, store_app=False)        # started by the app itself
    assert not apps._is_part_of(2, {100}, store_app=True)     # a Windows service that happened to start
    assert apps._is_part_of(3, {100}, store_app=True)         # a Store app's own process
    assert not apps._is_part_of(3, {100}, store_app=False)    # ...but only when a Store window opened


def test_a_moved_game_opens_from_where_it_is_now_and_a_deleted_one_is_said_plainly(monkeypatch, tmp_path):
    # App sweep: AC Shadows' shortcut pointed to E:\shadows\Assassin Creed Shadows\ACShadows.exe but the game
    # is in E:\Assassin Creed Shadows - Windows' "Problem with Shortcut" box popped and the open hung on it.
    from assistant.tools import apps
    game = tmp_path / "Assassin Creed Shadows" / "ACShadows.exe"
    game.parent.mkdir()
    game.write_text("")
    real_moved = apps._moved
    monkeypatch.setattr(apps, "_moved", lambda target: real_moved(target, drive=tmp_path))
    launched = []
    monkeypatch.setattr(apps.subprocess, "Popen", lambda cmd, cwd=None: launched.append(cmd))
    monkeypatch.setattr(apps, "_start_menu_apps", lambda: {
        "Play Assassin Creed Shadows": str(tmp_path / "shadows" / "Assassin Creed Shadows" / "ACShadows.exe"),
        "God of War Ragnarok": str(tmp_path / "Games" / "God of War Ragnarok" / "GoWR.exe"),
        "Notepad": "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"})
    assert apps._real_open_shortcut_target("assassin creed shadows") == "Play Assassin Creed Shadows"
    assert launched == [[str(game)]]
    with pytest.raises(RuntimeError, match="isn't on this laptop any more"):
        apps._real_open_shortcut_target("god of war ragnarok")
    assert apps._real_open_shortcut_target("notepad") is None  # a normal entry: opened the usual way
    assert len(launched) == 1


@pytest.mark.parametrize("said, want", [
    ("force close notepad", {"app_name": "notepad", "force": True}),
    ("close notepad forcefully", {"app_name": "notepad", "force": True}),
    ("notepad zabardasti band karo", {"app_name": "notepad", "force": True}),
    ("close notepad", {"app_name": "notepad"}),  # never forced unless said - a save prompt is the user's call
])
def test_force_close_only_when_said(said, want):
    from assistant.llm import router
    assert router.fast_route(said) == ("tools", [("close_app", want)])
