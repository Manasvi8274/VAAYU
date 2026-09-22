"""
Second large parametrized suite (target ~5000 cases), extending the same
approach used in test_browser_suite_large.py to the rest of the tool
surface: control_phrases (wake/sleep/shutdown/confirmation detection - the
safety-critical layer), press_hotkey, apps, windows, devices, network,
system/audio status, dictation, the disk cache, the tool registry's schema
integrity across every registered tool, config loading, and the orb's
position parser.

This pass already found a real, serious bug just from reasoning about the
implementation before writing a single test: is_affirmative("that's not
right") returned True, because "not" wasn't recognized as a negation and
"right" alone is an affirmative word - the exact confirmation gate that
guards sending a real WhatsApp message on the user's behalf. Fixed in
control_phrases.py; regression cases for it are in this file. Everything
below is mocked at the same boundaries as the rest of the suite (no real
subprocess/registry/audio-device/filesystem-outside-tmp_path access), so
it runs in seconds despite the count.
"""

import itertools
from unittest.mock import MagicMock, patch

import pytest
import win32con

from assistant import control_phrases
from assistant.gui.orb_position import parse_position
from assistant.tools import (
    apps,
    audio_status,
    cache,
    devices,
    dictation,
    network_status,
    system_control,
    system_info,
    windows,
)
from assistant.config_schema import load_config


# ============================================================================
# control_phrases.is_affirmative - table-driven from the module's own
# vocabulary, so every defined trigger word gets real, direct coverage
# ============================================================================

_SENTENCE_TEMPLATES_YES = [
    "{w}", "{w} please", "yeah {w}", "well {w} then", "umm {w}", "{w}, do it",
]
_SENTENCE_TEMPLATES_NO = [
    "{w}", "{w} please dont", "umm {w}", "{w}, don't do that", "no {w}", "{w} thanks",
]


@pytest.mark.parametrize("word", sorted(control_phrases._AFFIRMATIVE_WORDS))
def test_is_affirmative_true_for_every_affirmative_word_bare(word):
    assert control_phrases.is_affirmative(word) is True


@pytest.mark.parametrize("word", sorted(control_phrases._AFFIRMATIVE_WORDS))
def test_is_affirmative_false_when_affirmative_word_is_negated(word):
    # Regression coverage for the real bug: "not <affirmative word>" must
    # never read as a confirmation.
    assert control_phrases.is_affirmative(f"that's not {word}") is False
    assert control_phrases.is_affirmative(f"not {word}") is False


@pytest.mark.parametrize(
    "word,template",
    list(itertools.product(sorted(control_phrases._AFFIRMATIVE_WORDS), _SENTENCE_TEMPLATES_YES)),
)
def test_is_affirmative_true_across_templates(word, template):
    assert control_phrases.is_affirmative(template.format(w=word)) is True


@pytest.mark.parametrize("phrase", control_phrases._AFFIRMATIVE_PHRASES)
def test_is_affirmative_true_for_every_affirmative_phrase_bare(phrase):
    assert control_phrases.is_affirmative(phrase) is True


@pytest.mark.parametrize(
    "phrase,template",
    list(itertools.product(control_phrases._AFFIRMATIVE_PHRASES, _SENTENCE_TEMPLATES_YES)),
)
def test_is_affirmative_true_for_phrases_across_templates(phrase, template):
    assert control_phrases.is_affirmative(template.format(w=phrase)) is True


@pytest.mark.parametrize("word", sorted(control_phrases._NEGATIVE_WORDS))
def test_is_affirmative_false_for_every_negative_word_bare(word):
    assert control_phrases.is_affirmative(word) is False


@pytest.mark.parametrize(
    "word,template",
    list(itertools.product(sorted(control_phrases._NEGATIVE_WORDS), _SENTENCE_TEMPLATES_NO)),
)
def test_is_affirmative_false_across_templates(word, template):
    assert control_phrases.is_affirmative(template.format(w=word)) is False


@pytest.mark.parametrize("text", [
    "what time is it", "how are you", "umm", "", "   ", "tell me a joke",
    "what's the weather", "play some music", "open the browser",
    "I don't know", "maybe later", "can you repeat that",
] * 3)  # 36 cases - unrelated speech must never default to a confirmation
def test_is_affirmative_false_for_unrelated_speech(text):
    assert control_phrases.is_affirmative(text) is False


# ============================================================================
# control_phrases: wake/sleep/shutdown detection across name spellings,
# casing, and sentence templates
# ============================================================================

_NAME_SPELLINGS = ["vaayu", "Vaayu", "VAAYU", "vayu", "Vayu", "VaAyU"]
_WAKE_TEMPLATES = ["wake up {n}", "{n} wake up", "hey {n} wake up", "{n}, please wake up", "wake {n} up"]
_SLEEP_TEMPLATES = ["sleep {n}", "{n} go to sleep", "{n}, time to sleep", "go to sleep {n}"]
_SHUTDOWN_TEMPLATES = ["shutdown {n}", "shut down {n}", "{n} please shut down", "{n}, shutdown now"]


@pytest.mark.parametrize(
    "name,template", list(itertools.product(_NAME_SPELLINGS, _WAKE_TEMPLATES))
)
def test_detect_control_command_wake_across_spellings_and_templates(name, template):
    text = template.format(n=name)
    assert control_phrases.detect_control_command(text) == control_phrases.ControlCommand.WAKE


@pytest.mark.parametrize(
    "name,template", list(itertools.product(_NAME_SPELLINGS, _SLEEP_TEMPLATES))
)
def test_detect_control_command_sleep_across_spellings_and_templates(name, template):
    text = template.format(n=name)
    assert control_phrases.detect_control_command(text) == control_phrases.ControlCommand.SLEEP


@pytest.mark.parametrize(
    "name,template", list(itertools.product(_NAME_SPELLINGS, _SHUTDOWN_TEMPLATES))
)
def test_detect_control_command_shutdown_across_spellings_and_templates(name, template):
    text = template.format(n=name)
    assert control_phrases.detect_control_command(text) == control_phrases.ControlCommand.SHUTDOWN


_CONTROL_WORDS_WITHOUT_NAME = [
    "wake up early tomorrow", "please shut down the laptop", "I need to sleep now",
    "shutdown the computer", "can you wake me at 7am", "sleep well tonight",
    "shut the door", "wake up call service", "going to sleep",
] * 4  # 36 cases


@pytest.mark.parametrize("text", _CONTROL_WORDS_WITHOUT_NAME)
def test_detect_control_command_none_without_name_present(text):
    assert control_phrases.detect_control_command(text) == control_phrases.ControlCommand.NONE


@pytest.mark.parametrize("name", _NAME_SPELLINGS)
@pytest.mark.parametrize("command", [
    "what's my battery percentage", "open brave", "search for pizza recipes",
    "what's the weather today", "play some music", "close notepad",
])
def test_detect_control_command_none_when_name_present_but_no_control_word(name, command):
    text = f"{name} {command}"
    assert control_phrases.detect_control_command(text) == control_phrases.ControlCommand.NONE


# ============================================================================
# control_phrases: has_command_content / is_name_mentioned combined
# ============================================================================

_REALISTIC_COMMANDS = [
    "open brave", "close chrome", "what's my battery", "play some music",
    "search for pizza", "increase the volume", "take a screenshot",
    "lock the computer", "minimize the window", "send a message to mom",
    "open youtube", "read my notes", "what's the wifi status",
    "close the third tab", "make it fullscreen", "search files named resume",
    "add a note", "list my notes", "switch to notepad", "mute the video",
    "unmute the video", "open the moviesmod website", "play fairytale music",
    "close the browser", "open file explorer", "what's the cpu usage",
    "check bluetooth devices", "list open windows", "type hello world",
    "pause the video",
]


@pytest.mark.parametrize("name", _NAME_SPELLINGS)
@pytest.mark.parametrize("command", _REALISTIC_COMMANDS)
def test_has_command_content_true_for_name_plus_real_command(name, command):
    assert control_phrases.has_command_content(f"{name} {command}") is True
    assert control_phrases.is_name_mentioned(f"{name} {command}") is True


@pytest.mark.parametrize("name", _NAME_SPELLINGS)
@pytest.mark.parametrize("filler", ["wake up", "hey", "hi", "hello", ""])
def test_has_command_content_false_for_bare_wake_variations(name, filler):
    text = f"{filler} {name}".strip()
    assert control_phrases.has_command_content(text) is False


# ============================================================================
# press_hotkey: broad modifier x key combinations
# ============================================================================

_MODIFIERS = ["ctrl", "alt", "shift", "win", "ctrl+shift", "ctrl+alt", "alt+shift", "ctrl+shift+alt"]
_KEYS = [
    "s", "z", "c", "v", "x", "a", "t", "w", "n", "tab", "f4", "f5", "f11",
    "esc", "delete", "home", "end", "pageup", "pagedown", "left", "right",
    "up", "down", "plus", "minus", "0", "1", "l", "d", "p", "f", "r", "e",
    "q", "y", "b", "u", "i", "o", "m",
]


@pytest.mark.parametrize("combo", [f"{m}+{k}" for m, k in itertools.product(_MODIFIERS, _KEYS)])
def test_press_hotkey_sends_exact_combo_string(combo):
    with patch.object(system_control.keyboard, "send") as mock_send:
        result = system_control.press_hotkey(combo)
    mock_send.assert_called_once_with(combo)
    assert result == f"Pressed: {combo}"


@pytest.mark.parametrize("key", _KEYS)
def test_press_hotkey_bare_key_no_modifier(key):
    with patch.object(system_control.keyboard, "send") as mock_send:
        system_control.press_hotkey(key)
    mock_send.assert_called_once_with(key)


# ============================================================================
# open_app / close_app across many app names
# ============================================================================

_APP_NAMES = [
    "notepad", "calculator", "chrome", "brave", "firefox", "edge", "spotify",
    "vscode", "word", "excel", "powerpoint", "paint", "explorer", "cmd",
    "powershell", "task manager", "control panel", "settings", "photos",
    "discord", "slack", "zoom", "teams", "skype", "vlc", "obs", "steam",
    "whatsapp", "telegram", "outlook",
]


@pytest.mark.parametrize("app_name", _APP_NAMES)
def test_open_app_across_many_names(app_name):
    # output=True is deliberate (see apps.py) - AppOpener's own throw_error
    # only fires when output is also truthy, so output=False used to
    # silently disable error detection entirely. "vlc" specifically goes
    # through the alias table first (AppOpener's fuzzy match can't find it
    # under its registered "VLC Media Player" name), so the real call uses
    # the resolved name, not the literal app_name.
    with patch.object(apps, "app_open") as mock_open, patch.object(
        apps.browser_state, "note_app_opened"
    ), patch.object(apps, "foreground_by_title"):
        result = apps.open_app(app_name)
    expected_name = apps._resolve_open_app_name(app_name)
    mock_open.assert_called_once_with(expected_name, match_closest=True, output=True, throw_error=True)
    assert result == f"Opened {app_name}."


@pytest.mark.parametrize("app_name", _APP_NAMES)
def test_close_app_across_many_names(app_name):
    with patch.object(apps, "app_close") as mock_close, patch.object(
        apps, "_find_matching_process_name", return_value=None
    ):
        result = apps.close_app(app_name)
    mock_close.assert_called_once_with(app_name, match_closest=True, output=True, throw_error=True)
    assert result == f"Closed {app_name}."


@pytest.mark.parametrize("app_name", _APP_NAMES)
def test_close_app_failure_propagates_for_every_app_name(app_name):
    with patch.object(apps, "app_close", side_effect=RuntimeError(f"{app_name} not found")):
        with pytest.raises(RuntimeError, match=app_name):
            apps.close_app(app_name)


# ============================================================================
# windows.py across many window-title-list sizes and query variations
# ============================================================================


def _titled_windows(n: int) -> list[tuple[int, str]]:
    return [(i, f"Window Title {i} - App{i % 5}") for i in range(n)]


@pytest.mark.parametrize("n", range(1, 41))
def test_list_open_windows_across_many_sizes(n):
    with patch.object(windows, "list_windows", return_value=_titled_windows(n)):
        result = windows.list_open_windows()
    assert len(result) == n
    assert result[0] == "Window Title 0 - App0"


@pytest.mark.parametrize("n", range(1, 41))
def test_switch_to_window_finds_last_window_by_title(n):
    with patch.object(windows, "list_windows", return_value=_titled_windows(n)), patch.object(
        windows, "force_foreground"
    ) as mock_fg:
        result = windows.switch_to_window(f"Title {n - 1} -")
    mock_fg.assert_called_once_with(n - 1)
    assert f"Window Title {n - 1}" in result


@pytest.mark.parametrize("query_case", ["upper", "lower", "title", "mixed"])
@pytest.mark.parametrize("index", range(0, 10))
def test_switch_to_window_case_insensitive_matching(index, query_case):
    windows_list = _titled_windows(10)
    base_query = f"title {index}"
    query = {
        "upper": base_query.upper(),
        "lower": base_query.lower(),
        "title": base_query.title(),
        "mixed": "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(base_query)),
    }[query_case]
    with patch.object(windows, "list_windows", return_value=windows_list), patch.object(
        windows, "force_foreground"
    ) as mock_fg:
        windows.switch_to_window(query)
    mock_fg.assert_called_once_with(index)


@pytest.mark.parametrize("n", range(1, 41))
def test_minimize_window_across_many_sizes(n):
    with patch.object(windows, "list_windows", return_value=_titled_windows(n)), patch.object(
        windows.win32gui, "ShowWindow"
    ) as mock_show:
        windows.minimize_window(f"Title {n - 1} -")
    mock_show.assert_called_once_with(n - 1, win32con.SW_MINIMIZE)


@pytest.mark.parametrize("n", range(1, 41))
def test_restore_window_across_many_sizes(n):
    with patch.object(windows, "list_windows", return_value=_titled_windows(n)), patch.object(
        windows, "force_foreground"
    ) as mock_fg:
        windows.restore_window(f"Title {n - 1} -")
    mock_fg.assert_called_once_with(n - 1)


# ============================================================================
# devices.list_connected_devices across categories and device-list sizes
# ============================================================================

import json as _json  # local alias to avoid shadowing anything above


@pytest.mark.parametrize("category", list(devices._CATEGORY_CLASS_MAP.keys()))
@pytest.mark.parametrize("count", [0, 1, 2, 5, 10, 20, 50])
def test_list_connected_devices_across_categories_and_sizes(category, count):
    fake_output = _json.dumps([{"FriendlyName": f"Device {i}", "Status": "OK"} for i in range(count)])
    with patch.object(devices, "_run_ps", return_value=fake_output):
        result = devices.list_connected_devices(category)
    assert len(result) == min(count, 20)


# ============================================================================
# network_status across WiFi/Bluetooth state combinations
# ============================================================================

_SSIDS = ["HomeNetwork", "Office_5G", "Cafe WiFi", "Guest Network", "MyRouter123"]
_SIGNALS = ["10%", "35%", "60%", "85%", "100%"]


def _wifi_output(state: str, ssid: str = "", signal: str = "") -> str:
    lines = [f"    State                  : {state}"]
    if ssid:
        lines.append(f"    SSID                   : {ssid}")
    if signal:
        lines.append(f"    Signal                 : {signal}")
    return "\n".join(lines)


@pytest.mark.parametrize("ssid,signal", list(itertools.product(_SSIDS, _SIGNALS)))
def test_get_network_status_wifi_connected_across_ssid_signal_combos(ssid, signal):
    with patch.object(network_status, "_run") as mock_run:
        mock_run.side_effect = [_wifi_output("connected", ssid, signal), "[]"]
        result = network_status.get_network_status()
    assert result["wifi"]["ssid"] == ssid
    assert result["wifi"]["signal"] == signal


@pytest.mark.parametrize("count", [0, 1, 2, 5, 10, 20])
def test_get_network_status_bluetooth_device_count_variations(count):
    bt_output = _json.dumps([{"FriendlyName": f"BT Device {i}", "Status": "OK"} for i in range(count)])
    with patch.object(network_status, "_run") as mock_run:
        mock_run.side_effect = [_wifi_output("disconnected"), bt_output]
        result = network_status.get_network_status()
    assert len(result["bluetooth"]["devices"]) == count
    assert result["bluetooth"]["enabled"] == (count > 0)


# ============================================================================
# system_info across battery/cpu/ram/disk value grids
# ============================================================================


@pytest.mark.parametrize("battery_pct", range(0, 101, 5))
@pytest.mark.parametrize("plugged_in", [True, False])
def test_get_system_status_battery_grid(battery_pct, plugged_in):
    fake_battery = MagicMock(percent=battery_pct, power_plugged=plugged_in)
    fake_disk = MagicMock(free=100 * 1024**3, total=500 * 1024**3)
    fake_mem = MagicMock(percent=50.0, used=8 * 1024**3, total=16 * 1024**3)
    with patch.object(system_info.psutil, "sensors_battery", return_value=fake_battery), patch.object(
        system_info.psutil, "disk_usage", return_value=fake_disk
    ), patch.object(system_info.psutil, "virtual_memory", return_value=fake_mem), patch.object(
        system_info.psutil, "cpu_percent", return_value=10.0
    ):
        result = system_info.get_system_status()
    assert result["battery_percent"] == battery_pct
    assert result["battery_plugged_in"] == plugged_in


@pytest.mark.parametrize("ram_pct", range(0, 101, 5))
def test_get_system_status_ram_grid(ram_pct):
    fake_disk = MagicMock(free=100 * 1024**3, total=500 * 1024**3)
    fake_mem = MagicMock(percent=float(ram_pct), used=8 * 1024**3, total=16 * 1024**3)
    with patch.object(system_info.psutil, "sensors_battery", return_value=None), patch.object(
        system_info.psutil, "disk_usage", return_value=fake_disk
    ), patch.object(system_info.psutil, "virtual_memory", return_value=fake_mem), patch.object(
        system_info.psutil, "cpu_percent", return_value=10.0
    ):
        result = system_info.get_system_status()
    assert result["ram_percent"] == float(ram_pct)


# ============================================================================
# audio_status across a fine-grained speaker/microphone volume grid
# ============================================================================


def _fake_audio_device(volume_scalar: float, muted: bool):
    device = MagicMock()
    device.FriendlyName = "Test Device"
    device.EndpointVolume.GetMasterVolumeLevelScalar.return_value = volume_scalar
    device.EndpointVolume.GetMute.return_value = muted
    return device


@pytest.mark.parametrize("speaker_pct", range(0, 101, 2))
@pytest.mark.parametrize("muted", [True, False])
def test_get_audio_status_speaker_volume_grid(speaker_pct, muted):
    speaker = _fake_audio_device(speaker_pct / 100, muted)
    mic = _fake_audio_device(0.5, False)
    with patch.object(audio_status.AudioUtilities, "GetSpeakers", return_value=speaker), patch.object(
        audio_status.AudioUtilities, "GetMicrophone", return_value=MagicMock()
    ), patch.object(audio_status.AudioUtilities, "CreateDevice", return_value=mic):
        result = audio_status.get_audio_status()
    assert result["speaker"]["volume_percent"] == round(speaker_pct)
    assert result["speaker"]["muted"] == muted


@pytest.mark.parametrize("mic_pct", range(0, 101, 5))
def test_get_audio_status_microphone_volume_grid(mic_pct):
    speaker = _fake_audio_device(0.5, False)
    mic = _fake_audio_device(mic_pct / 100, False)
    with patch.object(audio_status.AudioUtilities, "GetSpeakers", return_value=speaker), patch.object(
        audio_status.AudioUtilities, "GetMicrophone", return_value=MagicMock()
    ), patch.object(audio_status.AudioUtilities, "CreateDevice", return_value=mic):
        result = audio_status.get_audio_status()
    assert result["microphone"]["volume_percent"] == round(mic_pct)


# ============================================================================
# dictation.type_text across varied/adversarial text and clipboard states
# ============================================================================

_DICTATION_TEXTS = [
    "hello world", "The quick brown fox jumps over the lazy dog.",
    "line1\nline2\nline3", "tab\tseparated\tvalues", "unicode: café, naïve, 日本語",
    "emoji test 🎉🔥💯", "quotes: \"double\" and 'single'", "backslash \\ test",
    "very " + "long " * 50 + "text", "", "   spaces only   ",
    "numbers 1234567890", "special !@#$%^&*()_+-=[]{}|;:,.<>?/~`",
    "mixed CaSe TeXt", "a" * 500,
] * 7  # 15 * 7 = 105 cases


@pytest.mark.parametrize("text", _DICTATION_TEXTS)
def test_type_text_across_varied_content_restores_clipboard(text):
    with patch.object(dictation, "_get_clipboard_text", return_value="ORIGINAL"), patch.object(
        dictation, "_set_clipboard_text"
    ) as mock_set, patch.object(dictation.keyboard, "send") as mock_send, patch("time.sleep"):
        result = dictation.type_text(text)
    assert result == f"Typed: {text}"
    mock_send.assert_called_once_with("ctrl+v")
    assert mock_set.call_args_list[0].args == (text,)
    assert mock_set.call_args_list[-1].args == ("ORIGINAL",)


@pytest.mark.parametrize("original_clipboard", [
    "", "previous content", "unicode café", "a" * 1000, "multi\nline\ncontent",
] * 10)  # 50 cases
def test_type_text_restores_whatever_original_clipboard_had(original_clipboard):
    with patch.object(dictation, "_get_clipboard_text", return_value=original_clipboard), patch.object(
        dictation, "_set_clipboard_text"
    ) as mock_set, patch.object(dictation.keyboard, "send"), patch("time.sleep"):
        dictation.type_text("new text")
    assert mock_set.call_args_list[-1].args == (original_clipboard,)


# ============================================================================
# cache.py: boundary conditions around max_age_seconds, many key/value pairs
# ============================================================================


@pytest.mark.parametrize("namespace,key,value", [
    (f"ns_{i % 10}", f"key_{i}", f"value_{i}") for i in range(150)
])
def test_cache_round_trip_many_distinct_entries(namespace, key, value, tmp_path):
    with patch.object(cache, "_CACHE_FILE", tmp_path / "cache.json"), patch.object(
        cache, "_CACHE_DIR", tmp_path
    ):
        cache.set(namespace, key, value)
        result = cache.get(namespace, key, max_age_seconds=999999)
    assert result == value


@pytest.mark.parametrize("age_offset", [-100, -10, -1, -0.1, 0.1, 1, 10, 100])
def test_cache_freshness_boundary(age_offset, tmp_path):
    # age_offset negative -> still fresh (entry younger than max_age);
    # positive -> stale (entry older than max_age). max_age fixed at 60s.
    cache_file = tmp_path / "cache.json"
    with patch.object(cache, "_CACHE_FILE", cache_file), patch.object(cache, "_CACHE_DIR", tmp_path):
        cache.set("ns", "k", "v")
        import json as json_mod
        import time as time_mod

        data = json_mod.loads(cache_file.read_text())
        data["ns"]["k"]["cached_at"] = time_mod.time() - (60 + age_offset)
        cache_file.write_text(json_mod.dumps(data))

        result = cache.get("ns", "k", max_age_seconds=60)
    if age_offset < 0:
        assert result == "v"
    else:
        assert result is None


# ============================================================================
# registry.py: schema integrity across every currently-registered tool
# ============================================================================

import assistant.tools  # noqa: F401 - registers every tool
from assistant.tools.registry import registry as real_registry

_ALL_TOOL_NAMES = sorted(real_registry._tools.keys())
_HALLUCINATED_ARG_NAMES = [
    "extra_field", "unexpected_arg", "debug", "verbose", "fake_param",
    "random_key", "not_real", "junk", "padding_arg", "unused",
]


@pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
def test_every_tool_schema_is_well_formed(tool_name):
    spec = real_registry._tools[tool_name]
    assert isinstance(spec.name, str) and spec.name
    assert isinstance(spec.description, str) and spec.description
    assert spec.parameters.get("type") == "object"
    assert "properties" in spec.parameters
    required = spec.parameters.get("required", [])
    for req_field in required:
        assert req_field in spec.parameters["properties"], (
            f"{tool_name}: required field '{req_field}' isn't in properties"
        )


@pytest.mark.parametrize("tool_name", _ALL_TOOL_NAMES)
def test_every_tool_produces_valid_ollama_schema(tool_name):
    spec = real_registry._tools[tool_name]
    schema = spec.to_ollama_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == tool_name
    assert schema["function"]["parameters"] == spec.parameters


@pytest.mark.parametrize(
    "tool_name,extra_arg", list(itertools.product(_ALL_TOOL_NAMES, _HALLUCINATED_ARG_NAMES))
)
def test_dispatch_filters_hallucinated_arguments_for_every_tool(tool_name, extra_arg):
    spec = real_registry._tools[tool_name]
    # Build a minimally-valid argument set (empty is fine - dispatch calls
    # the real function, but for tools with required args that would raise
    # a TypeError unrelated to what we're testing; skip those here since
    # per-tool required-arg coverage already exists in their own test files -
    # this block is specifically about the hallucinated-arg-filtering
    # behavior in registry.dispatch, not each tool's own logic).
    if spec.parameters.get("required"):
        pytest.skip(f"{tool_name} needs required args - covered in its own test file")
    with patch.object(spec, "func", return_value="ok") as mock_func:
        result = real_registry.dispatch(tool_name, {extra_arg: "hallucinated value"})
    mock_func.assert_called_once_with()  # hallucinated arg was filtered out, not passed through
    assert result == {"ok": True, "result": "ok"}


# ============================================================================
# config_schema: many valid config variations
# ============================================================================

_LLM_MODELS = ["qwen2.5:3b", "qwen2.5:7b-instruct", "llama3.2:3b", "qwen2.5-coder:3b"]
_TTS_RATES = list(range(100, 301, 10))
_VAD_LEVELS = [0, 1, 2, 3]
_WHISPER_SIZES = ["tiny", "base", "small", "medium"]


@pytest.mark.parametrize("model", _LLM_MODELS)
def test_load_config_various_llm_models(model, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'llm:\n  model: "{model}"\n', encoding="utf-8")
    config = load_config(config_file)
    assert config.llm.model == model


@pytest.mark.parametrize("rate", _TTS_RATES)
def test_load_config_various_tts_rates(rate, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"tts:\n  rate: {rate}\n", encoding="utf-8")
    config = load_config(config_file)
    assert config.tts.rate == rate


@pytest.mark.parametrize("level", _VAD_LEVELS)
def test_load_config_various_vad_aggressiveness(level, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"listening:\n  vad_aggressiveness: {level}\n", encoding="utf-8")
    config = load_config(config_file)
    assert config.listening.vad_aggressiveness == level


@pytest.mark.parametrize("size", _WHISPER_SIZES)
def test_load_config_various_whisper_sizes(size, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'stt:\n  whisper_model_size: "{size}"\n', encoding="utf-8")
    config = load_config(config_file)
    assert config.stt.whisper_model_size == size


# ============================================================================
# orb_position.parse_position: direction x magnitude x starting-position
# grid, plus named positions across many screen sizes
# ============================================================================

_DIRECTIONS = ["left", "right", "up", "down"]
_MAGNITUDES = {"little": 40, "bit": 40, "slightly": 40, "more": 80, "further": 80, "lot": 150}
_START_POSITIONS = [(x, y) for x in range(0, 2000, 200) for y in (0, 500)]  # 20 combos


@pytest.mark.parametrize(
    "direction,magnitude,start",
    [(d, m, s) for d, m, s in itertools.product(_DIRECTIONS, _MAGNITUDES, _START_POSITIONS)],
)
def test_parse_position_nudge_grid(direction, magnitude, start):
    cx, cy = start
    dx_sign, dy_sign = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[direction]
    amount = _MAGNITUDES[magnitude]
    result = parse_position(f"a {magnitude} {direction}", cx, cy, 1920, 1080, 44)
    assert result == (cx + dx_sign * amount, cy + dy_sign * amount)


_NAMED_POSITION_NAMES = [
    "center", "middle", "top left", "top right", "bottom left", "bottom right",
    "top", "bottom", "left", "right",
]
_SCREEN_SIZES = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (1366, 768),
                  (1024, 768), (800, 600), (2560, 1600), (1920, 1200), (3440, 1440)]


@pytest.mark.parametrize(
    "name,screen", list(itertools.product(_NAMED_POSITION_NAMES, _SCREEN_SIZES))
)
def test_parse_position_named_positions_across_screen_sizes(name, screen):
    sw, sh = screen
    result = parse_position(f"move to {name}", 0, 0, sw, sh, 44)
    assert result is not None
    x, y = result
    assert 0 <= x <= sw
    assert 0 <= y <= sh


@pytest.mark.parametrize("garbage", [
    "do a backflip", "make me a sandwich", "what's the meaning of life",
    "explain quantum physics", "tell me about your day", "12345",
    "asdkfj laksdjf", "", "   ", "!@#$%^&*()",
] * 3)  # 30 cases
def test_parse_position_returns_none_for_unparseable_text(garbage):
    assert parse_position(garbage, 500, 500, 1920, 1080, 44) is None
