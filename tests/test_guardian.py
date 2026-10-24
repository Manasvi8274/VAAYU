from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from guardian import core, rules


@pytest.mark.parametrize("text", ["Vaayu die", "Vayu, die.", "vaayu die!", " Wayu die ", "VAAYU DIE", "Vaayu dies",
                                  "Veayu die now.", "Vailu die."])  # last two: real tiny-model mishearings
def test_kill_phrase_heard(text):
    assert core.is_kill_phrase(text)


@pytest.mark.parametrize(
    "text",
    ["vaayu did it", "I will die", "die hard is a movie", "sleep vaayu", "shutdown vaayu", "vaayu open notepad",
     "why you die", "you die", "", "vaayu, the die is cast" * 0 + "vaayu the die is cast"],
)
def test_ordinary_speech_does_not_kill(text):
    assert not core.is_kill_phrase(text)


def test_restart_policy_allows_three_crashes_then_gives_up():
    crashes = deque()
    assert [core.should_restart(crashes, t) for t in (0, 10, 20)] == [True, True, True]
    assert core.should_restart(crashes, 30) is False


def test_old_crashes_age_out_of_the_restart_window():
    crashes = deque()
    for t in (0, 10, 20):
        core.should_restart(crashes, t)
    assert core.should_restart(crashes, 20 + core.RESTART_WINDOW_SECONDS + 1) is True


def test_vaayu_is_launched_without_auto_elevation():
    with patch.object(core.subprocess, "Popen") as popen:
        core.start_vaayu()
    args, kwargs = popen.call_args
    assert args[0][1:] == ["-m", "assistant.voice_main"]
    assert kwargs["env"]["__COMPAT_LAYER"] == "RunAsInvoker"


@pytest.fixture
def quiet_guardian(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "DATA", tmp_path)
    monkeypatch.setattr(core, "PID_FILE", tmp_path / "guardian.pid")
    monkeypatch.setattr(core, "LOG_FILE", tmp_path / "guardian.log")
    for name in ("_hotkey_loop", "_phrase_loop", "_uac_watch"):
        monkeypatch.setattr(core, name, lambda *a: None)
    monkeypatch.setattr(core, "_speak", MagicMock())
    monkeypatch.setattr(core, "kill_tree", MagicMock())
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    return tmp_path


def _proc(returncode):
    p = MagicMock(pid=1234, returncode=returncode)
    p.poll.return_value = returncode
    return p


def test_normal_shutdown_stops_the_guardian_too(quiet_guardian):
    with patch.object(core, "start_vaayu", return_value=_proc(0)) as start:
        core.run()
    assert start.call_count == 1
    assert not core.PID_FILE.exists()


def test_crashing_vaayu_is_restarted_then_given_up_on(quiet_guardian):
    with patch.object(core, "start_vaayu", side_effect=lambda: _proc(1)) as start:
        core.run()
    assert start.call_count == core.MAX_RESTARTS + 1
    core._speak.assert_called_once()
    assert "keeps crashing" in core._speak.call_args.args[0]


def test_kill_trigger_kills_vaayu_and_says_so(quiet_guardian, monkeypatch):
    triggers = []
    monkeypatch.setattr(core, "_hotkey_loop", triggers.append)
    running = _proc(None)

    def start_then_press_hotkey():
        triggers[0]("hotkey")  # pressed while Vaayu is running
        return running

    with patch.object(core, "start_vaayu", side_effect=start_then_press_hotkey):
        core.run()
    core.kill_tree.assert_any_call(1234)
    core._speak.assert_called_once_with("Vaayu is off.")


def test_kill_all_stops_guardian_and_every_vaayu_process(quiet_guardian, monkeypatch):
    monkeypatch.setattr(core, "_guardian_pid", lambda: 111)
    monkeypatch.setattr(core, "_vaayu_pids", lambda: [222])
    assert core.kill_all() == "Stopped: guardian, vaayu."
    assert [c.args[0] for c in core.kill_tree.call_args_list] == [111, 222]


@pytest.mark.parametrize("target", ["phonepe", "PhonePe", "paytm.com", "google pay", "gpay", "onlinesbi.sbi",
                                    "hdfc bank netbanking", "https://www.paypal.com"])
def test_payment_targets_are_blocked(target):
    with pytest.raises(PermissionError):
        rules.check_not_payment(target)


@pytest.mark.parametrize("target", ["spotify", "notepad", "youtube.com", "pay rent reminder", "credits page"])
def test_ordinary_targets_are_not_blocked(target):
    rules.check_not_payment(target)


def test_open_app_refuses_payment_apps_before_launching_anything():
    from assistant.tools import apps
    with patch.object(apps, "app_open") as app_open:
        with pytest.raises(PermissionError):
            apps.open_app("PhonePe")
    app_open.assert_not_called()


class _LoudIsSpeech:
    """Stand-in for webrtcvad: 'speech' whenever the frame is loud."""

    def is_speech(self, frame, rate):
        return core.Segmenter.rms(frame) > 500


def _frames(level, n):
    import numpy as np
    rng = np.random.default_rng(0)
    return [(rng.standard_normal(480) * level).astype("int16").tobytes() for _ in range(n)]


def test_segmenter_emits_one_utterance_after_speech_then_silence():
    seg = core.Segmenter(_LoudIsSpeech(), floor=100)
    out = [seg.feed(f) for f in _frames(100, 30) + _frames(3000, 30) + _frames(100, 25)]
    utterances = [u for u in out if u]
    assert len(utterances) == 1
    assert len(utterances[0]) == (10 + 30 + 20) * 960  # pre-roll + speech + 600 ms tail


def test_segmenter_drops_long_speech_that_cannot_be_the_kill_phrase():
    seg = core.Segmenter(_LoudIsSpeech(), floor=100)
    out = [seg.feed(f) for f in _frames(3000, 250) + _frames(100, 25)]
    assert not any(out)


def test_room_noise_alone_never_becomes_an_utterance():
    # The live bug: webrtcvad called room noise speech; the loudness gate stops that.
    class NoiseIsSpeech:
        def is_speech(self, frame, rate):
            return True
    seg = core.Segmenter(NoiseIsSpeech(), floor=100)
    assert not any(seg.feed(f) for f in _frames(110, 300))
