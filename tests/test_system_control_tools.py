from unittest.mock import MagicMock, patch

import pytest

from assistant.tools import system_control


def test_press_hotkey_sends_the_given_keys():
    with patch.object(system_control.keyboard, "send") as mock_send:
        result = system_control.press_hotkey("ctrl+s")
    mock_send.assert_called_once_with("ctrl+s")
    assert result == "Pressed: ctrl+s"


def test_press_hotkey_passes_through_combo_keys_unmodified():
    with patch.object(system_control.keyboard, "send") as mock_send:
        system_control.press_hotkey("ctrl+shift+esc")
    mock_send.assert_called_once_with("ctrl+shift+esc")


def test_lock_computer_calls_lock_work_station():
    with patch.object(system_control.ctypes.windll.user32, "LockWorkStation", return_value=1) as mock_lock:
        result = system_control.lock_computer()
    mock_lock.assert_called_once()
    assert result == "Locked the computer."


def test_lock_computer_raises_on_failure():
    with patch.object(system_control.ctypes.windll.user32, "LockWorkStation", return_value=0):
        with pytest.raises(RuntimeError, match="Failed to lock"):
            system_control.lock_computer()


def test_take_screenshot_saves_and_returns_path(tmp_path):
    fake_image = MagicMock()

    with patch.object(system_control, "_SCREENSHOT_DIR", tmp_path), patch(
        "PIL.ImageGrab.grab", return_value=fake_image
    ):
        result = system_control.take_screenshot()

    assert str(tmp_path) in result
    assert result.startswith("Saved screenshot to ")
    fake_image.save.assert_called_once()
    saved_path = fake_image.save.call_args.args[0]
    assert saved_path.parent == tmp_path
    assert saved_path.suffix == ".png"
