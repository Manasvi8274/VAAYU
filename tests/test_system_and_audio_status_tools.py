from unittest.mock import MagicMock, patch

from assistant.tools import audio_status, system_info


# --- system_info.py ---


def test_get_system_status_reports_all_fields():
    fake_battery = MagicMock(percent=76, power_plugged=True)
    fake_disk = MagicMock(free=50 * 1024**3, total=200 * 1024**3)
    fake_mem = MagicMock(percent=42.5, used=8 * 1024**3, total=16 * 1024**3)

    with patch.object(system_info.psutil, "sensors_battery", return_value=fake_battery), patch.object(
        system_info.psutil, "disk_usage", return_value=fake_disk
    ), patch.object(system_info.psutil, "virtual_memory", return_value=fake_mem), patch.object(
        system_info.psutil, "cpu_percent", return_value=13.0
    ):
        result = system_info.get_system_status()

    assert result == {
        "cpu_percent": 13.0,
        "ram_percent": 42.5,
        "ram_used_gb": 8.0,
        "ram_total_gb": 16.0,
        "battery_percent": 76,
        "battery_plugged_in": True,
        "disk_free_gb": 50.0,
        "disk_total_gb": 200.0,
    }


def test_get_system_status_handles_no_battery_desktop_pc():
    fake_disk = MagicMock(free=50 * 1024**3, total=200 * 1024**3)
    fake_mem = MagicMock(percent=42.5, used=8 * 1024**3, total=16 * 1024**3)

    with patch.object(system_info.psutil, "sensors_battery", return_value=None), patch.object(
        system_info.psutil, "disk_usage", return_value=fake_disk
    ), patch.object(system_info.psutil, "virtual_memory", return_value=fake_mem), patch.object(
        system_info.psutil, "cpu_percent", return_value=5.0
    ):
        result = system_info.get_system_status()

    assert result["battery_percent"] is None
    assert result["battery_plugged_in"] is None


# --- audio_status.py ---


def _fake_device(name: str, volume_scalar: float, muted: bool):
    device = MagicMock()
    device.FriendlyName = name
    device.EndpointVolume.GetMasterVolumeLevelScalar.return_value = volume_scalar
    device.EndpointVolume.GetMute.return_value = muted
    return device


def test_get_audio_status_reports_speaker_and_microphone():
    speaker = _fake_device("Speakers (Realtek)", 0.65, False)
    mic_raw = MagicMock()
    mic_wrapped = _fake_device("Microphone (Realtek)", 0.80, True)

    with patch.object(audio_status.AudioUtilities, "GetSpeakers", return_value=speaker), patch.object(
        audio_status.AudioUtilities, "GetMicrophone", return_value=mic_raw
    ), patch.object(audio_status.AudioUtilities, "CreateDevice", return_value=mic_wrapped):
        result = audio_status.get_audio_status()

    assert result == {
        "speaker": {"name": "Speakers (Realtek)", "volume_percent": 65, "muted": False},
        "microphone": {"name": "Microphone (Realtek)", "volume_percent": 80, "muted": True},
    }
