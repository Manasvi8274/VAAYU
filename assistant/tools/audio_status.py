from pycaw.pycaw import AudioUtilities

from assistant.tools.registry import registry


def _device_status(device) -> dict:
    volume = device.EndpointVolume
    return {
        "name": device.FriendlyName,
        "volume_percent": round(volume.GetMasterVolumeLevelScalar() * 100),
        "muted": bool(volume.GetMute()),
    }


@registry.register(
    name="get_audio_status",
    description=(
        "Check the default speaker (output) and microphone (input): device name, "
        "current volume level, and whether either is muted. This is volume, NOT battery - for any "
        "battery/charge question (including Hinglish 'battery kitni hai'), use get_system_status "
        "instead, never this one."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def get_audio_status() -> dict:
    speaker = AudioUtilities.GetSpeakers()
    # GetMicrophone() returns a raw, unwrapped device pointer (unlike
    # GetSpeakers()) - wrap it with CreateDevice() to get FriendlyName/EndpointVolume.
    microphone = AudioUtilities.CreateDevice(AudioUtilities.GetMicrophone())
    return {
        "speaker": _device_status(speaker),
        "microphone": _device_status(microphone),
    }
