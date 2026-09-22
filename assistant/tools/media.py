import keyboard

from assistant.tools.registry import registry

_KEY_MAP = {
    "play_pause": "play/pause media",
    "next": "next track",
    "previous": "previous track",
    "volume_up": "volume up",
    "volume_down": "volume down",
    "mute": "volume mute",
}


@registry.register(
    name="control_media",
    description=(
        "Control media playback or volume: play/pause, skip to the next/previous "
        "track, adjust volume up/down, or mute."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_KEY_MAP.keys()),
                "description": "The media control action to perform.",
            }
        },
        "required": ["action"],
    },
)
def control_media(action: str) -> str:
    if action not in _KEY_MAP:
        raise ValueError(f"Unknown media action '{action}'. Valid actions: {list(_KEY_MAP)}")
    keyboard.send(_KEY_MAP[action])
    return f"Sent media command: {action}"
