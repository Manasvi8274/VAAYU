"""
Play/pause/next/previous and volume - aimed at whatever the user is
actually watching.

Found live: with a YouTube video playing, "volume down to 75%" and "pause"
went to the system volume / a media key, while the user was looking at
YouTube's own slider (still at 90%) - and nothing confirmed anything. So:
if a YouTube video is playing in the browser Vaayu controls, these act on
YouTube through its player API (verified - see youtube_player.py).
Otherwise they act on Windows itself: the system volume is set and read
back through the audio endpoint (not volume keys, which only move 2% at a
time and can't be verified), and play/pause/next/previous use media keys.
"""

import keyboard
from pycaw.pycaw import AudioUtilities

from assistant.tools.registry import registry

_MEDIA_KEYS = {"play_pause": "play/pause media", "play": "play/pause media", "pause": "play/pause media",
               "next": "next track", "previous": "previous track"}
_ACTIONS = ["play_pause", "play", "pause", "next", "previous", "volume_up", "volume_down", "volume_set", "mute", "unmute"]
# Model sends these despite the enum (issues_log: "band karo" -> 'pause' x5).
_ALIASES = {"resume": "play", "stop": "pause", "toggle": "play_pause"}
_STEP = 10  # percentage points per "volume up/down"


def _youtube_video():
    """(youtube_player module, tab, is_playing) for a YouTube watch tab in the
    controlled browser, or None - including when no such browser is open."""
    try:
        from assistant.tools import youtube_player as yp
        tab = yp.youtube_tab(require_watch=True)
        return (yp, tab, yp.is_playing(tab)) if tab else None
    except Exception:
        return None


def _youtube_in_foreground() -> bool:
    """The window in front is a browser showing YouTube ("... - YouTube - Brave")."""
    try:
        import win32gui
        title = win32gui.GetWindowText(win32gui.GetForegroundWindow())
        return "YouTube" in title and any(b in title for b in ("Brave", "Chrome", "Edge", "Comet"))
    except Exception:
        return False


def _on_youtube(action: str, level: int | None):
    found = _youtube_video()
    if not found:
        return None
    yp, tab, playing = found
    if action == "pause":
        return yp.pause(tab) if playing else None
    if action in ("play", "play_pause"):
        if playing:
            return yp.pause(tab) if action == "play_pause" else "It's already playing."
        return yp.play(tab)
    # A paused YouTube tab in the background shouldn't capture volume/next meant
    # for something else - but if YouTube is the window the user is looking at,
    # it's what they mean (found live: "reduce the volume to 0 percent" with
    # YouTube paused on screen set the LAPTOP to 0%).
    if not playing and not _youtube_in_foreground():
        return None
    if action == "next":
        return yp.next_video(tab)
    if action == "previous":
        return yp.previous_video(tab)
    if action in ("volume_up", "volume_down"):
        return yp.set_volume(delta=_STEP if action == "volume_up" else -_STEP, tab=tab)
    if action == "volume_set":
        return yp.set_volume(level=level, tab=tab)
    if action in ("mute", "unmute"):
        return yp.set_muted(action == "mute", tab=tab)
    return None


def _endpoint():
    return AudioUtilities.GetSpeakers().EndpointVolume


def _system_volume(action: str, level: int | None) -> str:
    ep = _endpoint()
    if action in ("mute", "unmute"):
        want = action == "mute"
        ep.SetMute(1 if want else 0, None)
        if bool(ep.GetMute()) != want:
            raise RuntimeError(f"I tried to {action} the laptop, but it didn't change.")
        return "Muted the laptop." if want else "Unmuted the laptop."
    current = round(ep.GetMasterVolumeLevelScalar() * 100)
    target = level if action == "volume_set" else current + (_STEP if action == "volume_up" else -_STEP)
    target = max(0, min(100, int(target)))
    ep.SetMasterVolumeLevelScalar(target / 100, None)
    if ep.GetMute() and target > 0:
        ep.SetMute(0, None)
    actual = round(ep.GetMasterVolumeLevelScalar() * 100)
    if abs(actual - target) > 1:
        raise RuntimeError(f"I tried to set the laptop volume to {target}%, but it's at {actual}%.")
    return f"Laptop volume is now {target}%."


@registry.register(
    name="control_media",
    description=(
        "Play, pause, next/previous track, and volume (volume_up, volume_down, volume_set with level "
        "0-100, mute, unmute). If a YouTube video is playing in the browser, this controls YouTube; "
        "otherwise it controls the laptop's own volume and media keys."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": _ACTIONS, "description": "The media control action."},
            "level": {"type": "integer", "description": "Volume percentage 0-100, only for volume_set."},
        },
        "required": ["action"],
    },
)
def control_media(action: str, level: int | None = None) -> str:
    action = _ALIASES.get(action, action)
    if action not in _ACTIONS:
        raise ValueError(f"Unknown media action '{action}'. Valid actions: {_ACTIONS}")
    if action == "volume_set" and level is None:
        raise ValueError("volume_set needs a level (0-100).")
    from assistant.tools import music
    if music.player.active:  # a song Vaayu is playing himself (no browser): pause/next/volume go to it
        return music.control(action, level)
    on_youtube = _on_youtube(action, level)
    if on_youtube:
        return on_youtube
    if action.startswith("volume") or action in ("mute", "unmute"):
        return _system_volume(action, level)
    keyboard.send(_MEDIA_KEYS[action])
    # A media key can't be read back - say what was sent, don't claim the result.
    return {"next": "Sent next track.", "previous": "Sent previous track."}.get(action, "Sent play/pause.")
