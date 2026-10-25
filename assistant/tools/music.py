"""
Songs straight to the speakers - no browser, no YouTube page.

yt-dlp finds the song (a YouTube search, done by Vaayu himself) and gives
its audio stream; PyAV (already here for faster-whisper) decodes it, and
sounddevice plays it. Measured: ~4 s to find a song, then sound in ~0.2 s.
Pause / resume / stop / next / volume all work by voice (control_media
routes here while this player is active), and every song played feeds the
taste history (youtube_taste) that recommendations use.
"""

import shutil
import threading
import time

import numpy as np

from assistant.tools.registry import registry

_RATE, _CHANNELS = 48000, 2
_BLOCK = 1024


class _Player:
    def __init__(self):
        self.lock = threading.Lock()
        self.track: dict | None = None
        self.queue: list[dict] = []  # the other search results: "next" plays these
        self.volume = 0.8
        self.duck = 1.0  # lowered while Vaayu speaks
        self.paused = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._buf = np.zeros((0, _CHANNELS), np.float32)
        self._stream = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self, track: dict) -> None:
        self.stop()
        self._stop.clear()
        self.paused.clear()
        self.track = track
        self._thread = threading.Thread(target=self._run, args=(track,), daemon=True, name="music")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.paused.clear()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    def _callback(self, outdata, frames, _time, _status):
        with self.lock:
            if self.paused.is_set() or len(self._buf) == 0:
                outdata.fill(0)
                return
            n = min(frames, len(self._buf))
            outdata[:n] = self._buf[:n] * (self.volume * self.duck)
            outdata[n:] = 0
            self._buf = self._buf[n:]

    def _run(self, track: dict) -> None:
        import av
        import sounddevice as sd
        try:
            container = av.open(track["url"], options={"reconnect": "1", "reconnect_streamed": "1"}, timeout=15)
            resampler = av.AudioResampler(format="flt", layout="stereo", rate=_RATE)
            with sd.OutputStream(samplerate=_RATE, channels=_CHANNELS, dtype="float32", blocksize=_BLOCK,
                                 callback=self._callback):
                for frame in container.decode(container.streams.audio[0]):
                    for out in resampler.resample(frame):
                        pcm = out.to_ndarray().reshape(-1, _CHANNELS)
                        while not self._stop.is_set():  # keep ~2 s buffered, no more
                            with self.lock:
                                if len(self._buf) < _RATE * 2:
                                    self._buf = np.concatenate([self._buf, pcm])
                                    break
                            time.sleep(0.05)
                        if self._stop.is_set():
                            return
                while not self._stop.is_set() and (len(self._buf) > 0 or self.paused.is_set()):
                    time.sleep(0.1)  # let the last of the song play out
            container.close()
        except Exception as exc:
            from assistant import interaction_log
            interaction_log.log_error("music", exc, title=track.get("title"))
        finally:
            with self.lock:
                self._buf = np.zeros((0, _CHANNELS), np.float32)
        if not self._stop.is_set() and self.queue:  # finished on its own: the next one, like autoplay
            nxt = self.queue.pop(0)
            try:
                self.play(_resolve(nxt))
            except Exception:
                pass


player = _Player()


def _ydl_opts(**extra) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": "bestaudio[ext=m4a]/bestaudio/best"}
    if shutil.which("node"):  # YouTube needs a JS runtime for some formats
        opts["js_runtimes"] = {"node": {}}
    return opts | extra


_ANY_MUSIC = {"some music", "music", "some songs", "a song", "songs", "koi gaana", "kuch gaane", "some", "koi", "kuch"}


def _search(query: str, n: int = 5) -> list[dict]:
    import yt_dlp
    with yt_dlp.YoutubeDL(_ydl_opts(extract_flat=True)) as y:
        info = y.extract_info(f"ytsearch{n}:{query}", download=False)
    return [{"id": e["id"], "title": e.get("title") or "", "channel": e.get("channel") or e.get("uploader") or "",
             "duration": e.get("duration") or 0} for e in info.get("entries") or [] if e.get("id")]


def _resolve(track: dict) -> dict:
    import yt_dlp
    with yt_dlp.YoutubeDL(_ydl_opts()) as y:
        info = y.extract_info(f"https://www.youtube.com/watch?v={track['id']}", download=False)
    return track | {"url": info["url"], "title": info.get("title") or track["title"],
                    "channel": info.get("channel") or info.get("uploader") or track.get("channel", "")}


def _spoken(title: str) -> str:
    """"Kesariya - Brahmastra | Ranbir Kapoor, Alia Bhatt | Pritam | ... | 4K" -> "Kesariya - Brahmastra"."""
    import re
    return re.split(r"\s[|(\[]|\s\|\s", title)[0].strip() or title


def _remember(track: dict) -> None:
    from assistant.tools import youtube_taste
    youtube_taste.record(track["id"], track["title"], track.get("channel", ""))


@registry.register(
    name="play_music",
    description=("Play a song or music through the laptop speakers WITHOUT opening YouTube or a browser - Vaayu "
                 "searches for it himself and streams only the audio. Use for 'play <song/artist> song', "
                 "'<song> gaana chalao/bajao', 'play some music'. For videos, or when the user says 'on youtube', "
                 "use play_video_on_youtube instead."),
    parameters={"type": "object", "properties": {"query": {"type": "string", "description": "Song, artist or genre."}},
                "required": ["query"]},
)
def play_music(query: str) -> str:
    query = query.strip()
    if query.lower() in _ANY_MUSIC or query.lower().startswith("something"):  # "play some music": what they like (watch history), else popular Hindi songs
        from assistant.tools import youtube_taste
        channels, topics = youtube_taste.profile()
        query = (f"{channels.most_common(1)[0][0]} songs" if channels else "top hindi songs")
    results = _search(query if len(query.split()) > 1 or "song" in query.lower() else f"{query} song")
    music = [r for r in results if 60 <= (r["duration"] or 0) <= 20 * 60] or results  # skip shorts and 3-hour mixes
    if not music:
        raise RuntimeError(f"I couldn't find a song for {query}.")
    track = _resolve(music[0])
    player.queue = music[1:]
    player.play(track)
    _remember(track)
    by = f" by {track['channel']}" if track.get("channel") else ""
    return f"Playing {_spoken(track['title'])}{by}."


def control(action: str, level: int | None = None) -> str:
    """Pause/resume/stop/next/volume for the song this player is playing."""
    if not player.active and action not in ("stop",):
        raise RuntimeError("No song is playing.")
    if action in ("pause", "play_pause") and not player.paused.is_set():
        player.paused.set()
        return "Paused the song."
    if action in ("play", "resume", "play_pause"):
        player.paused.clear()
        return "Playing."
    if action == "pause":
        return "It's already paused."
    if action == "stop":
        player.queue = []
        player.stop()
        return "Stopped the music."
    if action == "next":
        if not player.queue:
            raise RuntimeError("That was the last song from that search - ask me for another.")
        track = _resolve(player.queue.pop(0))
        player.play(track)
        _remember(track)
        return f"Next: {_spoken(track['title'])}."
    if action in ("volume_up", "volume_down", "volume_set", "mute", "unmute"):
        if action == "volume_set":
            player.volume = max(0, min(int(level or 0), 100)) / 100
        elif action == "mute":
            player.volume = 0
        elif action == "unmute":
            player.volume = max(player.volume, 0.5)
        else:
            player.volume = max(0.0, min(1.0, player.volume + (0.1 if action == "volume_up" else -0.1)))
        return f"Music volume is now {round(player.volume * 100)}%."
    if action == "now_playing":
        t = player.track or {}
        state = "paused" if player.paused.is_set() else "playing"
        return f"{_spoken(t.get('title', 'Nothing'))} by {t.get('channel', 'unknown')}, {state}."
    raise RuntimeError(f"I can't do '{action}' with the music player.")


@registry.register(
    name="music_control",
    description=("Control the song Vaayu is playing through the speakers (play_music): pause, resume, stop, next, "
                 "volume_up, volume_down, volume_set (level 0-100), mute, unmute, now_playing."),
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "enum": ["pause", "resume", "stop", "next", "volume_up", "volume_down",
                                               "volume_set", "mute", "unmute", "now_playing"]},
        "level": {"type": "integer"}}, "required": ["action"]},
)
def music_control(action: str, level: int | None = None) -> str:
    return control(action, level)
