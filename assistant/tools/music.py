"""
Songs straight to the speakers - no browser, no YouTube page.

yt-dlp finds the song (a YouTube search, done by Vaayu himself) and gives
its audio stream; PyAV (already here for faster-whisper) decodes it, and
sounddevice plays it. Measured: ~4 s to find a song, then sound in ~0.2 s.
Pause / resume / stop / next / volume / skip to 1:10 / "play 15 seconds"
all work by voice (control_media routes here while this player is active),
"who sang this song?" asks about whatever is playing, and every song
played feeds the taste history (youtube_taste) that recommendations use.
"""

import re
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
        self.start_at = 0.0       # where this run of the decoder started, in the song (s)
        self._played = 0          # frames sent to the speakers since then
        self.stop_at: float | None = None  # "play it for 15 seconds": pause at this point in the song

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def position(self) -> float:
        return self.start_at + self._played / _RATE

    def play(self, track: dict, start_at: float = 0.0) -> None:
        self.stop()
        self._stop.clear()
        self.paused.clear()
        self.track, self.start_at, self._played = track, max(0.0, start_at), 0
        self._thread = threading.Thread(target=self._run, args=(track,), daemon=True, name="music")
        self._thread.start()

    def seek(self, seconds: float) -> None:
        """Jump within the current song: a new decoder run from there (the stream supports ranges)."""
        if not self.track:
            raise RuntimeError("No song is playing.")
        stop_at, was_paused = self.stop_at, self.paused.is_set()
        self.play(self.track, start_at=seconds)
        self.stop_at = stop_at if stop_at and stop_at > seconds else None
        if was_paused:
            self.paused.set()

    def stop(self) -> None:
        self._stop.set()
        self.paused.clear()
        if self._thread is not None and self._thread is not threading.current_thread():
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
            self._played += n
            if self.stop_at is not None and self.position >= self.stop_at:
                self.stop_at = None
                self.paused.set()  # "play 15 seconds" done - paused, so "resume" carries on

    def _run(self, track: dict) -> None:
        import av
        import sounddevice as sd
        finished = False
        try:
            container = av.open(track["url"], options={"reconnect": "1", "reconnect_streamed": "1"}, timeout=15)
            stream = container.streams.audio[0]
            if self.start_at > 0:
                container.seek(int(self.start_at * 1_000_000), any_frame=False, backward=True)  # microseconds
            resampler = av.AudioResampler(format="flt", layout="stereo", rate=_RATE)
            with sd.OutputStream(samplerate=_RATE, channels=_CHANNELS, dtype="float32", blocksize=_BLOCK,
                                 callback=self._callback):
                for frame in container.decode(stream):
                    if frame.time is not None and frame.time < self.start_at - 0.05:
                        continue  # seek lands on the keyframe before the target - skip up to it
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
                finished = not self._stop.is_set()
            container.close()
        except Exception as exc:
            from assistant import interaction_log
            interaction_log.log_error("music", exc, title=track.get("title"))
            # Real log: "403 Forbidden" halfway through - YouTube's stream link expired. Get a fresh link
            # and carry on from the same second (twice at most, so a dead song can't loop forever).
            if not self._stop.is_set() and track.get("_retries", 0) < 2:
                resume_at = self.position
                threading.Thread(target=_resume_fresh, args=(track, resume_at), daemon=True).start()
                return
        finally:
            with self.lock:
                self._buf = np.zeros((0, _CHANNELS), np.float32)
        if finished and self.queue:  # ended on its own: the next one, like autoplay
            threading.Thread(target=_autoplay_next, daemon=True).start()  # not from this thread: play() joins it


player = _Player()


def _resume_fresh(track: dict, at: float) -> None:
    try:
        fresh = _resolve(track) | {"_retries": track.get("_retries", 0) + 1}
        stop_at, paused = player.stop_at, player.paused.is_set()
        player.play(fresh, start_at=at)
        player.stop_at = stop_at
        if paused:
            player.paused.set()
    except Exception:
        pass


def _autoplay_next() -> None:
    try:
        _play_first(player.queue)
    except Exception:
        pass


def _ydl_opts(**extra) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": "bestaudio[ext=m4a]/bestaudio/best"}
    if shutil.which("node"):  # YouTube needs a JS runtime for some formats
        opts["js_runtimes"] = {"node": {}}
    return opts | extra


_ANY_MUSIC = {"some music", "music", "some songs", "a song", "songs", "koi gaana", "kuch gaane", "some", "koi", "kuch"}
# Spoken filler that sank real searches (log): "bang bang which is from the bollywood movie" played a FULL MOVIE,
# "which is named bang bang ... starring rithik roshan" played the trailer, "a new song by Eminem" played Kid Rock.
_FILLER = re.compile(r"\b(?:which|that) (?:is|was)(?: from| in| named| called| sung by| by)?\b|\b(?:the )?song (?:named|called)\b|"
                     r"\b(?:named|called|starring|featuring|from the movie|from the film|from the|of the movie|"
                     r"the movie|the film|bollywood movie|hindi movie|movie|film)\b|^(?:a|the|some|any|that)\s+", re.I)
_NOT_MUSIC = re.compile(r"\b(?:trailer|teaser|full movie|full film|movie scene|scene|reaction|review|interview|explained|"
                        r"behind the scenes|making of|news|podcast|shorts?)\b", re.I)


def _clean_query(query: str) -> str:
    q = _FILLER.sub(" ", query)
    q = re.sub(r"\b(?:koi|ek|accha|achha|acha|badhiya|badiya|wadya|sa|si|wala|wali|wale|mujhe|mere liye)\b", " ", q, flags=re.I)
    q = re.sub(r"\b(?:new|latest|newest)\s+(?:song|songs|track)\s+(?:by|of|from)\s+(.+)", r"\1 latest song", q, flags=re.I)
    q = re.sub(r"\b(?:song|songs|track)\s+(?:by|of|from)\s+", " ", q, flags=re.I)
    q = re.sub(r"\brithik\b", "hrithik", q, flags=re.I)  # how Whisper writes Hrithik
    return re.sub(r"\s+", " ", q).strip()


def _search(query: str, n: int = 8) -> list[dict]:
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
                    "channel": info.get("channel") or info.get("uploader") or track.get("channel", ""),
                    "duration": info.get("duration") or track.get("duration") or 0}


def _spoken(title: str) -> str:
    """"Kesariya - Brahmastra | Ranbir Kapoor, Alia Bhatt | Pritam | ... | 4K" -> "Kesariya - Brahmastra";
    "Bang Bang Title Track Full Video" -> "Bang Bang"."""
    t = re.split(r"\s[|(\[]|\s\|\s", title)[0].strip() or title
    t = re.sub(r"\b(?:title track|full video(?: song)?|official (?:music )?video|official audio|lyrical(?: video)?|"
               r"lyric video|lyrics|video song|full song|audio song|hd|4k)\b", " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip(" -") or title


def _remember(track: dict) -> None:
    from assistant.tools import youtube_taste
    youtube_taste.record(track["id"], track["title"], track.get("channel", ""))


def _play_first(candidates: list[dict], seconds: float | None = None) -> dict:
    """Play the first candidate that YouTube will actually stream (live: "This video is not available")."""
    last_error = None
    while candidates:
        cand = candidates.pop(0)
        try:
            track = _resolve(cand)
        except Exception as exc:
            last_error = exc
            continue
        player.queue = candidates
        player.play(track)
        player.stop_at = seconds
        _remember(track)
        return track
    raise RuntimeError(f"None of the songs I found would play{f' ({last_error})' if last_error else ''}.")


def _clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


@registry.register(
    name="play_music",
    description=("Play a song or music through the laptop speakers WITHOUT opening YouTube or a browser - Vaayu "
                 "searches for it himself and streams only the audio. Use for 'play <song/artist> song', "
                 "'<song> gaana chalao/bajao', 'play some music'. seconds: play only that long (e.g. 'play kesariya "
                 "for 15 seconds'). For videos, or when the user says 'on youtube', use play_video_on_youtube."),
    parameters={"type": "object", "properties": {"query": {"type": "string", "description": "Song, artist or genre."},
                                                 "seconds": {"type": "integer"}},
                "required": ["query"]},
)
def play_music(query: str, seconds: int | None = None) -> str:
    query = _clean_query(query.strip())
    if not query or query.lower() in _ANY_MUSIC or query.lower().startswith("something"):  # "play some music": taste
        from assistant.tools import youtube_taste
        channels, topics = youtube_taste.profile()
        query = (f"{channels.most_common(1)[0][0]} songs" if channels else "top hindi songs")
    results = _search(query if len(query.split()) > 1 or "song" in query.lower() else f"{query} song")
    songs = [r for r in results if 60 <= (r["duration"] or 0) <= 12 * 60 and not _NOT_MUSIC.search(r["title"])]
    songs = songs or [r for r in results if not _NOT_MUSIC.search(r["title"])] or results
    if not songs:
        raise RuntimeError(f"I couldn't find a song for {query}.")
    track = _play_first(songs, seconds)
    by = f" by {track['channel']}" if track.get("channel") else ""
    for_ = f" for {seconds} seconds" if seconds else ""
    return f"Playing {_spoken(track['title'])}{by}{for_}."


def control(action: str, level: int | None = None) -> str:
    """Pause/resume/stop/next/volume/seek for the song this player is playing."""
    if not player.track and action != "stop":
        raise RuntimeError("No song is playing.")
    if action in ("pause", "play_pause") and not player.paused.is_set():
        player.paused.set()
        return "Paused the song."
    if action in ("play", "resume", "play_pause"):
        if not player.active:  # it ended (or a "play 15 seconds" stopped it): start it again
            player.play(player.track)
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
        track = _play_first(player.queue)
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
    if action in ("seek_to", "seek_forward", "seek_back", "restart"):
        duration = (player.track or {}).get("duration") or 0
        target = {"seek_to": float(level or 0), "restart": 0.0,
                  "seek_forward": player.position + float(level or 10),
                  "seek_back": player.position - float(level or 10)}[action]
        if duration and target >= duration - 1:
            raise RuntimeError(f"The song is only {_clock(duration)} long.")
        player.seek(max(0.0, target))
        return f"Playing from {_clock(player.position)}."
    if action == "play_for":
        player.stop_at = player.position + float(level or 15)
        player.paused.clear()
        if not player.active:
            player.play(player.track, start_at=player.position)
            player.stop_at = player.start_at + float(level or 15)
        return f"Playing {int(level or 15)} seconds."
    if action == "now_playing":
        t = player.track or {}
        state = "paused" if player.paused.is_set() or not player.active else "playing"
        at = f", at {_clock(player.position)} of {_clock(t['duration'])}" if t.get("duration") else ""
        return f"{_spoken(t.get('title', 'Nothing'))} by {t.get('channel', 'unknown')}, {state}{at}."
    raise RuntimeError(f"I can't do '{action}' with the music player.")


@registry.register(
    name="music_control",
    description=("Control the song Vaayu is playing through the speakers (play_music): pause, resume, stop, next, "
                 "volume_up, volume_down, volume_set (level 0-100), mute, unmute, seek_to (level = seconds into the "
                 "song, e.g. 70 for 1:10), seek_forward / seek_back (level = seconds), restart, play_for (level = "
                 "seconds to play then pause), now_playing."),
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "enum": ["pause", "resume", "stop", "next", "volume_up", "volume_down",
                                               "volume_set", "mute", "unmute", "seek_to", "seek_forward", "seek_back",
                                               "restart", "play_for", "now_playing"]},
        "level": {"type": "integer"}}, "required": ["action"]},
)
def music_control(action: str, level: int | None = None) -> str:
    return control(action, level)


def owns_controls() -> bool:
    """True when this player is what "pause / skip to 1:10 / volume 40" should act on: a song of ours is
    loaded and no YouTube video is open in the browser. Real log: "pause the video" while a song played
    here answered "I can't control the browser's tabs"."""
    if not player.active:
        return False
    try:
        from assistant.tools import youtube_player
        return youtube_player.youtube_tab(require_watch=True) is None
    except Exception:
        return True  # no controllable browser at all


# youtube_control / video-control actions -> (music action, level)
FROM_VIDEO = {
    "play": lambda v: ("play", None), "pause": lambda v: ("pause", None), "toggle_play": lambda v: ("play_pause", None),
    "volume_set": lambda v: ("volume_set", int(float(v))), "volume_up": lambda v: ("volume_up", None),
    "volume_down": lambda v: ("volume_down", None), "mute": lambda v: ("mute", None), "unmute": lambda v: ("unmute", None),
    "seek_forward": lambda v: ("seek_forward", int(float(v or 10))), "seek_back": lambda v: ("seek_back", int(float(v or 10))),
    "seek_to": lambda v: ("seek_to", int(float(v or 0))), "restart": lambda v: ("restart", None),
    "next": lambda v: ("next", None), "now_playing": lambda v: ("now_playing", None),
}


_THIS_SONG = re.compile(r"\b(?:this|the current|current|that) (?:song|track|music|gaana)\b|\b(?:ye|yeh|is|iss) gaan[ae]\b|"
                        r"\bthe song (?:that'?s|that is|you'?re) playing\b", re.I)


@registry.register(
    name="ask_about_song",
    description="Answer a question about the song playing right now ('which movie is this song from?', 'who sang "
                "this song?') - looks it up online with the song's real title.",
    parameters={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
)
def ask_about_song(question: str):
    title = _current_title()
    if not title:
        raise RuntimeError("Nothing is playing, so I don't know which song you mean - say its name.")
    from assistant.tools.web_answer import answer_question
    return answer_question(_THIS_SONG.sub(f'the song "{title}"', question))


def _current_title() -> str | None:
    if player.track:
        return _spoken(player.track["title"])
    try:  # a song playing on YouTube in the browser
        from assistant.tools import youtube_player
        tab = youtube_player.youtube_tab(require_watch=True)
        if tab:
            return _spoken(youtube_player._ev(tab, f"{youtube_player._P}.getVideoData().title") or "") or None
    except Exception:
        pass
    return None
