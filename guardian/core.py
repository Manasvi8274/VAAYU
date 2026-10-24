"""
The guardian supervisor: starts Vaayu as a child process, restarts it if it
crashes, and stops it completely on any kill path. Deliberately imports
nothing from assistant/ - it must keep working even if Vaayu's own code is
broken by a bad self-upgrade.

"Protected against accidents, not guaranteed" (agreed trade-off): Vaayu runs
as the same Windows user, so a determined process could still touch these
files - the protection is that Vaayu's self-upgrade gate refuses changes to
guardian/ (rules.PROTECTED_PATHS), and that this process doesn't depend on
Vaayu's code to stop it.
"""

import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
PID_FILE = DATA / "guardian.pid"
LOG_FILE = DATA / "guardian.log"

VAAYU_MODULE = "assistant.voice_main"
KILL_HOTKEY = "Ctrl+Alt+Shift+F12"
# Crash restarts: at most this many inside the window, then give up rather
# than loop forever on a broken build.
MAX_RESTARTS = 3
RESTART_WINDOW_SECONDS = 600
# Kill-phrase listener: how much louder than room noise a frame must be to
# count as speech. Calibration knob - raise it in a noisy room, lower it if
# "Vaayu die" from across the room isn't picked up.
_SPEECH_OVER_FLOOR = 2.5
_KILL_MODEL = "base"  # "tiny" misheard the TTS-through-speaker test badly; base is already cached for Vaayu

log = logging.getLogger("guardian")

# --- "Vaayu die" ---------------------------------------------------------------
# Whisper hears the made-up name many ways (PLAN.md: "Vaayu" never transcribed
# reliably), so the name side is a list of observed/likely spellings. The name
# must come right before "die" so ordinary speech with "die" in it can't fire.
# Spellings seen live include "Veayu" and "Vailu", so beyond the listed ones
# the name is matched by its sound shape: v/w/b + vowels + optional y/l/h + o/u.
_NAME = (r"(?:vaa?yu|vayu|vaayoo|vaiyu|vaayou|vayou|waa?yu|bayu|bhayu|vahu|vio|viu|vayo|va yu|vai yu|vaayu's|"
         r"[vwb]h?[aeiy]{1,3}[ylh]?[ou]{1,2})")
_DIE = r"(?:die|dye|dai|dies|died)"
_KILL_RE = re.compile(rf"\b{_NAME}\s+{_DIE}\b")


def is_kill_phrase(text: str) -> bool:
    t = re.sub(r"[^a-z' ]+", " ", (text or "").lower())
    return bool(_KILL_RE.search(re.sub(r"\s+", " ", t)))


def should_restart(crash_times: deque, now: float) -> bool:
    """Record a crash at `now`; True if Vaayu may be restarted again."""
    crash_times.append(now)
    while crash_times and now - crash_times[0] > RESTART_WINDOW_SECONDS:
        crash_times.popleft()
    return len(crash_times) <= MAX_RESTARTS


# --- processes -------------------------------------------------------------------

def _python_exe() -> str:
    # pythonw: no console window for Vaayu when the guardian runs at login.
    w = Path(sys.executable).with_name("pythonw.exe")
    return str(w if w.exists() else sys.executable)


def start_vaayu() -> subprocess.Popen:
    env = dict(os.environ)
    # Programs Vaayu launches run with Vaayu's own (non-admin) rights instead
    # of auto-requesting elevation. Elevation still always needs a human click
    # on the UAC secure desktop, which no user-level program can press.
    env["__COMPAT_LAYER"] = "RunAsInvoker"
    env["VAAYU_GUARDED"] = "1"
    return subprocess.Popen([_python_exe(), "-m", VAAYU_MODULE], cwd=REPO, env=env)


def kill_tree(pid: int) -> None:
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = root.children(recursive=True) + [root]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _gone, alive = psutil.wait_procs(procs, timeout=3)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _cmdline(p: psutil.Process) -> str:
    try:
        return " ".join(p.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return ""


def _guardian_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        return pid if "guardian" in _cmdline(psutil.Process(pid)) else None
    except psutil.NoSuchProcess:
        return None


def _vaayu_pids() -> list[int]:
    return [p.pid for p in psutil.process_iter() if VAAYU_MODULE in _cmdline(p)]


def kill_all() -> str:
    """The desktop shortcut's action: stop the guardian and every Vaayu
    process, whether or not the guardian itself is still responsive."""
    killed = []
    gpid = _guardian_pid()
    if gpid and gpid != os.getpid():
        kill_tree(gpid)
        killed.append("guardian")
    for pid in _vaayu_pids():
        kill_tree(pid)
        killed.append("vaayu")
    PID_FILE.unlink(missing_ok=True)
    if killed:
        _speak("Vaayu is off.")
    return f"Stopped: {', '.join(killed)}." if killed else "Vaayu wasn't running."


def status() -> str:
    gpid, vpids = _guardian_pid(), _vaayu_pids()
    return f"guardian: {'running (pid %d)' % gpid if gpid else 'not running'}; vaayu: " + (
        f"running (pid {', '.join(map(str, vpids))})" if vpids else "not running")


def _speak(text: str) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:  # a voice failure must never block a kill
        log.exception("speak failed")


# --- kill triggers (each on its own daemon thread) --------------------------------

def _hotkey_loop(trigger) -> None:
    import win32con
    import win32gui
    mod_norepeat = 0x4000
    try:
        win32gui.RegisterHotKey(None, 1, win32con.MOD_CONTROL | win32con.MOD_ALT | win32con.MOD_SHIFT | mod_norepeat,
                                win32con.VK_F12)
    except Exception:
        log.exception("could not register %s (another program may own it)", KILL_HOTKEY)
        return
    log.info("kill hotkey %s registered", KILL_HOTKEY)
    while True:
        msg = win32gui.GetMessage(None, 0, 0)
        if msg[1][1] == win32con.WM_HOTKEY:
            trigger("hotkey")
            return


class Segmenter:
    """Cuts a stream of 30 ms int16 frames into utterances. Pure logic (no
    mic) so it can be tested with recorded audio - the laptop's mic array
    cancels its own speaker output, so speaker->mic self-tests can't stand
    in for a real voice.

    A frame is speech only if webrtcvad says so AND it's louder than the
    room's noise floor: webrtcvad alone called room noise "speech", so an
    utterance never ended (found live: 1 of 5 kill phrases got through).
    ~300 ms of pre-roll keeps the soft "V" of Vaayu, which sits under the
    loudness gate and was being cut off."""

    RATE = 16000
    FRAME = 480

    def __init__(self, vad, floor: float):
        self.vad, self.floor = vad, max(floor, 1.0)
        self.voiced: list[bytes] = []
        self.silent = 0
        self.preroll: deque = deque(maxlen=10)
        self.skipping = False  # inside an over-long utterance: ignore until a real pause

    @staticmethod
    def rms(frame: bytes) -> float:
        import numpy as np
        return float(np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2)))

    def feed(self, frame: bytes) -> bytes | None:
        """Returns a finished utterance short enough to be the kill phrase."""
        level = self.rms(frame)
        is_speech = level > self.floor * _SPEECH_OVER_FLOOR and self.vad.is_speech(frame, self.RATE)
        done = None
        if self.skipping:
            self.silent = 0 if is_speech else self.silent + 1
            if self.silent >= 20:
                self.skipping, self.silent = False, 0
            self.preroll.append(frame)
            return None
        if is_speech:
            if not self.voiced:
                self.voiced = list(self.preroll)
            self.voiced.append(frame)
            self.silent = 0
        elif self.voiced:
            self.voiced.append(frame)
            self.silent += 1
            if self.silent >= 20:  # 600 ms of silence ends the utterance
                audio, self.voiced, self.silent = b"".join(self.voiced), [], 0
                if 0.3 <= len(audio) / (2 * self.RATE) <= 4.0:
                    done = audio
        else:
            self.floor = 0.98 * self.floor + 0.02 * level  # follow the room when nobody's talking
        if len(self.voiced) > 200:  # >6 s of speech: not the kill phrase - skip the rest of it
            self.voiced, self.silent, self.skipping = [], 0, True
        self.preroll.append(frame)
        return done


def transcribe_short(model, audio: bytes) -> str:
    import numpy as np
    samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(samples))) or 1.0
    samples *= min(0.9 / peak, 20.0)  # quiet mic (35% here): boost before Whisper
    segments, _ = model.transcribe(samples, language="en", beam_size=1, initial_prompt="Vaayu die.")
    return " ".join(s.text for s in segments)


def _phrase_loop(trigger, stop: threading.Event) -> None:
    """Independent "Vaayu die" listener: own mic stream, own Whisper model.
    Only short utterances are transcribed - the kill phrase is two words, so
    normal-length speech is skipped to keep CPU use low."""
    try:
        import numpy as np
        import sounddevice as sd
        import webrtcvad
        from faster_whisper import WhisperModel

        model = WhisperModel(_KILL_MODEL, device="cpu", compute_type="int8")
        frames: "queue.Queue[bytes]" = queue.Queue()
        with sd.RawInputStream(samplerate=Segmenter.RATE, blocksize=Segmenter.FRAME, dtype="int16", channels=1,
                               callback=lambda data, n, t, s: frames.put(bytes(data))):
            floor = float(np.median([Segmenter.rms(frames.get()) for _ in range(33)]))  # ~1 s of room
            seg = Segmenter(webrtcvad.Vad(2), floor)
            log.info('"Vaayu die" listener running (noise floor %.0f)', floor)
            while not stop.is_set():
                try:
                    audio = seg.feed(frames.get(timeout=0.5))
                except queue.Empty:
                    continue
                if audio and is_kill_phrase(text := transcribe_short(model, audio)):
                    log.info("kill phrase heard: %r", text)
                    trigger("phrase")
                    return
    except Exception:
        log.exception('"Vaayu die" listener stopped - hotkey and shortcut still work')


def _uac_watch(stop: threading.Event, vaayu_alive) -> None:
    """Logs any UAC prompt (consent.exe) that appears while Vaayu runs, so an
    elevation attempt is visible in data/guardian.log."""
    seen = set()
    while not stop.wait(2):
        for p in psutil.process_iter(["name"]):
            if (p.info.get("name") or "").lower() == "consent.exe" and p.pid not in seen and vaayu_alive():
                seen.add(p.pid)
                log.warning("UAC elevation prompt appeared while Vaayu was running (pid %d)", p.pid)


# --- main loop ---------------------------------------------------------------------

def run() -> None:
    DATA.mkdir(exist_ok=True)
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("faster_whisper", "httpx", "comtypes"):  # keep guardian.log to the guardian's own events
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if (other := _guardian_pid()) and other != os.getpid():
        log.info("guardian already running (pid %d) - not starting a second one", other)
        return
    PID_FILE.write_text(str(os.getpid()))

    kill = threading.Event()
    reason = {}

    def trigger(source: str) -> None:
        reason.setdefault("source", source)
        kill.set()

    proc = None
    threading.Thread(target=_hotkey_loop, args=(trigger,), daemon=True).start()
    threading.Thread(target=_phrase_loop, args=(trigger, kill), daemon=True).start()
    threading.Thread(target=_uac_watch, args=(kill, lambda: proc is not None and proc.poll() is None),
                     daemon=True).start()

    crashes: deque = deque()
    try:
        while not kill.is_set():
            proc = start_vaayu()
            log.info("started Vaayu (pid %d)", proc.pid)
            while proc.poll() is None and not kill.wait(0.5):
                pass
            if kill.is_set():
                break
            if proc.returncode == 0:  # "shutdown Vaayu" - an intended exit
                log.info("Vaayu exited normally - guardian stopping too")
                return
            log.warning("Vaayu crashed (exit code %s)", proc.returncode)
            if not should_restart(crashes, time.time()):
                log.error("Vaayu crashed %d times in %d s - not restarting", MAX_RESTARTS + 1, RESTART_WINDOW_SECONDS)
                _speak("Vaayu keeps crashing, so I've stopped it. Check the guardian log.")
                return
            time.sleep(3)
        log.info("kill requested via %s", reason.get("source"))
        if proc is not None:
            kill_tree(proc.pid)
        _speak("Vaayu is off.")
    finally:
        if proc is not None and proc.poll() is None:
            kill_tree(proc.pid)
        PID_FILE.unlink(missing_ok=True)
