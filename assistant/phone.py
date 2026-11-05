"""
Vaayu from the phone (#13; user, 2026-11-05: "I also want to talk to him on my phone and control my laptop with him").

A Telegram bot the user makes (free, works anywhere, voice notes included), run inside Vaayu: a message typed or
spoken to the bot goes through the same brain and the same Guardian rules as a spoken command, and the answer comes
back as a message - not said aloud at home. "/screen" sends a screenshot. Anything outward asks "yes?" in the chat.

Only ONE Telegram chat may use it: the first one to send "/start <pairing code>" (the code is in
data/telegram.json and Vaayu's log) - every other chat is ignored without a word. The bot token stays in
data/telegram.json (never committed). Plain urllib long-polling: no new package, no open port on the laptop.
"""

import io
import json
import queue
import random
import threading
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "data" / "telegram.json"
API = "https://api.telegram.org"
CONFIRM_SECONDS = 90


def load_config(path: Path = CONFIG) -> dict:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if cfg.get("token") and not cfg.get("chat_id") and not cfg.get("pair_code"):
        cfg["pair_code"] = f"{random.SystemRandom().randrange(10**5, 10**6)}"
        path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return cfg


class TelegramBridge:
    def __init__(self, handle, stt=None, screenshot=None, config_path: Path = CONFIG, call=None, log=print):
        """handle(text, confirm, audio) -> reply text (Orchestrator.handle_remote)."""
        self.path, self.handle, self.stt, self.screenshot, self.log = config_path, handle, stt, screenshot, log
        self.cfg = load_config(config_path)
        self._call = call or self._http
        self._answers: "queue.Queue[str]" = queue.Queue()
        self._waiting = threading.Event()  # a "yes?" is out: the next message answers it
        self._busy = threading.Lock()

    # --- Telegram API -------------------------------------------------------------------------------------------
    def _http(self, method: str, files: dict | None = None, **params) -> dict:
        url = f"{API}/bot{self.cfg['token']}/{method}"
        if files:
            boundary = uuid.uuid4().hex
            body = io.BytesIO()
            for k, v in params.items():
                body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
            for k, (name, data) in files.items():
                body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\n"
                           f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
            body.write(f"--{boundary}--\r\n".encode())
            req = urllib.request.Request(url, body.getvalue(), {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        else:
            req = urllib.request.Request(url, json.dumps(params).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=float(params.get("timeout", 0)) + 30) as r:  # long poll + margin
            return json.load(r)

    def send(self, text: str) -> None:
        if self.cfg.get("chat_id"):
            self._call("sendMessage", chat_id=self.cfg["chat_id"], text=text[:4000])

    # --- messages -----------------------------------------------------------------------------------------------
    def on_update(self, update: dict) -> None:
        msg = update.get("message") or {}
        chat, text = (msg.get("chat") or {}).get("id"), (msg.get("text") or "").strip()
        if not chat:
            return
        if not self.cfg.get("chat_id"):
            if self.cfg.get("pair_code") and text in (f"/start {self.cfg['pair_code']}", self.cfg["pair_code"]):
                self.cfg["chat_id"] = chat
                self.cfg.pop("pair_code", None)
                self.path.write_text(json.dumps(self.cfg, indent=1), encoding="utf-8")
                self.send("Paired. It's Vaayu - tell me what to do on your laptop. /screen shows you the screen.")
            return  # not paired: never a word to a stranger
        if chat != self.cfg["chat_id"]:
            return
        if self._waiting.is_set() and text:
            self._answers.put(text)
            return
        if text == "/screen":
            threading.Thread(target=self._send_screen, daemon=True).start()
            return
        if text in ("/start", "/help"):
            self.send("Type or send a voice note - I'll do it on the laptop. /screen shows you the screen.")
            return
        voice = msg.get("voice") or msg.get("audio")
        if not text and not voice:
            return
        threading.Thread(target=self._do, args=(text, voice), daemon=True, name="phone-command").start()

    def _do(self, text: str, voice: dict | None) -> None:
        with self._busy:
            audio = None
            try:
                if voice:
                    audio = self._voice_audio(voice["file_id"])
                    text = self.stt.transcribe(audio) if self.stt else ""
                    if not text.strip():
                        self.send("I couldn't make out that voice note.")
                        return
                    self.send(f"Heard: {text}")
                reply = self.handle(text, self.confirm, audio)
            except Exception as exc:
                reply = f"Sorry, that went wrong: {exc}"
            self.send(reply or "Done.")

    def confirm(self, name: str, arguments: dict) -> bool:
        from assistant.control_phrases import is_affirmative
        from assistant.tools.registry import registry
        while not self._answers.empty():
            self._answers.get_nowait()
        self._waiting.set()
        try:
            self.send(registry.get_confirmation_prompt(name, arguments) + " (yes / no)")
            answer = self._answers.get(timeout=CONFIRM_SECONDS)
        except queue.Empty:
            self.send("No answer - not doing it.")
            return False
        finally:
            self._waiting.clear()
        return is_affirmative(answer)

    def _voice_audio(self, file_id: str):
        from faster_whisper import decode_audio
        path = self._call("getFile", file_id=file_id)["result"]["file_path"]
        with urllib.request.urlopen(f"{API}/file/bot{self.cfg['token']}/{urllib.parse.quote(path)}", timeout=60) as r:
            return decode_audio(io.BytesIO(r.read()), sampling_rate=16000)

    def _send_screen(self) -> None:
        try:
            img = self.screenshot()
            img.thumbnail((1600, 1600))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            self._call("sendPhoto", files={"photo": ("screen.jpg", buf.getvalue())}, chat_id=self.cfg["chat_id"])
        except Exception as exc:
            self.send(f"I couldn't take the screenshot: {exc}")

    # --- loop ---------------------------------------------------------------------------------------------------
    def run_forever(self) -> None:
        import time
        offset = 0
        while True:
            try:
                updates = self._call("getUpdates", timeout=55, offset=offset, allowed_updates=["message"]).get("result", [])
                for u in updates:
                    offset = max(offset, u["update_id"] + 1)
                    self.on_update(u)
            except Exception as exc:  # no internet for a while - keep trying, quietly
                self.log(f"Telegram: {exc!r}")
                time.sleep(10)

    def start(self) -> threading.Thread | None:
        if not self.cfg.get("token"):
            return None
        if not self.cfg.get("chat_id"):
            self.log(f"Telegram: send '/start {self.cfg.get('pair_code')}' to your bot to pair your phone.")
        thread = threading.Thread(target=self.run_forever, daemon=True, name="telegram")
        thread.start()
        return thread
