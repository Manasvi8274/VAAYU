"""
Seeing the screen (user, 2026-11-05: "what's on my screen", "click the blue button", "read this error").

A screenshot (scaled to 1280 px wide) is looked at by Claude through the user's Pro login - measured 6.4 s, and it
read the page right ("Microsoft Edge ... a Google search ... comparing camera phones"). The screenshot leaves the
laptop for that, like a "think about" question does; LOCAL_ONLY = True keeps it here. The local fallback is
Qwen2.5-VL 3B in Ollama (models/ollama): private, 3 s once loaded, but ~40 s to load on the 4 GB GPU next to the
everyday brain - so it is the fallback, not the default.

Clicking: the same look returns where the thing is; the mouse is moved there and clicked (plain Win32, no new
package). What is clicked is judged by the Guardian like any click ("click Pay" is refused - action_rules).
"""

import base64
import ctypes
import io
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from assistant.tools.registry import registry

LOCAL_ONLY = False      # calibration knob: True = the screen never leaves the laptop (slower)
LOCAL_MODEL = "qwen2.5vl:3b"
AFTER_LOCAL_LOOK = None  # set by the orchestrator: re-check that the brain is whole on the GPU
_VISION_CTX = 4096
WIDTH = 1280
_DIR = Path(tempfile.gettempdir()) / "vaayu-screen"  # only the screenshot is in here, deleted after each look


def _grab_full():
    from PIL import ImageGrab
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # real pixels, so a click lands where it was seen
    except (AttributeError, OSError):
        pass
    return ImageGrab.grab()


def _grab():
    img = _grab_full()
    full = img.size
    img.thumbnail((WIDTH, WIDTH))
    return img, full


def _ask_claude(img, prompt: str) -> str | None:
    from assistant.tools import think
    claude = shutil.which("claude")
    if not claude:
        return None
    _DIR.mkdir(exist_ok=True)
    path = _DIR / f"screen-{int(time.time() * 1000)}.png"
    img.save(path)
    try:
        run = subprocess.run([claude, "-p", "--model", think.MODEL, "--output-format", "text",
                              *think._LEAN[:-2], "--tools", "Read", "--allowedTools", "Read"],
                             input=f"Look at the screenshot {path.name} in the current folder (only that file). "
                                   f"{prompt}",
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=think.TIMEOUT,
                             cwd=str(_DIR), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        path.unlink(missing_ok=True)
    answer = (run.stdout or "").strip()
    return answer if run.returncode == 0 and answer and not think._LIMIT.search(answer) else None


def _ask_local(img, prompt: str) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    body = {"model": LOCAL_MODEL, "stream": False, "keep_alive": 0,  # unloaded at once: the brain needs the GPU
            "options": {"num_ctx": _VISION_CTX, "num_predict": 200},  # its own model: no brain reload to avoid
            "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(buf.getvalue()).decode()]}]}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return (json.load(r)["message"].get("content") or "").strip()
    except OSError as exc:
        raise RuntimeError("I couldn't look at the screen just now - my vision model isn't answering.") from exc
    finally:
        if AFTER_LOCAL_LOOK:
            AFTER_LOCAL_LOOK()  # the vision model pushed part of the brain to the CPU - put it back (orchestrator)


def look(prompt: str, img=None) -> str:
    img = img if img is not None else _grab()[0]
    answer = None if LOCAL_ONLY else _ask_claude(img, prompt)
    return answer if answer is not None else _ask_local(img, prompt)


@registry.register(
    name="look_at_screen",
    description="Look at the user's screen and answer about it: what's on it, read an error or text, what an app "
                "or page shows.",
    parameters={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
)
def look_at_screen(question: str = "What's on the screen?") -> str:
    answer = look(f"Answer for a voice assistant, in 1-3 short spoken sentences, no markdown: {question}")
    return re.sub(r"[*#`]+", "", answer)


_POINT = re.compile(r"\{[^{}]*\}")


def _click(x: int, y: int) -> None:
    user32 = ctypes.windll.user32
    user32.SetCursorPos(int(x), int(y))
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up


def find_on_screen(target: str, img=None, full=None) -> tuple[int, int] | None:
    """Where `target` is, in real screen pixels - or None if it isn't visible."""
    if img is None:
        img, full = _grab()
    w, h = img.size
    answer = look(f"Find this on the screen: \"{target}\". The image is {w}x{h} pixels. Reply with JSON only: "
                  '{"x": <pixel x of its centre>, "y": <pixel y>} or {"none": "<why>"} if it is not visible.', img)
    m = _POINT.search(answer or "")
    try:
        point = json.loads(m.group(0)) if m else {}
    except ValueError:
        point = {}
    if "x" not in point or "y" not in point:
        return None
    sx, sy = (full[0] / w, full[1] / h) if full else (1.0, 1.0)
    return round(float(point["x"]) * sx), round(float(point["y"]) * sy)


@registry.register(
    name="click_on_screen",
    description="Click something the user describes on the screen ('click the blue Submit button', 'click the "
                "second video') - anywhere, in any app.",
    parameters={"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]},
)
def click_on_screen(target: str) -> str:
    from assistant.tools import screen_memory as sm
    img = _grab_full()
    app, g = sm.foreground_app(), sm.gray(img)
    where = sm.recall(app, target, g)  # seen here before: ~0.1 s, no screenshot leaves the laptop
    if where is None:
        thumb = img.copy()
        thumb.thumbnail((WIDTH, WIDTH))
        where = find_on_screen(target, thumb, img.size)
        if where is None:
            raise RuntimeError(f"I can't see {target} on the screen.")
        sm.remember(app, target, g, *where)
    _click(*where)
    return f"Clicked {target}."
