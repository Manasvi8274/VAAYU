"""
Where things are on the screen, remembered (user, 2026-11-05: "recognise my button positions which rarely change -
only when he doesn't know, share the screenshot with Claude and update the positions").

After a thing is found (screen.find_on_screen - Claude, ~6 s), its spot and a small picture of it (48x48, grey) are
kept per app in data/screen_targets.json. Next time:
  1. the picture still at the remembered spot      -> click at once        (~0.1 s)
  2. it moved (window dragged, page scrolled)      -> found by its picture  (an FFT match over the screen, < 1 s)
  3. not on the screen                             -> Claude looks again and the memory is updated
Only numpy - no OpenCV.
"""

import base64
import io
import json
import re
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parent.parent.parent / "data" / "screen_targets.json"
HALF = 24              # the picture kept: 48x48 around the point
SAME = 0.92            # calibration knob: correlation for "it's still right there"
MOVED = 0.85           # calibration knob: correlation for "that's it, somewhere else"
_lock = threading.Lock()


def _key(app: str, target: str) -> str:
    words = re.findall(r"[a-z0-9]+", target.lower())
    return f"{app.lower()}|{' '.join(w for w in words if w not in ('the', 'a', 'an', 'on', 'button', 'icon'))}"


def foreground_app() -> str:
    try:
        import ctypes
        import psutil
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return psutil.Process(pid.value).name()
    except Exception:
        return "screen"


def gray(img) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _patch(g: np.ndarray, x: int, y: int) -> np.ndarray | None:
    if HALF <= x < g.shape[1] - HALF and HALF <= y < g.shape[0] - HALF:
        return g[y - HALF:y + HALF, x - HALF:x + HALF]
    return None


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def _search(g: np.ndarray, patch: np.ndarray) -> tuple[float, int, int]:
    """Best normalised match of `patch` anywhere in `g` (both at half size for speed): (score, x, y) in full pixels."""
    G, P = g[::2, ::2].astype(np.float64), patch[::2, ::2].astype(np.float64)  # float32 sums lost the detail
    ph, pw = P.shape
    P = P - P.mean()
    pn = float(np.sqrt((P * P).sum()))
    if pn < 1e-6:
        return 0.0, 0, 0
    H, W = G.shape
    fh, fw = H + ph, W + pw
    corr = np.fft.irfft2(np.fft.rfft2(G, (fh, fw)) * np.conj(np.fft.rfft2(P, (fh, fw))), (fh, fw))[:H - ph + 1, :W - pw + 1]
    # local energy of each window (integral images), so a bright area doesn't win by brightness
    S = np.pad(G, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    S2 = np.pad(G * G, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(T):
        return T[ph:, pw:] - T[:-ph, pw:] - T[ph:, :-pw] + T[:-ph, :-pw]
    n = ph * pw
    var = box(S2) - box(S) ** 2 / n
    flat = var < (0.05 * pn) ** 2  # a blank area can't be a button
    score = np.where(flat, 0.0, corr / (np.sqrt(np.maximum(var, 1e-9)) * pn))
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(score[iy, ix]), int(ix * 2 + pw), int(iy * 2 + ph)  # top-left at half size -> centre, full size


def _load() -> dict:
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def recall(app: str, target: str, g: np.ndarray) -> tuple[int, int] | None:
    """Where `target` is now, from memory - or None (never seen here, or not on the screen any more)."""
    entry = _load().get(_key(app, target))
    if not entry:
        return None
    from PIL import Image
    saved = np.asarray(Image.open(io.BytesIO(base64.b64decode(entry["patch"]))), dtype=np.float32)
    here = _patch(g, entry["x"], entry["y"])
    if here is not None and here.shape == saved.shape and _ncc(here, saved) >= SAME:
        return entry["x"], entry["y"]
    score, x, y = _search(g, saved)
    if score >= MOVED:
        remember(app, target, g, x, y)
        return x, y
    return None


def remember(app: str, target: str, g: np.ndarray, x: int, y: int) -> None:
    patch = _patch(g, x, y)
    if patch is None:
        return
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(patch.astype(np.uint8)).save(buf, "PNG")
    with _lock:
        data = _load()
        data[_key(app, target)] = {"x": int(x), "y": int(y), "patch": base64.b64encode(buf.getvalue()).decode(),
                                   "seen": datetime.now().isoformat(timespec="seconds")}
        try:
            PATH.parent.mkdir(parents=True, exist_ok=True)
            PATH.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass


def forget(app: str, target: str) -> None:
    with _lock:
        data = _load()
        if data.pop(_key(app, target), None) is not None:
            PATH.write_text(json.dumps(data), encoding="utf-8")
