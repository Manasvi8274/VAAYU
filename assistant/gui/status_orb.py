"""
A marble-sphere status orb: a radial-gradient base (concentric rings, cyan
in the middle fading to pink at the edge - a 2D-canvas approximation of a
glass marble's color blend) with a few colored pie-slice "swirl" wedges
layered on top and rotating.

Rotation IS the sleep/awake indicator, not decoration: spinning means Vaayu
is awake and actually listening for commands; stationary means it's asleep
(still running, only listening for its name). The window closes entirely on
shutdown - this needed an explicit fix, not just "let the thread end":
Tkinter's mainloop() runs on the main thread while the orchestrator runs on
a background thread, so the orchestrator finishing does NOT stop the GUI on
its own (confirmed by reasoning through the threading model, not assumed) -
orchestrator.py pushes a "shutdown" message here instead.

Runs its own Tkinter event loop and must be started on the main thread;
the orchestrator (which does the actual listening/thinking/speaking) runs on
a background thread instead and pushes state updates through a thread-safe
queue.Queue, which this polls periodically via root.after() - this is the
standard safe pattern for mixing Tkinter with other threads (Tkinter widgets
must only ever be touched from the thread running mainloop()).

Known cosmetic limitation: attempts a true circular look via Windows'
-transparentcolor window attribute, but this did NOT visibly take effect in
testing (verified via screenshot, tried several standard recipes) - falls
back to a small dark square panel behind the sphere. The things that matter
functionally - rotation as the sleep/awake signal, real voice-amplitude
reactivity, dragging, voice-controlled positioning - are confirmed working.
"""

import colorsys
import queue
import time
import tkinter as tk

from assistant.gui.orb_position import parse_position

_SIZE = 60
_MARGIN = 24
_POLL_MS = 30
_ROTATION_DEGREES_PER_SEC = 45.0  # steady spin while awake
_RING_COUNT = 9
_STREAK_COUNT = 5
_STREAK_EXTENT = 22  # degrees wide per wedge
_HUE_MIN = 0.5  # cyan
_HUE_MAX = 0.92  # pink
_SATURATION = 0.8


class StatusOrb:
    def __init__(self, state_queue: "queue.Queue"):
        self.state_queue = state_queue
        self._amplitude = 0.0
        self._awake = False  # starts asleep, matching Orchestrator's boot state
        self._rotation_deg = 0.0
        self._last_frame_time = time.time()
        self._drag_start = (0, 0)

        self.root = tk.Tk()
        self.root.overrideredirect(True)  # no title bar/border - just the orb
        self.root.attributes("-topmost", True)
        self.root.geometry(f"{_SIZE}x{_SIZE}+{_MARGIN}+{_MARGIN}")

        try:
            self.root.attributes("-transparentcolor", "black")
            bg = "black"
        except tk.TclError:
            bg = "#1e1e1e"  # fallback if transparency isn't supported

        self.root.configure(bg=bg)
        self.canvas = tk.Canvas(self.root, width=_SIZE, height=_SIZE, bg=bg, highlightthickness=0)
        self.canvas.pack()

        self._streak_ids: list[int] = []
        self._build_sphere()

        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_motion)

        self._poll_queue()
        self._animate()

    def _hsv_color(self, h: float, s: float, v: float) -> str:
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, min(1.0, max(0.0, v)))
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def _build_sphere(self) -> None:
        cx = cy = _SIZE / 2
        max_r = _SIZE / 2 - 2

        # Radial-gradient base: concentric rings, cyan (edge) -> pink (center).
        for i in range(_RING_COUNT):
            t = i / (_RING_COUNT - 1)
            r = max_r * (1 - t * 0.88)
            hue = _HUE_MIN + t * (_HUE_MAX - _HUE_MIN)
            color = self._hsv_color(hue, _SATURATION, 0.45 + t * 0.4)
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="")

        # Swirl wedges - pie slices layered on top of the rings, repositioned
        # (their `start` angle) every frame to animate rotation.
        for i in range(_STREAK_COUNT):
            hue = (_HUE_MIN + (i / _STREAK_COUNT) * (_HUE_MAX - _HUE_MIN)) % 1.0
            color = self._hsv_color(hue, 0.9, 0.95)
            streak_id = self.canvas.create_arc(
                2, 2, _SIZE - 2, _SIZE - 2,
                start=i * (360 / _STREAK_COUNT),
                extent=_STREAK_EXTENT,
                style=tk.PIESLICE,
                fill=color,
                outline="",
            )
            self._streak_ids.append(streak_id)

        # Fixed glossy highlight (doesn't rotate) - gives the 3D-sphere illusion.
    def _on_drag_start(self, event: "tk.Event") -> None:
        self._drag_start = (event.x, event.y)

    def _on_drag_motion(self, event: "tk.Event") -> None:
        x = self.root.winfo_x() + (event.x - self._drag_start[0])
        y = self.root.winfo_y() + (event.y - self._drag_start[1])
        self.root.geometry(f"+{x}+{y}")

    def _handle_move(self, position_text: str) -> None:
        current_x = self.root.winfo_x()
        current_y = self.root.winfo_y()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        result = parse_position(position_text, current_x, current_y, screen_w, screen_h, _SIZE)
        if result is None:
            return
        x = max(0, min(result[0], screen_w - _SIZE))
        y = max(0, min(result[1], screen_h - _SIZE))
        self.root.geometry(f"+{x}+{y}")

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.state_queue.get_nowait()
                kind = msg.get("kind")
                if kind == "amplitude":
                    self._amplitude = float(msg.get("value", 0.0))
                elif kind == "move":
                    self._handle_move(msg.get("position", ""))
                elif kind == "state":
                    self._awake = msg.get("value") == "awake"
                elif kind == "shutdown":
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(_POLL_MS, self._poll_queue)

    def _animate(self) -> None:
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        if self._awake:
            # Spins faster while actively speaking, for a more "alive" look.
            speed = _ROTATION_DEGREES_PER_SEC * (1.0 + self._amplitude * 6.0)
            self._rotation_deg = (self._rotation_deg + speed * dt) % 360
            for i, streak_id in enumerate(self._streak_ids):
                angle = (self._rotation_deg + i * (360 / _STREAK_COUNT)) % 360
                self.canvas.itemconfig(streak_id, start=angle)

        self.root.after(_POLL_MS, self._animate)

    def run(self) -> None:
        self.root.mainloop()
