"""Real voice I/O entry point. Run with `python -m assistant.voice_main`.

Also the target for auto-start (see PLAN.md/README for the scheduled task
that launches this via pythonw.exe at login).

The status orb GUI must run on the main thread (Tkinter requirement), so the
actual listening/thinking/speaking orchestrator runs on a background thread
instead, communicating with the orb through a thread-safe queue.
"""

from assistant.log_setup import ensure_output_stream

ensure_output_stream()

import queue  # noqa: E402
import threading  # noqa: E402

from assistant.config_schema import load_config  # noqa: E402
from assistant.gui import orb_bridge  # noqa: E402
from assistant.gui.status_orb import StatusOrb  # noqa: E402
from assistant.orchestrator import Orchestrator  # noqa: E402


def main() -> None:
    config = load_config()
    gui_queue: "queue.Queue" = queue.Queue()
    orb_bridge.set_gui_queue(gui_queue)  # lets move_status_orb reach the orb
    orchestrator = Orchestrator(config, gui_queue=gui_queue)

    orchestrator_thread = threading.Thread(target=orchestrator.run_forever, daemon=True)
    orchestrator_thread.start()

    StatusOrb(gui_queue).run()


if __name__ == "__main__":
    main()
