"""
Lets tool functions (which don't have direct access to the Orchestrator
instance or its gui_queue) send messages to the status orb - set once by
voice_main.py right after creating the queue, used by
tools/orb_control.py's move_status_orb. If the orb isn't running (e.g. text
mode), send() just returns False rather than erroring.
"""

import queue

_gui_queue: "queue.Queue | None" = None


def set_gui_queue(q: "queue.Queue") -> None:
    global _gui_queue
    _gui_queue = q


def send(message: dict) -> bool:
    if _gui_queue is None:
        return False
    _gui_queue.put(message)
    return True
