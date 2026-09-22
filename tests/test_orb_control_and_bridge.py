import queue

import pytest

from assistant.gui import orb_bridge
from assistant.tools import orb_control


@pytest.fixture(autouse=True)
def _reset_bridge():
    orb_bridge.set_gui_queue(None)
    yield
    orb_bridge.set_gui_queue(None)


# --- orb_bridge.py ---


def test_send_returns_false_when_no_queue_set():
    assert orb_bridge.send({"kind": "move", "position": "center"}) is False


def test_send_puts_message_on_queue_when_set():
    q = queue.Queue()
    orb_bridge.set_gui_queue(q)

    result = orb_bridge.send({"kind": "move", "position": "center"})

    assert result is True
    assert q.get_nowait() == {"kind": "move", "position": "center"}


# --- orb_control.py ---


def test_move_status_orb_raises_when_orb_not_running():
    with pytest.raises(RuntimeError, match="isn't running"):
        orb_control.move_status_orb("center")


def test_move_status_orb_sends_position_through_bridge():
    q = queue.Queue()
    orb_bridge.set_gui_queue(q)

    result = orb_control.move_status_orb("a little left")

    assert result == "Moved the status orb: a little left"
    assert q.get_nowait() == {"kind": "move", "position": "a little left"}
