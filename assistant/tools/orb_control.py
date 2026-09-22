"""Lets the user reposition the on-screen status orb by voice."""

from assistant.gui import orb_bridge
from assistant.tools.registry import registry


@registry.register(
    name="move_status_orb",
    description=(
        "Moves the on-screen status orb. Accepts either a named position ('center', 'top left', "
        "'top right', 'bottom left', 'bottom right', 'top', 'bottom', 'left', 'right') or a "
        "relative nudge from where it currently is ('left', 'right', 'up', 'down', optionally with "
        "a magnitude word like 'a little left' or 'further right'). Pass exactly what the user said "
        "about direction/position, e.g. 'center' or 'a little left'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "position": {
                "type": "string",
                "description": "Where to move the orb, e.g. 'center' or 'a little left'.",
            }
        },
        "required": ["position"],
    },
)
def move_status_orb(position: str) -> str:
    sent = orb_bridge.send({"kind": "move", "position": position})
    if not sent:
        raise RuntimeError("The status orb isn't running right now.")
    return f"Moved the status orb: {position}"
