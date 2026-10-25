"""Voice tools for assistant/alarms.py."""

from assistant import alarms
from assistant.tools.registry import registry


@registry.register(
    name="set_alarm",
    description="Set an alarm: Vaayu calls the user at that time until they say they're awake. time: as said, "
                "e.g. '1:30 pm', '13:30', '7 am'.",
    parameters={"type": "object", "properties": {"time": {"type": "string"}, "label": {"type": "string"}},
                "required": ["time"]},
)
def set_alarm(time: str, label: str = "") -> str:
    return alarms.set_alarm(time, label)


@registry.register(
    name="set_timer",
    description="Set a timer for a number of seconds (10 minutes = 600).",
    parameters={"type": "object", "properties": {"seconds": {"type": "integer"}, "label": {"type": "string"}},
                "required": ["seconds"]},
)
def set_timer(seconds: int, label: str = "") -> str:
    return alarms.set_timer(int(seconds), label)


@registry.register(
    name="list_alarms",
    description="Say which alarms and timers are set.",
    parameters={"type": "object", "properties": {}},
)
def list_alarms() -> str:
    return alarms.list_alarms()


@registry.register(
    name="cancel_alarm",
    description="Cancel alarms/timers: which = 'all', a time like '7 am', or 'timer'.",
    parameters={"type": "object", "properties": {"which": {"type": "string"}}},
)
def cancel_alarm(which: str = "all") -> str:
    return alarms.cancel(which)


@registry.register(
    name="stop_alarm",
    description="Stop the alarm that is ringing now (or snooze it: 'five more minutes').",
    parameters={"type": "object", "properties": {"said": {"type": "string"}}},
)
def stop_alarm(said: str = "") -> str:
    return alarms.dismiss(said)
