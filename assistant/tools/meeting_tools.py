"""Voice tools for holding the user's place in a meeting (assistant/meeting.py)."""

from assistant import meeting
from assistant.tools.registry import registry


def _ask(args: dict) -> str:
    when = f" at {args['time']}" if args.get("time") else ""
    return (f"Join the meeting{when} in your name - mic and camera off - and post \"{meeting.LATE_LINE}\" in its chat?")


@registry.register(
    name="join_meeting",
    description="Join a Google Meet for the user until they arrive: mic and camera off, a 'running late' line in the "
                "chat, notes from the live captions. link: the meet.google.com link (empty = the one on the "
                "clipboard); time: when to join (empty = now).",
    parameters={"type": "object", "properties": {"link": {"type": "string"}, "time": {"type": "string"}}},
    requires_confirmation=True,
    confirmation_prompt=_ask,
)
def join_meeting(link: str = "", time: str = "") -> str:
    link = link or meeting.clipboard_link() or ""
    if not meeting.MEET_LINK.match(link):
        raise ValueError("I need the Google Meet link - copy it, or send it to me on the phone.")
    return meeting.join_later(link, time) if time else meeting.join(link)


@registry.register(
    name="meeting_summary",
    description="The user has arrived in the meeting Vaayu joined for them (or wants to know what happened): stop "
                "taking notes and summarise what they missed.",
    parameters={"type": "object", "properties": {}},
)
def meeting_summary() -> str:
    return meeting.stop_and_summarise()
