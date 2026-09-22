"""
Voice control over the daily check-in time (see assistant/self_review/ and
orchestrator.py's _maybe_run_daily_checkin) - "set the update time to 9pm".
Times are interpreted in IST unless the user names a different timezone (per
explicit instruction); this tool only handles the time-of-day itself, not
timezone changes - config/config.yaml's self_review.timezone covers that.
"""

import re

from assistant.config_schema import load_config, set_check_in_time
from assistant.tools.registry import registry

_TZ_LABELS = {"Asia/Kolkata": "IST"}

_TIME_RE = re.compile(
    r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$",
    re.IGNORECASE,
)


def _parse_time_to_hhmm(text: str) -> str | None:
    """Parses a spoken/typed time ('9pm', '9:30 pm', '21:00', '9') into a
    24-hour 'HH:MM' string, or None if it can't be understood."""
    match = _TIME_RE.match(text.strip())
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()

    if not (0 <= minute <= 59):
        return None

    if meridiem:
        if not (1 <= hour <= 12):
            return None
        if meridiem == "am":
            hour = 0 if hour == 12 else hour
        else:  # pm
            hour = hour if hour == 12 else hour + 12
    else:
        if not (0 <= hour <= 23):
            return None

    return f"{hour:02d}:{minute:02d}"


@registry.register(
    name="set_daily_update_time",
    description=(
        "Change what time Vaayu checks in each day about issues/bugs it hit while working - e.g. "
        "'set the update time to 9pm', 'change the check-in time to 21:30'. Interpreted in IST "
        "unless the user names a different timezone."
    ),
    parameters={
        "type": "object",
        "properties": {
            "time": {
                "type": "string",
                "description": "The time to set, e.g. '9pm', '9:30 pm', '21:00'.",
            }
        },
        "required": ["time"],
    },
)
def set_daily_update_time(time: str) -> str:
    hhmm = _parse_time_to_hhmm(time)
    if hhmm is None:
        raise ValueError(f"Couldn't understand the time '{time}' - try something like '9pm' or '21:00'.")

    config = load_config()
    tz_label = _TZ_LABELS.get(config.self_review.timezone, config.self_review.timezone)
    set_check_in_time(hhmm)
    return f"Daily check-in time set to {hhmm} {tz_label}."
