"""
Parses a spoken position/direction into actual screen coordinates for the
status orb - "change position to center" (absolute) vs "a little left"
(relative nudge from wherever it currently is) need different handling, so
this distinguishes them by the presence of a magnitude word ("little",
"bit", "more", "a lot", ...) rather than requiring an exact phrase.
"""

import re

_MARGIN = 24

_NUDGE_MAGNITUDES = {
    "little": 40,
    "bit": 40,
    "slightly": 40,
    "more": 80,
    "further": 80,
    "lot": 150,
}
_DEFAULT_NUDGE = 60

# (dx_sign, dy_sign)
_DIRECTIONS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}

_NAMED_POSITIONS = {
    "center": lambda sw, sh, size: ((sw - size) // 2, (sh - size) // 2),
    "middle": lambda sw, sh, size: ((sw - size) // 2, (sh - size) // 2),
    "top left": lambda sw, sh, size: (_MARGIN, _MARGIN),
    "top right": lambda sw, sh, size: (sw - size - _MARGIN, _MARGIN),
    "bottom left": lambda sw, sh, size: (_MARGIN, sh - size - _MARGIN),
    "bottom right": lambda sw, sh, size: (sw - size - _MARGIN, sh - size - _MARGIN),
    "top": lambda sw, sh, size: ((sw - size) // 2, _MARGIN),
    "bottom": lambda sw, sh, size: ((sw - size) // 2, sh - size - _MARGIN),
    "left": lambda sw, sh, size: (_MARGIN, (sh - size) // 2),
    "right": lambda sw, sh, size: (sw - size - _MARGIN, (sh - size) // 2),
}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


def parse_position(
    text: str,
    current_x: int,
    current_y: int,
    screen_w: int,
    screen_h: int,
    size: int,
) -> tuple[int, int] | None:
    """Returns the new (x, y) top-left position, or None if the text
    couldn't be parsed into a position/direction at all."""
    normalized = _normalize(text)
    words = set(normalized.split())

    direction = next((d for d in _DIRECTIONS if d in words), None)
    magnitude_word = next((w for w in _NUDGE_MAGNITUDES if w in words), None)

    # A direction word alone (no magnitude, no "top"/"bottom" pairing) with a
    # magnitude qualifier, or any direction + magnitude combo, is a relative
    # nudge from the current position.
    if direction and magnitude_word:
        amount = _NUDGE_MAGNITUDES[magnitude_word]
        dx, dy = _DIRECTIONS[direction]
        return (current_x + dx * amount, current_y + dy * amount)

    # Named absolute positions - check two-word combos before single words.
    for name in ("top left", "top right", "bottom left", "bottom right"):
        if all(w in words for w in name.split()):
            x, y = _NAMED_POSITIONS[name](screen_w, screen_h, size)
            return (x, y)
    for name in ("center", "middle", "top", "bottom", "left", "right"):
        if name in words:
            x, y = _NAMED_POSITIONS[name](screen_w, screen_h, size)
            return (x, y)

    # A bare direction with no magnitude and no absolute-position match -
    # still treat as a default-sized relative nudge (e.g. just "left").
    if direction:
        dx, dy = _DIRECTIONS[direction]
        return (current_x + dx * _DEFAULT_NUDGE, current_y + dy * _DEFAULT_NUDGE)

    return None
