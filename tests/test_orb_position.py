from assistant.gui.orb_position import parse_position

SCREEN_W, SCREEN_H, SIZE = 1920, 1080, 44


def test_center():
    x, y = parse_position("center", 0, 0, SCREEN_W, SCREEN_H, SIZE)
    assert x == (SCREEN_W - SIZE) // 2
    assert y == (SCREEN_H - SIZE) // 2


def test_named_corner():
    x, y = parse_position("top right", 500, 500, SCREEN_W, SCREEN_H, SIZE)
    assert x == SCREEN_W - SIZE - 24
    assert y == 24


def test_relative_nudge_with_magnitude():
    x, y = parse_position("a little left", 500, 500, SCREEN_W, SCREEN_H, SIZE)
    assert x == 460  # 500 - 40
    assert y == 500


def test_relative_nudge_bare_direction():
    x, y = parse_position("right", 500, 500, SCREEN_W, SCREEN_H, SIZE)
    # "right" alone matches the named absolute right-edge position, not a nudge
    assert x == SCREEN_W - SIZE - 24
    assert y == (SCREEN_H - SIZE) // 2


def test_unparseable_returns_none():
    assert parse_position("do a backflip", 500, 500, SCREEN_W, SCREEN_H, SIZE) is None


def test_large_nudge():
    x, y = parse_position("move a lot to the right", 500, 500, SCREEN_W, SCREEN_H, SIZE)
    assert x == 650  # 500 + 150
    assert y == 500
