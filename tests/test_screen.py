"""Seeing and clicking the screen (assistant/tools/screen.py)."""
import pytest

from assistant.tools import screen


class _Img:
    size = (1280, 720)


def test_a_found_point_is_scaled_to_real_pixels(monkeypatch):
    monkeypatch.setattr(screen, "look", lambda prompt, img=None: 'Here: {"x": 290, "y": 700}')
    assert screen.find_on_screen("the start button", _Img(), (1920, 1080)) == (435, 1050)  # measured live: on it
    monkeypatch.setattr(screen, "look", lambda prompt, img=None: '{"none": "not visible"}')
    assert screen.find_on_screen("a unicorn", _Img(), (1920, 1080)) is None


def test_claude_first_then_the_local_model(monkeypatch):
    monkeypatch.setattr(screen, "_ask_claude", lambda img, prompt: None)  # not there / limit reached
    assert screen.look("what's here", _Img()) == "Local look."
    monkeypatch.setattr(screen, "_ask_claude", lambda img, prompt: "Edge, a Google search.")
    assert screen.look("what's here", _Img()) == "Edge, a Google search."
    monkeypatch.setattr(screen, "LOCAL_ONLY", True)
    assert screen.look("what's here", _Img()) == "Local look."


@pytest.mark.parametrize("said, tool", [("what is on my screen", "look_at_screen"), ("read this error", "look_at_screen"),
                                        ("screen pe kya hai", "look_at_screen"),
                                        ("click the blue submit button", "click_on_screen"), ("press enter", None)])
def test_screen_routes(said, tool):
    from assistant.llm import router
    routed = router.fast_route(said)
    assert (routed[1][0][0] if routed else None) == tool


def test_what_is_clicked_is_judged_like_any_click():
    from guardian.action_rules import DANGEROUS, OUTWARD, CHANGE, tier
    assert tier("click_on_screen", {"target": "Pay now"}) == DANGEROUS
    assert tier("click_on_screen", {"target": "Send"}) == OUTWARD
    assert tier("click_on_screen", {"target": "the second video"}) == CHANGE


def _screen_with_button(x, y, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    g = np.full((540, 960), 240, np.float32) + rng.normal(0, 2, (540, 960)).astype(np.float32)
    g[y - 12:y + 12, x - 30:x + 30] = 60  # a dark button...
    g[y - 4:y + 4, x - 20:x + 10] = 250   # ...with a light label
    return g


def test_a_button_seen_once_is_clicked_from_memory_then_found_again_when_it_moves(tmp_path, monkeypatch):
    # User (2026-11-05): "recognise my button positions which rarely change - Claude only when he doesn't know".
    from assistant.tools import screen_memory as sm
    monkeypatch.setattr(sm, "PATH", tmp_path / "targets.json")
    g = _screen_with_button(300, 200)
    sm.remember("notepad.exe", "the Save button", g, 300, 200)
    assert sm.recall("notepad.exe", "save", g) == (300, 200)          # same spot, same look: at once
    moved = sm.recall("notepad.exe", "save button", _screen_with_button(620, 410, seed=1))
    assert moved is not None and abs(moved[0] - 620) <= 2 and abs(moved[1] - 410) <= 2  # window moved: found
    assert sm.recall("notepad.exe", "save", _screen_with_button(620, 410)) == moved      # ...and remembered there
    blank = _screen_with_button(300, 200) * 0 + 240
    assert sm.recall("notepad.exe", "save", blank) is None              # gone: Claude looks again
    assert sm.recall("chrome.exe", "save", g) is None                   # per app


def test_clicking_uses_memory_before_asking(monkeypatch, tmp_path):
    from PIL import Image
    from assistant.tools import screen_memory as sm
    monkeypatch.setattr(sm, "PATH", tmp_path / "targets.json")
    img = Image.fromarray(_screen_with_button(300, 200).astype("uint8"))
    monkeypatch.setattr(screen, "_grab_full", lambda: img)
    monkeypatch.setattr(sm, "foreground_app", lambda: "notepad.exe")
    asked, clicks = [], []
    monkeypatch.setattr(screen, "find_on_screen", lambda t, thumb, full: asked.append(t) or (300, 200))
    monkeypatch.setattr(screen, "_click", lambda x, y: clicks.append((x, y)))
    screen.click_on_screen("save")
    screen.click_on_screen("save")
    assert asked == ["save"] and clicks == [(300, 200), (300, 200)]  # Claude once, memory the second time
