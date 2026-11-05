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
