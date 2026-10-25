import json

import pytest

from guardian import action_rules as rules
from assistant.tools.registry import ToolRegistry


@pytest.mark.parametrize("name, args, want", [
    ("open_app", {"app_name": "notepad"}, rules.SAFE),
    ("type_text", {"text": "hello"}, rules.CHANGE),
    ("send_whatsapp_message", {"contact": "x", "message": "hi"}, rules.OUTWARD),
    ("app_click", {"app": "whatsapp", "target": "Send"}, rules.OUTWARD),
    ("app_click", {"app": "amazon", "target": "Place order"}, rules.DANGEROUS),
    ("app_click", {"app": "photos", "target": "Delete"}, rules.DANGEROUS),
    ("app_menu", {"app": "notepad", "path": "File > Save As"}, rules.CHANGE),
    ("app_click", {"app": "calculator", "target": "Equals"}, rules.CHANGE),
    # a video's description, not a button: its words don't make it dangerous
    ("click_on_page", {"description": "the video about how to install python on windows"}, rules.CHANGE),
    ("some_new_mcp_tool", {}, rules.CHANGE),  # unknown: at least logged and rate-limited
])
def test_tiers(name, args, want):
    assert rules.tier(name, args) == want


def test_dangerous_is_refused_and_outward_after_reading_content_is_refused():
    assert "never do myself" in rules.refusal("app_click", {"target": "Pay now"})
    assert rules.refusal("send_whatsapp_message", {"message": "hi"}) is None
    assert "what I just read" in rules.refusal("send_whatsapp_message", {"message": "hi"}, after_content=True)
    assert rules.refusal("type_text", {"text": "x"}, after_content=True) is None  # not outward


def test_rate_limit(monkeypatch):
    monkeypatch.setattr(rules, "_PER_MINUTE", {rules.CHANGE: 2})
    monkeypatch.setitem(rules._recent, rules.CHANGE, __import__("collections").deque())
    assert rules.refusal("type_text") is None and rules.refusal("type_text") is None
    assert "too many" in rules.refusal("type_text")


def test_every_action_is_logged_and_a_refused_one_never_runs():
    reg, ran = ToolRegistry(), []

    @reg.register(name="app_click", description="", parameters={"type": "object", "properties": {
        "app": {"type": "string"}, "target": {"type": "string"}}, "required": ["app", "target"]})
    def app_click(app, target):
        ran.append(target)
        return "clicked"

    assert reg.dispatch("app_click", {"app": "calc", "target": "Equals"})["ok"]
    out = reg.dispatch("app_click", {"app": "shop", "target": "Buy now"})
    assert out["refused"] and ran == ["Equals"]
    rows = [json.loads(line) for line in rules.LOG_PATH.read_text(encoding="utf-8").splitlines()][-2:]
    assert [(r["tool"], r["ok"], r["tier"]) for r in rows] == [("app_click", True, 1), ("app_click", False, 3)]
    assert reg.requires_confirmation("app_click", {"app": "whatsapp", "target": "Send"})
    assert not reg.requires_confirmation("app_click", {"app": "calc", "target": "Equals"})
