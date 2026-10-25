"""mcp_servers/windows_apps.py without touching a real window (the live check is scratch-run: Calculator
5 + 3 = 8, a throwaway text file typed/read/saved)."""
import pytest

from mcp_servers import windows_apps as w


@pytest.mark.parametrize("said, name, better_than", [
    ("save", "Save", "Save as"),
    ("equals", "Equals", "Memory recall"),
    ("five", "Five", "Fifteen"),
])
def test_the_control_said_wins(said, name, better_than):
    assert w._similar(said, name) > w._similar(said, better_than)


def test_icon_glyphs_are_dropped():
    assert w._clean(" Display is 8") == "Display is 8"


def test_every_failure_is_said_not_hidden():
    from mcp.server.mcpserver.exceptions import ToolError

    @w._said
    def f(app):
        raise ValueError(f"I can't see {app} open on the screen.")
    with pytest.raises(ToolError, match="can't see notepad"):
        f("notepad")


def test_payment_apps_are_refused_before_any_window_is_looked_at(monkeypatch):
    monkeypatch.setattr(w, "_top_windows", lambda: (_ for _ in ()).throw(AssertionError("looked")))
    with pytest.raises(PermissionError):
        w._window("paytm")


def test_the_six_tools_and_their_read_only_marks():
    import asyncio
    tools = {t.name: t for t in asyncio.run(w.server.list_tools())}
    assert set(tools) == {"app_controls", "app_click", "app_menu", "app_type", "app_keys", "app_read"}
    assert tools["app_read"].annotations.read_only_hint and tools["app_controls"].annotations.read_only_hint
    assert tools["app_click"].input_schema["required"] == ["app", "target"]


@pytest.fixture
def open_apps(monkeypatch):
    from assistant.llm import router
    monkeypatch.setattr(router, "_open_window_matches", lambda app: app.lower() in ("notepad", "calculator", "paint",
                                                                                     "whatsapp"))
    return router


@pytest.fixture
def loaded(monkeypatch):
    from assistant.tools.registry import registry
    for name in ("app_controls", "app_click", "app_menu", "app_type", "app_keys", "app_read"):
        monkeypatch.setitem(registry._tools, name, object())


@pytest.mark.parametrize("said, want", [
    ("click save in notepad", ("app_click", {"app": "notepad", "target": "save"})),
    ("in calculator press equals", ("app_click", {"app": "calculator", "target": "equals"})),
    ("calculator mein five dabao", ("app_click", {"app": "calculator", "target": "five"})),
    ("open file then save as in notepad", ("app_menu", {"app": "notepad", "path": "file then save as"})),
    ("type hello world in notepad", ("app_type", {"app": "notepad", "text": "hello world"})),
    ("notepad mein hello likho", ("app_type", {"app": "notepad", "text": "hello"})),
    ("press ctrl+s in paint", ("app_keys", {"app": "paint", "keys": "ctrl+s"})),
    ("what does notepad say", ("app_read", {"app": "notepad"})),
    ("what can you do in calculator", ("app_controls", {"app": "calculator"})),
])
def test_in_app_commands(open_apps, loaded, said, want):
    assert open_apps.fast_route(said) == ("tools", [want])


def test_not_for_an_app_that_isnt_open_nor_a_web_page(open_apps, loaded):
    assert open_apps.fast_route("type I live in delhi") == ("tools", [("type_text", {"text": "I live in delhi"})])
    assert open_apps.fast_route("click the second video on youtube") is None


def test_before_the_upgrade_is_approved_typing_works_the_old_way_and_the_rest_says_whats_waiting(open_apps):
    assert open_apps.fast_route("type hello in notepad") == ("tools", [("type_text", {"text": "hello"})])
    kind, said = open_apps.fast_route("click save in notepad")
    assert kind == "reply" and "approve" in said


def test_an_outward_click_is_asked_first_in_plain_words():
    from assistant.tools.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register(name="app_click", description="", parameters={"type": "object", "properties": {}})(lambda **a: "")
    assert reg.requires_confirmation("app_click", {"app": "WhatsApp", "target": "Send"})
    assert reg.get_confirmation_prompt("app_click", {"app": "WhatsApp", "target": "Send"}) == \
        "Should I click Send in WhatsApp?"
