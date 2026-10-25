"""assistant/mcp_builder.py - a free MCP server first, else Vaayu's own; nothing real is searched, downloaded or
started here (conftest stubs acquire; these use _real_acquire with every outside step mocked)."""
import io
import json

import pytest

from assistant import mcp_builder
from guardian import upgrades


def _registry(*entries):
    return io.BytesIO(json.dumps({"servers": list(entries)}).encode())


def _entry(name, description, packages=None, remotes=None):
    server = {"name": name, "description": description}
    if packages:
        server["packages"] = packages
    if remotes:
        server["remotes"] = remotes
    return {"server": server, "_meta": {"io.modelcontextprotocol.registry/official": {"status": "active",
                                                                                     "isLatest": True}}}


def test_only_local_key_free_servers_that_fit_are_offered(monkeypatch):
    entries = [
        _entry("x/translate-remote", "Translate text", remotes=[{"type": "streamable-http", "url": "https://x"}]),
        _entry("x/translate-keyed", "Translate text", packages=[{"registryType": "npm", "identifier": "tk",
               "environmentVariables": [{"name": "API_KEY", "isRequired": True}]}]),
        _entry("x/translate-py", "Translate text between languages", packages=[{"registryType": "pypi",
               "identifier": "translate-mcp", "version": "1.0"}]),
        _entry("x/translate-js", "Offline translate", packages=[{"registryType": "npm", "identifier": "tjs",
               "version": "2.0"}]),
        _entry("x/weather", "Weather forecasts", packages=[{"registryType": "pypi", "identifier": "w"}]),
    ]
    monkeypatch.setattr(mcp_builder.urllib.request, "urlopen", lambda url, timeout: _registry(*entries))
    found = mcp_builder.find_free("translate text")
    assert [c["package"] for c in found] == ["translate-mcp", "tjs"]  # no remote, no key, nothing unrelated


def test_a_pip_packages_start_command_comes_from_its_console_script(tmp_path):
    info = tmp_path / "translate_mcp-1.0.dist-info"
    info.mkdir()
    (info / "entry_points.txt").write_text("[console_scripts]\ntranslate-mcp = translate_mcp.server:main\n")
    assert mcp_builder._entry_point(tmp_path) == ("translate_mcp.server", "main")


def test_a_free_one_is_downloaded_and_proposed_never_loaded_before_the_yes(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_builder, "LIST", tmp_path / "servers.json")
    monkeypatch.setattr(mcp_builder, "find_free", lambda need: [{"name": "x/translate-py", "about": "Translate",
                        "type": "pypi", "package": "translate-mcp", "version": "1.0", "score": 2}])
    monkeypatch.setattr(mcp_builder, "download", lambda c: {"name": "translate-py", "about": "Translate",
                        "files": ["mcp_servers/vendor/translate-py/t.py"], "command": ["{python}", "-c", "x"],
                        "tools": [], "source": "MCP Registry: x/translate-py (pypi translate-mcp==1.0)"})
    proposed = []
    monkeypatch.setattr(upgrades, "propose", lambda kind, name, files, claim: proposed.append(name) or {"id": 5, "harm": 2})
    monkeypatch.setattr(mcp_builder, "build", lambda need: pytest.fail("built although a free one fits"))
    report, source = mcp_builder._real_acquire("translate text")
    assert proposed == ["translate-py"] and "free one" in source
    assert json.loads((tmp_path / "servers.json").read_text())[0]["name"] == "translate-py"


def test_when_nothing_free_fits_vaayu_builds_his_own(monkeypatch):
    monkeypatch.setattr(mcp_builder, "find_free", lambda need: [])
    monkeypatch.setattr(mcp_builder, "build", lambda need: {"id": 6, "harm": 1})
    report, source = mcp_builder._real_acquire("read my pdfs aloud")
    assert report["id"] == 6 and source == "one I built myself"


def test_the_request_answers_at_once_and_says_what_happened_later(monkeypatch):
    said = []
    monkeypatch.setattr(mcp_builder, "ANNOUNCE", said.append)
    reply = mcp_builder.get_new_ability("translate text")
    assert "Vaayu Upgrades" in reply
    for t in __import__("threading").enumerate():
        if t.name == "new-ability":
            t.join(5)
    assert said and said[0].startswith("I couldn't get a tool to translate text")  # conftest: no real work


def test_router_sends_new_ability_requests():
    from assistant.llm import router
    assert router.fast_route("find a tool to translate text") == ("tools", [("get_new_ability", {"need": "translate text"})])
    assert router.fast_route("make yourself a tool that can read pdfs") == \
        ("tools", [("get_new_ability", {"need": "read pdfs"})])
