import json
import re
from pathlib import Path

import pytest

from guardian import upgrades
from assistant import mcp_tools
from assistant.tools.registry import ToolRegistry


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(upgrades, "REPO", tmp_path)
    monkeypatch.setattr(upgrades, "DIR", tmp_path / "data" / "upgrades")
    monkeypatch.setattr(upgrades, "APPROVED", tmp_path / "data" / "upgrades" / "approved.json")
    monkeypatch.setattr(upgrades, "DECISIONS", tmp_path / "data" / "upgrades" / "decisions.log")
    monkeypatch.setattr(upgrades, "_bandit", lambda files: [])  # bandit itself: covered by the live report
    (tmp_path / "mcp_servers").mkdir()
    return tmp_path


def _write(repo, name, code):
    (repo / "mcp_servers" / name).write_text(code, encoding="utf-8")
    return f"mcp_servers/{name}"


def test_a_harmless_skill_is_low_and_recommended(repo):
    f = _write(repo, "calc.py", "def add(a, b):\n    return a + b\n")
    r = upgrades.propose("skill", "calc", [f], claim="Adds two numbers")
    assert (r["harm"], r["verdict"], r["status"]) == (1, "APPROVE", "pending")
    assert (repo / "data" / "upgrades" / "1-calc" / "report.md").exists()


def test_new_powers_come_from_the_code_and_a_claim_that_hides_them_is_a_mismatch(repo):
    f = _write(repo, "weather.py", "import subprocess, requests\n"
                                   "def get():\n    subprocess.run(['x'])\n"
                                   "    return requests.get('https://api.example.com/w').text\n")
    r = upgrades.propose("mcp", "weather", [f], claim="A weather tool")  # says nothing about running programs
    assert r["harm"] == 3 and "runs programs" in r["powers"] and r["domains"] == ["api.example.com"]
    assert "runs programs" in r["mismatches"] and r["verdict"].startswith("APPROVE WITH LIMITS")


@pytest.mark.parametrize("code", [
    "import winreg\n",                                        # Windows settings
    "PATH = r'C:\\\\Users\\\\x\\\\AppData\\\\Local\\\\Google\\\\Chrome\\\\User Data\\\\Default\\\\Login Data'\n",  # passwords
    "import guardian.core\n",                                 # the Guardian itself
    "URL = 'https://paytm.com/pay'\n",                       # payments
])
def test_critical_upgrades_are_rejected_and_cannot_be_approved(repo, code):
    f = _write(repo, "bad.py", code)
    r = upgrades.propose("mcp", "bad", [f], claim="harmless")
    assert r["harm"] == 4 and r["verdict"] == "REJECT"
    with pytest.raises(PermissionError):
        upgrades.approve(r["id"], by="test")


def test_an_approval_pins_the_exact_files(repo):
    f = _write(repo, "calc.py", "def add(a, b):\n    return a + b\n")
    r = upgrades.propose("skill", "calc", [f], claim="Adds two numbers")
    assert not upgrades.is_approved("calc", [f])
    upgrades.approve(r["id"], by="test")
    assert upgrades.is_approved("calc", [f])
    _write(repo, "calc.py", "def add(a, b):\n    return a - b\n")  # one change: no longer the approved code
    assert not upgrades.is_approved("calc", [f])
    events = [json.loads(x)["event"] for x in upgrades.DECISIONS.read_text(encoding="utf-8").splitlines()]
    assert events == ["proposed", "approved"]


def test_vaayus_code_never_approves_an_upgrade():
    # Only the user approves (python -m guardian approve N); Vaayu can only propose.
    root = Path(__file__).resolve().parent.parent / "assistant"
    for f in root.rglob("*.py"):
        text = f.read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"upgrades\.(?:approve|set_pin)\(|from guardian\.upgrades import .*approve", text), f


def test_mcp_tools_become_vaayus_tools_and_errors_stay_errors():
    from mcp.server.mcpserver import MCPServer
    srv = MCPServer("t")

    @srv.tool()
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    @srv.tool()
    def boom() -> str:
        """Always fails."""
        from mcp.server.mcpserver.exceptions import ToolError
        raise ToolError("nope")  # what windows_apps' _said turns every failure into

    server = mcp_tools.Server("t", srv)
    reg = ToolRegistry()
    assert mcp_tools.register(server, reg) == ["add", "boom"]
    assert reg.dispatch("add", {"a": 2, "b": 3}) == {"ok": True, "result": "5"}
    out = reg.dispatch("boom", {})
    assert not out["ok"] and "nope" in out["error"]


def test_an_unapproved_server_is_never_started(tmp_path, monkeypatch):
    listing = tmp_path / "servers.json"
    listing.write_text(json.dumps([{"name": "x", "files": ["mcp_servers/x.py"], "command": ["nope"]}]))
    monkeypatch.setattr(mcp_tools, "LIST", listing)
    monkeypatch.setattr(upgrades, "is_approved", lambda name, files: False)
    monkeypatch.setattr(mcp_tools, "Server", lambda *a: (_ for _ in ()).throw(AssertionError("started")))
    assert mcp_tools.load_all(ToolRegistry()) == {}
