"""
MCP servers -> Vaayu's tools. Each server in mcp_servers/servers.json is started (stdio) and its tools
are registered like any built-in one - but ONLY if the Guardian approved exactly these files
(guardian/upgrades.py: report, the user's yes, SHA-256 pinned). A server changed by even one byte since
its approval is not loaded until it's approved again.

Built-in tools always win a name clash: a server can add abilities, never replace one.
"""

import asyncio
import json
import logging
import sys
import threading
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path

from guardian import upgrades
from assistant.tools.registry import registry as default_registry

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent
LIST = REPO / "mcp_servers" / "servers.json"
_CALL_TIMEOUT = 30.0
_START_TIMEOUT = 30.0
LOADED: dict[str, "Server"] = {}


def _python() -> str:
    exe = Path(sys.executable)
    quiet = exe.with_name("pythonw.exe")  # no console window from a background Vaayu
    return str(quiet if quiet.exists() else exe)


class Server:
    """One MCP server on its own event-loop thread; call() is synchronous, for the tool registry."""

    def __init__(self, name: str, target):
        """target: StdioServerParameters (a real server process) or an MCPServer (in-process, tests)."""
        self.name, self._target = name, target
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name=f"mcp-{name}").start()
        self.tools = self._run(self._connect(), _START_TIMEOUT)

    def _run(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self):
        from mcp import Client
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(self._target))
        return (await self._client.list_tools()).tools

    def call(self, tool: str, **arguments) -> str:
        result = self._run(self._client.call_tool(tool, arguments), _CALL_TIMEOUT)
        text = " ".join(getattr(c, "text", "") for c in result.content).strip()
        if result.is_error:
            raise RuntimeError(text or f"{tool} failed")
        return text


def register(server: Server, registry=default_registry) -> list[str]:
    added = []
    for t in server.tools:
        if t.name in registry._tools:
            log.warning("MCP %s: %s clashes with a built-in tool - skipped", server.name, t.name)
            continue
        schema = dict(t.input_schema or {"type": "object", "properties": {}})
        schema.pop("title", None)
        registry.register(name=t.name, description=t.description or t.name, parameters=schema)(
            partial(server.call, t.name))
        added.append(t.name)
    return added


def waiting_for(tool: str) -> str | None:
    """What to tell the user when a tool they asked for comes from a server still waiting for their yes."""
    try:
        entries = json.loads(LIST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for entry in entries:
        if tool in entry.get("tools", []) and entry["name"] not in LOADED:
            report = next((r for r in reversed(upgrades.reports()) if r["name"] == entry["name"]), None)
            if report and report["status"] == "pending":
                return (f"I can do that once you approve my '{entry['name']}' upgrade. The Guardian's report is "
                        f"ready - open 'Vaayu Upgrades' on your desktop to read it and say yes or no.")
            return f"That needs my '{entry['name']}' upgrade, which isn't approved yet."
    return None


def load_all(registry=default_registry) -> dict[str, list[str]]:
    """Start every approved server; {server: [its tools]}. Never raises - a broken server is skipped."""
    from mcp.client.stdio import StdioServerParameters
    out = {}
    try:
        entries = json.loads(LIST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("MCP list unreadable: %s", exc)
        return out
    for entry in entries:
        name = entry["name"]
        if name in LOADED:
            continue
        if not upgrades.is_approved(name, entry["files"]):
            log.warning("MCP %s: not approved by the Guardian (or changed since) - not loaded", name)
            continue
        cmd = [c.replace("{python}", _python()) for c in entry["command"]]
        try:
            server = Server(name, StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=str(REPO)))
        except Exception as exc:
            log.warning("MCP %s didn't start: %s", name, exc)
            continue
        LOADED[name] = server
        out[name] = register(server, registry)
        log.info("MCP %s loaded: %s", name, ", ".join(out[name]))
    return out
