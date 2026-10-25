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
import os
import re
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
ANNOUNCE = None  # the orchestrator sets this to Vaayu's voice: "On it - give me a minute" before a slow tool
_SAID_WAITING: set[str] = set()


def _python() -> str:
    exe = Path(sys.executable)
    quiet = exe.with_name("pythonw.exe")  # no console window from a background Vaayu
    return str(quiet if quiet.exists() else exe)


class Server:
    """One MCP server on its own event-loop thread; call() is synchronous, for the tool registry."""

    def __init__(self, name: str, target, timeout: float = _CALL_TIMEOUT, slow=()):
        """target: StdioServerParameters (a real server process) or an MCPServer (in-process, tests).
        timeout: how long one call may take (a design: minutes); slow: tools announced before they run."""
        self.name, self._target, self.timeout, self.slow = name, target, timeout, set(slow)
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
        if tool in self.slow and ANNOUNCE:
            ANNOUNCE("On it - give me a minute.")
        result = self._run(self._client.call_tool(tool, arguments), self.timeout)
        text = " ".join(getattr(c, "text", "") for c in result.content).strip()
        if result.is_error:  # said as is: "I can't see notepad open", not the SDK's "Error executing tool..."
            raise RuntimeError(re.sub(r"^Error executing tool \w+: ", "", text) or f"{tool} failed")
        return text


class Lazy:
    """A server started on its FIRST call, not at Vaayu's start: 15 servers idling took ~1.5 GB of RAM on a
    16 GB laptop (found: 0.6 GB free during the checks). Its tools come from data/mcp_schemas.json, saved the
    first time it ran."""

    def __init__(self, name: str, tools: list, make):
        self.name, self.tools, self._make, self._server = name, tools, make, None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._server is not None

    def call(self, tool: str, **arguments) -> str:
        with self._lock:
            if self._server is None:
                self._server = self._make()
        return self._server.call(tool, **arguments)


SCHEMAS = REPO / "data" / "mcp_schemas.json"


def _schemas() -> dict:
    try:
        return json.loads(SCHEMAS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remember_schemas(name: str, pin: str, tools: list) -> None:
    cache = _schemas()
    cache[name] = {"pin": pin, "tools": [{"name": t.name, "description": t.description or "",
                                          "input_schema": t.input_schema or {}} for t in tools]}
    SCHEMAS.parent.mkdir(parents=True, exist_ok=True)
    SCHEMAS.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def register(server, registry=default_registry, exclude=()) -> list[str]:
    added = []
    for t in server.tools:
        if t.name in exclude:  # a tool the listing leaves out (e.g. Playwright's "run any code in the page")
            continue
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


_RELOAD_SECONDS = 60


def keep_loading(registry=default_registry) -> None:
    """Load what's approved now, and again every minute - a server the user approves later (in 'Vaayu
    Upgrades') starts working without restarting Vaayu ("reload plugins")."""
    import time
    while True:
        try:
            load_all(registry)
        except Exception as exc:  # never let a bad listing end the loop
            log.warning("MCP reload failed: %s", exc)
        time.sleep(_RELOAD_SECONDS)


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
            if name not in _SAID_WAITING:  # checked every minute - said once
                _SAID_WAITING.add(name)
                log.warning("MCP %s: not approved by the Guardian (or changed since) - not loaded", name)
            continue
        cmd = [c.replace("{python}", _python()) for c in entry["command"]]
        # Vaayu's own environment: the SDK's default is a bare few variables - the designer couldn't even find
        # Edge (no ProgramFiles(x86)), and the storage paths (guardian/__init__.py) must reach them too
        params = StdioServerParameters(command=cmd[0], args=cmd[1:], cwd=str(REPO), env=dict(os.environ))

        def make(name=name, params=params, entry=entry):
            return Server(name, params, timeout=float(entry.get("timeout", _CALL_TIMEOUT)), slow=entry.get("slow", ()))
        pin = upgrades.approved_pin(name)
        cached = _schemas().get(name)
        if cached and cached.get("pin") == pin:  # known tools: nothing starts until one is used
            from types import SimpleNamespace
            server = Lazy(name, [SimpleNamespace(**t) for t in cached["tools"]], make)
        else:  # first time since its approval: start it once to learn its tools
            try:
                running = make()
            except Exception as exc:
                log.warning("MCP %s didn't start: %s", name, exc)
                continue
            _remember_schemas(name, pin, running.tools)
            server = Lazy(name, running.tools, lambda running=running: running)
        LOADED[name] = server
        out[name] = register(server, registry, exclude=entry.get("exclude", ()))
        log.info("MCP %s loaded: %s", name, ", ".join(out[name]))
    return out
