"""
New abilities on request (user's rule, 2026-10-25): "first he checks if an MCP is available for free - if
yes, load it; if not, he creates his own, tests it, loads it - either way an upgrade that goes through the
Guardian once."

    get_new_ability("translate text between languages")
      1. find_free(): the official MCP Registry - LOCAL packages only (PyPI or npm, they run on this laptop)
         that need no API key. Hosted ("remote") servers are skipped: the user's data would leave the laptop.
      2. download(): into mcp_servers/vendor/<name>/ - wheels only / npm --ignore-scripts, so none of its code
         runs before the user's yes; every cache in this folder.
      3. if nothing fits, build(): Claude (the user's Pro login, as in the daily review) writes
         mcp_servers/<name>.py and its tests on its own branch; the whole test suite must pass.
      4. the Guardian's report -> pending; the user says yes in 'Vaayu Upgrades'; Vaayu loads it by himself
         within a minute (mcp_tools.keep_loading) - no restart.
"""

import json
import re
import shutil
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from pathlib import Path

from assistant.tools.registry import registry

REPO = Path(__file__).resolve().parent.parent
REGISTRY = "https://registry.modelcontextprotocol.io/v0/servers"
VENDOR = REPO / "mcp_servers" / "vendor"
LIST = REPO / "mcp_servers" / "servers.json"
ANNOUNCE = print  # the orchestrator sets this to Vaayu's voice
_STOP = {"the", "and", "for", "that", "can", "with", "from", "into", "this", "tool", "tools", "mcp", "server",
         "make", "get", "find", "able", "ability", "something", "things", "thing", "want", "need", "you", "your",
         "me", "my", "use", "using", "help", "which", "who", "what", "how", "about", "between", "some", "any"}
_lock = threading.Lock()


def _keywords(need: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", need.lower()) if w not in _STOP and len(w) > 2][:4]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "tool"


def find_free(need: str) -> list[dict]:
    """Local, key-free MCP servers in the official registry that fit the need, best first."""
    words, seen, out = _keywords(need), set(), []
    for word in words or [need]:
        url = f"{REGISTRY}?search={urllib.parse.quote(word)}&limit=100"
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)
        for entry in data.get("servers", []):
            server = entry.get("server", {})
            meta = entry.get("_meta", {}).get("io.modelcontextprotocol.registry/official", {})
            if meta.get("status", "active") != "active" or meta.get("isLatest") is False:
                continue
            for pkg in server.get("packages") or []:
                if pkg.get("registryType") not in ("pypi", "npm"):
                    continue
                if any(v.get("isRequired") for v in pkg.get("environmentVariables") or []):
                    continue  # needs an API key or account - not free to use as is
                if any(a.get("isRequired") and not (a.get("default") or a.get("value"))
                       for a in pkg.get("packageArguments") or []):
                    continue
                key = (pkg["registryType"], pkg["identifier"])
                if key in seen:
                    continue
                seen.add(key)
                text = f"{server.get('name', '')} {server.get('title', '')} {server.get('description', '')}".lower()
                out.append({"name": server.get("name", pkg["identifier"]), "about": server.get("description", ""),
                            "type": pkg["registryType"], "package": pkg["identifier"],
                            "version": pkg.get("version") or server.get("version") or "",
                            "score": sum(bool(re.search(rf"\b{re.escape(w)}", text)) for w in words)})
    need_score = max(1, (len(words) + 1) // 2)  # at least half the need's words in its name/description
    return sorted((c for c in out if c["score"] >= need_score), key=lambda c: -c["score"])


def _entry_point(dest: Path, package: str = "") -> tuple[str, str] | None:
    """The package's own first console script: (module, function). Its own - found live: the first script in
    the folder belonged to a dependency (cffi), and every server was started as cffi's code generator."""
    own = re.sub(r"[-_.]+", "_", package.lower())
    infos = sorted(dest.glob("*.dist-info/entry_points.txt"))
    if own:
        infos = [p for p in infos if re.sub(r"[-_.]+", "_", p.parent.name.lower()).startswith(own + "_")]
    for ep in infos:
        section = ""
        for line in ep.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("["):
                section = line
            elif section == "[console_scripts]" and "=" in line:
                mod, _, fn = line.split("=", 1)[1].strip().partition(":")
                return mod.strip(), (fn.strip().split()[0] if fn.strip() else "main")
    return None


def _own_files(dest: Path, package: str) -> list[Path]:
    """The .py files the package itself installed (its dist-info RECORD) - not its dependencies'."""
    own = re.sub(r"[-_.]+", "_", package.lower())
    for record in dest.glob("*.dist-info/RECORD"):
        if re.sub(r"[-_.]+", "_", record.parent.name.lower()).startswith(own + "_"):
            rows = (line.split(",")[0] for line in record.read_text(encoding="utf-8").splitlines())
            return sorted(dest / r for r in rows if r.endswith(".py") and (dest / r).exists())
    return []


def download(c: dict) -> dict:
    """Fetch it without running any of it; the servers.json entry to add (not loaded until approved)."""
    slug = _slug(c["name"].split("/")[-1])
    dest = VENDOR / slug
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    rel = dest.relative_to(REPO).as_posix()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if c["type"] == "pypi":
        spec = f"{c['package']}=={c['version']}" if c["version"] else c["package"]
        run = subprocess.run([sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--target", str(dest),
                              spec], capture_output=True, text=True, timeout=600, creationflags=flags)
        if run.returncode != 0:
            raise RuntimeError(f"pip couldn't fetch {spec}: {run.stderr.strip()[-200:]}")
        ep = _entry_point(dest, c["package"])
        if ep is None:
            raise RuntimeError(f"{c['package']} has no command to start it.")
        mod, fn = ep
        command = ["{python}", "-c", f"import sys; sys.path.insert(0, r'{rel}'); from {mod} import {fn}; sys.exit({fn}())"]
        files = _own_files(dest, c["package"])  # its own code: scanned by the Guardian, pinned
        files += sorted(dest.glob("*.dist-info/RECORD"))  # every installed file's hash, dependencies included
    else:
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("npm isn't installed, so I can't use that one.")
        spec = f"{c['package']}@{c['version']}" if c["version"] else c["package"]
        env = dict(__import__("os").environ, npm_config_cache=str(REPO / ".cache" / "npm"))
        run = subprocess.run([npm, "install", "--prefix", str(dest), "--ignore-scripts", "--no-audit", "--no-fund",
                              spec], capture_output=True, text=True, timeout=600, env=env, creationflags=flags)
        if run.returncode != 0:
            raise RuntimeError(f"npm couldn't fetch {spec}: {run.stderr.strip()[-200:]}")
        pkg_dir = dest / "node_modules" / c["package"]
        meta = json.loads((pkg_dir / "package.json").read_text(encoding="utf-8"))
        binary = meta.get("bin")
        script = binary if isinstance(binary, str) else next(iter((binary or {}).values()), meta.get("main"))
        if not script:
            raise RuntimeError(f"{c['package']} has no command to start it.")
        command = ["node", (pkg_dir / script).relative_to(REPO).as_posix()]
        # ponytail: its own code + the lock file are pinned and scanned, not every dependency's file
        files = sorted(p for p in pkg_dir.rglob("*") if p.suffix in (".js", ".mjs", ".cjs") and
                       "node_modules" not in p.relative_to(pkg_dir).parts)
        files += [p for p in (dest / "package-lock.json",) if p.exists()]
    return {"name": slug, "about": c["about"][:200], "files": [f.relative_to(REPO).as_posix() for f in files],
            "command": command, "tools": [], "source": f"MCP Registry: {c['name']} ({c['type']} {spec})"}


def _add_to_list(entry: dict) -> None:
    entries = json.loads(LIST.read_text(encoding="utf-8")) if LIST.exists() else []
    entries = [e for e in entries if e["name"] != entry["name"]] + [entry]
    LIST.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _claude_prompt(need: str, slug: str) -> str:
    return f"""You are adding a new ability to Vaayu, a local voice assistant (this repo). The user asked for: "{need}".
No free MCP server fits, so write Vaayu's own:

1. mcp_servers/{slug.replace('-', '_')}.py - an MCP server like mcp_servers/windows_apps.py: `from mcp.server.mcpserver import
   MCPServer`, one @server.tool() per action, every failure raised as a ToolError with a short plain sentence
   (copy windows_apps.py's _said decorator), and `server.run("stdio")` under `if __name__ == "__main__":`.
   Only the Python standard library and packages already in requirements.txt. Nothing outside this folder,
   no payments, no passwords, never touch guardian/. The internet only if the ability truly needs it.
2. tests/test_{slug.replace('-', '_')}.py - pytest tests of every tool with NO network and NO real side effects
   (mock them), like tests/test_windows_apps.py.
3. Add an entry to mcp_servers/servers.json: name "{slug}", about, files (just the server file), command
   ["{{python}}", "-m", "mcp_servers.{slug.replace('-', '_')}"], tools (the tool names).
4. Run the full suite: .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider - fix until it passes.
5. git commit everything on this branch (never push). Your commit message is shown to the user as the claim.
"""


def build(need: str) -> dict:
    """Vaayu's own server, written by Claude on its own branch; the Guardian's pending report."""
    from assistant import daily_review as dr
    from guardian import upgrades
    claude = shutil.which("claude")
    if not claude:
        raise RuntimeError("I'd need the Claude CLI to build one, and it isn't installed.")
    slug = _slug(" ".join(_keywords(need)) or need)
    branch = f"vaayu-mcp-{slug}"
    if dr._git("rev-parse", "--verify", "--quiet", branch).returncode == 0:
        branch += f"-{__import__('time').strftime('%H%M%S')}"
    tree = REPO.parent / f"{REPO.name}-{branch}"
    made = dr._git("worktree", "add", "-b", branch, str(tree), "vaayu-v2")
    if made.returncode != 0:
        raise RuntimeError(f"couldn't make a worktree: {made.stderr.strip()[:160]}")
    keep = False
    try:
        if (REPO / "config" / "config.yaml").exists():
            shutil.copy(REPO / "config" / "config.yaml", tree / "config" / "config.yaml")
        run = subprocess.run([claude, "-p", _claude_prompt(need, slug), "--permission-mode", "acceptEdits",
                              "--output-format", "text", "--allowedTools", *dr._CLAUDE_TOOLS,
                              "--disallowedTools", *dr._CLAUDE_DENY], cwd=tree, capture_output=True, text=True,
                             timeout=dr.CLAUDE_TIMEOUT, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        said = (run.stdout or "") + (run.stderr or "")
        if dr._git("rev-list", "--count", f"vaayu-v2..{branch}").stdout.strip() in ("", "0"):
            if dr._LIMIT_HIT.search(said):
                raise RuntimeError("Claude's usage limit is reached - I'll need to try after it resets.")
            raise RuntimeError("Claude didn't manage to write it.")
        changed = dr._git("diff", "--name-only", f"vaayu-v2...{branch}").stdout.split()
        if dr.protected_changes(changed):
            raise RuntimeError("the code it wrote touched the Guardian, so I threw it away.")
        tests = subprocess.run([str(dr.PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=tree,
                               capture_output=True, text=True, timeout=dr.TEST_TIMEOUT, encoding="utf-8",
                               errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tail = ((tests.stdout or "").strip().splitlines() or ["(no output)"])[-1]
        if tests.returncode != 0:
            raise RuntimeError(f"its tests failed ({tail}), so it isn't offered.")
        claim = dr._git("log", "--format=%B", f"vaayu-v2..{branch}").stdout.strip()[:600] or f"A tool to {need}"
        keep = True
        return upgrades.propose_patch(f"new ability: {need[:40]}", branch, claim, tests_passed=True, test_line=tail)
    finally:
        dr._git("worktree", "remove", "--force", str(tree))
        if not keep:
            dr._git("branch", "-D", branch)


def acquire(need: str) -> tuple[dict, str]:
    """A free server if there is one, else Vaayu's own: (the Guardian's report, where it came from)."""
    from guardian import upgrades
    with _lock:  # one at a time - two Claude sessions or pip runs at once would only fight
        try:
            candidates = find_free(need)
        except OSError:
            candidates = []  # registry unreachable: building our own still works offline-ish (Claude)
        for c in candidates[:3]:
            try:
                entry = download(c)
            except Exception:
                continue  # next candidate
            _add_to_list(entry)
            report = upgrades.propose("mcp", entry["name"], entry["files"],
                                      claim=f"A free MCP server from the official registry ({entry['source']}): "
                                            f"{c['about'][:300]}")
            return report, f"a free one from the MCP Registry, {c['name']}"
        return build(need), "one I built myself"


def _work(need: str) -> None:
    try:
        report, source = acquire(need)
    except Exception as exc:
        ANNOUNCE(f"I couldn't get a tool to {need}: {exc}")
        return
    ANNOUNCE(f"A tool to {need} is ready - {source}. The Guardian's report is waiting for your yes in "
             f"'Vaayu Upgrades' (upgrade {report['id']}, harm level {report['harm']}).")


@registry.register(
    name="get_new_ability",
    description=("When the user asks Vaayu for an ability or tool he doesn't have yet ('find a tool to translate', "
                 "'make yourself a tool that can read PDFs', 'get an MCP for weather'): looks for a free MCP server "
                 "that does it, else builds his own - the Guardian checks it and the user approves it once."),
    parameters={"type": "object", "properties": {"need": {"type": "string", "description": "What the new tool should do."}},
                "required": ["need"]},
)
def get_new_ability(need: str) -> str:
    threading.Thread(target=_work, args=(need,), daemon=True, name="new-ability").start()
    return (f"On it - I'll look for a free tool that can {need}, and build one myself if there isn't. The Guardian "
            "checks it first; then it waits for your yes in 'Vaayu Upgrades'.")
