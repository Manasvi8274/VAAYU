"""
Fetch the chosen free MCP servers into mcp_servers/vendor/ and list them in mcp_servers/servers.json.
Nothing here approves or loads anything - each still needs the Guardian's report and the user's yes.

    .venv/Scripts/python.exe scripts/add_mcp_servers.py            # all of them
    .venv/Scripts/python.exe scripts/add_mcp_servers.py fetch time  # just these

Chosen (free, run on this laptop, no account or API key, light enough for the GTX 1650 laptop):
  thinking  sequential-thinking (official)        files     filesystem (official; the user's own folders)
  web       fetch (official), DuckDuckGo search, Wikipedia
  websites  Playwright (Microsoft) - use sites like a person, in its own Edge profile (never the user's logins)
  office    Word, PowerPoint, Excel               other     time zones (official), calculator, git (official)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from assistant import mcp_builder  # noqa: E402

HOME = Path(os.environ["USERPROFILE"])
USER_FOLDERS = [str(HOME / d) for d in ("Documents", "Desktop", "Downloads", "Pictures", "Music", "Videos")
                if (HOME / d).exists()] + [str(ROOT / "designs")]

# (short name, registry type, package, version, about, extra: args appended / tools left out / settings)
SERVERS = [
    ("sequential-thinking", "npm", "@modelcontextprotocol/server-sequential-thinking", "2.0.0",
     "Think a hard problem through step by step, revising as it goes", {}),
    ("filesystem", "npm", "@modelcontextprotocol/server-filesystem", "2.0.1",
     "Read, write, move, search and list files - only in the user's own folders (no deleting)",
     {"args": USER_FOLDERS}),
    ("playwright", "npm", "@playwright/mcp", "0.0.83",
     "Use websites like a person: open pages, read them, click, type, fill forms - in its own Edge profile",
     {"args": ["--browser", "msedge", "--user-data-dir", "mcp_servers/vendor/playwright/profile",
               "--output-dir", "designs/browser", "--image-responses", "omit"],
      "exclude": ["browser_install", "browser_evaluate", "browser_run_code"], "timeout": 120}),
    ("fetch", "pypi", "mcp-server-fetch", "2026.10.24", "Read any web page as plain text", {}),
    ("duckduckgo", "pypi", "duckduckgo-mcp-server", "0.7.0", "Search the web with DuckDuckGo (no account)", {}),
    ("wikipedia", "pypi", "wikipedia-mcp", "2.0.1", "Look things up on Wikipedia", {}),
    ("time", "pypi", "mcp-server-time", "2026.10.19", "Current time anywhere, and converting between time zones",
     {"args": ["--local-timezone", "Asia/Kolkata"]}),
    ("calculator", "pypi", "mcp-server-calculator", "0.2.1", "Exact arithmetic and math expressions", {}),
    ("git", "pypi", "mcp-server-git", "2026.10.19", "Work with git repositories: status, diffs, commits, branches", {}),
    ("word", "pypi", "office-word-mcp-server", "1.1.11", "Create and edit Word documents", {"timeout": 120}),
    ("powerpoint", "pypi", "office-powerpoint-mcp-server", "2.0.7", "Create and edit PowerPoint presentations",
     {"timeout": 120}),
    ("excel", "pypi", "excel-mcp-server", "2.2.2", "Create, read and edit Excel workbooks and charts",
     {"args": ["stdio"], "timeout": 120}),
]


def main(only: list[str]) -> None:
    (ROOT / "designs").mkdir(exist_ok=True)
    for name, kind, package, version, about, extra in SERVERS:
        if only and name not in only:
            continue
        print(f"{name}: fetching {package} {version} ...", flush=True)
        try:
            entry = mcp_builder.download({"name": name, "about": about, "type": kind, "package": package,
                                          "version": version})
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            continue
        entry["name"] = name
        entry["command"] += extra.get("args", [])
        for key in ("exclude", "timeout"):
            if key in extra:
                entry[key] = extra[key]
        mcp_builder._add_to_list(entry)
        print(f"  ok - {len(entry['files'])} file(s) pinned; command: {' '.join(entry['command'][:3])} ...", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
