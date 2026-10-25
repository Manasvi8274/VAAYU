"""
The Guardian's upgrade gate (build plan: "Self-upgrade pipeline" and "The Guardian's upgrade report").

Anything that gives Vaayu a new ability - an MCP server (a free one found, or one he wrote), a skill, a
code patch - comes here first. The report is built from what the code DOES (fixed Python checks and
bandit), not from what Vaayu says; his description is shown as a claim, checked against the evidence.
Nothing loads until the user approves it:

    .venv/Scripts/python.exe -m guardian upgrades        # the list, with harm levels
    .venv/Scripts/python.exe -m guardian show 3          # one full report
    .venv/Scripts/python.exe -m guardian approve 3       # yes (harm 3 also asks for the PIN)
    .venv/Scripts/python.exe -m guardian reject 3

An approval pins the exact files (SHA-256) - change one byte and it is a new upgrade. Vaayu's code never
calls approve() (tests/test_upgrades.py checks); he can only propose.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DIR = REPO / "data" / "upgrades"
APPROVED = DIR / "approved.json"
DECISIONS = DIR / "decisions.log"  # append-only: every proposal and every yes/no
PIN_FILE = REPO / "data" / "guardian_pin.json"

HARM = {0: "Cosmetic", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
_NETWORK = {"socket", "requests", "httpx", "urllib", "aiohttp", "websockets", "http", "ftplib", "smtplib",
            "paramiko", "telnetlib", "urllib3"}
_CONTROL = {"keyboard", "pyautogui", "pynput", "uiautomation", "pywinauto", "mouse"}
_RUNS = {"subprocess", "os.system", "os.startfile", "os.popen", "os.execv", "os.spawnv", "Popen"}
_DELETES = {"os.remove", "os.unlink", "unlink", "rmtree", "os.rmdir", "rmdir"}
# A browser's saved passwords/cookies FILES (not the word "cookies" - every web client handles cookies; found:
# the DuckDuckGo server was rejected for it), autostart, Windows' security and boot settings.
_CRITICAL_TEXT = re.compile(r"\bLogin Data\b|\bLocal State\b|User Data[\\/]+[^'\"\n]{0,40}?[\\/]+(?:Network[\\/]+)?Cookies\b|"
                            r"cookies\.sqlite|logins\.json|key4\.db|CurrentVersion\\\\?Run\b|"
                            r"Start Menu\\\\?Programs\\\\?Startup|\brunas\b|\bDisableAntiSpyware\b|\bSet-MpPreference\b|"
                            r"\bnetsh\b|\bbcdedit\b|\bschtasks\b|\breg add\b|\bvssadmin\b", re.I)
# What a claim must mention for a power it really has - else the report shows a mismatch.
_CLAIM_WORDS = {
    "network access": r"internet|web|online|network|download|api|http|search|fetch|url",
    "controls the laptop (types, clicks, keys)": r"click|type|key|control|button|menu|app",
    "runs programs": r"run|launch|start|open|program|command",
    "deletes files": r"delet|remov|clean",
    "file access outside Vaayu's folder": r"file|folder|disk|document",
    "low-level Windows calls (ctypes)": r"window|system|low.level",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sha(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        # line endings ignored: git may rewrite LF as CRLF on a checkout - the same code, not a new upgrade
        h.update(f.relative_to(REPO).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _decide(event: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), **event}, ensure_ascii=False) + "\n")


def scan(files: list[Path], root: Path | None = None) -> dict:
    """What the code can do, from the code itself (paths judged relative to root: the repo, or a snapshot)."""
    from guardian.rules import is_payment_target
    powers, domains, critical, notes = set(), set(), [], []
    for f in files:
        rel = f.relative_to(root or REPO).as_posix()
        if rel.startswith("guardian/"):
            critical.append(f"changes the Guardian itself ({rel})")
        if f.suffix in (".js", ".mjs", ".cjs"):  # an npm server: no parser here, so its words are read
            _scan_js(f.read_text(encoding="utf-8", errors="replace"), rel, powers, domains, critical)
            continue
        if f.suffix != ".py":
            continue
        source = f.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            critical.append(f"{rel} doesn't even parse: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for m in mods:
                    top = m.split(".")[0]
                    if top in _NETWORK:
                        powers.add("network access")
                    if top in _CONTROL:
                        powers.add("controls the laptop (types, clicks, keys)")
                    if top == "subprocess":
                        powers.add("runs programs")
                    if top in ("winreg", "_winreg"):
                        critical.append(f"changes Windows settings (registry) in {rel}")
                    if top == "ctypes":
                        powers.add("low-level Windows calls (ctypes)")
                    if top == "guardian" and m not in ("guardian.rules", "guardian.action_rules"):
                        critical.append(f"reaches into the Guardian ({m}) in {rel}")
            elif isinstance(node, ast.Call):
                fn = node.func
                name = (f"{fn.value.id}.{fn.attr}" if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else "")
                if name in _RUNS or name.split(".")[-1] in _RUNS:
                    powers.add("runs programs")
                if name in _DELETES or name.split(".")[-1] in _DELETES:
                    powers.add("deletes files")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value
                for url in re.findall(r"https?://([a-z0-9.-]+)", s, re.I):
                    domains.add(url.lower())
                if _CRITICAL_TEXT.search(s):
                    critical.append(f"touches a protected part of Windows ('{_CRITICAL_TEXT.search(s).group(0)}') "
                                    f"in {rel}")
                if is_payment_target(s):
                    critical.append(f"mentions a payment/banking target ('{s[:40]}') in {rel}")
                if re.match(r"^[A-Za-z]:[\\/]", s) and not s.lower().startswith(str(REPO).lower()):
                    powers.add("file access outside Vaayu's folder")
                    notes.append(f"path outside the repo: {s[:80]}")
    if domains:
        powers.add("network access")
    return {"powers": sorted(powers), "domains": sorted(domains), "critical": critical, "notes": notes,
            "bandit": _bandit(files)}


_JS_POWERS = [
    (r"\bchild_process\b|\bexecSync\s*\(|\bspawn(?:Sync)?\s*\(|\bexecFile\s*\(", "runs programs"),
    (r"\bfs(?:\.promises)?\.(?:unlink|rm|rmdir)(?:Sync)?\s*\(|\brimraf\b", "deletes files"),
    (r"""\brequire\(\s*['"](?:node:)?(?:https?|net|dgram|tls)['"]\s*\)|\bfrom\s+['"](?:node:)?(?:https?|net)['"]|"""
     r"\bfetch\s*\(|\baxios\b|\bWebSocket\b", "network access"),
    (r"\brobotjs\b|\bnut-tree\b|\bSendKeys\b", "controls the laptop (types, clicks, keys)"),
]


def _scan_js(source: str, rel: str, powers: set, domains: set, critical: list) -> None:
    from guardian.rules import is_payment_target
    for pattern, power in _JS_POWERS:
        if re.search(pattern, source):
            powers.add(power)
    for host in re.findall(r"https?://([a-z0-9.-]+)", source, re.I):
        domains.add(host.lower())
    if m := _CRITICAL_TEXT.search(source):
        critical.append(f"touches a protected part of Windows ('{m.group(0)}') in {rel}")
    for literal in re.findall(r"""['"]([^'"\n]{3,80})['"]""", source):
        if is_payment_target(literal):
            critical.append(f"mentions a payment/banking target ('{literal[:40]}') in {rel}")
            break


def _bandit(files: list[Path]) -> list[str]:
    py = [str(f) for f in files if f.suffix == ".py"]
    if not py:
        return []
    try:
        out = subprocess.run([sys.executable, "-m", "bandit", "-q", "-f", "json", *py], capture_output=True,
                             text=True, timeout=120).stdout
        results = json.loads(out or "{}").get("results", [])
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return [f"bandit didn't run: {exc}"]
    return [f"{r['issue_severity']}: {r['issue_text']} ({Path(r['filename']).name}:{r['line_number']})"
            for r in results if r.get("issue_severity") in ("MEDIUM", "HIGH")]


def harm_level(found: dict, kind: str) -> tuple[int, list[str]]:
    if found["critical"]:
        return 4, found["critical"]
    high = [p for p in found["powers"] if p != "network access"]
    if high:
        return 3, [f"new power: {p}" for p in high]
    if "network access" in found["powers"]:
        return 2, [f"contacts {', '.join(found['domains']) or 'the internet'}"]
    if kind == "cosmetic":
        return 0, ["changes only how Vaayu talks"]
    return 1, ["a new ability with no outside effect"]


def _run_tests(tests: list[str]) -> tuple[bool, str]:
    if not tests:
        return True, "no tests given"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=REPO,
                          capture_output=True, text=True, timeout=900)
    last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    return proc.returncode == 0, last


def propose(kind: str, name: str, files: list[str | Path], claim: str, tests: list[str] | None = None) -> dict:
    """A new upgrade: scanned, tested, tagged for rollback, reported - and left PENDING for the user."""
    paths = [Path(f) if Path(f).is_absolute() else REPO / f for f in files]
    passed, test_line = _run_tests(tests or [])
    return _file(kind, name, [p.relative_to(REPO).as_posix() for p in paths], claim, scan(paths), passed, test_line,
                 {"sha256": _sha(paths)})


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, timeout=120,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _snapshot(ref: str, files: list[str], dest: Path) -> list[Path]:
    """The files as they are at a git ref, written under dest (a file the ref doesn't have is skipped)."""
    out = []
    for rel in files:
        shown = _git("show", f"{ref}:{rel}")
        if shown.returncode == 0:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_bytes(shown.stdout)
            out.append(dest / rel)
    return out


def propose_patch(name: str, branch: str, claim: str, tests_passed: bool, test_line: str,
                  base: str = "vaayu-v2") -> dict:
    """A code patch to Vaayu (the daily review's fixes, on their own branch): judged by what it ADDS - the
    powers the changed files have after it and didn't before - and merged only on the user's yes."""
    import shutil
    files = _git("diff", "--name-only", f"{base}...{branch}").stdout.decode().split()
    tmp = DIR / "_scan"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        before = scan(_snapshot(base, files, tmp / "before"), tmp / "before")
        after = scan(_snapshot(branch, files, tmp / "after"), tmp / "after")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    found = {"powers": sorted(set(after["powers"]) - set(before["powers"])),
             "domains": sorted(set(after["domains"]) - set(before["domains"])),
             "critical": [c for c in after["critical"] if c not in before["critical"] or "the Guardian itself" in c],
             "notes": [n for n in after["notes"] if n not in before["notes"]],
             "bandit": [b for b in after["bandit"] if b.split(" (")[0] not in {x.split(" (")[0] for x in before["bandit"]}]}
    commit = _git("rev-parse", branch).stdout.decode().strip()
    return _file("patch", name, files, claim, found, tests_passed, test_line,
                 {"branch": branch, "base": base, "commit": commit, "sha256": ""})


def _file(kind: str, name: str, files: list[str], claim: str, found: dict, passed: bool, test_line: str,
          extra: dict) -> dict:
    level, reasons = harm_level(found, kind)
    mismatches = [p for p in found["powers"] if p in _CLAIM_WORDS and not re.search(_CLAIM_WORDS[p], claim, re.I)]
    if level == 4 or not passed or any(b.startswith("HIGH") for b in found["bandit"]):
        verdict = "REJECT"
    elif level == 3 or mismatches:
        verdict = "APPROVE WITH LIMITS (read the new powers first)"
    else:
        verdict = "APPROVE"
    DIR.mkdir(parents=True, exist_ok=True)
    uid = 1 + max((int(p.name.split("-")[0]) for p in DIR.iterdir() if p.is_dir() and p.name.split("-")[0].isdigit()),
                  default=0)
    tag = f"pre-upgrade-{uid}"
    subprocess.run(["git", "tag", "-f", tag], cwd=REPO, capture_output=True)
    report = {"id": uid, "name": name, "kind": kind, "status": "pending", "created": _now(), "files": files,
              "claim": claim, "harm": level, "harm_reasons": reasons, "mismatches": mismatches,
              "tests_passed": passed, "tests": test_line, "rollback": f"git checkout {tag}",
              "verdict": verdict, **found, **extra}
    folder = DIR / f"{uid}-{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')}"
    folder.mkdir()
    (folder / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (folder / "report.md").write_text(render(report), encoding="utf-8")
    _decide({"event": "proposed", "id": uid, "name": name, "harm": level, "verdict": verdict})
    return report


def render(r: dict) -> str:
    lines = [f"UPGRADE #{r['id']} - \"{r['name']}\" ({r['kind']})", "",
             f"Harm level: {r['harm']} ({HARM[r['harm']]}) - " + "; ".join(r["harm_reasons"]),
             f"Summary: {r['kind']} '{r['name']}' - {len(r['files'])} file(s): {', '.join(r['files'])}",
             f"Vaayu's claim (a claim, checked below): {r['claim']}",
             "Claim vs actual: " + ("MATCH" if not r["mismatches"] else
                                    "MISMATCH - the claim doesn't mention: " + ", ".join(r["mismatches"])),
             "New powers: " + (", ".join(r["powers"]) or "none"),
             "Network: " + (", ".join(r["domains"]) or "no web addresses in the code"),
             f"Tests: {'passed' if r['tests_passed'] else 'FAILED'} - {r['tests']}",
             "Code scan (bandit): " + ("; ".join(r["bandit"]) or "no findings")]
    lines += [f"Note: {n}" for n in r["notes"][:10]]
    if r.get("branch"):
        lines.append(f"Branch: {r['branch']} at {r['commit'][:10]} - joins {r['base']} only on your yes")
    lines += [f"Rollback: {r['rollback']}", f"Recommendation: {r['verdict']}", f"Status: {r['status']}"]
    return "\n".join(lines) + "\n"


def reports() -> list[dict]:
    if not DIR.exists():
        return []
    out = []
    for p in sorted(DIR.glob("*/report.json"), key=lambda p: int(p.parent.name.split("-")[0])):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _save(r: dict) -> None:
    folder = next(p.parent for p in DIR.glob(f"{r['id']}-*/report.json"))
    (folder / "report.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    (folder / "report.md").write_text(render(r), encoding="utf-8")


def _approved() -> dict:
    try:
        return json.loads(APPROVED.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def approve(uid: int, by: str) -> dict:
    """The user's yes - from the CLI below (never from Vaayu's code). Pins the files' exact contents."""
    r = next(x for x in reports() if x["id"] == uid)
    if r["harm"] == 4:
        raise PermissionError("Harm level 4 is rejected automatically - only you can make such a change, by hand.")
    if r["kind"] == "patch":
        return _merge(r, by)
    paths = [REPO / f for f in r["files"]]
    if _sha(paths) != r["sha256"]:
        raise PermissionError("The files changed after the report was made - it needs a new report.")
    approved = _approved()
    approved[r["name"]] = {"id": uid, "sha256": r["sha256"], "files": r["files"], "approved": _now(), "by": by,
                           "harm": r["harm"]}
    APPROVED.write_text(json.dumps(approved, indent=1, ensure_ascii=False), encoding="utf-8")
    r["status"] = "approved"
    _save(r)
    _decide({"event": "approved", "id": uid, "name": r["name"], "by": by})
    return r


def _merge(r: dict, by: str) -> dict:
    """A patch's yes: its branch joins Vaayu - only if it is still exactly the commit the report checked."""
    if _git("rev-parse", r["branch"]).stdout.decode().strip() != r["commit"]:
        raise PermissionError(f"Branch {r['branch']} changed after the report was made - it needs a new report.")
    if _git("branch", "--show-current").stdout.decode().strip() != r["base"]:
        raise RuntimeError(f"Check out {r['base']} first - the fixes go onto it.")
    merged = _git("merge", "--ff-only", r["branch"])
    if merged.returncode != 0:
        raise RuntimeError(f"Couldn't fast-forward {r['base']}: {merged.stderr.decode().strip()[:200]}")
    _pin_servers_in(r)
    r["status"] = "approved"
    _save(r)
    _decide({"event": "approved", "id": r["id"], "name": r["name"], "by": by, "merged": r["commit"]})
    return r


def _pin_servers_in(r: dict) -> None:
    """An approved patch that adds an MCP server (one Vaayu built): the server's files are pinned too, so it
    loads - it was part of what the user just approved."""
    try:
        entries = json.loads((REPO / "mcp_servers" / "servers.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    approved = _approved()
    for entry in entries:
        if entry.get("files") and set(entry["files"]) <= set(r["files"]):
            paths = [REPO / f for f in entry["files"]]
            approved[entry["name"]] = {"id": r["id"], "sha256": _sha(paths), "files": entry["files"],
                                       "approved": _now(), "by": f"with patch #{r['id']}", "harm": r["harm"]}
    APPROVED.parent.mkdir(parents=True, exist_ok=True)
    APPROVED.write_text(json.dumps(approved, indent=1, ensure_ascii=False), encoding="utf-8")


def reject(uid: int, by: str) -> dict:
    r = next(x for x in reports() if x["id"] == uid)
    r["status"] = "rejected"
    _save(r)
    approved = _approved()
    if approved.get(r["name"], {}).get("id") == uid:
        approved.pop(r["name"])
        APPROVED.write_text(json.dumps(approved, indent=1, ensure_ascii=False), encoding="utf-8")
    _decide({"event": "rejected", "id": uid, "name": r["name"], "by": by})
    return r


def approved_pin(name: str) -> str:
    """The approved files' SHA-256 ('' if not approved) - what a cached tool list is valid for."""
    return _approved().get(name, {}).get("sha256", "")


def is_approved(name: str, files: list[str]) -> bool:
    """Approved, and still exactly the files that were approved."""
    entry = _approved().get(name)
    if not entry:
        return False
    paths = [REPO / f for f in files]
    return all(p.exists() for p in paths) and sorted(entry["files"]) == sorted(files) and _sha(paths) == entry["sha256"]


def pending() -> list[dict]:
    return [r for r in reports() if r["status"] == "pending"]


# --- the PIN for harm-3 upgrades (build plan: "Yes plus your PIN") --------------------------------------

def set_pin(pin: str) -> None:
    import os
    salt = os.urandom(16)
    PIN_FILE.write_text(json.dumps({"salt": salt.hex(), "hash": hashlib.pbkdf2_hmac(
        "sha256", pin.encode(), salt, 200_000).hex()}), encoding="utf-8")


def pin_ok(pin: str) -> bool:
    try:
        d = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(d["salt"]), 200_000).hex() == d["hash"]


def cli(args: list[str]) -> int:
    import getpass
    cmd = args[0] if args else "upgrades"
    if cmd == "review":  # the "Vaayu Upgrades" desktop shortcut: every pending report - all at once, or one by one
        waiting = pending()
        if not waiting:
            print("Nothing is waiting for your approval.")
        elif len(waiting) > 1:
            print(f"{len(waiting)} upgrades are waiting for you:\n")
            for r in waiting:
                print(f"  #{r['id']:<3} harm {r['harm']} ({HARM[r['harm']]:<8}) {r['name']:<22} {r['verdict'][:30]:<30} "
                      f"new powers: {', '.join(r['powers']) or 'none'}")
            choice = input("\nApprove ALL of them (all), go one by one (one), or none now (no)? ").strip().lower()
            if choice in ("all", "a", "yes", "y"):
                if any(r["harm"] >= 3 for r in waiting):
                    if not PIN_FILE.exists():
                        print("Some are harm 3 - choose your Guardian PIN now (at least 4 characters).")
                        if cli(["set-pin"]):
                            return 1
                    if not pin_ok(getpass.getpass("Your Guardian PIN: ")):
                        print("Wrong PIN - nothing approved.")
                        input("\nPress Enter to close.")
                        return 1
                for r in waiting:
                    if r["harm"] == 4:
                        print(f"  #{r['id']} {r['name']}: harm 4 is never approved this way - skipped.")
                        continue
                    try:
                        approve(r["id"], by="user (guardian review, approve all)")
                        print(f"  #{r['id']} {r['name']}: approved.")
                    except Exception as exc:
                        print(f"  #{r['id']} {r['name']}: NOT approved - {exc}")
                print("\nVaayu loads them by himself within a minute.")
                input("\nPress Enter to close.")
                return 0
            if choice not in ("one", "o", "one by one"):
                input("\nNothing approved. Press Enter to close.")
                return 0
        for r in waiting:
            print("\n" + "=" * 70 + "\n" + render(r))
            answer = input(f"Approve #{r['id']} '{r['name']}'? (yes / no / later) ").strip().lower()
            if answer in ("n", "no"):
                cli(["reject", str(r["id"])])
            elif answer in ("y", "yes"):
                if r["harm"] >= 3 and not PIN_FILE.exists():
                    print("Harm 3 needs your PIN - choose one now (at least 4 characters).")
                    if cli(["set-pin"]):
                        continue
                cli(["approve", str(r["id"]), "--confirmed"])
        input("\nPress Enter to close.")
        return 0
    if cmd == "upgrades":
        for r in reports():
            print(f"#{r['id']:<3} {r['status']:<9} harm {r['harm']} ({HARM[r['harm']]:<8}) {r['name']} - {r['verdict']}")
        return 0
    if cmd == "set-pin":
        pin = getpass.getpass("New Guardian PIN: ")
        if pin != getpass.getpass("Again: ") or len(pin) < 4:
            print("Didn't match (or shorter than 4) - no PIN set.")
            return 1
        set_pin(pin)
        print("PIN set.")
        return 0
    uid = int(args[1])
    r = next((x for x in reports() if x["id"] == uid), None)
    if r is None:
        print(f"No upgrade #{uid}.")
        return 1
    if cmd == "show":
        print(render(r))
        return 0
    if cmd == "reject":
        reject(uid, by="user (guardian CLI)")
        print(f"Rejected #{uid}.")
        return 0
    if cmd == "approve":
        if "--confirmed" not in args:  # from review: the report was just shown and the yes just said
            print(render(r))
        if r["harm"] >= 3:
            if not PIN_FILE.exists():
                print("Harm 3 needs your PIN - set one first: python -m guardian set-pin")
                return 1
            if not pin_ok(getpass.getpass("Your Guardian PIN: ")):
                print("Wrong PIN - not approved.")
                return 1
        elif "--confirmed" not in args and input("Approve? (yes/no) ").strip().lower() not in ("y", "yes"):
            print("Not approved.")
            return 1
        approve(uid, by="user (guardian CLI)")
        print(f"Approved #{uid}. Vaayu loads it on his next start.")
        return 0
    print(__doc__)
    return 1
