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
_CRITICAL_TEXT = re.compile(r"Login Data|Cookies|Local State|CurrentVersion\\\\?Run|Start Menu\\\\?Programs\\\\?Startup|"
                            r"runas|UAC|DisableAntiSpyware|Set-MpPreference|netsh|bcdedit|schtasks|"
                            r"reg add|vssadmin", re.I)
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
        h.update(f.relative_to(REPO).as_posix().encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def _decide(event: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), **event}, ensure_ascii=False) + "\n")


def scan(files: list[Path]) -> dict:
    """What the code can do, from the code itself."""
    from guardian.rules import is_payment_target
    powers, domains, critical, notes = set(), set(), [], []
    for f in files:
        rel = f.relative_to(REPO).as_posix()
        if rel.startswith("guardian/"):
            critical.append(f"changes the Guardian itself ({rel})")
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
    found = scan(paths)
    level, reasons = harm_level(found, kind)
    passed, test_line = _run_tests(tests or [])
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
    report = {"id": uid, "name": name, "kind": kind, "status": "pending", "created": _now(),
              "files": [p.relative_to(REPO).as_posix() for p in paths], "sha256": _sha(paths),
              "claim": claim, "harm": level, "harm_reasons": reasons, "mismatches": mismatches,
              "tests_passed": passed, "tests": test_line, "rollback": f"git checkout {tag}",
              "verdict": verdict, **found}
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
    if cmd == "review":  # the "Vaayu Upgrades" desktop shortcut: every pending report, one by one
        waiting = pending()
        if not waiting:
            print("Nothing is waiting for your approval.")
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
