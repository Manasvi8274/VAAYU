"""
Open and close every app on this laptop with Vaayu's own open_app / close_app.

    .venv/Scripts/python.exe scripts/app_sweep.py            # all of them (~10 s each)
    .venv/Scripts/python.exe scripts/app_sweep.py notepad calc   # just these

Safe by design:
  * skips uninstallers, setup/repair/reset tools, help/links, and admin tools (they pop a Windows
    permission prompt - which the guardian also treats as a threat)
  * never closes an app that was already running before it opened it (your editor, browser, chats)
Results: data/app_sweep.json and a summary printed at the end.
"""

import json
import re
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from assistant.tools import apps  # noqa: E402

OUT = ROOT / "data" / "app_sweep.json"
_SKIP = re.compile(
    r"uninstall|setup|install|repair|reset|update|upgrade|remove|recovery|readme|read me|help|documentation|manual|"
    r"license|website|web site|support|feedback|report a problem|release notes|what's new|register|activate|crash|"
    r"debug|diagnostic|safe mode|elevated|administrator|registry|disk cleanup|defragment|optimize drives|"
    r"computer management|services|task scheduler|event viewer|system configuration|security policy|firewall|odbc|"
    r"iscsi|component services|print management|memory diagnostic|hyper-v|system information|resource monitor|"
    r"performance monitor|windows tools|local security|group policy|bitlocker|backup|restore|shutdown|restart|"
    r"sign out|log off|python|pythonw|idle \(|vaayu|guardian|claude code|uninstaller|benchmark|task manager|"
    # remote access / side effects / takes over input / Vaayu's own brain:
    r"getscreen|quick assist|anydesk|teamviewer|utorrent|bittorrent|narrator|voice access|ollama|lp-app", re.I)
_OPEN_WAIT, _HEAVY_WAIT = 6.0, 30.0
_HEAVY = re.compile(r"tekken|steam|nvidia|epic|unity|visual studio(?! code)|android studio|blender|game|god of war|assassin|"
                    r"wukong|black myth|stellar blade|minecraft|pycharm|xbox", re.I)


def windowed_pids() -> dict[int, str]:
    import win32gui
    import win32process
    out = {}

    def add(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            pid = win32process.GetWindowThreadProcessId(hwnd)[1]
            out[pid] = win32gui.GetWindowText(hwnd)
    win32gui.EnumWindows(add, None)
    return out


def uac_prompt() -> bool:
    return any((p.info.get("name") or "").lower() == "consent.exe" for p in psutil.process_iter(["name"]))


def sweep(only: list[str]) -> list[dict]:
    names = apps._start_menu_apps()
    todo = sorted(n for n, app_id in names.items()
                  if not _SKIP.search(n) and (app_id.startswith("steam://") or "://" not in app_id[:10])
                  and not re.search(r"\.(?:url|chm|pdf|txt|html?|md)$", app_id, re.I)
                  and (not only or any(o.lower() in n.lower() for o in only)))
    todo.sort(key=lambda n: bool(_HEAVY.search(n)))  # games and big IDEs last
    skipped = sorted(set(names) - set(todo))
    print(f"{len(todo)} apps to open and close ({len(skipped)} skipped as unsafe/not apps)\n", flush=True)
    results = []
    for i, name in enumerate(todo, 1):
        row = {"app": name, "opened": False, "closed": None, "note": ""}
        before_windows = windowed_pids()
        t = time.time()
        try:
            reply = apps.open_app(name, foreground=False)
            row["reply"] = reply
        except Exception as exc:
            row["note"] = f"open failed: {exc}"[:200]
            results.append(row)
            print(f"[{i}/{len(todo)}] {name}: OPEN FAILED - {row['note']}", flush=True)
            continue
        wait = _HEAVY_WAIT if _HEAVY.search(name) else _OPEN_WAIT
        new = {}
        while time.time() - t < wait:
            time.sleep(0.5)
            if uac_prompt():
                row["note"] = "asked for admin permission (Windows prompt) - answer it 'No'"
                break
            new = {p: w for p, w in windowed_pids().items() if p not in before_windows}
            # a slow app's first new window can be someone else's (Steam's "Launching..." before TEKKEN's own)
            if new and time.time() - t > 2.5 and (apps._opened_entry(name) or not _HEAVY.search(name)):
                break
        time.sleep(15.0 if _HEAVY.search(name) else 1.0)  # a game ignores X while still loading; tracker catch-up
        entry = apps._opened_entry(name)
        row["opened"] = bool(new or entry)  # a Store app's window belongs to an already-running host
        row["window"] = next(iter(new.values()), "")
        if not entry:
            row["note"] = row["note"] or "no new window of its own (already open, or it lives in the tray)"
            row["closed"] = None  # never close what the user already had open
        else:
            import win32gui
            try:
                row["close_reply"] = apps.close_app(name)
                time.sleep(1.0)
                left_windows = [h for h in entry["hwnds"] if win32gui.IsWindow(h) and win32gui.IsWindowVisible(h)]
                left_procs = [p for p in entry["pids"] if psutil.pid_exists(p) and apps._proc_name(p) not in apps._SHARED_HOSTS]
                row["closed"] = not left_windows and not left_procs
                if not row["closed"]:
                    row["note"] = f"still open: {len(left_windows)} window(s), {[apps._proc_name(p) for p in left_procs]}"
            except Exception as exc:
                row["closed"] = False
                row["note"] = f"close failed: {exc}"[:200]
                for p in entry["pids"]:  # tidy up so the next app starts clean (never a shared Windows host)
                    if apps._proc_name(p) not in apps._SHARED_HOSTS:
                        try:
                            psutil.Process(p).terminate()
                        except psutil.Error:
                            pass
        results.append(row)
        print(f"[{i}/{len(todo)}] {name}: opened {'yes' if row['opened'] else 'NO '} | "
              f"closed {'-' if row['closed'] is None else 'yes' if row['closed'] else 'NO '} | {row['note']}",
              flush=True)
        OUT.write_text(json.dumps({"results": results, "skipped": skipped}, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    return results


if __name__ == "__main__":
    res = sweep(sys.argv[1:])
    opened = sum(r["opened"] for r in res)
    tried_close = [r for r in res if r["closed"] is not None]
    closed = sum(bool(r["closed"]) for r in tried_close)
    print(f"\nopened {opened}/{len(res)} | closed {closed}/{len(tried_close)} of those it opened")
    for r in res:
        if not r["opened"] or r["closed"] is False:
            print("  -", r["app"], ":", r["note"] or "no window seen")
