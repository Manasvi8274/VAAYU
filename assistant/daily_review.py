"""
The daily log review: scan the day's logs, update Vaayu from them, then delete the logs.

User's request: "scan them once a day and then update vaayu on that basis and
then delete the logs... triggered manually by me, or when i say to u, and at a
particular time of each day. when all the scanning and update is done, then the
logs should be deleted."

    1. snapshot   - a small digest (data/reviews/YYYY-MM-DD/digest.md) of what
                    happened: failures, slow turns, mishearings, new names...
    2. local      - facts learned from the day's chats (memory), names the user
                    keeps using added to the known list (names.py)
    3. Claude     - if the `claude` CLI is installed (the user's Claude Pro
                    login, never API credits): a fix session in a separate git
                    worktree on branch vaayu-review-YYYYMMDD. Its commits reach
                    vaayu-v2 only if they don't touch the guardian
                    (guardian.rules.PROTECTED_PATHS) and the full test suite
                    passes again here ("fully automatic once the claude access
                    is given"). Never pushed. Vaayu picks the changes up on its
                    next restart.
    4. delete     - the raw logs go only after 1-3 finished. data/reviews/,
                    memory and the watch history are kept; the newest 300 voice
                    clips are kept for voice training. data/guardian.log is the
                    guardian's and is never touched.

Triggers: the daily self_review.check_in_time (orchestrator), "Vaayu, review your
logs" (tools/self_review_control.py), or `python -m assistant.daily_review`.
"""

import json
import re
import shutil
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INTERACTIONS = DATA / "interactions.jsonl"
ISSUES = DATA / "issues_log.jsonl"
VAAYU_LOG = DATA / "vaayu.log"
AUDIO_DIR = DATA / "audio"
REVIEWS = DATA / "reviews"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
AUDIO_KEEP = 300  # newest voice clips kept for training the speech model on the user's voice
SLOW_MS = 8000
# Build plan's latency budget: a simple command's first spoken word within ~2 s on the 1650 - measured as
# hearing (STT) + deciding (brain) for the turns the fast rules answered, i.e. before the voice starts.
FIRST_WORD_BUDGET_MS = 2000
CLAUDE_TIMEOUT = 30 * 60
TEST_TIMEOUT = 20 * 60
_LEARN_MAX = 50  # chat turns learned from per review (each is one model call)
_NAME_MIN_USES = 3  # a name used this often in successful commands is worth knowing
_lock = threading.Lock()

_CLAUDE_TOOLS = ["Read", "Edit", "Write", "Grep", "Glob", "Bash(git *)", f"Bash({PYTHON.as_posix()} *)",
                 "Bash(.venv/Scripts/python.exe *)", "Bash(ls *)"]
# Edit(...) rules cover every file-editing tool (the CLI rejects Write(path) deny rules).
_CLAUDE_DENY = ["Edit(guardian/**)", "Bash(git push *)", "Bash(git push)"]
_LIMIT_HIT = re.compile(r"hit your (?:session|usage|weekly) limit|usage limit|rate limit", re.I)


def _log(review_dir: Path, line: str) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    with (review_dir / "review.log").open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


# --- 1. snapshot ------------------------------------------------------------------------------------------

def _failed(turn: dict) -> bool:
    reply = turn.get("reply") or ""
    return bool(turn.get("failed")) or reply.startswith(("That didn't work", "Sorry"))


def digest(events: list[dict], issues: list[dict], vaayu_log: str) -> tuple[str, dict]:
    """(markdown report, the numbers) - small on purpose: < 50 KB however long the day was."""
    from assistant.tools import names
    turns = [e for e in events if e.get("kind") == "turn"]
    kinds = Counter(e.get("kind") for e in events)
    failures: dict[str, list[str]] = defaultdict(list)
    for t in turns:
        if _failed(t):
            failures[(t.get("reply") or "")[:90]].append(t.get("text") or "")
    errors = Counter(e.get("where", "?") for e in events if e.get("kind") == "error")
    slow = sorted((t for t in turns if (t.get("brain_ms") or 0) > SLOW_MS), key=lambda t: -(t.get("brain_ms") or 0))
    misheard, mentioned = Counter(), Counter()
    known = {k: {n.lower() for n in v} for k, v in names.load().items()}
    for t in turns:
        for step in t.get("trace") or []:
            if fix := step.get("name_corrected"):
                for key, heard in (fix.get("from") or {}).items():
                    if heard != (fix.get("to") or {}).get(key):
                        misheard[f"{heard} -> {fix['to'][key]}"] += 1
            for (tool, arg), k in names.ARG_KINDS.items():
                value = (step.get("args") or {}).get(arg) if step.get("tool") == tool else None
                if isinstance(value, str) and value.lower() not in known[k]:
                    mentioned[(k, value, bool(step.get("ok")))] += 1
    simple = [(t.get("stt_ms") or 0) + (t.get("brain_ms") or 0) for t in turns
              if any(step.get("path") in ("fast", "fast_reply") for step in t.get("trace") or [])]
    tracebacks = re.findall(r"^(\w+(?:Error|Exception)[^\n]{0,160})$", vaayu_log or "", re.M)
    open_issues = [i for i in issues if not i.get("acknowledged")]
    stats = {"turns": len(turns), "failed": sum(len(v) for v in failures.values()), "errors": sum(errors.values()),
             "ignored_media": kinds["ignored_media"], "barge_in": kinds["barge_in"], "stops": kinds["stop"],
             "name_corrected": sum(misheard.values()), "slow": len(slow), "issues": len(open_issues),
             "tracebacks": len(tracebacks),
             "simple_within_budget": sum(ms <= FIRST_WORD_BUDGET_MS for ms in simple), "simple": len(simple),
             "simple_median_ms": sorted(simple)[len(simple) // 2] if simple else 0}

    out = [f"# Vaayu daily review - {datetime.now():%A %d %B %Y, %H:%M}", "",
           f"- Turns: {stats['turns']} | failed replies: {stats['failed']} | errors logged: {stats['errors']}",
           f"- Ignored as video audio: {stats['ignored_media']} | 'stop' while talking: {stats['barge_in']} | "
           f"Vaayu stopped: {stats['stops']}",
           f"- Names corrected by sound: {stats['name_corrected']} | slow turns (> {SLOW_MS // 1000} s): {stats['slow']}",
           f"- Open issues (issues_log): {stats['issues']} | tracebacks in vaayu.log: {stats['tracebacks']}",
           f"- Latency budget (simple commands, heard + decided within {FIRST_WORD_BUDGET_MS / 1000:.0f} s): "
           + (f"{stats['simple_within_budget']}/{stats['simple']}, median {stats['simple_median_ms'] / 1000:.1f} s"
              if simple else "no simple commands today"), ""]
    if failures:
        out += ["## Failed replies (most common first)", ""]
        for reply, texts in sorted(failures.items(), key=lambda kv: -len(kv[1]))[:25]:
            examples = "; ".join(f"'{t[:80]}'" for t in list(dict.fromkeys(texts))[:3])
            out.append(f"- {len(texts)}x \"{reply}\" - said: {examples}")
        out.append("")
    if errors:
        out += ["## Errors", ""] + [f"- {n}x {where}" for where, n in errors.most_common(15)] + [""]
    if slow:
        out += ["## Slowest turns", ""] + [f"- {t['brain_ms'] / 1000:.1f} s: '{(t.get('text') or '')[:90]}'"
                                           for t in slow[:10]] + [""]
    if misheard:
        out += ["## Names corrected by sound (heard -> used)", ""] + [f"- {n}x {pair}" for pair, n in misheard.most_common(20)] + [""]
    if mentioned:
        out += ["## Names not in the known list", ""] + [
            f"- {n}x {kind} '{value}'" + ("" if ok else " (the command failed)")
            for (kind, value, ok), n in mentioned.most_common(20)] + [""]
    if open_issues:
        out += ["## Issues logged during use", ""] + [
            f"- [{i.get('kind', 'tool_failure')}] '{(i.get('user_said') or '')[:80]}' -> {i.get('tool') or '-'}: "
            f"{(i.get('error') or '')[:140]}" for i in open_issues[-25:]] + [""]
    if tracebacks:
        out += ["## Exceptions in vaayu.log", ""] + [f"- {n}x {line}" for line, n in Counter(tracebacks).most_common(10)] + [""]
    text = "\n".join(out)
    return text[:49_000], stats | {"mentioned": mentioned}


# --- 2. local updates ------------------------------------------------------------------------------------

def learn_locally(events: list[dict], review_dir: Path, mentioned: Counter) -> list[str]:
    """Facts from the day's chats into memory; names the user keeps using into the known list."""
    done = []
    chats = [t for t in events if t.get("kind") == "turn" and any(s.get("path") == "chat" for s in t.get("trace") or [])]
    if chats:
        try:
            import ollama
            from assistant import memory
            from assistant.config_schema import load_config
            cfg = load_config().llm
            client = ollama.Client(host=cfg.host, timeout=60)
            learned = sum(memory.learn(client, cfg.model, t.get("text") or "", t.get("reply") or "")
                          for t in chats[-_LEARN_MAX:])
            done.append(f"learned {learned} new fact(s) from {min(len(chats), _LEARN_MAX)} chat turn(s)")
        except Exception as exc:
            _log(review_dir, f"memory learning skipped: {exc!r}")
    from assistant.tools import names
    for (kind, value, ok), n in mentioned.items():
        if ok and n >= _NAME_MIN_USES:
            names.remember_name(kind, value)
            done.append(f"now knows the {kind} '{value}' (used {n} times)")
    return done


# --- 3. Claude fix session -----------------------------------------------------------------------------------

def protected_changes(changed: list[str]) -> list[str]:
    """Files a review may never change (the guardian) - any of these and the whole branch is discarded."""
    from guardian.rules import PROTECTED_PATHS
    return [f for f in changed if f.replace("\\", "/").startswith(PROTECTED_PATHS)]


def _git(*args, cwd: Path = ROOT, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _claude_prompt(review_dir: Path, branch: str) -> str:
    return f"""You are reviewing yesterday's real use of Vaayu, the user's local voice assistant (this repo).
Read {review_dir / 'digest.md'} first, then the raw logs in {review_dir / 'raw'} (interactions.jsonl: one JSON
event per line - 'heard', 'turn' with the reply and a 'trace' of what the router/LLM/tools did, 'error'...).

Find REAL problems the user hit - wrong actions, failed or slow replies, mishearings the router should
tolerate, hallucinated answers - and fix the root cause of each in assistant/, scripts/ or tests/ ONLY.
Never edit anything under guardian/ (the user's untouchable safety layer) - a change there discards
everything. Never touch config/, data/ or models/. Never push.

For every fix add a regression test using the real utterance from the log. Run the full suite with
{PYTHON} -m pytest -q -x . Commit only when it passes, on the current branch ({branch}), with a clear message
ending with: Co-Authored-By: Claude <noreply@anthropic.com>. If nothing in the logs needs fixing, change nothing.

Finally write a short plain-English summary for the user - what you found, what you fixed, what you left and
why - to {review_dir / 'claude_summary.md'}."""


class ClaudeNotRun(Exception):
    """The fix session didn't happen (usage limit, timeout, CLI error) - the logs are kept for the next review."""


def claude_session(review_dir: Path, day: str) -> str:
    """One fix session in a separate worktree; returns what happened, in a line."""
    claude = shutil.which("claude")
    if not claude:
        return "Claude CLI not installed - skipped the fix session."
    review_dir.mkdir(parents=True, exist_ok=True)
    branch = f"vaayu-review-{day.replace('-', '')}"
    if _git("rev-parse", "--verify", "--quiet", branch).returncode == 0:
        branch += datetime.now().strftime("-%H%M%S")
    tree = ROOT.parent / f"{ROOT.name}-{branch}"
    made = _git("worktree", "add", "-b", branch, str(tree), "vaayu-v2")
    if made.returncode != 0:
        raise ClaudeNotRun(f"couldn't make the review worktree: {made.stderr.strip()[:200]}")
    commits = "0"
    try:
        if (ROOT / "config" / "config.yaml").exists():
            shutil.copy(ROOT / "config" / "config.yaml", tree / "config" / "config.yaml")
        cmd = [claude, "-p", _claude_prompt(review_dir, branch), "--permission-mode", "acceptEdits",
               "--output-format", "text", "--add-dir", str(review_dir),
               "--allowedTools", *_CLAUDE_TOOLS, "--disallowedTools", *_CLAUDE_DENY]
        try:
            run = subprocess.run(cmd, cwd=tree, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT,
                                 encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            said = (run.stdout or "") + "\n--- stderr ---\n" + (run.stderr or "")
            code = run.returncode
        except subprocess.TimeoutExpired:
            said, code = "(timed out)", "timeout"
        (review_dir / "claude_output.txt").write_text(said, encoding="utf-8")
        commits = _git("rev-list", "--count", f"vaayu-v2..{branch}").stdout.strip() or "0"
        if commits == "0":
            if _LIMIT_HIT.search(said):  # found live: "You've hit your session limit - resets 11am"
                raise ClaudeNotRun("Claude usage limit reached - " + said.strip().splitlines()[0][:120])
            if code != 0:
                raise ClaudeNotRun(f"claude exited with {code}: {said.strip()[:160]}")
            return "Claude found nothing to change." if (review_dir / "claude_summary.md").exists() \
                else "The Claude session made no commits."
        changed = _git("diff", "--name-only", f"vaayu-v2...{branch}").stdout.split()
        if bad := protected_changes(changed):
            commits = "0"  # discard the whole branch below
            return f"Discarded Claude's changes: they touched the guardian ({', '.join(bad[:5])})."
        tests = subprocess.run([str(PYTHON), "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=tree,
                               capture_output=True, text=True, timeout=TEST_TIMEOUT, encoding="utf-8", errors="replace",
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tail = (tests.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        if tests.returncode != 0:
            return f"Kept Claude's fixes on branch {branch} only - the tests failed here: {tail[0]}"
        if _git("branch", "--show-current").stdout.strip() != "vaayu-v2":
            return f"Claude's fixes pass the tests ({tail[0]}) - left on {branch}: vaayu-v2 isn't checked out."
        merged = _git("merge", "--ff-only", branch)
        if merged.returncode != 0:
            return f"Claude's fixes pass ({tail[0]}) but couldn't fast-forward vaayu-v2: {merged.stderr.strip()[:200]}"
        return f"Applied {commits} commit(s) from Claude to vaayu-v2 ({tail[0]}); Vaayu uses them after its next restart."
    finally:
        _git("worktree", "remove", "--force", str(tree))
        if commits == "0":  # nothing worth keeping (or guardian changes): no leftover branch
            _git("branch", "-D", branch)


# --- 4. delete the logs ------------------------------------------------------------------------------------

def _clear(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        path.unlink()
    except PermissionError:  # the running Vaayu holds it open (vaayu.log): empty it in place
        with open(path, "r+b") as f:
            f.truncate(0)
    return True


def delete_logs(review_dir: Path) -> list[str]:
    gone = [p.name for p in (INTERACTIONS, ISSUES, VAAYU_LOG) if _clear(p)]
    clips = sorted(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []
    for old in clips[:-AUDIO_KEEP] if len(clips) > AUDIO_KEEP else []:
        old.unlink(missing_ok=True)
    shutil.rmtree(review_dir / "raw", ignore_errors=True)
    removed = max(0, len(clips) - AUDIO_KEEP)
    return gone + ([f"{removed} older voice clips (kept the newest {AUDIO_KEEP} for voice training)"] if removed else [])


# --- the whole review ----------------------------------------------------------------------------------------

def run_review(trigger: str = "manual", use_claude: bool | None = None) -> Path | None:
    """Scan, update, delete. None if a review is already running."""
    if not _lock.acquire(blocking=False):
        return None
    try:
        day = datetime.now().date().isoformat()
        review_dir = REVIEWS / day
        if (review_dir / "digest.md").exists():
            review_dir = REVIEWS / f"{day}_{datetime.now():%H%M%S}"
        _log(review_dir, f"review started ({trigger})")
        events, issues = _read_jsonl(INTERACTIONS), _read_jsonl(ISSUES)
        try:
            vaayu_log = VAAYU_LOG.read_text(encoding="utf-8", errors="replace")
        except OSError:
            vaayu_log = ""
        report, stats = digest(events, issues, vaayu_log)
        (review_dir / "digest.md").write_text(report, encoding="utf-8")  # must exist before anything is deleted
        raw = review_dir / "raw"
        raw.mkdir(exist_ok=True)
        for p in (INTERACTIONS, ISSUES, VAAYU_LOG):
            if p.exists():
                shutil.copy(p, raw / p.name)

        updates = []
        try:
            updates = learn_locally(events, review_dir, stats["mentioned"])
        except Exception as exc:
            _log(review_dir, f"local update failed: {exc!r}")
        if use_claude is None:
            try:
                from assistant.config_schema import load_config
                use_claude = load_config().self_review.claude_fix
            except Exception:
                use_claude = True
        claude, keep = "Claude fix session off (self_review.claude_fix: false).", False
        if use_claude and stats["turns"]:
            try:
                claude = claude_session(review_dir, day)
            except ClaudeNotRun as exc:  # "update done" isn't true yet - keep the logs for the next review
                claude, keep = f"Claude fix session didn't run ({exc}) - kept the logs for the next review.", True
            except Exception as exc:
                claude = f"Claude fix session failed: {exc!r}"
        elif use_claude:
            claude = "No turns to review - skipped the Claude session."
        _log(review_dir, claude)

        if keep:
            shutil.rmtree(review_dir / "raw", ignore_errors=True)  # the originals are still there
            deleted = []
        else:
            deleted = delete_logs(review_dir)
        with (review_dir / "digest.md").open("a", encoding="utf-8") as f:
            f.write("\n## What the review did\n\n" + "".join(f"- {u}\n" for u in updates)
                    + f"- {claude}\n- Deleted: {', '.join(deleted) or ('nothing - kept for the next review' if keep else 'nothing (no logs)')}\n")
        _log(review_dir, "review finished")
        return review_dir
    finally:
        _lock.release()


def start_in_background(trigger: str) -> bool:
    """False if one is already running."""
    if _lock.locked():
        return False
    threading.Thread(target=run_review, args=(trigger,), daemon=True, name="daily-review").start()
    return True


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    where = run_review("cli", use_claude=False if "--no-claude" in sys.argv else None)
    print(f"review written to {where / 'digest.md'}" if where else "a review is already running")
