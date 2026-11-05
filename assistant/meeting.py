"""
Holding the user's place in a meeting (#15; user, 2026-11-05: "I can also ask him to join a meeting until I arrive").

"Join the meeting I copied" (a Google Meet link on the clipboard), or the link itself sent from the phone, at once or
"at 3 pm": Vaayu opens it in the browser he controls (CDP, port 9222), turns the mic and camera OFF before joining,
joins, posts one line in the meeting chat ("I'm running a few minutes late..."), turns on Meet's own live captions and
writes down what is said (data/meetings/<time>.txt). "I'm here" / "summarise the meeting" stops the notes and gives a
summary of what the user missed (Claude through the Pro login; the local model if not).

Joining and the chat line act in the user's name, so it is an outward action: the Guardian asks "yes?" first, every
time (guardian/action_rules.py). Vaayu never unmutes, never turns the camera on, never speaks in the meeting.

The page is read by its accessibility labels ("Turn off microphone", "Join now", "Captions") - if Meet renames them, a
step fails and Vaayu says which one.
"""

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

DIR = Path(__file__).resolve().parent.parent / "data" / "meetings"
LATE_LINE = "Hi everyone, I'm running a few minutes late - I'll be with you shortly."
MEET_LINK = re.compile(r"https?://meet\.google\.com/[a-z]{3,4}-[a-z]{4}-[a-z]{3,4}\S*", re.I)
_state: dict = {}
_lock = threading.Lock()

# --- the page --------------------------------------------------------------------------------------------------
_CLICK_JS = """(labels => {
  const all = [...document.querySelectorAll('button, [role=button]')].filter(b => b.offsetParent !== null);
  for (const want of labels) {
    const b = all.find(b => ((b.getAttribute('aria-label') || '') + ' ' + (b.innerText || '')).toLowerCase()
                              .includes(want.toLowerCase()));
    if (b) { b.click(); return want; }
  }
  return '';
})(%s)"""
_CAPTIONS_JS = """(() => {
  const r = document.querySelector('[role=region][aria-label*="aption"], [aria-label="Captions"]');
  return r ? r.innerText : '';
})()"""
_CHAT_JS = """(line => {
  const box = document.querySelector('textarea[aria-label*="message" i], textarea');
  if (!box) return false;
  const set = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
  set.call(box, line);
  box.dispatchEvent(new Event('input', {bubbles: true}));
  const send = [...document.querySelectorAll('button')].find(b => /send/i.test(b.getAttribute('aria-label') || ''));
  if (send) { send.disabled = false; send.click(); return true; }
  return false;
})(%s)"""


def _eval(ws: str, js: str):
    from assistant.tools.browser_automation import _send_cdp_command
    out = _send_cdp_command(ws, "Runtime.evaluate", {"expression": js, "returnByValue": True})
    return ((out.get("result") or {}).get("result") or {}).get("value")


def _click(ws: str, *labels: str) -> str:
    return _eval(ws, _CLICK_JS % json.dumps(list(labels))) or ""


def _wait_click(ws: str, labels: tuple, seconds: float, sleep=time.sleep) -> str:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if hit := _click(ws, *labels):
            return hit
        sleep(1)
    return ""


def join(link: str, line: str = LATE_LINE, sleep=time.sleep) -> str:
    from assistant.tools.browser import _new_tab
    tab = _new_tab(link)
    ws = tab["webSocketDebuggerUrl"]
    sleep(6)  # the pre-join page
    _wait_click(ws, ("Turn off microphone",), 10, sleep)  # off BEFORE joining - never heard
    _click(ws, "Turn off camera")
    joined = _wait_click(ws, ("Join now", "Ask to join", "Switch here"), 20, sleep)
    if not joined:
        raise RuntimeError("I opened the meeting but couldn't find its Join button.")
    left_waiting = _wait_click(ws, ("Turn on captions",), 120, sleep)  # in once captions can be turned on
    said_late = False
    if line and _click(ws, "Chat with everyone", "Open chat"):
        sleep(1.5)
        said_late = bool(_eval(ws, _CHAT_JS % json.dumps(line)))
        _click(ws, "Close")
    DIR.mkdir(parents=True, exist_ok=True)
    notes = DIR / f"{datetime.now():%Y-%m-%d_%H%M}.txt"
    with _lock:
        _state.clear()
        _state.update(ws=ws, notes=notes, link=link, stop=threading.Event(), lines=[])
    threading.Thread(target=_take_notes, args=(_state["stop"],), daemon=True, name="meeting-notes").start()
    waiting = "" if left_waiting else " I'm still waiting to be let in."
    return ("I've joined with your mic and camera off" + (" and told them you're on your way" if said_late else "")
            + f". I'm taking notes - say 'I'm here' when you arrive.{waiting}")


def _take_notes(stop: threading.Event, every: float = 2.0) -> None:
    seen = ""
    while not stop.wait(every):
        try:
            text = _eval(_state["ws"], _CAPTIONS_JS) or ""
        except Exception:
            continue
        # the caption box shows the last few lines; keep only what wasn't there last time
        new = text[len(seen):] if text.startswith(seen) else text
        if new.strip():
            with _lock:
                _state["lines"].append(new.strip())
                _state["notes"].write_text("\n".join(_state["lines"]), encoding="utf-8")
        seen = text


def stop_and_summarise() -> str:
    with _lock:
        if not _state:
            return "I'm not in a meeting for you right now."
        _state["stop"].set()
        transcript, notes = "\n".join(_state["lines"]), _state["notes"]
        _state.clear()
    if not transcript.strip():
        return "Welcome. I couldn't read anything from the captions, so I have no notes - sorry."
    prompt = ("These are live captions from a meeting the user joined late. In 3-5 short spoken sentences tell them what "
              "they missed: the main points, any decisions, and anything asked of them. No markdown.\n\n"
              + transcript[-12000:])
    from assistant.tools import think
    summary = think.ask_claude(prompt) or think._local(prompt)
    return f"Welcome. {re.sub(r'[*#`]+', '', summary)} The full notes are in {notes.name}."


def clipboard_link() -> str | None:
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
    except Exception:
        return None
    m = MEET_LINK.search(text or "")
    return m.group(0) if m else None


def join_later(link: str, when: str) -> str:
    from assistant import alarms
    at = alarms.parse_time(when)
    delay = max(0.0, (at - datetime.now()).total_seconds())

    def go():
        from assistant import agent
        try:
            said = join(link)
        except Exception as exc:
            said = f"I couldn't join the meeting: {exc}"
        if agent.ANNOUNCE:
            agent.ANNOUNCE(said)
    threading.Timer(delay, go).start()
    return f"Okay, I'll join it at {at:%I:%M %p}.".replace(" 0", " ")
