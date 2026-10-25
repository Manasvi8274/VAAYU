"""
Action rules: every tool call Vaayu makes passes these plain-Python checks - no AI model, well under
1 ms (build plan, "The Guardian: Vaayu's firewall"). In guardian/ so no self-upgrade can change them;
only the user, by hand.

Tiers (build plan):
  0 safe       open an app, read the time, search the web, play music     runs at once
  1 change     type into apps, click in them, close, volume, notes        runs, logged
  2 outward    send a message, post, submit, email                        needs the user's "yes" every time
  3 dangerous  pay, buy, delete, install/uninstall, format                blocked - the user does it himself

Also: rate limits (a runaway loop can't fire 100 clicks), never an outward action right after Vaayu
read a web page / file / message in the same turn (the instruction may be the page's, not the user's),
and an append-only log of every action: data/actions.log (kept by the daily review, never deleted).
"""

import json
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

SAFE, CHANGE, OUTWARD, DANGEROUS = 0, 1, 2, 3
LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "actions.log"

_SAFE = {
    "answer_question", "get_weather", "get_audio_status", "get_network_status", "get_system_status",
    "list_alarms", "list_browser_tabs", "list_connected_devices", "list_notes", "list_open_windows",
    "read_file", "read_page", "search_files", "take_screenshot", "open_app", "open_website", "browse_to",
    "play_music", "music_control", "ask_about_song", "play_video_on_youtube", "play_latest_video_from_channel",
    "play_recommended", "youtube_control", "control_video_playback", "seek_video", "set_video_fullscreen",
    "set_video_volume", "switch_to_browser_tab", "switch_to_window", "restore_window", "minimize_window",
    "set_alarm", "set_timer", "cancel_alarm", "stop_alarm", "set_voice", "remember_name", "move_status_orb",
    "review_logs", "set_daily_update_time", "app_controls", "app_read", "focus_input_field",
    # MCP servers' read-only tools (mcp_servers/servers.json)
    "sequentialthinking", "list_allowed_directories", "read_text_file", "read_media_file", "read_multiple_files",
    "list_directory", "list_directory_with_sizes", "directory_tree", "get_file_info", "browser_snapshot",
    "browser_navigate", "browser_tabs", "browser_take_screenshot", "browser_close", "browser_wait_for", "fetch",
    "search", "fetch_content", "search_wikipedia", "get_article", "get_summary", "get_related_topics",
    "get_sections", "get_links", "get_coordinates", "get_current_time", "convert_time", "calculate", "git_status",
    "git_log", "git_diff", "git_diff_staged", "git_diff_unstaged", "git_show", "git_branch", "get_document_text",
    "get_document_info", "read_data_from_excel", "list_presentations", "list_designs", "show_plan", "list_plans",
    "list_tasks", "think_deeply", "get_new_ability",
}
_OUTWARD = {"send_whatsapp_message", "type_sensitive_field", "browser_file_upload"}
# Tools whose risk is in WHAT they click: "click Send" is outward, "click Pay" is blocked - and which of their
# arguments name that thing (what is TYPED is not judged: "I want to buy..." typed in a box is just text).
_CLICK_KEYS = {"click_on_page": ("description",), "app_click": ("target",), "app_menu": ("path",),
               "browser_click": ("element",), "browser_select_option": ("element",), "browser_type": ("element",),
               "browser_fill_form": ("fields",), "browser_press_key": ("key",)}
_CLICKS = set(_CLICK_KEYS)
_OUTWARD_WORDS = re.compile(r"\b(?:send|post|submit|publish|reply|share|tweet|email|e-mail|upload|forward|"
                            r"bhejo|bhej do|book|confirm order)\b", re.I)
_DANGEROUS_WORDS = re.compile(r"\b(?:pay|payment|buy|purchase|checkout|check out|place order|add money|transfer|"
                              r"delete|remove|uninstall|install|format|erase|wipe|factory reset|"
                              r"empty recycle bin|sign out of all)\b", re.I)
# Tools that bring outside text into the turn - an outward action right after one is refused.
CONTENT_TOOLS = {"read_page", "read_file", "answer_question", "app_read", "search_files", "get_weather",
                 "fetch", "fetch_content", "search", "browser_snapshot", "browser_navigate", "read_text_file",
                 "read_multiple_files", "get_article", "get_summary", "get_document_text", "read_data_from_excel"}

_PER_MINUTE = {CHANGE: 30, OUTWARD: 3}  # calibration knobs
_recent: dict[int, deque] = {CHANGE: deque(), OUTWARD: deque()}


def tier(name: str, args: dict | None = None) -> int:
    args = args or {}
    from guardian.rules import is_payment_target
    # a payment site/app as WHERE to go or WHAT to act on, by any tool (a browser told to go to paytm.com) - not
    # what something is about ("a presentation on UPI payments" is fine)
    if any(isinstance(v, str) and is_payment_target(v) for k, v in args.items()
           if k in ("url", "url_or_query", "app", "app_name", "target", "element")):
        return DANGEROUS
    if name in _CLICKS:
        what = " ".join(str(v) for k, v in args.items() if k in _CLICK_KEYS[name] + ("label",))
        # ponytail: buttons are short ("Pay now", "Delete", "Place order"); a long description is a link or
        # a video ("the one about how to install python") - judged by its words only when it's button-sized
        if len(re.split(r"\s+|>", what.strip())) > 5:
            return CHANGE
        if _DANGEROUS_WORDS.search(what):
            return DANGEROUS
        if _OUTWARD_WORDS.search(what):
            return OUTWARD
        return CHANGE
    if name in _OUTWARD:
        return OUTWARD
    if name in _SAFE:
        return SAFE
    return CHANGE  # anything unknown (a new MCP tool...) is at least logged and rate-limited


def needs_yes(name: str, args: dict | None = None) -> bool:
    return tier(name, args) == OUTWARD


def refusal(name: str, args: dict | None = None, after_content: bool = False) -> str | None:
    """Why this call may NOT run (said to the user as is), or None if it may."""
    level = tier(name, args)
    if level == DANGEROUS:
        return ("That's something I never do myself - paying, buying, deleting or installing. "
                "Please do that one yourself.")
    if level >= OUTWARD and after_content:
        return ("I won't do that - it came from what I just read, not from you. "
                "If you want it, tell me yourself.")
    limit = _PER_MINUTE.get(level)
    if limit:
        now, q = time.time(), _recent[level]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            return "That's too many actions too fast - I've paused for a moment. Ask me again in a minute."
        q.append(now)
    return None


def log(name: str, args: dict | None, ok: bool, error: str = "", path: str = "") -> None:
    """Append-only: one line per action - time, tool, tier, what, outcome."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        what = {k: (str(v)[:200]) for k, v in (args or {}).items()}
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "tool": name,
                                "tier": tier(name, args), "args": what, "ok": ok, "error": error[:200],
                                "path": path}, ensure_ascii=False) + "\n")
    except OSError:
        pass  # a full disk must not stop Vaayu from working
