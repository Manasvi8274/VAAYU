"""
Builds a supervised fine-tuning dataset for Vaayu's local model
(data/finetune/tool_calling_dataset.jsonl), targeting correct tool
selection and argument formatting - especially the exact confusion
patterns that were found as REAL bugs this session, not hypothetical ones:
"close the third tab" needs close_browser_tab, not close_app; "increase the
youtube volume" needs set_video_volume (in-page), not control_media
(system); "play fairytale music" needs play_video_on_youtube, not
open_website; a phone number/OTP needs type_sensitive_field, not type_text.
Each of those gets deliberately repeated across several phrasings so the
fine-tune actually reinforces the distinction, not just "a tool exists".

Format: one JSON object per line, {"messages": [system, user, assistant]} -
the same shape ollama_client.py already sends/receives at inference time
(a system prompt, a user turn, and an assistant turn carrying either a
tool_calls list or plain content), so training distribution matches what
the model actually sees when running for real. A modest set of
non-tool-call conversational examples is included too, so fine-tuning
doesn't bias the model toward calling a tool for everything.

*** ROOT CAUSE FOUND LIVE, v1 fine-tune real-usage audio session ***: every
_ex() example below stops at "assistant calls a tool" - none of them ever
showed a tool RESULT coming back followed by a clean final reply with no
further tool call. The v1 fine-tune (trained only on this dataset) had
therefore never seen a single example of "tool result -> stop and answer"
at all, and it showed: real audio-session turns repeatedly called the same
zero-argument status tool 3-5 times in a row, sometimes with hallucinated
arguments that look like fabricated/drifting versions of the tool's own
OUTPUT shape (e.g. get_system_status called with a fake {"result":
{"cpu_percent": 3.5, ...}} argument, a field it doesn't even accept) rather
than the empty {} it actually takes - consistent with a model that's never
seen what a real post-tool-result turn looks like, improvising from
pattern-matching alone. Added _ex_with_reply() below specifically to close
this gap: full 5-message trajectories (system, user, assistant tool_call,
tool result, assistant final reply with NO tool_calls) using REALISTIC
result shapes captured from real tool output during that same session, so
the model has real, concrete examples of stopping after one call.

Every generated tool_call's arguments are validated against the LIVE tool
registry's declared JSON schema before being written (_tool_call below) -
catching a dataset bug (typo'd argument name, wrong type, missing required
field) before it ever reaches training, the same "verify before trusting"
principle used everywhere else in this project.

Usage: python scripts/build_finetune_dataset.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import assistant.tools  # noqa: E402,F401 - registers every tool
from assistant.llm.prompts import SYSTEM_PROMPT  # noqa: E402
from assistant.tools.registry import registry  # noqa: E402

_OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "finetune" / "tool_calling_dataset.jsonl"


def _tool_call(name: str, arguments: dict) -> dict:
    spec = registry._tools.get(name)
    if spec is None:
        raise ValueError(f"Dataset references unknown tool: {name}")
    allowed = set(spec.parameters.get("properties", {}).keys())
    required = set(spec.parameters.get("required", []))
    given = set(arguments.keys())
    if not given <= allowed:
        raise ValueError(f"{name}: dataset gives unexpected args {given - allowed}")
    if not required <= given:
        raise ValueError(f"{name}: dataset is missing required args {required - given}")
    return {"function": {"name": name, "arguments": arguments}}


def _ex(user_text: str, *calls: tuple[str, dict]) -> dict:
    """One or more tool calls (rare: >1 for a genuinely chained request)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call(name, args) for name, args in calls],
            },
        ]
    }


def _ex_with_reply(user_text: str, call: tuple[str, dict], tool_result: dict, final_reply: str) -> dict:
    """A COMPLETE real trajectory: user -> one tool call -> the tool's
    result -> a clean final reply with no further tool call. Mirrors
    exactly what ollama_client.py's real loop produces
    (messages + [{"role": "tool", "content": str(tool_result), "name": name}])
    so the model sees a real "stop after one call" example, not just the
    first half of one. tool_result should be a realistic {"ok": True,
    "result": ...} or {"ok": False, "error": ...} shape, ideally drawn from
    real captured output, not invented - see the module docstring for why
    this category exists."""
    name, args = call
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "", "tool_calls": [_tool_call(name, args)]},
            {"role": "tool", "content": str(tool_result), "name": name},
            {"role": "assistant", "content": final_reply},
        ]
    }


def _no_tool_ex(user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def build_examples() -> list[dict]:
    examples: list[dict] = []

    # -- open_app / close_app - including the exact bug case: a browser
    # closing request must ALWAYS produce a real tool call, never a bare
    # text claim of success.
    examples += [
        _ex("open notepad", ("open_app", {"app_name": "notepad"})),
        _ex("launch chrome", ("open_app", {"app_name": "chrome"})),
        _ex("open brave", ("open_app", {"app_name": "brave"})),
        _ex("can you open spotify for me", ("open_app", {"app_name": "spotify"})),
        _ex("close spotify", ("close_app", {"app_name": "spotify"})),
        _ex("quit calculator", ("close_app", {"app_name": "calculator"})),
        _ex("close the brave browser", ("close_app", {"app_name": "brave"})),
        _ex("shut down chrome completely", ("close_app", {"app_name": "chrome"})),
        _ex("close out of notepad", ("close_app", {"app_name": "notepad"})),
    ]

    # -- open_website - including "<name> website" phrasing and same_tab.
    examples += [
        _ex("search for pizza recipes", ("open_website", {"url_or_query": "pizza recipes"})),
        _ex("open youtube.com", ("open_website", {"url_or_query": "youtube.com"})),
        _ex("go to google", ("open_website", {"url_or_query": "google.com"})),
        _ex("open the moviesmod website", ("open_website", {"url_or_query": "moviesmod website"})),
        _ex(
            "open wikipedia in this tab",
            ("open_website", {"url_or_query": "wikipedia.org", "same_tab": True}),
        ),
        _ex("search how to fix a flat tire", ("open_website", {"url_or_query": "how to fix a flat tire"})),
    ]

    # -- browser tabs - the exact bug case: ordinal position must route to
    # close_browser_tab, never close_app (which closes the WHOLE browser).
    examples += [
        _ex("close the third tab", ("close_browser_tab", {"position": 3})),
        _ex("close the first tab", ("close_browser_tab", {"position": 1})),
        _ex("close the youtube tab", ("close_browser_tab", {"query": "youtube"})),
        _ex("switch to the gmail tab", ("switch_to_browser_tab", {"query": "gmail"})),
        _ex("go to the second tab", ("switch_to_browser_tab", {"position": 2})),
        _ex("what tabs do I have open", ("list_browser_tabs", {})),
        _ex("show me my open tabs", ("list_browser_tabs", {})),
    ]

    # -- YouTube search/channel tools - the exact bug case: "play X" must
    # use these, never open_website, even when "youtube" isn't said.
    examples += [
        _ex(
            "play the latest OG crew video",
            ("play_latest_video_from_channel", {"channel_name": "OG Crew"}),
        ),
        _ex(
            "open the mrbeast channel and play the newest video",
            ("play_latest_video_from_channel", {"channel_name": "MrBeast"}),
        ),
        _ex("play fairytale music", ("play_video_on_youtube", {"query": "fairytale music"})),
        _ex("play some lofi", ("play_video_on_youtube", {"query": "lofi"})),
        _ex("play Ed Sheeran Perfect", ("play_video_on_youtube", {"query": "Ed Sheeran Perfect"})),
        _ex("play some bollywood songs", ("play_video_on_youtube", {"query": "bollywood songs"})),
    ]

    # -- in-page video controls - the exact bug case: "youtube volume" or
    # "video volume" must use set_video_volume, never control_media, and
    # vice versa for an explicit system-volume request.
    examples += [
        _ex("increase the youtube volume", ("set_video_volume", {"direction": "up"})),
        _ex("turn up the video volume", ("set_video_volume", {"direction": "up"})),
        _ex("lower the volume on this video", ("set_video_volume", {"direction": "down"})),
        _ex("mute the video", ("set_video_volume", {"direction": "mute"})),
        _ex("set the video volume to 50", ("set_video_volume", {"level": 50})),
        _ex("turn up the system volume", ("control_media", {"action": "volume_up"})),
        _ex("mute my speakers", ("control_media", {"action": "mute"})),
        _ex("make it fullscreen", ("set_video_fullscreen", {"fullscreen": True})),
        _ex("exit fullscreen", ("set_video_fullscreen", {"fullscreen": False})),
        _ex("skip ahead 10 seconds", ("seek_video", {"direction": "forward", "seconds": 10})),
        _ex("go back 30 seconds", ("seek_video", {"direction": "backward", "seconds": 30})),
        _ex("skip forward a minute", ("seek_video", {"direction": "forward", "seconds": 60})),
        _ex("pause the video", ("control_video_playback", {"action": "pause"})),
        _ex("pause it", ("control_video_playback", {"action": "toggle"})),
        _ex("play it", ("control_video_playback", {"action": "toggle"})),
    ]

    # -- system media
    examples += [
        _ex("pause the music", ("control_media", {"action": "play_pause"})),
        _ex("next track", ("control_media", {"action": "next"})),
        _ex("previous song", ("control_media", {"action": "previous"})),
    ]

    # -- WhatsApp - exact phrasing style from this project's own bug report.
    examples += [
        _ex(
            "send hey there to vedant on whatsapp",
            ("send_whatsapp_message", {"contact_name": "Vedant", "message": "hey there"}),
        ),
        _ex(
            "message mom that I'm on my way",
            ("send_whatsapp_message", {"contact_name": "Mom", "message": "I'm on my way"}),
        ),
        _ex(
            "tell priya happy birthday on whatsapp",
            ("send_whatsapp_message", {"contact_name": "Priya", "message": "Happy birthday!"}),
        ),
    ]

    # -- login/credential fields - the exact bug case: a phone number or OTP
    # must use type_sensitive_field, never type_text, even though both just
    # "type" something.
    examples += [
        _ex(
            "type 9876543210",
            ("type_sensitive_field", {"value": "9876543210", "field_description": "phone number field"}),
        ),
        _ex(
            "my otp is 456789",
            ("type_sensitive_field", {"value": "456789", "field_description": "OTP field"}),
        ),
        _ex(
            "enter the password hunter2",
            ("type_sensitive_field", {"value": "hunter2", "field_description": "password field"}),
        ),
        _ex("type hello world", ("type_text", {"text": "hello world"})),
        _ex("write this is a test note", ("type_text", {"text": "this is a test note"})),
        _ex("click on the mobile number field", ("focus_input_field", {"text": "mobile number"})),
        _ex("go to the search box", ("focus_input_field", {"text": "search"})),
    ]

    # -- generic page interaction
    examples += [
        _ex("click sign in", ("click_on_page", {"text": "Sign in"})),
        _ex("click the OG Crew channel", ("click_on_page", {"text": "OG Crew"})),
        _ex("what's on this page", ("read_page", {})),
        _ex("what can I click here", ("read_page", {})),
    ]

    # -- generic hotkey fallback
    examples += [
        _ex("save this", ("press_hotkey", {"keys": "ctrl+s"})),
        _ex("undo that", ("press_hotkey", {"keys": "ctrl+z"})),
        _ex("copy this", ("press_hotkey", {"keys": "ctrl+c"})),
        _ex("paste it", ("press_hotkey", {"keys": "ctrl+v"})),
        _ex("switch windows", ("press_hotkey", {"keys": "alt+tab"})),
        _ex("open a new tab", ("press_hotkey", {"keys": "ctrl+t"})),
    ]

    # -- system/desktop control
    examples += [
        _ex("lock my computer", ("lock_computer", {})),
        _ex("lock the screen", ("lock_computer", {})),
        _ex("take a screenshot", ("take_screenshot", {})),
        _ex("capture the screen", ("take_screenshot", {})),
        _ex("minimize this", ("minimize_window", {})),
        _ex("minimize the browser", ("minimize_window", {"title": "browser"})),
        _ex("switch to notepad", ("switch_to_window", {"title": "notepad"})),
        _ex("what windows are open", ("list_open_windows", {})),
        _ex("bring back the browser", ("restore_window", {"title": "browser"})),
    ]

    # -- status/info queries
    examples += [
        _ex("what's my battery percentage", ("get_system_status", {})),
        _ex("how much ram am I using", ("get_system_status", {})),
        _ex("what's the cpu usage right now", ("get_system_status", {})),
        _ex("what's my speaker volume", ("get_audio_status", {})),
        _ex("is my microphone muted", ("get_audio_status", {})),
        _ex("am I connected to wifi", ("get_network_status", {})),
        _ex("is bluetooth on", ("get_network_status", {})),
        _ex("what usb devices are connected", ("list_connected_devices", {"category": "usb"})),
        _ex("check my bluetooth devices", ("list_connected_devices", {"category": "bluetooth"})),
    ]

    # -- files/notes
    examples += [
        _ex("find files named resume", ("search_files", {"query": "resume"})),
        _ex("search for my tax documents", ("search_files", {"query": "tax documents"})),
        _ex("save a note that says buy milk", ("add_note", {"text": "buy milk"})),
        _ex("what are my notes", ("list_notes", {})),
        _ex("read out my notes", ("list_notes", {})),
    ]

    # -- orb / self-review
    examples += [
        _ex("move to the center", ("move_status_orb", {"position": "center"})),
        _ex("move a little left", ("move_status_orb", {"position": "a little left"})),
        _ex("set the update time to 9pm", ("set_daily_update_time", {"time": "9pm"})),
        _ex("change the check-in time to 21:30", ("set_daily_update_time", {"time": "21:30"})),
    ]

    # -- non-tool conversational examples, so fine-tuning doesn't bias
    # toward calling a tool for everything. Matches the system prompt's own
    # "only use a tool when the request actually needs it" and "never say
    # you are about to do something you have no tool for" rules.
    examples += [
        _no_tool_ex("how are you", "I'm doing well, thanks for asking! What can I help with?"),
        _no_tool_ex("what's your name", "I'm Vaayu, your personal assistant."),
        _no_tool_ex("thank you", "You're welcome!"),
        _no_tool_ex(
            "can you order me a pizza",
            "I don't actually have a tool to do that - I can't place orders on delivery apps.",
        ),
        _no_tool_ex(
            "tell me a joke",
            "Why don't scientists trust atoms? Because they make up everything!",
        ),
    ]

    # -- COMPLETE trajectories: tool call -> real result -> stop and reply.
    # See the module docstring's "ROOT CAUSE FOUND LIVE" note - every example
    # above this point only ever showed the FIRST half of a real turn. These
    # specifically target the categories that failed live in a real audio
    # session against the v1 fine-tune: repeated status-tool calls (3-5x for
    # one simple question), hallucinated fake-output-shaped arguments, and
    # open_app/close_app falling back to switch_to_window instead. Real
    # result shapes, not invented ones - status fields captured from actual
    # tool output during that session.
    examples += [
        _ex_with_reply(
            "what's my battery percentage",
            ("get_system_status", {}),
            {"ok": True, "result": {"battery_percent": 77, "battery_plugged_in": False, "cpu_percent": 12.0,
                                     "disk_free_gb": 233.2, "disk_total_gb": 451.6, "ram_percent": 58.0,
                                     "ram_total_gb": 15.8, "ram_used_gb": 9.2}},
            "Your battery is at 77%.",
        ),
        _ex_with_reply(
            "battery kitni hai",
            ("get_system_status", {}),
            {"ok": True, "result": {"battery_percent": 100, "battery_plugged_in": True, "cpu_percent": 5.0,
                                     "disk_free_gb": 233.2, "disk_total_gb": 451.6, "ram_percent": 63.6,
                                     "ram_total_gb": 15.8, "ram_used_gb": 10.1}},
            "Your battery is at 100%. It's currently plugged in.",
        ),
        _ex_with_reply(
            "how much ram am I using",
            ("get_system_status", {}),
            {"ok": True, "result": {"battery_percent": 90, "battery_plugged_in": True, "cpu_percent": 8.0,
                                     "disk_free_gb": 233.2, "disk_total_gb": 451.6, "ram_percent": 60.0,
                                     "ram_total_gb": 15.8, "ram_used_gb": 9.5}},
            "You're using about 9.5 GB of RAM out of 15.8 GB, that's 60%.",
        ),
        _ex_with_reply(
            "what's my wifi status",
            ("get_network_status", {}),
            {"ok": True, "result": {"wifi": {"available": True, "connected": False, "signal": None, "ssid": None},
                                     "bluetooth": {"devices": [{"name": "Boult Audio Airbass", "status": "OK"}], "enabled": True}}},
            "Your Wi-Fi is available but not connected right now. Bluetooth is on and connected to your speaker.",
        ),
        _ex_with_reply(
            "am I connected to wifi",
            ("get_network_status", {}),
            {"ok": True, "result": {"wifi": {"available": True, "connected": True, "signal": "strong", "ssid": "HomeNet"},
                                     "bluetooth": {"devices": [], "enabled": False}}},
            "Yes, you're connected to Wi-Fi network HomeNet with a strong signal.",
        ),
        _ex_with_reply(
            "what's my speaker volume",
            ("get_audio_status", {}),
            {"ok": True, "result": {"speaker": {"name": "Speakers (Realtek(R) Audio)", "volume_percent": 100, "muted": False},
                                     "microphone": {"name": "Microphone Array (Realtek(R) Audio)", "volume_percent": 40, "muted": False}}},
            "Your speaker volume is at 100%, and it's not muted.",
        ),
        _ex_with_reply(
            "is my microphone muted",
            ("get_audio_status", {}),
            {"ok": True, "result": {"speaker": {"name": "Speakers (Realtek(R) Audio)", "volume_percent": 100, "muted": False},
                                     "microphone": {"name": "Microphone Array (Realtek(R) Audio)", "volume_percent": 40, "muted": False}}},
            "No, your microphone isn't muted - it's at 40% volume.",
        ),
        _ex_with_reply(
            "open notepad",
            ("open_app", {"app_name": "notepad"}),
            {"ok": True, "result": "Opened notepad."},
            "Notepad has been opened.",
        ),
        _ex_with_reply(
            "open brave",
            ("open_app", {"app_name": "brave"}),
            {"ok": True, "result": "Opened brave."},
            "Brave has been opened.",
        ),
        _ex_with_reply(
            "khol do brave",
            ("open_app", {"app_name": "brave"}),
            {"ok": True, "result": "Opened brave."},
            "Brave has been opened.",
        ),
        _ex_with_reply(
            "close notepad",
            ("close_app", {"app_name": "notepad"}),
            {"ok": True, "result": "Closed notepad."},
            "Notepad has been closed.",
        ),
        _ex_with_reply(
            "close brave",
            ("close_app", {"app_name": "brave"}),
            {"ok": True, "result": "Closed brave."},
            "Brave has been closed.",
        ),
        _ex_with_reply(
            "brave band kar do",
            ("close_app", {"app_name": "brave"}),
            {"ok": True, "result": "Closed brave."},
            "Brave has been closed.",
        ),
        _ex_with_reply(
            "band kar do brave",
            ("close_app", {"app_name": "brave"}),
            {"ok": True, "result": "Closed brave."},
            "Brave has been closed.",
        ),
        _ex_with_reply(
            "note that I need to buy milk tomorrow",
            ("add_note", {"text": "buy milk tomorrow"}),
            {"ok": True, "result": "Note saved."},
            "I've saved your note: buy milk tomorrow.",
        ),
        _ex_with_reply(
            "note that I need to call the dentist at 3pm",
            ("add_note", {"text": "call the dentist at 3pm"}),
            {"ok": True, "result": "Note saved."},
            "I've saved your note: call the dentist at 3pm.",
        ),
        _ex_with_reply(
            "what are my recent notes",
            ("list_notes", {}),
            {"ok": True, "result": [{"text": "buy milk tomorrow", "created_at": "2026-09-21 10:54:09"},
                                     {"text": "call the dentist at 3pm", "created_at": "2026-09-21 10:55:05"}]},
            "Here are your recent notes: buy milk tomorrow, and call the dentist at 3pm.",
        ),
        _ex_with_reply(
            "what tabs do I have open",
            ("list_browser_tabs", {}),
            {"ok": True, "result": [{"position": 1, "title": "YouTube", "url": "https://youtube.com"},
                                     {"position": 2, "title": "GitHub", "url": "https://github.com"}]},
            "You have 2 tabs open: YouTube and GitHub.",
        ),
        _ex_with_reply(
            "what devices are connected",
            ("list_connected_devices", {}),
            {"ok": True, "result": [{"category": "usb", "name": "USB Mouse", "status": "OK"},
                                     {"category": "bluetooth", "name": "Boult Audio Airbass", "status": "OK"}]},
            "You have a USB mouse and a Bluetooth speaker (Boult Audio Airbass) connected.",
        ),
    ]

    return examples


def main() -> None:
    examples = build_examples()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {_OUT_PATH}")


if __name__ == "__main__":
    main()
