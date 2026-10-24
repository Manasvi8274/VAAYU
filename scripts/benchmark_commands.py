"""
Command-understanding benchmark against the REAL local model, with every
tool's execution stubbed out (nothing opens, closes, types or sends - the
stub only checks the call against the tool's schema, like real dispatch).

Measures, per case: which tool was called first, whether its key arguments
are right, whether a no-tool request correctly called nothing, latency, and
whether it took the fast path or the LLM.

    python scripts/benchmark_commands.py --model vaayu-finetuned --mode new
    python scripts/benchmark_commands.py --model vaayu-finetuned --mode old --sample 120

--mode old = fast_paths=False (all 37 tools, no router, two LLM calls) -
the pre-router baseline on the same code.
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import assistant.tools  # noqa: E402,F401 - registers every tool
from assistant.llm.ollama_client import OllamaBrain  # noqa: E402
from assistant.llm.prompts import SYSTEM_PROMPT  # noqa: E402
from assistant.tools.registry import ToolRegistry, registry, repair_arguments  # noqa: E402

APPS = ["notepad", "calculator", "chrome", "brave", "spotify", "whatsapp", "telegram", "discord",
        "vs code", "edge", "firefox", "paint", "file explorer", "settings", "word", "excel", "steam",
        "tekken 7", "zoom", "task manager"]


def case(_cat, _utterance, _tool, **args):
    return {"cat": _cat, "text": _utterance, "tool": _tool, "args": args}


def build_cases():
    r = random.Random(7)
    c = []
    # --- apps, English ---
    for a in APPS:
        for t in r.sample(["open {a}", "launch {a}", "start {a}", "can you open {a}", "please open {a} for me",
                           "Vaayu, open {a}.", "fire up {a}", "open the {a} app", "I want to use {a}, open it"], 3):
            c.append(case("app_open_en", t.format(a=a), "open_app", app_name=a.split()[0]))
        for t in r.sample(["close {a}", "quit {a}", "exit {a}", "close the {a} app", "please close {a}",
                           "shut {a}", "kill {a}"], 2):
            c.append(case("app_close_en", t.format(a=a), "close_app", app_name=a.split()[0]))
    for a in r.sample(APPS, 8):
        c.append(case("app_background", f"open {a} in the background", "open_app", app_name=a.split()[0], foreground="false"))
    # --- apps, Hinglish ---
    for a in APPS:
        c.append(case("app_hinglish", r.choice(["{a} kholo", "{a} khol do", "{a} open kro", "{a} chalu karo",
                                                  "zara {a} khol do"]).format(a=a), "open_app", app_name=a.split()[0]))
        c.append(case("app_hinglish", r.choice(["{a} band karo", "{a} band kar do", "{a} close kro",
                                                  "{a} ko band karo"]).format(a=a), "close_app", app_name=a.split()[0]))
    # --- system media / volume ---
    for t, act in [("pause", "play_pause"), ("play", "play_pause"), ("resume the music", "play_pause"),
                   ("pause the music", "play_pause"), ("stop the song", "play_pause"), ("gaana roko", "play_pause"),
                   ("next song", "next"), ("skip this track", "next"), ("play the next track", "next"),
                   ("agla gaana chalao", "next"), ("previous song", "previous"), ("go back to the last song", "previous"),
                   ("pichla gaana", "previous"), ("increase the volume", "volume_up"), ("turn up the volume", "volume_up"),
                   ("volume badhao", "volume_up"), ("awaz tez karo", "volume_up"), ("make it louder", "volume_up"),
                   ("raise the system volume", "volume_up"), ("decrease the volume", "volume_down"),
                   ("turn down the volume", "volume_down"), ("volume kam karo", "volume_down"),
                   ("awaz dheere karo", "volume_down"), ("make it quieter", "volume_down"), ("mute", "mute"),
                   ("mute the sound", "mute"), ("mute my laptop", "mute"), ("sound band karo", "mute")]:
        c.append(case("media_system", t, "control_media", action=act))
    # --- status ---
    for t in ["battery kitni hai", "how much battery is left", "what's my battery percentage", "kitna charge hai",
              "is my laptop charging", "check battery", "how much ram is being used", "what's the cpu usage",
              "how much disk space do I have", "system status", "ram kitni use ho rahi hai", "battery status batao",
              "is the laptop plugged in", "how's my cpu doing", "storage kitna bacha hai"]:
        c.append(case("status", t, "get_system_status"))
    for t, tool in [("is my wifi connected", "get_network_status"), ("internet chal raha hai kya", "get_network_status"),
                    ("am I online", "get_network_status"), ("which wifi am I on", "get_network_status"),
                    ("is bluetooth on", "list_connected_devices"), ("what usb devices are connected", "list_connected_devices"),
                    ("is my printer connected", "list_connected_devices"), ("what displays are connected", "list_connected_devices"),
                    ("what's my speaker volume", "get_audio_status"), ("is my mic muted", "get_audio_status"),
                    ("mic on hai kya", "get_audio_status"), ("what is the microphone level", "get_audio_status")]:
        c.append(case("status_other", t, tool))
    # --- quick system actions ---
    for t, tool in [("take a screenshot", "take_screenshot"), ("screenshot lo", "take_screenshot"),
                    ("capture the screen", "take_screenshot"), ("grab a screenshot", "take_screenshot"),
                    ("screenshot le lo", "take_screenshot"), ("save a picture of my screen", "take_screenshot"),
                    ("lock my laptop", "lock_computer"), ("lock the computer", "lock_computer"),
                    ("laptop lock karo", "lock_computer"), ("lock the screen", "lock_computer"),
                    ("lock it, I'm leaving", "lock_computer"), ("minimize this window", "minimize_window"),
                    ("minimize", "minimize_window"), ("minimize chrome", "minimize_window"),
                    ("send this window to the taskbar", "minimize_window"), ("ye minimize karo", "minimize_window"),
                    ("bring back the notepad window", "restore_window"), ("restore the spotify window", "restore_window"),
                    ("switch to the calculator window", "switch_to_window"), ("go to the discord window", "switch_to_window"),
                    ("what windows are open", "list_open_windows"), ("list all open windows", "list_open_windows"),
                    ("press ctrl s", "press_hotkey"), ("press alt tab", "press_hotkey"), ("press the escape key", "press_hotkey")]:
        c.append(case("system_actions", t, tool))
    # --- time/date (answered without a tool) ---
    for t in ["what time is it", "what's the time", "time kya hai", "kitne baje hain", "time batao",
              "what's the date today", "what day is it today", "aaj kya date hai"]:
        c.append(case("time_date", t, None))
    # --- YouTube ---
    songs = ["perfect by ed sheeran", "shape of you", "kesariya", "lofi hindi songs", "arijit singh songs",
             "bollywood party songs", "believer by imagine dragons", "tum hi ho", "hollywood songs", "old hindi songs",
             "punjabi songs", "workout music", "naatu naatu", "blinding lights by the weeknd", "relaxing piano music"]
    for s in songs:
        c.append(case("youtube", r.choice(["play {s} on youtube", "youtube pe {s} chalao", "play {s}",
                                           "put on {s} on youtube", "youtube par {s} lagao"]).format(s=s),
                      "play_video_on_youtube", query=s.split()[0]))
    for ch in ["MrBeast", "Carryminati", "OG Crew", "Dhruv Rathee", "Technical Guruji", "BB Ki Vines", "T-Series",
               "Mortal", "Total Gaming", "Ranveer Allahbadia"]:
        c.append(case("youtube_channel", r.choice(["play the latest video from {c}", "play {c}'s newest video",
                                                   "{c} ka latest video chalao", "open the {c} channel and play their newest video"]).format(c=ch),
                      "play_latest_video_from_channel", channel_name=ch.split()[0].lower()))
    # --- in-browser video control ---
    for t, tool, args in [("pause the video", "control_video_playback", {}), ("play the video", "control_video_playback", {}),
                          ("video roko", "control_video_playback", {}), ("pause the dune trailer", "control_video_playback", {"tab_hint": "dune"}),
                          ("resume the youtube video", "control_video_playback", {}),
                          ("make it fullscreen", "set_video_fullscreen", {"fullscreen": "true"}),
                          ("full screen this video", "set_video_fullscreen", {"fullscreen": "true"}),
                          ("exit fullscreen", "set_video_fullscreen", {"fullscreen": "false"}),
                          ("shrink the video", "set_video_fullscreen", {"fullscreen": "false"}),
                          ("skip ahead 30 seconds", "seek_video", {"direction": "forward", "seconds": "30"}),
                          ("go back 10 seconds", "seek_video", {"direction": "backward"}),
                          ("skip forward a minute", "seek_video", {"direction": "forward", "seconds": "60"}),
                          ("video 20 second aage karo", "seek_video", {"direction": "forward"}),
                          ("rewind 15 seconds", "seek_video", {"direction": "backward", "seconds": "15"}),
                          ("turn up the youtube volume", "set_video_volume", {"direction": "up"}),
                          ("mute the video", "set_video_volume", {"direction": "mute"}),
                          ("unmute the video", "set_video_volume", {"direction": "unmute"}),
                          ("lower the video volume", "set_video_volume", {"direction": "down"}),
                          ("set the video volume to 50", "set_video_volume", {"level": "50"}),
                          ("turn down the hotstar volume", "set_video_volume", {"tab_hint": "hotstar"})]:
        c.append(case("video_control", t, tool, **args))
    # --- browser / web ---
    for t, tool, args in [("open google.com", "open_website", {"url_or_query": "google"}),
                          ("go to github.com", "open_website", {"url_or_query": "github"}),
                          ("search for weather in delhi", "open_website", {"url_or_query": "weather"}),
                          ("google best laptops under 50000", "open_website", {"url_or_query": "laptop"}),
                          ("open amazon in a new tab", "open_website", {"url_or_query": "amazon"}),
                          ("open flipkart website", "open_website", {"url_or_query": "flipkart"}),
                          ("search python tutorials", "open_website", {"url_or_query": "python"}),
                          ("cricket score dikhao", "open_website", {"url_or_query": "cricket"}),
                          ("what tabs are open", "list_browser_tabs", {}), ("list my browser tabs", "list_browser_tabs", {}),
                          ("close this tab", "close_browser_tab", {}), ("close the youtube tab", "close_browser_tab", {"query": "youtube"}),
                          ("close the second tab", "close_browser_tab", {"position": "2"}),
                          ("switch to the gmail tab", "switch_to_browser_tab", {"query": "gmail"}),
                          ("go to the first tab", "switch_to_browser_tab", {"position": "1"}),
                          ("youtube wale tab pe jao", "switch_to_browser_tab", {"query": "youtube"}),
                          ("read this page", "read_page", {}), ("what's on this page", "read_page", {}),
                          ("click on sign in", "click_on_page", {"text": "sign in"}),
                          ("click the subscribe button", "click_on_page", {"text": "subscribe"})]:
        c.append(case("browser", t, tool, **args))
    # --- WhatsApp ---
    for who, msg in [("SAVED MESSAGES", "test one"), ("mom", "I'll be late"), ("Rahul", "call me"),
                     ("SAVED MESSAGES", "buy milk"), ("papa", "reached home"), ("Priya", "happy birthday"),
                     ("SAVED MESSAGES", "meeting at 5"), ("Raman Garg", "send the report"), ("didi", "kal milte hain"),
                     ("SAVED MESSAGES", "hello from vaayu")]:
        for t in r.sample(["send {m} to {w} on whatsapp", "whatsapp {w} saying {m}", "message {w} {m} on whatsapp",
                           "{w} ko whatsapp pe {m} bhejo", "tell {w} on whatsapp that {m}"], 3):
            c.append(case("whatsapp", t.format(m=msg, w=who), "send_whatsapp_message",
                          contact_name=who.split()[0].lower(), message=msg.split()[0].lower()))
    # --- notes / typing / files ---
    for t, tool, args in [("note that I have a meeting at 5", "add_note", {"text": "meeting"}),
                          ("remember to buy groceries", "add_note", {"text": "groceries"}),
                          ("add a note: call the bank tomorrow", "add_note", {"text": "bank"}),
                          ("yaad rakhna kal doctor ke paas jana hai", "add_note", {"text": "doctor"}),
                          ("save a note saying project deadline friday", "add_note", {"text": "deadline"}),
                          ("show my notes", "list_notes", {}), ("what are my notes", "list_notes", {}),
                          ("read my last 3 notes", "list_notes", {}), ("mere notes dikhao", "list_notes", {}),
                          ("type hello world", "type_text", {"text": "hello"}),
                          ("write dear sir in notepad", "type_text", {"text": "dear"}),
                          ("type my email is test at gmail dot com", "type_text", {"text": "email"}),
                          ("likho good morning", "type_text", {"text": "good"}),
                          ("find my resume file", "search_files", {"query": "resume"}),
                          ("search for invoice in my documents", "search_files", {"query": "invoice"}),
                          ("where is my budget spreadsheet", "search_files", {"query": "budget"}),
                          ("photos folder dhundo", "search_files", {"query": "photo"}),
                          ("read the file notes.txt on my desktop", "read_file", {"path": "notes"}),
                          ("open and read todo.txt", "read_file", {"path": "todo"}),
                          ("move the orb to the top right", "move_status_orb", {"position": "right"}),
                          ("put the ball in the bottom left corner", "move_status_orb", {"position": "left"}),
                          ("set my daily update time to 9 pm", "set_daily_update_time", {"time": "21"}),
                          ("change check-in time to 10:30", "set_daily_update_time", {"time": "10"})]:
        c.append(case("notes_files_typing", t, tool, **args))
    # --- speech-to-text noise on otherwise-simple commands ---
    for t, tool, args in [("Open WhatsApp.", "open_app", {"app_name": "whatsapp"}),
                          ("um, open notepad", "open_app", {"app_name": "notepad"}),
                          ("uh can you like open chrome", "open_app", {"app_name": "chrome"}),
                          ("OPEN SPOTIFY!", "open_app", {"app_name": "spotify"}),
                          ("Vayu, close brave.", "close_app", {"app_name": "brave"}),
                          ("Vaayu open calculator", "open_app", {"app_name": "calculator"}),
                          ("open... uh... discord", "open_app", {"app_name": "discord"}),
                          ("Close. Notepad.", "close_app", {"app_name": "notepad"}),
                          ("battery?", "get_system_status", {}), ("Battery kitni hai?", "get_system_status", {}),
                          ("Pause.", "control_media", {"action": "play_pause"}),
                          ("next song please", "control_media", {"action": "next"}),
                          ("hey vaayu take a screenshot", "take_screenshot", {}),
                          ("okay vaayu lock my laptop", "lock_computer", {}),
                          ("Volume. Up.", "control_media", {"action": "volume_up"}),
                          ("open the brave browser", "open_app", {"app_name": "brave"}),
                          ("could you please open file explorer for me", "open_app", {"app_name": "explorer"}),
                          ("whatsapp kholo yaar", "open_app", {"app_name": "whatsapp"}),
                          ("chrome ko band kar do please", "close_app", {"app_name": "chrome"}),
                          ("play shape of you on youtube please", "play_video_on_youtube", {"query": "shape"})]:
        c.append(case("stt_noise", t, tool, **args))
    # --- negatives: should call NO tool ---
    for t in ["how are you", "tell me a joke", "who made you", "what's your name", "thank you", "good morning",
              "you're awesome", "what can you do", "kya haal hai", "what is the capital of france",
              "explain what an api is", "how many days are in a leap year", "what's 25 times 4", "ok", "never mind",
              "shukriya", "who is the prime minister of india", "tell me something interesting", "nothing, forget it",
              "are you there"]:
        c.append(case("negative_chat", t, None))
    for t in ["order a pizza", "book a flight to mumbai", "transfer 500 rupees to rahul", "pay my electricity bill",
              "call mom", "turn on the lights", "set an alarm for 6 am", "delete all my files", "format my c drive",
              "install photoshop", "what's the weather tomorrow", "send an email to my boss",
              "uninstall chrome", "make a phone call", "turn off my phone"]:
        c.append(case("negative_unsupported", t, None))
    # --- rarer phrasings the fast path deliberately doesn't cover (LLM path) ---
    for a in r.sample(APPS, 15):
        c.append(case("app_paraphrase", r.choice(["bring up {a}", "get {a} running", "I need {a}",
                                                  "can I have {a} open", "{a} chahiye"]).format(a=a),
                      "open_app", app_name=a.split()[0]))
        c.append(case("app_paraphrase", r.choice(["get rid of {a}", "I'm done with {a}, close it",
                                                  "{a} hata do", "shut down {a} for me"]).format(a=a),
                      "close_app", app_name=a.split()[0]))
    # --- multi-intent: the FIRST step must be right ---
    for t, tool, args in [("open notepad and type hello", "open_app", {"app_name": "notepad"}),
                          ("open chrome and search for news", "open_app", {"app_name": "chrome"}),
                          ("open spotify and play some music", "open_app", {"app_name": "spotify"}),
                          ("take a screenshot and then lock the laptop", "take_screenshot", {}),
                          ("close chrome and open brave", "close_app", {"app_name": "chrome"}),
                          ("notepad kholo aur hello likho", "open_app", {"app_name": "notepad"}),
                          ("open youtube and play kesariya", "play_video_on_youtube", {"query": "kesariya"}),
                          ("mute and lock the laptop", "control_media", {"action": "mute"}),
                          ("check my battery and wifi", "get_system_status", {}),
                          ("open whatsapp and send hi to SAVED MESSAGES", "open_app", {"app_name": "whatsapp"}),
                          ("pause the music and open calculator", "control_media", {"action": "play_pause"}),
                          ("open discord then minimize it", "open_app", {"app_name": "discord"}),
                          ("screenshot lo aur laptop lock karo", "take_screenshot", {}),
                          ("open edge in the background and open notepad", "open_app", {"app_name": "edge"}),
                          ("note that I called rahul and then show my notes", "add_note", {"text": "rahul"})]:
        c.append(case("multi_intent", t, tool, **args))
    # --- speech-to-text noise injected into a sample of the clean cases above ---
    noisy = []
    for base in r.sample([x for x in c if x["tool"] and x["cat"].startswith(("app_", "media", "status"))], 50):
        t = base["text"]
        t = r.choice([lambda s: s.upper(), lambda s: s.capitalize() + ".", lambda s: "um " + s,
                      lambda s: s + " please", lambda s: "Vaayu, " + s + "?", lambda s: s.replace(" ", " uh ", 1),
                      lambda s: "okay so " + s, lambda s: s + " na"])(t)
        noisy.append({**base, "cat": "stt_noise_auto", "text": t})
    c += noisy
    return c


def build_holdout():
    """Fresh phrasings/apps/contacts written AFTER the round-2 router fixes and
    never used to tune them - measures generalization, not memorization."""
    H = []
    for t, tool, args in [
        ("open vlc", "open_app", {"app_name": "vlc"}), ("launch outlook", "open_app", {"app_name": "outlook"}),
        ("open the photos app", "open_app", {"app_name": "photos"}), ("start obs", "open_app", {"app_name": "obs"}),
        ("camera kholo", "open_app", {"app_name": "camera"}), ("open microsoft teams", "open_app", {"app_name": "teams"}),
        ("powerpoint open karo", "open_app", {"app_name": "powerpoint"}), ("can you launch slack", "open_app", {"app_name": "slack"}),
        ("open epic games launcher", "open_app", {"app_name": "epic"}), ("fire up onenote", "open_app", {"app_name": "onenote"}),
        ("I want to play tekken, open it", "open_app", {"app_name": "tekken"}), ("open control panel", "open_app", {"app_name": "control"}),
        ("open snipping tool", "open_app", {"app_name": "snipping"}), ("start the clock app", "open_app", {"app_name": "clock"}),
        ("open armoury crate", "open_app", {"app_name": "armoury"}), ("open ms paint in the background", "open_app", {"app_name": "paint", "foreground": "false"}),
        ("close vlc", "close_app", {"app_name": "vlc"}), ("outlook band karo", "close_app", {"app_name": "outlook"}),
        ("quit obs", "close_app", {"app_name": "obs"}), ("teams ko band kar do", "close_app", {"app_name": "teams"}),
        ("shut slack", "close_app", {"app_name": "slack"}), ("close the camera", "close_app", {"app_name": "camera"}),
        ("I'm done with powerpoint", "close_app", {"app_name": "powerpoint"}), ("exit epic games", "close_app", {"app_name": "epic"}),
        ("turn the volume up a bit", "control_media", {"action": "volume_up"}), ("thoda awaz kam karo", "control_media", {"action": "volume_down"}),
        ("silence the laptop", "control_media", {"action": "mute"}), ("play next", "control_media", {"action": "next"}),
        ("change the song", "control_media", {"action": "next"}), ("music band karo", "control_media", {"action": "play_pause"}),
        ("go to the previous track", "control_media", {"action": "previous"}), ("resume", "control_media", {"action": "play_pause"}),
        ("how much charge is left", "get_system_status", {}), ("is my pc running hot on cpu", "get_system_status", {}),
        ("memory kitni free hai", "get_system_status", {}), ("am I connected to wifi", "get_network_status", {}),
        ("internet hai kya", "get_network_status", {}), ("are my headphones connected over bluetooth", "list_connected_devices", {}),
        ("is the microphone on", "get_audio_status", {}), ("take a quick screenshot", "take_screenshot", {}),
        ("screen ki photo lo", "take_screenshot", {}), ("lock the pc", "lock_computer", {}),
        ("lock it up", "lock_computer", {}), ("minimise this", "minimize_window", {}),
        ("what's the time right now", None, {}), ("aaj kaunsa din hai", None, {}),
        ("play arijit singh on youtube", "play_video_on_youtube", {"query": "arijit"}),
        ("youtube par coke studio songs chalao", "play_video_on_youtube", {"query": "coke"}),
        ("play the latest video from Linus Tech Tips", "play_latest_video_from_channel", {"channel_name": "linus"}),
        ("Samay Raina ka naya video chalao", "play_latest_video_from_channel", {"channel_name": "samay"}),
        ("pause the youtube video", "control_video_playback", {}), ("go fullscreen", "set_video_fullscreen", {"fullscreen": "true"}),
        ("exit full screen", "set_video_fullscreen", {"fullscreen": "false"}), ("skip ahead 45 seconds", "seek_video", {"direction": "forward", "seconds": "45"}),
        ("go back 2 minutes", "seek_video", {"direction": "backward", "seconds": "120"}), ("mute the netflix video", "set_video_volume", {"direction": "mute"}),
        ("turn up the netflix volume", "set_video_volume", {"direction": "up"}),
        ("open stackoverflow.com", "open_website", {"url_or_query": "stackoverflow"}), ("search for train tickets to jaipur", "open_website", {"url_or_query": "train"}),
        ("google how to cook paneer", "open_website", {"url_or_query": "paneer"}), ("open the irctc website", "open_website", {"url_or_query": "irctc"}),
        ("close the third tab", "close_browser_tab", {"position": "3"}), ("close the gmail tab", "close_browser_tab", {"query": "gmail"}),
        ("switch to the second tab", "switch_to_browser_tab", {"position": "2"}), ("go to the whatsapp tab", "switch_to_browser_tab", {"query": "whatsapp"}),
        ("which tabs do I have open", "list_browser_tabs", {}), ("summarize this page", "read_page", {}),
        ("send good night to SAVED MESSAGES on whatsapp", "send_whatsapp_message", {"contact_name": "saved", "message": "good"}),
        ("SAVED MESSAGES ko whatsapp pe kal 10 baje call bhejo", "send_whatsapp_message", {"contact_name": "saved", "message": "kal"}),
        ("whatsapp SAVED MESSAGES saying pick up laundry", "send_whatsapp_message", {"contact_name": "saved", "message": "laundry"}),
        ("tell Raman Garg on whatsapp that the file is ready", "send_whatsapp_message", {"contact_name": "raman", "message": "file"}),
        ("send running late to Ankit on whatsapp", "send_whatsapp_message", {"contact_name": "ankit", "message": "running"}),
        ("bhaiya ko whatsapp pe khana kha liya bhejo", "send_whatsapp_message", {"contact_name": "bhaiya", "message": "khana"}),
        ("note that the wifi password is on the fridge", "add_note", {"text": "fridge"}), ("remember to pay rent on the 5th", "add_note", {"text": "rent"}),
        ("yaad rakhna mummy ka birthday 12 ko hai", "add_note", {"text": "birthday"}), ("show me my notes", "list_notes", {}),
        ("type thank you for your time", "type_text", {"text": "thank"}), ("write regards manasvi", "type_text", {"text": "regards"}),
        ("find my aadhaar pdf", "search_files", {"query": "aadhaar"}), ("search for the tax file in downloads", "search_files", {"query": "tax"}),
        ("move the orb to the bottom right", "move_status_orb", {"position": "right"}),
        ("open notepad and write meeting notes", "open_app", {"app_name": "notepad"}),
        ("mute and then take a screenshot", "control_media", {"action": "mute"}),
        ("close spotify aur chrome kholo", "close_app", {"app_name": "spotify"}),
        ("OPEN VLC.", "open_app", {"app_name": "vlc"}), ("uh, close outlook", "close_app", {"app_name": "outlook"}),
        ("Vaayu, battery?", "get_system_status", {}), ("next track please", "control_media", {"action": "next"}),
        ("okay vaayu pause", "control_media", {"action": "play_pause"}),
        ("thanks vaayu", None, {}), ("good night", None, {}), ("what's your favourite colour", None, {}),
        ("how old are you", None, {}), ("acha theek hai", None, {}), ("who is virat kohli", None, {}),
        ("book an uber", None, {}), ("send money to papa on phonepe", None, {}),
        # label changed after set_radio was added on request (was None when no wifi tool existed)
        ("turn off wifi", "set_radio", {"kind": "wifi", "state": "off"}),
        ("shut down the laptop", None, {}), ("delete the downloads folder", None, {}), ("change my wallpaper", None, {}),
    ]:
        H.append(case("holdout", t, tool, **args))
    return H


def build_holdout2():
    """Written BEFORE the round-3 fixes (common phrasing, wifi/bluetooth,
    verbatim WhatsApp, refusals, mixed multi-step) and never used to tune them."""
    H = []
    for t, tool, args in [
        # everyday media/volume wording
        ("change this song", "control_media", {"action": "next"}), ("koi aur gaana lagao", "control_media", {"action": "next"}),
        ("gaana badlo", "control_media", {"action": "next"}), ("song change karo", "control_media", {"action": "next"}),
        ("skip it", "control_media", {"action": "next"}), ("play the song before this", "control_media", {"action": "previous"}),
        ("pichla wala gaana chalao", "control_media", {"action": "previous"}), ("back to previous song", "control_media", {"action": "previous"}),
        ("gaana band karo", "control_media", {"action": "play_pause"}), ("music pause karo", "control_media", {"action": "play_pause"}),
        ("stop the music", "control_media", {"action": "play_pause"}), ("hold on pause it", "control_media", {"action": "play_pause"}),
        ("music wapas chalao", "control_media", {"action": "play_pause"}), ("continue playing", "control_media", {"action": "play_pause"}),
        ("thoda volume badhao", "control_media", {"action": "volume_up"}), ("volume thoda kam kar do", "control_media", {"action": "volume_down"}),
        ("it's too loud", "control_media", {"action": "volume_down"}), ("I can't hear anything, louder", "control_media", {"action": "volume_up"}),
        ("bump up the volume", "control_media", {"action": "volume_up"}), ("lower the sound a little", "control_media", {"action": "volume_down"}),
        ("awaz band kar do", "control_media", {"action": "mute"}), ("silence everything", "control_media", {"action": "mute"}),
        ("mute it", "control_media", {"action": "mute"}), ("unmute the laptop", "control_media", {"action": "mute"}),
        # common app wording
        ("I'm done with chrome", "close_app", {"app_name": "chrome"}), ("chrome ki zarurat nahi, band karo", "close_app", {"app_name": "chrome"}),
        ("close it, the spotify one", "close_app", {"app_name": "spotify"}), ("open up notepad for me", "open_app", {"app_name": "notepad"}),
        ("mujhe calculator chahiye", "open_app", {"app_name": "calculator"}), ("notepad chalu kar", "open_app", {"app_name": "notepad"}),
        ("let's open vs code", "open_app", {"app_name": "vs"}), ("whatsapp band kar de", "close_app", {"app_name": "whatsapp"}),
        ("discord hatao", "close_app", {"app_name": "discord"}), ("open brave quickly", "open_app", {"app_name": "brave"}),
        # wifi / bluetooth on/off
        ("turn off wifi", "set_radio", {"kind": "wifi", "state": "off"}), ("turn on the wifi", "set_radio", {"kind": "wifi", "state": "on"}),
        ("wifi band karo", "set_radio", {"kind": "wifi", "state": "off"}), ("wifi on kar do", "set_radio", {"kind": "wifi", "state": "on"}),
        ("disable bluetooth", "set_radio", {"kind": "bluetooth", "state": "off"}), ("enable bluetooth", "set_radio", {"kind": "bluetooth", "state": "on"}),
        ("bluetooth chalu karo", "set_radio", {"kind": "bluetooth", "state": "on"}), ("switch off bluetooth", "set_radio", {"kind": "bluetooth", "state": "off"}),
        ("is wifi on", "get_network_status", {}), ("bluetooth on hai kya", "list_connected_devices", {}),
        # WhatsApp: verbatim contact/message, many shapes
        ('send message "bye bye" to saved messages in whatsapp', "send_whatsapp_message", {"contact_name": "saved messages", "message": "bye bye"}),
        ("send a message saying call me later to SAVED MESSAGES on whatsapp", "send_whatsapp_message", {"contact_name": "saved", "message": "call me later"}),
        ("on whatsapp send see you soon to SAVED MESSAGES", "send_whatsapp_message", {"contact_name": "saved", "message": "see you soon"}),
        ("whatsapp pe SAVED MESSAGES ko dinner ready bhejo", "send_whatsapp_message", {"contact_name": "saved", "message": "dinner"}),
        ("SAVED MESSAGES ko whatsapp karo ki main aa raha hoon", "send_whatsapp_message", {"contact_name": "saved", "message": "main aa raha"}),
        ("text SAVED MESSAGES on whatsapp saying done for today", "send_whatsapp_message", {"contact_name": "saved", "message": "done for today"}),
        ("send hi on whatsapp to SAVED MESSAGES", "send_whatsapp_message", {"contact_name": "saved", "message": "hi"}),
        ("whatsapp message to SAVED MESSAGES: lunch at 2", "send_whatsapp_message", {"contact_name": "saved", "message": "lunch"}),
        ("send ok to Raman Garg on whatsapp", "send_whatsapp_message", {"contact_name": "raman garg", "message": "ok"}),
        ("send thanks to SAVED MESSAGES and Raman Garg on whatsapp", None, {}),  # two recipients - must not guess
        ("message SAVED MESSAGES call mom on whatsapp", None, {}),  # ambiguous split - must not guess
        ("send a whatsapp", None, {}), ("whatsapp someone", None, {}),
        # refusals: things Vaayu must not guess at
        ("shut down the laptop", None, {}), ("restart my computer", None, {}), ("put the laptop to sleep", None, {}),
        ("hibernate the pc", None, {}), ("delete everything in downloads", None, {}), ("format the d drive", None, {}),
        ("install vlc", None, {}), ("uninstall spotify", None, {}), ("pay 200 on paytm", None, {}),
        ("open phonepe and send money", None, {}), ("log out of windows", None, {}),
        # mixed multi-step: first step must be right
        ("open spotify and play some music", "open_app", {"app_name": "spotify"}),
        ("open chrome and then show me the news", "open_app", {"app_name": "chrome"}),
        ("close discord and play kesariya on youtube", "close_app", {"app_name": "discord"}),
        ("lock the laptop after taking a screenshot", "take_screenshot", {}),
        ("open notepad then type the meeting is at 4", "open_app", {"app_name": "notepad"}),
        ("mute and open whatsapp", "control_media", {"action": "mute"}),
        ("turn off wifi and lock the laptop", "set_radio", {"kind": "wifi", "state": "off"}),
        ("whatsapp kholo aur SAVED MESSAGES ko hello bhejo", "open_app", {"app_name": "whatsapp"}),
        ("pause the music and tell me the battery", "control_media", {"action": "play_pause"}),
        ("take a screenshot and save a note saying bug in login", "take_screenshot", {}),
        # misc everyday
        ("what's the battery like", "get_system_status", {}), ("charging ho raha hai kya", "get_system_status", {}),
        ("take a ss", "take_screenshot", {}), ("lock kar do", "lock_computer", {}),
        ("show my open tabs", "list_browser_tabs", {}), ("close current tab", "close_browser_tab", {}),
        ("next tab", "switch_to_browser_tab", {}), ("go full screen", "set_video_fullscreen", {"fullscreen": "true"}),
        ("video aage badhao 10 second", "seek_video", {"direction": "forward"}), ("pause the video please", "control_video_playback", {}),
        ("note: buy a new charger", "add_note", {"text": "charger"}), ("type ok done", "type_text", {"text": "ok done"}),
        ("search youtube for funny cat videos", "play_video_on_youtube", {"query": "cat"}),
        ("bas", None, {}), ("chalo theek hai", None, {}), ("good job vaayu", None, {}), ("sorry", None, {}),
    ]:
        H.append(case("holdout2", t, tool, **args))
    return H


def _stub_dispatch(self, name, arguments):
    # Same contract as real ToolRegistry.dispatch, minus the side effects.
    spec = registry._tools.get(name)
    if spec is None:
        return {"ok": False, "error": f"Unknown tool '{name}'"}
    args = repair_arguments(spec, arguments)
    missing = [k for k in spec.parameters.get("required", []) if k not in args]
    if missing:
        return {"ok": False, "error": f"missing required argument(s): {missing}"}
    if name == "get_system_status":
        return {"ok": True, "result": {"cpu_percent": 12, "ram_percent": 61, "battery_percent": 80,
                                       "battery_plugged_in": True}}
    return {"ok": True, "result": f"{name} done."}


def _relabel(cs):
    """Labels written before two deliberate behaviour changes (kept visible, not silently edited):
    play/pause became explicit actions instead of one toggle, so "pause" can never START something;
    and "search youtube for X" now shows results instead of auto-playing the first one."""
    want = cs["tool"], dict(cs["args"])
    if want == ("control_media", {"action": "play_pause"}):
        is_pause = re.search(r"\b(pause|stop|roko|band|hold)\b", cs["text"], re.I)
        return [("control_media", {"action": "pause" if is_pause else "play"})]
    if cs["tool"] == "play_video_on_youtube" and re.match(r"^search youtube for ", cs["text"], re.I):
        return [want, ("youtube_control", {"action": "search"})]
    # Third change, on request: questions about the world are looked up online (tools/web_answer.py)
    # instead of answered from the model's memory or refused.
    if cs["text"] in _NOW_LOOKED_UP:
        return [_NOW_LOOKED_UP[cs["text"]]]
    return [want]


_NOW_LOOKED_UP = {
    "what is the capital of france": ("answer_question", {"question": "capital"}),
    "how many days are in a leap year": ("answer_question", {"question": "leap"}),
    "who is the prime minister of india": ("answer_question", {"question": "prime minister"}),
    "who is virat kohli": ("answer_question", {"question": "kohli"}),
    "what's the weather tomorrow": ("get_weather", {"day": "tomorrow"}),
}


def score(cs, calls):
    first = calls[0] if calls else None
    wants = _relabel(cs)
    if wants[0][0] is not None and first is not None:
        return any(first[0] == tool and all(str(v).lower() in str(first[1].get(k, "")).lower() for k, v in args.items())
                   for tool, args in wants)
    if wants[0][0] is None:
        return first is None
    if first is None or first[0] != cs["tool"]:
        return False
    for key, want in cs["args"].items():
        got = str(first[1].get(key, "")).lower()
        if want.lower() not in got:
            return False
    return True


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def report(paths):
    """Per-category accuracy/latency for each results file, side by side."""
    runs = {Path(p).stem: json.loads(Path(p).read_text(encoding="utf-8")) for p in paths}
    cats = sorted({r["cat"] for rs in runs.values() for r in rs})
    print("| category | " + " | ".join(f"{n} acc | {n} median s" for n in runs) + " |")
    print("|---|" + "---|---|" * len(runs))
    for cat in cats + ["ALL"]:
        row = [cat]
        for rs in runs.values():
            sub = [r for r in rs if cat == "ALL" or r["cat"] == cat]
            ok = sum(r["ok"] for r in sub)
            row += [f"{ok}/{len(sub)} ({100 * ok / max(1, len(sub)):.0f}%)", f"{_pct([r['secs'] for r in sub], .5):.1f}"]
        print("| " + " | ".join(row) + " |")
    for name, rs in runs.items():
        secs = [r["secs"] for r in rs]
        fast = [r for r in rs if r["path"] == "fast"]
        print(f"\n{name}: p50 {_pct(secs, .5):.1f}s  p90 {_pct(secs, .9):.1f}s  max {max(secs):.1f}s  "
              f"fast-path {len(fast)}/{len(rs)} ({sum(r['ok'] for r in fast)} correct)  "
              f"avg LLM calls {sum(r['llm_calls'] for r in rs) / len(rs):.2f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # model replies contain emoji; cp1252 console crashes on them
    if sys.argv[1:2] == ["--report"]:
        return report(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vaayu-finetuned")
    ap.add_argument("--mode", choices=["new", "old"], default="new")
    ap.add_argument("--sample", type=int, default=0, help="stratified random subset size (0 = all)")
    ap.add_argument("--out", default="data/benchmark")
    ap.add_argument("--set", choices=["main", "holdout", "holdout2"], default="main")
    a = ap.parse_args()

    cases = {"main": build_cases, "holdout": build_holdout, "holdout2": build_holdout2}[a.set]()
    if a.sample:
        by_cat = defaultdict(list)
        for cs in cases:
            by_cat[cs["cat"]].append(cs)
        frac = a.sample / len(cases)
        r = random.Random(1)
        cases = [x for cat in by_cat.values() for x in r.sample(cat, max(1, round(len(cat) * frac)))]

    brain = OllamaBrain("http://127.0.0.1:11434", a.model, fast_paths=(a.mode == "new"))
    calls = []
    real_dispatch = _stub_dispatch

    def recording_dispatch(self, name, arguments):
        # Score what would actually execute, i.e. after the real argument repair.
        spec = registry._tools.get(name)
        calls.append((name, repair_arguments(spec, arguments) if spec else arguments))
        return real_dispatch(self, name, arguments)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.model.replace(':', '_')}_{a.mode}{'' if a.set == 'main' else '_' + a.set}.json"
    results = []
    with patch.object(ToolRegistry, "dispatch", recording_dispatch), \
            patch("assistant.llm.ollama_client.issue_log.log_issue"):
        brain.respond([{"role": "user", "content": "hi"}], ToolRegistry())  # load the model
        for i, cs in enumerate(cases, 1):
            calls.clear()
            chat_calls = []
            orig_chat = brain.client.chat

            def counting_chat(*args, **kw):
                chat_calls.append(1)
                return orig_chat(*args, **kw)

            brain.client.chat = counting_chat
            t = time.perf_counter()
            try:
                reply, _ = brain.respond([{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": cs["text"]}], registry,
                                         confirm=lambda n, args: True)
            except Exception as exc:  # a crash is a failed case, not a dead benchmark
                reply = f"CRASH: {exc!r}"
            dt = time.perf_counter() - t
            brain.client.chat = orig_chat
            ok = score(cs, calls)
            results.append({**cs, "ok": ok, "secs": round(dt, 2), "llm_calls": len(chat_calls),
                            "path": "fast" if not chat_calls else "llm", "calls": calls[:3], "reply": reply[:160]})
            print(f"[{i}/{len(cases)}] {'OK ' if ok else 'BAD'} {dt:5.1f}s {results[-1]['path']:4} "
                  f"{cs['cat']:20} {cs['text']!r} -> {calls[:1]}", flush=True)
            path.write_text(json.dumps(results, indent=1), encoding="utf-8")  # survive a crash mid-run

    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
