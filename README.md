# Vaayu — Personal AI Assistant (Jarvis/Friday-style)

A local, privacy-first voice assistant for Windows, named **Vaayu** — built incrementally,
milestone by milestone. See [`PLAN.md`](PLAN.md) for the full architecture and roadmap.

**Status: M3 — real voice I/O, hands-free continuous conversation, wake/sleep/shutdown voice
commands, auto-starts at login, glowing status orb.**

## Status orb

A small marble-like sphere in the top-left corner is Vaayu's state indicator — a radial-gradient
color blend (cyan → blue → purple → pink, glass-marble style).

- **Rotating = awake and listening for real commands. Stationary = asleep** (still running, only
  listening for its name). This is the actual point of the widget, not decoration — verified for
  real: identical frames while asleep (confirmed static), visibly different wedge positions a second
  apart while awake (confirmed rotating), spins faster while actively speaking.
- **Disappears entirely on "shutdown Vaayu"**, until it's started again. This needed a real fix, not
  just "let it happen": the orb runs on the main thread while the orchestrator runs on a background
  thread, so the orchestrator finishing does *not* stop the GUI on its own — shutdown now explicitly
  signals the orb to close. Verified: the window closes and the process exits cleanly.
- **Drag it** with the mouse to reposition it anywhere on screen.
- **Move it by voice**: "move to center", "a little left", "top right", etc. — the `move_status_orb`
  tool accepts named positions (center, top/bottom, left/right, the four corners) or relative nudges
  with an optional magnitude word ("a little", "a lot", "further").
- One known cosmetic gap: it was meant to look like a borderless floating sphere (via Windows'
  window-transparency attribute), but that didn't visibly take effect in testing (tried several
  standard approaches, verified via screenshot) - you'll likely see a small dark square panel behind
  it instead. Everything that actually matters (rotation as the sleep/awake signal, real voice
  reactivity, dragging, voice-controlled positioning, disappearing on shutdown) is confirmed working.

## Vaayu's states

Vaayu auto-starts at every login (see **Auto-start** below), beginning **asleep** — it's running
and listening, but ignores everything except its own name:

- **"Vaayu"** (just the name, or "wake up Vaayu") → wakes up.
- **"Vaayu, open brave"** (name + a command in one breath) → wakes up **and** runs that command
  immediately, no need to wait for a confirmation first. Verified working (with the transcript
  controlled directly, to isolate this from the "Vaayu" mishearing issue below).
- **"Sleep Vaayu"** → goes back to sleep (still running, still listening for its name, just ignores
  everything else) — does *not* stop the process.
- **"Shutdown Vaayu"** → actually exits the running process.

**⚠️ Known unresolved issue**: recognizing the word "Vaayu" itself could not be made reliable in
testing — every test (multiple prompt/vocabulary variants tried) consistently misheard it as
unrelated words ("baby", "demon", "dream", "They"), never as "Vaayu" or "Vayu". This is very likely
because the robotic English TTS voice used for testing mispronounces this Sanskrit-origin word in a
way real speech never would (the same root cause noted in the Hinglish section below) — but **this
genuinely needs to be tested with your real voice**. If "Vaayu" doesn't work for you, tell me the exact
wrong text it heard (check `data/vaayu.log`, or run `python -m assistant.voice_main` in a terminal to
see it live) and I'll add that specific mis-hearing as an accepted alias, the same way "WhatsApp" got
fixed earlier. Both `Vaayu` and `Vayu` spellings are already accepted either way.

## Setup

1. Install Python 3.11 or 3.12 (64-bit), and [Ollama](https://ollama.com/download) for Windows.
2. Pull the model: `ollama pull qwen2.5:7b-instruct`
3. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. `config/config.yaml` already exists with sane defaults — edit it if you want a different model.
5. Sanity-check your setup:
   ```
   python scripts/setup_check.py
   ```
   Checks Ollama/model, microphone, and (informationally - not a hard failure) whether a browser is
   already reachable on the remote-debugging port that tab/YouTube/WhatsApp tools need.
6. Run the assistant, in either mode:
   - **Text mode** (type instead of speak — good for quick testing):
     ```
     python -m assistant.main
     ```
   - **Voice mode** (real microphone + spoken replies, hands-free):
     ```
     python -m assistant.voice_main
     ```
     Just start talking once it says "Ready" — no hotkey needed. The mic stays open continuously;
     it automatically detects when you start and stop speaking (voice activity detection), handles
     that as one command, speaks its reply, and immediately goes back to listening. Have a full
     back-and-forth conversation, e.g.: "open chrome" → (it responds) → "open youtube.com" →
     (it responds) → "what's my battery percentage" → ... Ctrl+C to quit.

   Try asking things like: "what's my battery percentage?", "open notepad", "is my bluetooth on?",
   "what's my speaker volume?", "search for files named resume", or "save a note that says X".
7. Run the automated test suite any time: `pytest tests/` (5235 tests as of this milestone, all
   real assertions against mocked CDP/AppOpener/Ollama/win32/psutil boundaries, runs in ~20s - see
   **Testing philosophy** below for why so many of these are parametrized into the thousands)

### Voice mode notes

- First run downloads the Whisper speech-to-text model (~150MB for the default multilingual `base`)
  — one-time, then cached locally.
- Speech-to-text runs on CPU by default even if you have an NVIDIA GPU — `faster-whisper`'s
  GPU auto-detection can find a GPU but then crash if the matching CUDA/cuBLAS runtime isn't
  installed, so CPU is forced explicitly for reliability.
- Text-to-speech uses Windows' built-in SAPI5 voices via `pyttsx3` — functional but robotic-sounding;
  a higher-quality option (Piper) is planned for M6.
- **App-name/word recognition**: improved via a vocabulary hint (`_VOCAB_PROMPT` in
  `stt/whisper_stt.py`) that biases Whisper toward expected app/command vocabulary (WhatsApp,
  Instagram, Gmail, Bluetooth, etc.) — tested across 10 varied commands, 9/10 correctly recognized
  with the hint vs. clear failures without it. Upgrading the model size (`base.en` → `small.en`) was
  tested too and made no real difference here, so `base.en` stays the default (faster, same accuracy
  for this).
  **Known remaining limitation**: some compound/coined words (tested case: "Paytm") still get
  misheard even with the hint - repeating the word several times in the prompt fixed it in testing,
  but a sentence mention didn't, and repeating *every* word isn't practical without diluting the
  hint's effect on everything else. If a specific word keeps failing, tell me and I can add targeted
  repetition for just that word.
- **Hinglish support (e.g. "whatsapp open kro")**: switched from the English-only `base.en` model
  to the multilingual `base` model (`base.en` is architecturally incapable of producing Hindi/Hinglish
  words no matter how clearly they're spoken), kept `language: "en"` so output stays in Roman script
  (how Hinglish is normally typed) instead of switching to Devanagari, and added common Hindi command
  words to the vocabulary hint. **Important caveat: this could not be properly tested** - there's no
  Hindi voice or language pack on the dev machine this was built on, so realistic Hindi/Hinglish
  pronunciation couldn't be verified end-to-end (an English voice mispronouncing Hindi words produces
  audio neither a human nor Whisper can reliably parse). A controlled comparison confirmed switching
  to the multilingual model doesn't clearly hurt plain English accuracy, so it's kept as the default -
  but the Hinglish recognition itself needs testing with your real voice. Tell me what gets misheard
  and I'll tune the vocabulary hint or try `language: "hi"` instead.
- **No wake word yet (that's M5)**: since the mic is always listening, the assistant can't tell your
  voice apart from any other audio in the room. Confirmed in testing: a video/audio playing nearby
  gets transcribed and treated as a command just like real speech - if that happens often, muting
  other audio sources while using voice mode helps, and a wake word would be the real fix.
- **Push-to-talk fallback still available**: set `activation.mode: "hotkey"` in `config.yaml` to go
  back to press-`Ctrl+Alt+J`-then-speak mode instead of continuous listening.
- **Speech rate**: `tts.rate` in `config.yaml` (default 210, ~1.2x normal pyttsx3 speed) - raise/lower
  to taste.
- **Dictation accuracy caveat**: short, soft-onset trigger words (e.g. "type") can occasionally be
  misheard as a similar-sounding word (e.g. "Time") by the speech-to-text model, especially with
  quieter/less clear speech. If a command like "type ..." isn't recognized, try again a bit louder
  and more clearly, or upgrade the Whisper model size in `stt/whisper_stt.py` for better accuracy.

## Auto-start (runs at every login)

A Windows Scheduled Task named **"Vaayu Assistant"** is registered to run `assistant.voice_main`
(via `pythonw.exe`, no console window) at every login for this user, starting asleep. It was created
with:
```
Register-ScheduledTask -TaskName "Vaayu Assistant" -Action <pythonw.exe -m assistant.voice_main, working dir = this repo> -Trigger <AtLogOn> -Force
```
- **Check its status**: `Get-ScheduledTask -TaskName "Vaayu Assistant"`, or open Task Scheduler
  (`taskschd.msc`) and look under the root folder.
- **Since it has no console, its output goes to `data/vaayu.log` instead** — check that file to see
  what Vaayu has heard/done, or if it failed to start.
- **Manually start/stop it**: `Start-ScheduledTask -TaskName "Vaayu Assistant"` /
  `Stop-ScheduledTask -TaskName "Vaayu Assistant"`.
- **Remove auto-start entirely**: `Unregister-ScheduledTask -TaskName "Vaayu Assistant" -Confirm:$false`.
- Config resolution (`config/config.yaml`) and all data paths are resolved relative to the repo's own
  location, not the current directory, specifically so this works regardless of how/where the process
  gets launched from.
- **This is not a true low-power wake word** — full speech-to-text still runs continuously even while
  "asleep" (to catch "wake up Vaayu"), which uses real CPU the whole time the laptop is on. Fine for a
  plugged-in laptop; worth knowing if you care about battery life while unplugged.

## Available tools

- `get_system_status` — CPU, RAM, battery, disk
- `get_network_status` — Wi-Fi (connected network, signal) and Bluetooth (on/off, paired devices)
- `get_audio_status` — default speaker/microphone name, volume, mute state
- `list_connected_devices` — USB, Bluetooth, display, or printer devices
- `open_app` / `close_app` — launch or close an application by name (this includes browsers - "open
  chrome"/"open brave" launch the app, don't search for the word). **Brought to the foreground by
  default** — say "in the background" to keep it from stealing focus.
- `control_media` — play/pause, next/previous track, volume up/down, mute
- `open_website` — open a URL or run a web search, **targeting whichever browser you're actually
  using** (see below), not always the OS default. Also foreground-by-default. Opens a **new** tab
  unless `same_tab=true` (for "open X in this tab"/"on the same tab"), which navigates the current
  tab in place via CDP instead. "open the &lt;name&gt; website" (e.g. "open the moviesmod website")
  is treated as a direct domain (moviesmod.com), not a Google search of that phrase.
- `search_files` / `read_file` — scoped to Desktop/Documents/Downloads
- `add_note` / `list_notes` — simple local note storage
- `list_open_windows` / `switch_to_window` — see and switch between open desktop windows
- `list_browser_tabs` / `switch_to_browser_tab` / `close_browser_tab` — see, switch to, and close
  one specific open browser tab (by title/URL text, or by position - "the third tab") without
  touching the browser itself or any other tab. Requires Chrome/Brave running with
  `--remote-debugging-port=9222` — see below.
- `minimize_window` / `restore_window` — send a window to the taskbar or bring it back; omit the
  title on `minimize_window` to minimize whatever's currently focused ("minimize this").
- `set_video_volume` / `set_video_fullscreen` / `seek_video` / `control_video_playback` —
  site-agnostic control of whatever `<video>` is playing in a browser tab: in-page volume up/down/
  absolute/mute (separate from the system volume — see below), fullscreen toggle, skip forward/
  backward by an exact, verified number of seconds, and play/pause/toggle. These work on *any*
  site's player (Hotstar, Netflix, YouTube, anything with a standard HTML5 video element), not
  hardcoded to YouTube — `tools/video_player.py`. Fullscreen uses a real simulated 'f' keypress
  (the de-facto standard shortcut across these players) rather than calling the browser's
  Fullscreen API directly from JS, since a JS-invoked `requestFullscreen()` has no genuine
  user-activation and Chromium silently rejects it (tested). All four verify the actual resulting
  page state before reporting success, and raise instead of claiming success if it didn't take
  effect. **Multi-video disambiguation**: if more than one tab has a video actively playing at once
  (e.g. music on YouTube and a show on Hotstar simultaneously), none of these guess — they check
  every open tab first and, if more than one qualifies, raise an error naming each by tab title and
  asking which one; the model relays that as a question, and the answer becomes `tab_hint` on the
  next call, which skips the ambiguity check and resolves straight to the matching tab.
- `type_text` — dictate text into whatever window currently has focus (e.g. "open notepad" then
  "type hello world")
- `type_sensitive_field` — like `type_text`, but for a phone number, OTP/verification code,
  password, or PIN going into a login form on any site. **Sensitive action**: always asks for a real
  spoken confirmation first (reading back the exact value and which field), before typing anything —
  see **Confirmation flow** above. There's deliberately no site-specific "log in" tool (e.g. for
  Hotstar) — logging in anywhere is just `open_website` → `read_page`/`click_on_page` to navigate
  the form, then `type_sensitive_field` for the phone number/OTP itself, which generalizes to every
  site instead of hardcoding one site's login form (which would also be exactly as unverified as the
  WhatsApp tool below, without adding real generality).
- `browse_to` / `read_page` / `click_on_page` / `focus_input_field` — real page interaction:
  navigate the active tab, read what's visible/clickable/fillable on it, click a link/button by its
  visible text, and click/focus a text input or textarea by whatever identifying text it actually
  has (`read_page`'s `input_fields`). This is what lets Vaayu actually follow through on "click the
  channel" or "log in with this phone number" instead of just running a search and stopping.
  `focus_input_field` exists as its own tool, not an extension of `click_on_page`, because a real
  field found live-testing (Hotstar's phone number field) had no placeholder/name/id at all - only a
  `title` attribute and matchable parent text - which `click_on_page`'s link/button-only selector
  can't reach regardless of matching logic. Requires `--remote-allow-origins=*` in addition to
  `--remote-debugging-port=9222` (see below) — a newer Chromium security requirement specifically
  for this WebSocket-based control, on top of the tab-listing HTTP endpoints.
- `play_latest_video_from_channel` / `play_video_on_youtube` — purpose-built YouTube tools (find a
  channel's newest video; search and play a song/artist/genre/specific title). Built after testing
  showed the local 3B model isn't reliably able to plan the underlying multi-step
  search → read → click sequence on its own (it kept hallucinating URLs and reaching for the wrong
  tools) — these do the actual navigation deterministically in Python instead, so the model only has
  to make one simple, reliable tool choice. `play_latest_video_from_channel` caches the channel
  name → channel URL mapping (`tools/cache.py`, `data/tool_cache.json`, 7-day freshness) since that
  mapping is stable — a repeat request for the same channel skips the search-page navigation
  entirely, falling back to a fresh search automatically if the cached URL ever turns out stale.
- `move_status_orb` — reposition the on-screen status orb by voice (see **Status orb** above).
- `press_hotkey` — press any keyboard shortcut on whatever window has focus (ctrl+s, alt+tab,
  ctrl+shift+esc, etc.) — a general fallback that covers most app functionality (save/undo/copy/
  paste/switch-window/close/new-tab/...) without needing a bespoke tool per app per feature. This is
  also how **VLC (or any other desktop media player)** gets controlled: `switch_to_window("vlc")`
  then `press_hotkey` with VLC's own shortcuts (space=play/pause, up/down=volume, left/right=seek).
  ⚠️ Unlike `seek_video`'s exact, verified seconds on a browser tab, VLC's arrow-key seek length
  depends on VLC's own configured jump setting (commonly, but not guaranteed to be, 10 seconds) —
  there's no way to make this precise through a keyboard shortcut alone, so it's left as an honest
  approximation rather than a false promise of an exact number.
- `lock_computer` — lock the Windows lock screen (same as Win+L).
- `take_screenshot` — capture the whole screen, saved to `data/screenshots/`.
- `send_whatsapp_message` — compose and send a WhatsApp message via WhatsApp Web (browser must
  already be logged in - scan the QR code once, same as normal). **Sensitive action**: Vaayu speaks
  the actual message back and requires a real spoken "yes" before it sends - see **Confirmation
  flow** below. ⚠️ Built against WhatsApp Web's commonly-referenced selectors but **could not be
  tested against a real logged-in session** in the environment this was built in - please test it
  live; if a step fails the error names exactly which one (search box / contact match / message box
  / send button / send verification), which is what to report back.
- `set_daily_update_time` — change when Vaayu's daily issue check-in happens ("set the update time
  to 9pm"). See **Daily check-in** below for what this actually does (writes a report, doesn't
  rewrite code).

### Browser targeting for `open_website`

A real reported bug: "open brave" then "open youtube" opened Brave, then silently opened YouTube in
Chrome instead (the OS default), not Brave. Fixed with `tools/browser_state.py`, which picks a
specific browser instead of always going to the OS default, in this order: (1) the browser Vaayu most
recently opened itself, (2) whichever registered browser currently owns the focused window, (3) any
registered browser that's running at all. This is **general, not a hardcoded browser list** — it reads
Windows' own registry of installed browsers (the same list the "choose your default browser" Settings
page uses), so it works for Opera, Vivaldi, or anything else installed, not just the ones anticipated
in advance. Verified for real: opened a browser manually (not via Vaayu) with another one also
running, confirmed the *foreground* one got targeted correctly, not just "any running browser."

### Enabling browser tab control and page interaction

These talk to Chrome/Brave's DevTools protocol, which is off by default. To enable it, close all
browser windows and relaunch with:
```
chrome.exe --remote-debugging-port=9222 --remote-allow-origins=*
```
(or add those flags to your normal browser shortcut). `--remote-debugging-port` alone is enough for
`list_browser_tabs`/`switch_to_browser_tab`; `--remote-allow-origins=*` is additionally required for
`browse_to`/`read_page`/`click_on_page`/the YouTube tools (a newer Chromium security requirement for
WebSocket-based control specifically - found by testing, not documented anywhere obvious). Without
these, the affected tools fail with a clear error rather than crashing — everything else works fine
without them.

### Confirmation flow (sensitive actions, e.g. WhatsApp)

Some tools are registered `requires_confirmation=True` (currently just `send_whatsapp_message`, per
the permanent safety rule in `PLAN.md`: anything sending a communication on the user's behalf must
be confirmation-gated). In voice mode, `Orchestrator._voice_confirm` handles this: it speaks the
*actual* confirmation question (reading back real argument values - "Send 'On my way' to Mom on
WhatsApp?", not a vague "are you sure?"), listens for one more spoken reply in the same turn, and
only proceeds on a recognizably affirmative answer (`control_phrases.is_affirmative` - "yes"/"yeah"/
"go ahead"/etc., with any negative word like "no"/"cancel"/"wait" always overriding). No response
heard, or an unrelated/negative reply, both decline the action. This replaces the old default
(`cli_confirm`, a blocking terminal `input()` prompt) which would have just hung forever in voice
mode - confirmed no tool used this path until now, so it was an untested gap, not a regression.

### Latency: two real, measured fixes

Real reported complaint: "his responses are taking too much time." Measured (not guessed) two
distinct causes in `llm/ollama_client.py`:

1. **Ollama unloads the model after 5 minutes idle by default.** Reloading it is slow - measured
   ~24s wall time on this machine for the 3B model vs ~0.3-4s once warm, entirely separate from
   actual inference time. Since Vaayu is meant to sit idle between commands for normal
   conversational use, every command after a few minutes' pause was paying this reload tax - very
   likely the real cause of the complaint. Fixed with `keep_alive=-1` on every call, keeping the
   model resident until Ollama itself restarts (trades ~2GB of continuously-held RAM for this).
2. **The default context window (4096 tokens) was nearly exhausted by tool schemas alone** - measured
   3214 tokens for the system prompt + one short user message + this project's tool set via
   `prompt_eval_count`, leaving under 900 tokens for the rest of any real conversation before Ollama
   silently drops earlier messages (including the system prompt) out of context. This isn't just
   slow, it's a correctness bug that gets worse as tools/conversation length grow - plausibly part of
   why tool selection sometimes seemed to "forget" earlier instructions. Fixed with
   `options={"num_ctx": 8192}` on every call.

Both verified for real: force-unloaded the model (`keep_alive: 0` via the raw API), confirmed the
fix keeps it resident afterward (`expires_at` far in the future via `/api/ps`) and that
`context_length` reports 8192. Covered by regression tests in `tests/test_ollama_client.py` so
these can't silently disappear in a later edit.

### Multi-step tool calling

A single spoken command can chain up to 5 tool calls (e.g. read a page, click something, read again,
click again) instead of being limited to one action per command - upgraded once page-interaction
tools made multi-step web tasks common. In testing, whether the model actually *uses* this reliably
varies a lot by task: simple chains (search → click) work, but it isn't yet reliable at planning
longer ad-hoc sequences on its own - which is exactly why `play_latest_video_from_channel` and
`play_video_on_youtube` exist as one-call tools instead of leaving that planning to the model.

### ⚠️ `close_app` closes whatever's running, not just what Vaayu opened

Real incident during testing: a bulk test that opened then closed Visual Studio Code closed the
user's own already-open VS Code window (with their real work in it), not a separate instance -
`close_app` (via AppOpener) matches by app name and closes whatever's running, with no way to tell
"the one I opened" apart from "the one you were already using." Be careful asking Vaayu to close
things you're actively using, especially editors/IDEs - it can't distinguish your work from a test.

### ⚠️ Grounding fix: no more "confirmed done" without a tool actually running

Real reported bug: asked Vaayu to close Brave, it replied that Brave was closed, but Brave stayed
open. Traced the actual cause (`AppOpener`'s fuzzy match + `taskkill` logic itself checked out fine
in isolation) to the small local model sometimes skipping the tool call entirely and just answering
in plain text as if the action had happened - the system prompt already said not to do this, but a
3B model doesn't reliably obey a pure text instruction under pressure.

Fixed at the code level, not just the prompt, in `llm/ollama_client.py`: if the user's message looks
like an action request (matches a curated action-verb list - open/close/play/send/increase/etc.) and
the model's first response calls no tool at all, it now gets exactly one corrective system message
("you did not call any tool - call the matching tool now, or say plainly you can't") before its
reply is accepted - bounded to one retry, so it can't loop, and it never fires for ordinary
conversation. Verified with `tests/test_ollama_client.py`, which directly reproduces this exact
scenario (model claims "Brave is now closed" with zero tool calls) and asserts the retry forces a
real `close_app` dispatch before the final reply is accepted.

The same principle now applies to every `video_player.py` tool (see above) - volume, fullscreen,
seek, and play/pause all re-read the actual page state after acting and raise an error instead of
returning a success message if the state didn't actually change.

### ⚠️ Fixed: "not right"/"not correct" could be misread as confirming a send

Found by reasoning through `control_phrases.is_affirmative` before writing a single test for it (the
subsequent large test pass then locked the fix in): `is_affirmative("that's not right")` returned
**True**. "not" wasn't in the negative-word set, so it was ignored, and "right" alone is a recognized
affirmative word - meaning a user declining a WhatsApp-send confirmation with "that's not right" or
"not correct" would have been read as a **yes**. This is the exact gate that decides whether a real
message actually gets sent on the user's behalf.

Fixed by adding "not" to the negative-word set, deliberately biased toward the safe failure mode:
this can now occasionally misread a genuine confirmation that happens to contain "not" in a
non-negating way (e.g. "yes, why not, go ahead") as a decline - but for a send-a-message gate, a
false affirmative (sends something the user declined) is far worse than a false negative (asks
again). See `control_phrases.py`'s `is_affirmative` docstring for the reasoning; regression coverage
for every affirmative word negated this way is in `tests/test_extended_suite_large.py`.

### Live-tested against real Hotstar - what actually broke, and what's fixed now

At the user's request, this was tested for real: a throwaway, isolated Chrome profile (not the
user's real Brave session - never touched it) launched with CDP enabled, pointed at
`hotstar.com` (redirects to `jiohotstar.com`), and driven with the actual tools, not mocks. Four
real things surfaced, all fixed:

1. **Any Unicode character Vaayu needs to print could crash the whole process.** Reading the real
   Hotstar homepage returned a rupee sign (₹) in its text; printing that via `orchestrator.py`'s
   normal `print(f"Assistant: {reply}")` crashed with `UnicodeEncodeError` when launched from a Git
   Bash terminal (`sys.stdout.encoding` reports `cp1252` there) - the *exact same command* from
   native PowerShell on the same machine reports `utf-8` and works fine, so this depends entirely on
   which terminal launches Vaayu, not the machine. Nothing in the main loop catches that exception,
   so it would have killed the process mid-conversation. `log_setup.py` previously only handled the
   *headless* (`pythonw.exe`/auto-start) case; now it also forces `stdout`/`stderr` to UTF-8 (with a
   `errors="replace"` last-resort fallback) whenever a real console is attached, for both entry
   points. Verified live that forcing UTF-8 doesn't just avoid the crash, it actually renders
   correctly - the Git Bash terminal could display the rupee sign fine once told to use UTF-8, so
   Python's auto-detected encoding was simply wrong, not a real limitation of that terminal.
2. **There was no way to actually reach a real login field.** Hotstar's phone-number input had no
   placeholder, name, or id text at all - `click_on_page` (links/buttons only) couldn't reach it, and
   `read_page` didn't surface input fields, so the model would have had no way to even discover the
   field existed, let alone target it. Real sites don't consistently label fields the same way (this
   one only had a `title="Mobile number"` attribute and matchable parent text) - fixed by having
   `read_page` also list `input_fields` (matching by title/aria-label/placeholder/name/label/parent
   text, whichever exists) and adding a new `focus_input_field` tool to get into one before typing.
   This is the actual missing link that made `type_sensitive_field` (built last round) reachable in
   practice, not just in theory.
3. **Clipboard paste can fail intermittently - a real, external Windows behavior, not a bug in this
   code.** `type_text`/`type_sensitive_field`'s clipboard set (`SetClipboardData`) sometimes failed
   with "The handle is invalid" testing live - confirmed genuinely intermittent, not deterministic
   (isolated every piece of the sequence separately, all fine every time; the exact same full
   sequence failed 3/3 runs in one stretch, then succeeded 3/3 moments later with nothing changed).
   This matches Windows' documented clipboard-contention behavior (its own Clipboard History/Cloud
   Clipboard service, or any other app, can transiently hold the clipboard). For `type_sensitive_field`
   specifically this matters - an OTP is time-sensitive and single-use, so a silent failure isn't a
   minor inconvenience. Fixed with retry-with-backoff (8 attempts, ~2s worst-case budget, sized by
   testing against a real failure that outlasted a shorter one) around every clipboard operation.
4. **The CDP debug connection itself dropped mid-session once**, unprompted, requiring the test
   browser to be relaunched - noted as an observed reliability characteristic worth being aware of
   (a `read_page`/`click_on_page`/tab-tool call can fail with a connection error even mid-session,
   not just at startup), not something with an identified fix here.

What worked correctly, live, no changes needed: `read_page` against real page content (2000+
characters, real clickable elements, real Unicode), `click_on_page`, tab listing/targeting,
`open_website`'s real-site detection, and the login-required redirect being handled the same way
any other page content would be (nothing crashed or hung on hitting Hotstar's auth wall). Actual
OTP submission was **not** tested - that needs a real phone number and a real, live-relayed SMS
code from the user, which isn't something to attempt unattended.

### Testing philosophy

5235 tests total (`pytest tests/`, ~20s). 184 of those give every registered tool module real,
direct coverage (URL construction, tab matching/position resolution, volume/fullscreen math and
state verification, the grounding-retry and voice-confirmation fixes, win32 foreground-forcing
including its AttachThreadInput-denied fallback path, the browser-targeting resolution order, disk
cache round-trips, and the state machine and orb positioning).

2308 (`tests/test_browser_suite_large.py`) are a large parametrized suite specifically
for browser functionality, at the user's explicit request for 1000+ browser test cases even after
an initial pushback on the number. The concern with a literal 1000+ suite is real end-to-end tests
(actual browser + actual Ollama calls) taking hours to run for mostly redundant coverage - that
doesn't apply here, since every case is generated by parametrizing real inputs (180 realistic
domains, 100 search-query phrasings, 50 tab-list sizes with position resolution at every boundary,
the full 0-100 YouTube volume range, ~150 relative volume deltas, adversarial/unicode click text,
WhatsApp confirmation-prompt formatting across contact/message combinations, cache round-trips,
browser-resolution order across registered-browser combinations) against the real tool logic, with
only the CDP/urlopen/subprocess boundary mocked - so the whole suite still runs in seconds, not
hours. Building it this way is also exactly what found two real bugs that smaller, hand-picked
examples had missed: `open_website`'s scheme check was case-sensitive (`HTTPS://x.com` silently
became `https://HTTPS://x.com`), and `" <name> web site"` phrasing matched the shorter `" site"`
suffix first and got stuck with a leftover space instead of ever trying the `" web site"` suffix
that actually fit. Both fixed, both now covered by regression cases in the large suite.

The remaining 2850 (`tests/test_extended_suite_large.py`) extend the same approach to everything
else: `is_affirmative`/`detect_control_command` table-driven from the module's own defined
vocabulary (every affirmative/negative word and phrase, every name spelling × control word ×
sentence template), `press_hotkey` across ~350 real modifier+key combinations, `open_app`/`close_app`
across 30 real app names, window management across list sizes 1-40, device/network status across
count and state grids, system/audio status across fine-grained value grids (battery 0-100, speaker
and mic volume at 2-5% steps), dictation across adversarial/unicode text, the cache's freshness
boundary exactly at/around the expiry threshold, every registered tool's JSON schema validity and
hallucinated-argument filtering (iterated directly over the live registry, so a new tool gets this
coverage automatically), config loading across many valid values, and the orb position parser's
full direction × magnitude × starting-position grid plus every named position across 10 screen
sizes. This is also where the `is_affirmative("not right")` bug above was found - before writing a
single test for it, just from reading the implementation while planning what to cover.

## Daily check-in (issue logging + a permission-gated report, not self-modifying code)

The user asked for Vaayu to "learn and rewrite his code using the internet" on a daily schedule.
Built the part of that which is real and safe; declined the literal ask, for two separate reasons
stated plainly rather than glossed over:

- **Capability**: Vaayu's brain is a local 3B-parameter model (qwen2.5:3b). This project's entire
  bug history is the evidence for why that model can't be trusted to diagnose, fix, test, and ship
  changes to its own source unsupervised - it needed the grounding-retry fix above just to reliably
  *call a tool* instead of lying about the result, and it hallucinates URLs planning more than a
  step or two ahead. Pointing it at its own codebase wouldn't produce self-improvement.
- **Safety**: an unattended process that edits and redeploys code controlling real actions (sending
  messages, typing passwords) with no human review is a real risk regardless of which model drives
  it - the same reason `send_whatsapp_message`/`type_sensitive_field` are confirmation-gated in the
  first place, just applied to code changes instead of a single action.

What's actually built (`assistant/self_review/`):

- **`issue_log.py`** - every real tool failure (a dispatch that returns `ok: False` - not a declined
  confirmation, that's expected behavior) and every time the grounding-retry fires gets appended to
  `data/issues_log.jsonl`, one JSON object per line, wired directly into `llm/ollama_client.py`'s
  tool-calling loop. Entries are marked acknowledged (never deleted) only after a real, confirmed
  check-in - a missed day or a "not now" leaves them exactly as they were, picked up automatically
  at the next check-in, per the explicit requirement that a missed update time must not lose data.
- **`Orchestrator._maybe_run_daily_checkin`** - runs once per day (in the configured timezone), at
  or after the configured time, regardless of sleep/awake state (this is Vaayu speaking up on its
  own). If there are unacknowledged issues, it speaks a short summary ("3 issues today, mostly
  around close_app, seek_video") and asks permission. On a real spoken "yes" (`is_affirmative`, same
  mechanism as every other confirmation in this project), it writes a human-readable markdown report
  to `data/daily_reports/YYYY-MM-DD.md` and says "Update's done." **"Done" means the report is
  ready for review, not that code changed.** On "no" or no response, everything stays exactly as
  logged for the next attempt.
- **`set_daily_update_time`** (tool) - "set the update time to 9pm" persists a new
  `self_review.check_in_time` into `config/config.yaml` via a targeted text edit (not a full
  re-serialize), specifically so the user's own comments and formatting in that file survive.
  Defaults: `21:00`, `Asia/Kolkata` (IST), per explicit instruction that times are IST unless a
  timezone is named.

Real bug found building this: `zoneinfo.ZoneInfo("Asia/Kolkata")` raised `ZoneInfoNotFoundError` on
this machine - confirmed live, not theoretical. Windows doesn't ship IANA timezone data the way
Linux/macOS do; the stdlib `zoneinfo` module needs the `tzdata` PyPI package there specifically.
Without it, the whole feature would have silently done nothing forever (the exception is a
`KeyError` subclass, so it was already being swallowed quietly rather than crashing - which meant
it could easily have gone unnoticed). Added `tzdata` to `requirements.txt`.

Also found, and fixed, while building this: adding a real file-write side effect inside the
tool-calling loop (`issue_log.log_issue`) made pre-existing tests that exercise that exact code
path - written before this feature existed, with no reason to expect a disk write - silently append
real entries to the actual `data/issues_log.jsonl` in this repo. Confirmed by finding that file
genuinely polluted with test data after a normal test run. Fixed with an autouse pytest fixture
(`tests/conftest.py`) that isolates every self_review data path for every test in the suite by
default, not just the ones that know they touch it - the systemic fix, since remembering to mock it
per-test doesn't protect a test written next month that happens to exercise the same path. That
fixture initially used `tmp_path` per test and roughly tripled the whole suite's runtime (~20s ->
~55s, measured) since it forced a real directory creation for all ~5000 tests regardless of whether
they touch self_review at all; switched to a session-scoped base directory with pure path
arithmetic per test (no I/O unless a test actually writes) to get back to ~20s.

## Model upgrade + fine-tuning

Asked, after the self-code-rewriting discussion: could Vaayu's own brain be upgraded, or even
fine-tuned, instead? Checked what's actually real on this machine before building anything.

**Model upgrade (`qwen2.5:3b` → `qwen2.5:7b-instruct`)**: this was the original plan's default
model all along (see `PLAN.md`) - `qwen2.5:3b` was only ever the fallback, used because the 7B
never finished pulling on a slow connection at the time. Pulled successfully this time (confirmed
via `ollama list`, 4.68GB, matching the registry's own manifest exactly) and tested for real - but
**`config/config.yaml`'s `llm.model` was deliberately left on `qwen2.5:3b`**, because the actual
comparison data doesn't support just switching by default:

- **Speed**: this laptop's GPU (NVIDIA GTX 1650, 4GB VRAM) can't fit the whole 7B model -
  `ollama ps` confirmed only ~2.2GB of it lands on the GPU, the rest runs on CPU. Measured warm
  latency for a simple query: ~4.2s on 3B vs ~16.5s on 7B, isolated and clean (only one model
  loaded at a time) - roughly 4x slower. Given the original complaint that started this whole
  latency investigation was specifically about response speed, that's a real regression, not a
  minor cost.
- **Reliability**: tested both models against the exact bug-case commands from this session
  ("close the third tab", "increase the youtube volume", "play fairytale music", "close the brave
  browser"), same moment, same real production system prompt. Genuinely mixed, not a clean win: 7B
  correctly handled "close the third tab" (3B called no tool at all here); 3B correctly handled
  "close the brave browser" via `close_app` (7B picked `close_browser_tab` instead - the wrong
  tool, since that only searches for a *tab* titled "brave" rather than closing the browser). Two
  different failure modes, not "bigger is strictly better," on a sample this small.
- Also found: running both models loaded simultaneously (`keep_alive=-1` on each) caused real VRAM
  thrashing on this 4GB card - latencies got dramatically worse (43s and 106s) than either model
  measured in isolation. Worth knowing if anyone considers comparing/switching models live rather
  than picking one and sticking with it.

**Update: switched to `qwen2.5:7b-instruct` on explicit request**, after the tradeoff above was
laid out. `config/config.yaml`'s `llm.model` now reads `qwen2.5:7b-instruct`; verified for real
through the actual `assistant.main` entry point (not just ad-hoc scripts) - startup banner shows
the new model, a real query answered correctly, and a real confirmation-gated flow (WhatsApp) still
prompts and responds as expected. `qwen2.5:3b` is still pulled and available as a fallback
(`ollama list`) if 7B doesn't hold up well in real day-to-day use - just change `llm.model` back.

### ⚠️ Found live while testing the switch: a real "confirmed but didn't happen" recurrence, now closed

Testing "send hey there to vedant on whatsapp" for real (no browser running, so the tool
genuinely failed) surfaced a materially worse version of the original grounding bug: the tool call
failed with a real `ConnectionError`, and the model's very next reply - no second tool call -
claimed **"Sending 'hey there' to Vedant on WhatsApp"** anyway. The existing grounding-nudge fix
didn't catch this because it only guards the very first "no tool called at all" case; once a real
tool call has happened at all in the turn (even a failed one), that protection no longer applied -
a gap in the highest-stakes category this project has (a real message, actually confirmed, silently
not sent while being reported as sent).

Fixed by making this deterministic rather than trying to detect-and-nudge the dishonesty after the
fact: `respond()` now tracks whether the most recent real tool attempt succeeded, failed, or was
declined by the user (declining is a normal, wanted outcome - the model's natural "okay, not
sending that" is left alone). If the last real attempt genuinely failed and the model's next reply
carries no new tool call, the reply is replaced with the tool's own real error message - the
model's generated text is never trusted in that specific situation, no matter what it says.
Re-ran the exact WhatsApp scenario after the fix: now correctly reports
`"That didn't work - Can't reach Chrome's remote debugging port (9222)..."` instead of a false
success. Covered by regression tests reproducing this exact scenario, plus checks that a genuine
decline and a genuine retry-then-succeed both still flow through normally.

### ⚠️ Found and fixed while running this comparison: a hallucinated fake tool call

The 3B model, when its grounding-nudge retry (see the grounding fix above) fires, sometimes
generates Qwen's own raw `<tool_call>{...}</tool_call>` text format itself instead of using the
real tool-calling API. Confirmed live: for "close the third tab," it produced
`{"name": "close_browser_tab", "arguments": {"position: 3}}` (note the missing closing quote after
`"position` - malformed JSON) followed by a stray `</tool_call>` tag. Ollama's own parser couldn't
extract that as a real structured tool call, so it fell through as plain message content - which
would have been spoken to the user verbatim, literal JSON syntax and all. Reproduced with a clean,
single-model GPU state specifically to rule out this being a symptom of the VRAM contention from
testing two models at once - it's a genuine model behavior, not a test artifact.

Fixed in `_sanitize_reply` (`llm/ollama_client.py`): text matching a `<tool_call>`/`</tool_call>`
tag or a bare `{"name": ...` JSON shape is now treated the same as any other failed action attempt
- replaced with the existing honest "I don't have a tool for that" fallback rather than spoken
as-is. Covered by regression tests using the exact captured garbage string, plus checks that normal
replies (including ones that happen to contain the word "name") aren't false-positived.

**Fine-tuning**: checked this machine's actual GPU before promising anything - an NVIDIA GTX 1650
with 4GB total VRAM (~2.2GB free at the time). QLoRA fine-tuning a 7B model needs meaningfully more
than that even in 4-bit (practical minimums are usually cited around 8-12GB) - it would fail to even
load, not just run slowly. So real training happens on a free-tier cloud GPU (Google Colab gives a
16GB T4 at no cost), not this laptop - this machine only prepares the dataset and, afterward, runs
the resulting model via Ollama for actual use.

- **`scripts/build_finetune_dataset.py`** - generates `data/finetune/tool_calling_dataset.jsonl`
  (98 examples as of this pass) from the live tool registry, not hand-typed JSON. Every generated
  tool call is validated against the real registry schema before being written (wrong argument
  name, missing required field, etc. fail loudly at generation time, not silently at training
  time). Deliberately concentrated on the *exact* mix-ups found as real bugs this session -
  "close the third tab" needing `close_browser_tab` not `close_app`, "increase the youtube volume"
  needing `set_video_volume` not `control_media`, "play fairytale music" needing
  `play_video_on_youtube` not `open_website`, a phone number/OTP needing `type_sensitive_field` not
  `type_text` - each repeated across several phrasings, plus a handful of non-tool conversational
  examples so the fine-tune doesn't bias toward calling a tool for everything. Deliberately not
  padded to a large count: for a dataset this targeted, correctness and coverage of real confusion
  patterns matter more than volume, the same reasoning as the earlier browser test suite - and
  unlike synthetic tests, wrong training data doesn't just fail a test, it teaches the model the
  wrong behavior.
- **`scripts/validate_finetune_dataset.py`** - checks the dataset against the *real*
  Qwen2.5-7B-Instruct tokenizer before spending any Colab GPU time on it. This already caught a
  real bug while building it: the training script's first draft used `MAX_SEQ_LENGTH=4096`, but the
  real measured max across this dataset - rendered with the actual ~36-tool schema block Qwen's
  chat template injects, which is several thousand tokens on its own - is 7192 tokens.
  `SFTTrainer` truncates silently past `max_seq_length`, which would have cut the assistant's tool
  call off the end of most training examples with no visible error, wasting the GPU time training
  on broken targets. Fixed by measuring for real and raising the budget to 8192 (with a note to
  re-check as more tools get added, since that schema block only grows).
- **`scripts/finetune_on_colab.py`** - the actual QLoRA training script, meant to run in a Colab
  notebook (free T4 GPU), not locally. Passes `tools=registry.get_ollama_tools()` into the chat
  template when formatting training examples - not optional: confirmed by rendering both ways that
  omitting it trains on a materially different (much shorter, tools-schema-free) prompt shape than
  `ollama_client.py` actually sends at inference time, which would undermine the fine-tune before it
  even started. Exports a `.gguf` file at the end, with instructions for turning that into an Ollama
  model (`ollama create`) and pointing `config/config.yaml` at it.
- **Honestly unverified**: this was never run end-to-end on a real GPU - there wasn't one available
  in the environment this was built in beyond this laptop's 4GB card, which the script itself
  declines to use. The dataset format, the tokenizer/chat-template behavior, and the sequence-length
  math are all verified for real (against the actual tokenizer, without needing a GPU for that part)
  - the training run itself, and whether the resulting model measurably improves tool-calling
  reliability over the base model, are not. Run it on Colab and report back exactly what happens,
  the same way everything else in this project has been tested for real rather than assumed.

### Retargeted from 7B to 3B, after real timing data made the tradeoff decisive

A later real-usage timing pass (12 representative real commands, spanning every category this
project has tested) measured `qwen2.5:7b-instruct` at 29-144 seconds per turn - **12/12 exceeded 5
seconds**, none came close. This confirmed what the Q3_K_M quantization investigation earlier in
this file already found in miniature: no quantization of the 7B model fits fully in this GPU's 4GB
VRAM, so no amount of config tuning gets a 7B-class model under a 5-second bar on this hardware -
it's a structural wall, not a setting. `qwen2.5:3b` already clears that bar on its own (~4.2s warm,
measured earlier this session) - its real weakness has always been tool-selection accuracy, not
speed, which is exactly what fine-tuning is suited to fixing. So: retargeted the fine-tuning
infrastructure above from 7B to 3B rather than continuing to chase an unreachable speed target on
7B. Concretely:

- `scripts/validate_finetune_dataset.py` and `scripts/finetune_on_colab.py` now point at
  `Qwen/Qwen2.5-3B-Instruct` / `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` (confirmed this repo exists on
  Hugging Face before hardcoding it - not assumed). The dataset itself (98 examples) is unchanged -
  it's built from the real tool registry and targets real tool-selection confusions independent of
  model size, arguably more relevant for 3B, which needs more help on exactly these mix-ups than 7B.
- Re-running the validation against the real Qwen2.5-3B-Instruct tokenizer (not estimated) surfaced
  a real, separate finding: the max token count across the dataset had grown from the
  previously-measured 7192 to **8541** - this session's own bug-fixing work added new tools and
  lengthened many tool descriptions (the "name the exact troublesome phrase" fixes throughout this
  file), which directly grows the `<tools>` schema block every real prompt carries. `MAX_SEQ_LENGTH`
  was 8192 in both scripts - already *below* the new real max, which would have silently truncated
  training targets. Raised to 10240 (real headroom above 8541, not just matched to it) in both files
  and re-verified clean: `[OK] Every example fits, with 1699 tokens of headroom on the longest one.`
- **Checked this wasn't also a live production bug**: fine-tuning goes through the HuggingFace/
  Unsloth pipeline (Jinja2 chat template, `AutoTokenizer`), which is a *different* tokenizer/
  templating path than Ollama's own internal one - confirmed live that they don't report the same
  token count for equivalent content. A real, cache-busted call through the actual
  `assistant.llm.ollama_client` path (real system prompt, real 37-tool schema, `num_ctx=8192`)
  measured `prompt_eval_count: 4098` - comfortably under budget. So the 8541-token finding is real
  and matters for training specifically, but does not mean live Vaayu usage is silently losing
  context; verified rather than assumed either way.
- Still not run end-to-end - same honest caveat as before, now against the retargeted base model:
  no GPU here capable of actually training even 3B (checked, not assumed), so the real training run
  and whether it measurably improves 3B's tool-selection accuracy over the base model remain
  unverified until someone runs it on Colab.

## Real-usage bug hunt (real browser profile, muted, text-mode) and the fixes from it

Ran Vaayu through ~35 real commands over ~40 minutes against a real (not isolated-test-profile)
Brave profile, in text mode with system audio muted throughout - "use it like a normal person would
for a day": system status, browsing, YouTube movies/anime/podcasts + controls, multi-video
disambiguation, a real (read-only) WhatsApp Web check, Notepad, Armoury Crate, Hinglish phrasing,
and deliberately bad/edge-case requests. Found 7 high-priority and 7 secondary issues; the machine
was restored to exactly how it was found afterward. Fixes below, each verified live against the
real code (not just asserted) after being made.

- ⚠️ **Fixed: `open_website`'s new-tab path could crash the whole browser, or silently lose the
  tab.** The old approach for "open in a new tab" (`browser_state.open_url_in_last_browser`) did a
  plain `subprocess.Popen([exe, url])` - a *second* browser process pointed at the same profile a
  CDP-debugged instance already has open. That's Vaayu's normal operating condition, not an edge
  case, and it broke two different ways live: once it crashed Brave outright (only the crash-handler
  processes were left running afterward), and separately it produced a "tab" that Vaayu reported as
  opened but which never actually existed (confirmed via `GET /json` immediately after - only the
  original tab was there). Fixed by having `open_website`'s new-tab path go through the DevTools
  HTTP endpoint itself first (`PUT /json/new?<url>`, added as `browser._new_tab`/`_activate_tab`) -
  genuinely creates a new tab in the *same* already-running instance, no second process - and only
  falling back to the old Popen-based approach when no CDP-debugged browser is reachable at all.
  Verified live: reproduced the exact original crash scenario (two consecutive new-tab opens) after
  the fix with no crash and both tabs genuinely present.
- Along the way: found `_CDP_BASE` used `http://localhost:9222`, and "localhost" resolving to both
  ::1 and 127.0.0.1 (Windows tries IPv6 first) was adding a real ~2s connect-refused delay on top of
  the IPv4 one for every CDP call made when no debugged browser is running - measured 4.06s vs 2.04s
  for the identical request, IPv4-only vs "localhost". Switched `_CDP_BASE` to `127.0.0.1` directly -
  halves that cost project-wide, not just for the new-tab path.
- ⚠️ **Fixed: a genuinely empty model reply used to pass straight through as `""`.** `close the
  seventeenth tab` (no tab at that position) produced a literal empty string reply - confirmed the
  grounding-nudge *did* fire and retry once, and the second pass still ended with no tool call and
  empty content. Root cause in `_sanitize_reply`: it only forced the fallback message when `cleaned`
  was empty *and* the original text was non-empty (i.e. real content got stripped away) - an
  originally-empty reply skipped that check entirely. In voice mode this is dead silence with zero
  indication anything happened; in text mode, a blank line. Fixed to fall back unconditionally
  whenever nothing usable is left, regardless of why.
- ⚠️ **Fixed: a hallucinated markdown link grafted onto an otherwise fully-correct reply.** A real,
  successful `play_video_on_youtube` call (video genuinely started, verified by every subsequent
  pause/seek/volume call working against it) got a reply like `"...here's the trailer:
  [play](chrome://dino)"` - a fabricated link to Chrome's offline dinosaur game, unrelated to
  anything in the tool result. Not covered by the existing fake-tool-call guard, which only catches
  missing/malformed tool calls, not garbage appended to a legitimate one. This assistant has no tool
  that returns a URL for it to relay, so any markdown link in a reply is by construction fabricated -
  `_sanitize_reply` now strips the link markup unconditionally (keeping the link's own text, usually
  still a genuine part of the sentence).
- Recurring pattern, partially fixed: **no dedicated tool -> silent `open_website` search fallback ->
  confidently false success.** Hit three times live (Spotify web player, WhatsApp, YouTube/podcast
  "in a new tab") - each time `open_website` ran a plain Google search for the phrase, and the reply
  described the *named service* as opened rather than what actually happened. Added explicit
  guidance (system prompt) to describe what the tool result actually says happened, not the intended
  destination, whenever `open_website` is used as a last-resort fallback. Verified live: "play music
  on spotify web player" now replies "I've opened a web browser and searched for the Spotify Web
  Player. You should be able to find it from the search results..." - honest about what happened,
  not a false claim of success.
- **Multi-video disambiguation confirmed working live for the first time** (previously only
  unit-tested) - deliberately got two real videos playing in two tabs and asked the ambiguous
  "pause it"; the tool correctly raised the "which one do you mean" question naming both tabs. The
  bug found: the *follow-up* answer ("pause the dune trailer") didn't reliably populate `tab_hint`,
  silently fell back to the most-recently-active tab, paused the wrong video, and confidently
  reported the *named* one as paused. Fixed by strengthening `tab_hint`'s parameter description
  (`video_player.py`) and the system prompt to set it proactively whenever the user's wording names
  a tab/site/video, not only reactively after the disambiguation question.
- Cosmetic, found alongside the above: raw HTML entities (`&amp;amp;`) and a leading YouTube
  notification-badge count (`"(576) "`) from `document.title` leaked verbatim into the
  disambiguation question read back to the user. Added `_clean_title()` (decodes entities, strips
  the badge prefix) used everywhere `video_player.py` surfaces a tab title.
- `list_connected_devices` had a required `category` argument, and the *same* phrase ("what devices
  are connected") got a missing-argument failure on one run and a correctly-filled category on
  another - consistent with this project's already-documented finding that this model's tool-arg
  completion isn't fully deterministic run to run. Made `category` optional; omitting it now queries
  every category and tags each result, instead of depending on the model to reliably supply one.
- "mute the volume" (no video/tab named) was routing to `set_video_volume` (the in-page player
  tool, which needs a reachable video/tab) instead of `control_media` (the plain system-wide mute
  key, which works regardless of what's open). Fixed via the system prompt: a bare mute/volume
  request with nothing playing/named defaults to `control_media`; only routes to `set_video_volume`
  when the wording actually references the video/player/a specific site. Verified live.
- "battery kitni hai" (Hinglish) answered with `get_audio_status`'s speaker volume (62%) mistaken
  for the battery percentage - both are percentages, but completely different numbers. Fixed by
  adding explicit cross-references in *both* tools' descriptions (`system_info.py`/`audio_status.py`)
  naming the Hinglish phrase directly - a tool-description fix, not just a system-prompt one, since
  descriptions sit right next to the schema during selection and proved more effective live: the
  same fix attempted only in the system prompt did not change the model's behavior on a live rerun,
  but adding it to the tool descriptions did (verified both "battery kitni hai" and "kitna charge
  hai" now correctly call `get_system_status`).
- "brave band kar do" ("close brave" in Hinglish) called `open_app`, the opposite of what was asked
  - stated with full confidence ("Brave is now open"). Added the same phrase-anchored fix to
  `open_app`/`close_app`'s descriptions. Live retest: grammatically complete phrasings ("brave ko
  band karo", "band kar do brave ko") now reliably resolve to `close_app`, but the exact terse
  original phrasing ("brave band kar do", no object marker) is still inconsistent - a real,
  substantially-improved-but-not-fully-closed gap, most likely this model's inherent Hindi/Hinglish
  parsing limits at this quantization level rather than something further prompt tuning alone fixes.
- A compound request ("open armoury crate and change my performance mode setting") did neither part
  and replied with an unrelated non-sequitur, despite the achievable half (opening the app) working
  fine on its own. Also confirmed a genuine capability gap, not just a model miss: there is no tool
  for clicking inside a native desktop app's UI (only `open_app`/`close_app`/`press_hotkey`/
  `take_screenshot` exist at the desktop level), and Armoury Crate throws a UAC elevation prompt on
  open - the same kind of isolated-session wall already documented for the lock screen, a hard
  constraint a future UI-automation tool would hit too, not a missing-feature gap alone. Added
  system-prompt guidance to do the achievable part of a compound request and say plainly that the
  rest isn't covered, rather than doing neither.
- `close armoury crate` failed ("might not be referring to something on your computer") right after
  the model had itself just successfully opened something by that exact name two turns earlier in
  the same conversation; `close the armoury crate app` worked immediately. Added guidance to treat a
  later "close <name>" as the same target already confirmed real by having just opened it.
- All fixes above are covered by unit tests (`tests/test_web_tools.py`,
  `tests/test_devices_and_network_tools.py`, existing `ollama_client`/`video_player` suites) and the
  full suite (5000 passed, 239 skipped) was re-run clean after every change in this pass. The new-tab
  CDP fix and the two Hinglish/battery description fixes were additionally verified live against the
  real running model and (for the browser fix) a real Chrome DevTools Protocol session, not just
  asserted from the unit tests alone.

### Second fix pass: pushed further until genuinely diminishing returns

Kept going past the first pass - re-tested every live-model-dependent fix repeatedly (not just
once) and chased down what was still soft, until further attempts stopped producing real
improvement. Everything below was verified live against the real running model, several of them
multiple times, not just asserted.

- ⚠️ **`set_video_fullscreen`'s one-off failure, root-caused as a timing race, not a structural
  bug.** Reproduced a real video playing and toggled fullscreen 3/3 times successfully - couldn't
  reproduce the original failure directly, consistent with a transition/overlay/buffering race at
  the single fixed 0.3s check rather than something broken. Hardened anyway with one retry after a
  longer wait before treating it as a real failure, same bounded-retry shape already used for the
  cold-start DOM-query race - cheap, safe, and covers the class of issue even without a pinned-down
  single cause.
- ⚠️ **Closed the YouTube "new tab" capability gap for real**, instead of just being honest about
  not having it (the first pass's fix). `play_video_on_youtube` and `play_latest_video_from_channel`
  now take a real `new_tab` parameter - resolves a target tab once (via the same CDP `_new_tab`
  used for the browser-crash fix, not a second process) and threads it through every navigation in
  that call, so a channel lookup's multi-step search -> videos page -> video sequence all lands on
  the *same* new tab instead of drifting. Verified live: two videos requested back to back (one
  default, one `new_tab=true`) produced exactly the right tab count and content each time, including
  the multi-step channel-lookup path.
- Extended the already-documented cold-start DOM-query race: bumped `_eval_js_retry_if_empty`'s
  default retries 1 -> 2 after finding that a single retry (one extra 2s wait) still wasn't enough
  specifically on a genuinely fresh, never-before-used browser profile's very first page load.
  Verified against the *realistic* scenario that actually matters (a cold restart of an
  already-established profile, not a brand-new one - the real user's profile has been used for
  months) - succeeded in 3.6s post-fix.
- Found and fixed along the way: `_CDP_BASE` used `http://localhost:9222`, and "localhost"
  resolving to both `::1` and `127.0.0.1` (Windows tries IPv6 first) was adding a real ~2s
  connect-refused delay on top of the IPv4 one for every CDP call made when no debugged browser is
  reachable - measured 4.06s vs 2.04s for the identical request, IPv4-only vs "localhost". Switched
  to `127.0.0.1` directly - this alone took the full touched-module test run from ~25s to ~13s, and
  a further systemic fix (an autouse fixture in `conftest.py` mocking the new CDP-new-tab call by
  default, same reasoning as the existing self-review-isolation fixture) took the full suite back to
  its normal ~20s.
- ⚠️ **The exact original failing phrase ("brave band kar do") is now fully fixed, not just
  "substantially improved" as the first pass left it.** The first pass's generic Hinglish
  verb-mapping fix worked for grammatically complete phrasing ("brave ko band karo") but not the
  terser original wording. Found the actual lever: the earlier successful fixes (battery/audio,
  generic verb mapping) worked specifically when the *exact* troublesome phrase was named directly
  in the tool description, not just described as a pattern - applying that same specificity here
  (an explicit "app name before OR after the verb" example using the literal phrase) resolved it
  completely: 4/4 across repeated identical live attempts, including the exact original wording.
- ⚠️ **Found and fixed a new, more general bug while chasing the above**: "open armoury crate"
  (bare, no "the"/"app"/"launch") was unreliable in a genuinely concerning way - sometimes called no
  tool at all and speculated about video games ("could you tell me which game?"), and at least once
  called `open_app` successfully and then *contradicted the successful result* in its reply ("it
  seems that 'armoury crate' is not an application installed on your computer"). This is the mirror
  image of the original "confirmed but didn't happen" bug this project already fixed once - a false
  *failure* claim after real success, not a false success claim - and wasn't covered by any existing
  guard. Fixed two ways: (1) an explicit system-prompt rule that a successful tool result is ground
  truth and must never be contradicted by a guess about whether a name "sounds real"; (2) naming
  "armoury crate" directly and explicitly in `open_app`/`close_app`'s own descriptions as a real
  ASUS utility, not a game feature - the same "name the exact phrase" lever that fixed the Hinglish
  case. Verified live: 5/5 open, 2/2 close, all correct and none contradicting a successful result.
- **A genuine regression, found and reverted the same session.** Tried a further fix for a
  compound-request case fabricating a fake keyboard shortcut ("press P") for a setting Vaayu can't
  actually change - the fix (an explicit "don't invent specific instructions" system-prompt
  addition) measurably made things *worse*: 0/3 live attempts could even do the achievable half
  anymore (opening the app), cycling through unrelated tools instead across up to 5 chained calls
  before giving up. Reverted it immediately and re-verified the reverted state 3/3 clean (opens the
  app correctly, doesn't fabricate anything for the rest either, in fresh live samples) - direct,
  measured evidence that the system prompt is long enough now that further additions carry a real
  destabilization risk elsewhere, not just a chance of not helping. Documented here rather than
  quietly dropped, since "a fix that didn't work" is exactly the kind of finding worth keeping
  visible.
- Also chased what looked like a related bug (replies inventing a wrong app name, "I've opened
  Spotify" after actually calling `open_app` with `armoury crate`) - confirmed this was an artifact
  of an overly-generic mock result string in the *test harness*, not a real product bug: retested
  with a mock matching the real tool's actual return shape (which always includes the real app name)
  and got the correct name back 3/3. Worth recording as a testing-methodology lesson as much as a
  product one - a vague mock can manufacture a bug that doesn't exist against the real tool.
- **Final consolidated regression check**, all five previously-fixed live-model-dependent behaviors
  re-verified together in the code's final state (battery/audio, Hinglish close, mute routing,
  armoury crate open, armoury crate close): 5/5 clean. Full test suite re-run clean throughout
  (5004 passed, 239 skipped, ~20s).
- **Reached genuine diminishing returns and stopped here, deliberately** - not because the list of
  theoretically-improvable behavior ran out, but because further prompt-engineering attempts on this
  specific local model started trading a rare failure mode for a worse, more frequent one (see the
  reverted regression above), and re-testing showed real remaining variance is more consistent with
  this model's inherent non-determinism (already documented earlier in this file) than with a fixable
  root cause. What's left unfixed - `set_video_fullscreen`'s no-longer-reproducible one-off, and
  the hard, non-negotiable walls (no native-desktop-UI-control tool, Armoury Crate's own UAC prompt)
  - are either already hardened against recurrence or are genuine constraints of "current conditions"
  (this hardware, this locally-hosted 7B model), not gaps a few more description tweaks would close.

### Third round: new territory (notes, files, tab management, VLC) - and a real library bug

A second real-usage bug hunt, deliberately targeting tools/scenarios the first two rounds never
touched: notes, files, ordinal browser-tab references, VLC desktop control, dictation, self-review
controls. Found 7 real bugs, including one with a much bigger blast radius than it first looked -
not just a "wrong tool" issue, but a genuine bug in the third-party `AppOpener` library that meant
`open_app`/`close_app` could **never** detect a real failure, at all, in this project's
configuration. Every fix below was verified live against the real system (real processes, real
files, real database rows), not just asserted.

- ⚠️⚠️ **The big one: `open_app`/`close_app` could never actually detect failure.** Reading
  AppOpener's own source (`.venv/Lib/site-packages/AppOpener/features.py`) found that its
  `raise AppNotFound(...)` line, in every code path including the one this project always uses
  (`match_closest=True`), is nested inside `if output: if throw_error: raise ...` - meaning
  `throw_error=True` only works when `output` is *also* truthy. This project's `apps.py` was built
  passing `output=False` (deliberately, to suppress console spam) - which silently disabled
  `throw_error` completely, with no exception, ever, regardless of what actually happened. Confirmed
  directly: `app_close("definitely_not_a_real_app", output=False, throw_error=True)` raised nothing
  at all. This explains two real-usage findings in one shot: "close notepad" claiming success while
  Notepad kept running, and "open vlc" claiming success while no `vlc.exe` process ever existed -
  both are the exact same root cause, not two unrelated bugs. Fixed by passing `output=True` (so
  the library's own detection actually runs) while capturing/discarding its prints via
  `contextlib.redirect_stdout`, keeping the same quiet console behavior `output=False` was meant to
  provide without disabling error detection to get it. Verified live: closing/opening a nonexistent
  app now genuinely raises `AppNotFound` every time.
- ⚠️ **`close_app` now also verifies independently**, on top of the library fix above - real process
  termination isn't always instant, so trusting "no exception" alone still wasn't quite enough.
  Added `_find_matching_process_name()` (exact, case/space/`.exe`-insensitive match against real
  running processes, with a short retry) and raise a clear error if the process is still there after
  it. Deliberately NOT the same fuzzy `difflib` matching AppOpener itself uses for guessing app
  names - found live that fuzzy matching here produces false positives ("brave" scored high enough
  against the unrelated, always-present `BraveCrashHandler.exe`) - this check needs precision, a
  best-effort guess is the wrong tool for it.
- Separately: AppOpener's own fuzzy match against its registered-app database can't find "vlc" for
  "VLC Media Player" (too short for its 0.6 similarity cutoff) - confirmed live (bare "vlc" raises
  `AppNotFound`; "vlc media player" opens correctly). Added a tiny alias table, **`open_app` only**
  - found live that applying the same alias to `close_app` actively broke it, since `close_app`
  matches against real *running process names* (where bare "vlc" already matches "vlc.exe"
  perfectly on its own), a completely different matching source than `open_app` uses.
- "What are my recent notes" called `add_note` again instead of `list_notes`, with the two prior
  notes' text concatenated as fabricated new content - a read request that both never actually read
  anything and had the side effect of writing a bogus row. Fixed via clearer, more explicit
  `add_note`/`list_notes` descriptions (mirroring the "name the exact confusion" pattern that's
  worked best throughout this project) - each states plainly what it is not for, with example
  phrasings for the other. Cleaned up the 3 stray test rows this left in the real notes database.
- Found a deeper, related issue while re-verifying the note-taking fix live: a single "note that X"
  request could trigger *several* separate `add_note` calls in one turn, each a slightly different
  paraphrase of the same note, before ever producing a real reply - confirmed via real database
  timestamps (4-5 near-duplicate rows per single user turn). Added a general "stop once a tool call
  already satisfies the request, don't call it again with reworded arguments" rule. Verified this is
  a genuine, non-regressing improvement (the simple first-turn case now saves exactly one row with
  matching content, live-tested twice) but **not fully resolved** - the same repeated-calling pattern
  still recurs as a conversation grows longer, consistent with this being context/attention
  degradation in this model rather than something more prompt text reliably fixes. Documented
  honestly as a known, partially-improved limitation rather than claimed as solved.
- `read_file` given a bare filename (no directory) resolved it against the Python process's own
  working directory, not any safe root - so a file genuinely sitting in `~/Documents` was reported
  "not found" even though `search_files` would have found it. Fixed: try each safe root for a bare
  filename before giving up, and raise a direct `FileNotFoundError` (not a misleading "outside the
  safe folders" `PermissionError`) if truly not found anywhere.
- "Search my files for X" called `read_file` with a hallucinated filename/path instead of calling
  `search_files` at all - fixed via clearer descriptions on both tools (search_files: "call this
  BEFORE read_file... never guess/invent a path"; read_file: "if you don't already have one
  confirmed, call search_files first").
- "What tabs do I have open" called `list_open_windows` (OS-level desktop windows) instead of
  `list_browser_tabs`, returning a list of unrelated window titles that didn't answer the question
  at all. Fixed by making `list_open_windows`'s description explicitly redirect any "tab(s)" wording
  to `list_browser_tabs`.
- **Investigated but not further chased**: "close the third browser tab" produced inconsistent,
  sometimes-empty tool selection immediately after the symmetric "switch to the second browser tab"
  worked perfectly - re-tested live 3 more times and got 3 different wrong-but-safe outcomes (an
  unrelated tool, a wrong parameter name, a confused clarifying question), never a silent wrong
  action or false success. Traced the "wrong parameter name" case (`close_browser_tab(index=2)`
  instead of `position=2`) to the registry's existing argument-filtering safety net, which already
  handles it correctly (drops the unrecognized key, the tool then raises its own clear "need a
  query or position" error rather than crashing or guessing) - confirmed this is genuine model
  inconsistency that the existing safety nets already contain safely, not a new gap to close, and
  consistent with this session's established finding that further prompt tuning risks net-negative
  regressions more than it reliably fixes narrow phrasing variance like this.
- "Pause the VLC video" was routing to the browser-tab video-control tool instead of following the
  system prompt's own explicit VLC pattern (switch_to_window + press_hotkey) - likely because the
  word "video" in the request pattern-matches the browser tool's own description strongly regardless
  of "VLC" being named. Added an explicit exclusion directly to `control_video_playback`'s
  description naming VLC and this exact phrasing.
- One background test batch was killed mid-run by the harness's own low-memory protection (twice,
  once during the bug-hunt fork and once during this fix-verification pass) - noted as an
  environment condition, not a Vaayu issue, and not retried per the harness's own guidance.
- Full test suite re-run clean after every change in this round (5009 passed, 239 skipped, ~20s).
  Two pre-existing tests (`test_apps_tools.py`, plus a *duplicate* parametrized battery covering the
  same `open_app`/`close_app` calls in `test_extended_suite_large.py` that was missed on the first
  pass and caught by a full-suite run) needed updating for the `output=True` change - both fixed and
  re-verified.

## Roadmap

- **M1**: text loop + one tool (`get_system_status`)
- **M2**: full tool set above
- **M3** (current): real voice I/O, hands-free continuous conversation, Sleeping/Awake state machine
  ("wake up Vaayu" / "sleep Vaayu" / "shutdown Vaayu"), auto-start at login, real browser page
  interaction (navigate/read/click), multi-step tool calling, colorful/draggable status orb
- **M4**: speaker verification (voice lock) — **postponed** until continuous listening is solid
- **M5**: a true lightweight wake word (would replace the "always transcribing, even asleep" design
  above with a purpose-built low-power model) — would also fix the "responds to any audio" limitation
  noted above
- **M6**: polish (higher-quality TTS, logging)
- **Not planned yet**: continuous self-improvement/accent adaptation over time (would need collecting
  speech + corrections and periodically retraining/fine-tuning - a genuinely separate project, scope
  it out explicitly if you want to pursue it)
- **Done since the last update**: WhatsApp message sending (confirmation-gated, see above - built
  and unit-tested, but not yet verified against a real logged-in session), the two latency fixes
  above (verified for real against the local Ollama instance), `close_browser_tab` +
  ordinal/positional tab targeting, `same_tab` navigation, in-page YouTube volume/fullscreen
  control, `minimize_window`/`restore_window`, and the grounding-retry fix for false "action
  confirmed" replies.
- **Still not attempted**: the wider "every feature of every app" ambition and full autonomous
  self-control of the machine - deliberately not pursued; see the note this was added alongside for
  why (unsupervised self-modifying code and unprompted autonomous action on a real machine are
  outside what this project takes on, regardless of how far the rest of it grows).
- **Phase 2** (later): Android companion app
