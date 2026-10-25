"""
Autouse fixture redirecting every self_review data path to an isolated
location, for every test in the suite - not just the ones that know they
touch issue logging.

Found the hard way: adding real file-write side effects inside
ollama_client.py's grounding-nudge branch (assistant/self_review/issue_log.py)
made pre-existing, otherwise-pure tests in test_ollama_client.py (written
before that logging existed, exercising the exact same code path with no
reason to expect a disk write) silently append real entries to the actual
data/issues_log.jsonl in the repo - confirmed by finding that file polluted
with test data after a normal test run. A single global fixture, applied to
every test automatically, is the systemic fix - it's not enough to remember
to mock this per-test, since any future test exercising this code path has
the same exposure by default.

Uses a session-scoped base directory (one real mkdir for the whole run) and
pure path arithmetic per test (no I/O) rather than the per-test `tmp_path`
fixture - found that using `tmp_path` here, applied automatically to all
~5000+ tests, forced a real directory creation for every single one
regardless of whether it touches self_review at all, and roughly tripled
the whole suite's runtime (~20s -> ~55s, confirmed by measuring both). The
actual writes this isolates only happen in the small minority of tests that
call issue_log functions directly - issue_log itself creates its parent
directory lazily on first write, so no eager mkdir is needed here either.
"""

import hashlib

import pytest

from assistant.self_review import issue_log
from assistant.tools import web


@pytest.fixture(scope="session")
def _self_review_base_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("self_review_isolation")


@pytest.fixture(autouse=True)
def _isolate_self_review_data(_self_review_base_dir, monkeypatch, request):
    # A hash of the full node id (not request.node.name) keeps this
    # filesystem-safe - raw parametrize IDs can contain characters like '/'
    # that aren't valid in a path segment.
    safe_name = hashlib.md5(request.node.nodeid.encode()).hexdigest()
    test_dir = _self_review_base_dir / safe_name
    monkeypatch.setattr(issue_log, "LOG_PATH", test_dir / "issues_log.jsonl")
    monkeypatch.setattr(issue_log, "REPORTS_DIR", test_dir / "daily_reports")
    monkeypatch.setattr(issue_log, "LAST_CHECKIN_PATH", test_dir / "last_checkin_date.txt")


@pytest.fixture(autouse=True)
def _no_real_cdp_calls(monkeypatch):
    # open_website's new-tab path tries a real CDP call (browser._new_tab)
    # against 127.0.0.1:9222 before falling back to the older
    # browser_state-based approach - the right behavior in production (see
    # web.py), but in a normal test run nothing is actually listening on
    # that port, so every un-mocked call was paying a real ~2s
    # connection-refused delay. Same systemic-fixture reasoning as
    # _isolate_self_review_data above: default every test to the "no
    # CDP-debugged browser reachable" outcome instantly, and let any test
    # that specifically wants to exercise the CDP path override it with its
    # own patch.object(web, "_new_tab", ...), same as it already does today.
    def _no_cdp_browser(url):
        raise ConnectionError("no debugged browser (test default)")

    monkeypatch.setattr(web, "_new_tab", _no_cdp_browser)


@pytest.fixture(autouse=True, scope="session")
def _no_real_radio_queries():
    # get_network_status's Bluetooth check asks the real Windows radio via
    # PowerShell (~1.5 s each) - older tests that only mock _run would hit
    # real hardware. Default to "On" once per session (a per-test fixture
    # costs more than it saves, see above); radio tests patch it themselves.
    from assistant.tools import network_status
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(network_status, "_radio_ps", lambda kind, state: "On")
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_real_browser_launches():
    # open_app launches Chromium browsers directly (with the DevTools port)
    # via subprocess.Popen - found the hard way: an older open_app test
    # started the user's real Brave. Nothing in the suite may launch a real
    # program; browser-launch tests assert on this stub instead.
    from unittest.mock import MagicMock
    from assistant.tools import apps
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(apps, "_launch_with_port", MagicMock())
        mp.setattr(apps, "_cdp_up", lambda: False)
        yield


@pytest.fixture(autouse=True, scope="session")
def _isolated_interaction_log(tmp_path_factory):
    # Same reason as the self_review isolation above: tests must never write
    # into the real data/interactions.jsonl that real sessions are logged to.
    from assistant import interaction_log
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(interaction_log, "LOG_PATH", tmp_path_factory.mktemp("interactions") / "interactions.jsonl")
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_real_volume_or_youtube_changes():
    # control_media now sets the REAL Windows volume (read back via the audio
    # endpoint) and acts on a real YouTube tab if one is open - no test may do
    # either. Media tests patch these per test with fakes.
    from unittest.mock import MagicMock
    from assistant.tools import media
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(media, "_youtube_video", lambda: None)
        mp.setattr(media, "_youtube_in_foreground", lambda: False)
        mp.setattr(media, "_endpoint", lambda: MagicMock(GetMasterVolumeLevelScalar=lambda: 0.5, GetMute=lambda: 0))
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_real_audio_meter():
    # The orchestrator reads the real Windows output meter to tell whether
    # media is playing - a test's outcome must not depend on whether the
    # laptop happens to be playing sound. Media-gate tests patch it per test.
    from assistant import orchestrator
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator, "_media_playing", lambda: False)
        yield


@pytest.fixture(autouse=True, scope="session")
def _isolated_known_names(tmp_path_factory):
    # remember_name writes the user's real data/known_names.json - tests may not.
    from assistant.tools import names
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(names, "PATH", tmp_path_factory.mktemp("names") / "known_names.json")
        yield


@pytest.fixture(autouse=True, scope="session")
def _isolated_memory(tmp_path_factory):
    # Vaayu's memory of the user (conversations, learned facts) and the notes
    # DB the chat path reads are the user's real data - tests may not touch
    # them, and learning must never make a real LLM call from a test.
    from assistant import memory
    from assistant.storage import db
    base = tmp_path_factory.mktemp("memory")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(memory, "CONVERSATIONS_PATH", base / "conversations.jsonl")
        mp.setattr(memory, "FACTS_PATH", base / "memory_facts.json")
        mp.setattr(db, "_DB_PATH", base / "assistant.db")
        mp.setattr(memory, "_real_learn_async", memory.learn_async, raising=False)
        mp.setattr(memory, "learn_async", lambda *a, **k: None)
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_real_audio_saving():
    # handle_one_turn saves each utterance to the real data/audio/ - tests may not.
    from assistant import orchestrator
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator, "_real_save_audio", orchestrator._save_audio, raising=False)
        mp.setattr(orchestrator, "_save_audio", lambda audio: None)
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_real_internet_lookups():
    # Questions now route to web_answer (weather, "who is the PM...") - the
    # suite must never depend on the network. Web tests patch these per test.
    from assistant.tools import web_answer

    def offline(*args, **kwargs):
        raise ConnectionError(web_answer._OFFLINE)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(web_answer, "_get_json", offline)
        mp.setattr(web_answer, "_ddg_text", offline)
        yield


@pytest.fixture(autouse=True, scope="session")
def _isolated_watch_history(tmp_path_factory):
    # The user's YouTube watch history (their taste) - tests may not read or write it,
    # and the background logger must never poll a browser from a test.
    from assistant.tools import youtube_taste
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(youtube_taste, "HISTORY_PATH", tmp_path_factory.mktemp("taste") / "watch_history.jsonl")
        mp.setattr(youtube_taste, "start_watch_logger", lambda: None)
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_real_start_menu_launch():
    # open_app's fallback reads Windows' Start menu (PowerShell) and LAUNCHES the match
    # (os.startfile / explorer) - tests may never start a real app.
    from assistant.tools import apps
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(apps, "_real_open_from_start_menu", apps._open_from_start_menu, raising=False)
        mp.setattr(apps, "_open_from_start_menu", lambda name: None)
        yield


@pytest.fixture(autouse=True, scope="session")
def _isolated_daily_review(tmp_path_factory):
    # The daily review DELETES the real logs and runs a real `claude` session - tests get
    # temp paths, no Claude, and no model calls (originals kept as _real_* for its own tests).
    from assistant import daily_review
    base = tmp_path_factory.mktemp("review")
    with pytest.MonkeyPatch.context() as mp:
        for name, rel in (("INTERACTIONS", "interactions.jsonl"), ("ISSUES", "issues_log.jsonl"),
                          ("VAAYU_LOG", "vaayu.log"), ("AUDIO_DIR", "audio"), ("REVIEWS", "reviews")):
            mp.setattr(daily_review, name, base / rel)
        mp.setattr(daily_review, "_real_claude_session", daily_review.claude_session, raising=False)
        mp.setattr(daily_review, "claude_session", lambda review_dir, day: "Claude session stubbed in tests.")
        mp.setattr(daily_review, "_real_learn_locally", daily_review.learn_locally, raising=False)
        mp.setattr(daily_review, "learn_locally", lambda events, review_dir, mentioned: [])
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_real_kokoro():
    # Loading Kokoro takes the GPU (~1.1 GB) and a few seconds - tests use the plain voice engine.
    from assistant import orchestrator
    from assistant.tts.pyttsx3_engine import Pyttsx3Engine
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator, "_real_make_tts", orchestrator._make_tts, raising=False)
        mp.setattr(orchestrator, "_make_tts", lambda cfg: Pyttsx3Engine(rate=cfg.rate))
        yield


def fake_embed(texts):
    """Bag-of-words hashing, 384 dims like the real model: same words -> same vector, no download."""
    import re
    import zlib
    import numpy as np
    out = np.zeros((len(texts), 384), np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            out[i, zlib.crc32(word.encode()) % 384] += 1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture(autouse=True, scope="session")
def _isolated_knowledge(tmp_path_factory):
    # Vaayu's knowledge (facts about the user, research notes, answer cache) is real user data,
    # and the real embedding model is a 130 MB download - tests use a temp dir and a fake embedder,
    # and the idle-time research thread never runs.
    from assistant import knowledge, research
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(knowledge, "DIR", tmp_path_factory.mktemp("knowledge"))
        mp.setattr(knowledge, "_embed", fake_embed)
        mp.setattr(research, "_real_start", research.start, raising=False)
        mp.setattr(research, "start", lambda config, client: None)
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_real_browser_takeover():
    # ensure_browser starts, or CLOSES AND REOPENS, the user's real Brave. Found the hard way: without
    # this stub one test run launched Brave on the user's desktop.
    from assistant.tools import apps
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(apps, "_real_ensure_browser", apps.ensure_browser, raising=False)
        mp.setattr(apps, "ensure_browser", lambda app_name="brave": "")
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_waiting_for_a_real_player():
    # play_video_on_youtube waits (up to 10 s) for YouTube's player to really be playing - tests have none.
    from assistant.tools import youtube
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(youtube, "_wait_until_playing", lambda video_url, tab, timeout=10.0: None)
        yield


@pytest.fixture(autouse=True, scope="session")
def _isolated_weather_place(tmp_path_factory):
    from assistant.tools import web_answer
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(web_answer, "PLACE_FILE", tmp_path_factory.mktemp("weather") / "weather_place.txt")
        yield


@pytest.fixture(autouse=True)
def _forget_weather_place():
    # The remembered city must not leak from one test into the next (cheap: one unlink per test).
    from assistant.tools import web_answer
    web_answer.PLACE_FILE.unlink(missing_ok=True)


@pytest.fixture(autouse=True, scope="session")
def _isolated_alarms(tmp_path_factory):
    # The user's real alarms - and the scheduler must never ring from a test.
    from assistant import alarms
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(alarms, "PATH", tmp_path_factory.mktemp("alarms") / "alarms.json")
        mp.setattr(alarms, "_real_start_scheduler", alarms.start_scheduler, raising=False)
        mp.setattr(alarms, "start_scheduler", lambda speak, set_volume=None: None)
        yield
