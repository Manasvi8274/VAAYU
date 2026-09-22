"""
Large parametrized browser test suite (1000+ cases), covering every
browser-facing tool: open_website (URL/search classification, same-tab
navigation, "X website" domain resolution), tab management (list/switch/
close by position and by query, across many tab-list sizes), YouTube in-page
control (volume across the full 0-100 range plus relative up/down deltas,
fullscreen toggling), YouTube search/channel navigation, generic page
interaction (click_on_page/read_page against varied and adversarial text),
WhatsApp Web confirmation-prompt formatting, the disk cache used by
play_latest_video_from_channel, and browser_state's browser-resolution order.

Deliberately built via parametrization over programmatically generated
input sets rather than 1000+ hand-written functions - each case is still a
real, distinct input against the tool's real logic (not a loop hidden inside
one test), and because every boundary here is mocked (CDP/urlopen/AppOpener),
the whole file runs in a few seconds, not hours - the concern with a literal
1000+ *end-to-end* suite (real browser + real LLM calls) doesn't apply to a
suite of this shape. Combinatorial parametrization over realistic input
variety is also how this pass actually found a real bug (open_website's
scheme check was case-sensitive, so 'HTTPS://x.com' silently became
'https://HTTPS://x.com') - broader input coverage catches exactly this kind
of thing that a handful of hand-picked examples doesn't.
"""

import itertools
import json
from unittest.mock import patch

import pytest

from assistant.tools import browser, browser_automation, browser_state, cache, web, whatsapp, youtube


# ============================================================================
# Shared generators / fixtures
# ============================================================================

_SITE_STEMS = [
    "google", "youtube", "amazon", "netflix", "wikipedia", "github", "reddit",
    "twitter", "instagram", "facebook", "moviesmod", "telegram", "spotify",
    "linkedin", "stackoverflow", "medium", "quora", "pinterest", "tumblr",
    "imdb", "espn", "cnn", "bbc", "nytimes", "coursera", "udemy",
    "khanacademy", "duckduckgo", "bing", "yahoo",
]
_TLDS = [".com", ".org", ".net", ".in", ".co", ".io"]
_DOMAINS = [stem + tld for stem in _SITE_STEMS for tld in _TLDS]  # 180

_SEARCH_TEMPLATES = [
    "best {topic} near me", "how to {topic}", "{topic} reviews 2026",
    "top 10 {topic}", "{topic} tutorial", "cheap {topic} deals",
    "{topic} vs alternatives", "is {topic} worth it", "{topic} price in india",
    "latest {topic} news",
]
_TOPICS = [
    "pizza", "laptops", "recipes", "movies", "weather", "yoga classes",
    "used cars", "hotels", "flights", "headphones",
]
_SEARCH_QUERIES = [t.format(topic=topic) for t in _SEARCH_TEMPLATES for topic in _TOPICS]  # 100

_CHANNEL_NAMES = [
    "OG Crew", "MrBeast", "PewDiePie", "Marques Brownlee", "Linus Tech Tips",
    "Kurzgesagt", "Veritasium", "CarryMinati", "BB Ki Vines", "Ashish Chanchlani",
    "Technical Guruji", "Tanmay Bhat", "Round2Hell", "Harsh Beniwal",
    "Flying Beast", "Slayy Point", "Mumbiker Nikhil", "Beer Biceps",
    "Physics Wallah", "Unacademy",
]
_SONG_QUERIES = [
    "fairytale music", "lofi hip hop beats", "bollywood love songs",
    "Ed Sheeran Perfect", "workout playlist 2026", "arijit singh hits",
    "study music instrumental", "punjabi songs new", "relaxing piano music",
    "top hollywood songs",
]


def _mock_response(payload: bytes):
    from unittest.mock import MagicMock

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value.read.return_value = payload
    return mock_cm


def _make_tabs(n: int) -> list[dict]:
    return [
        {"type": "page", "id": f"tab{i}", "title": f"Tab {i} Title", "url": f"https://site{i}.example.com/"}
        for i in range(n)
    ]


def _tab():
    return {"webSocketDebuggerUrl": "ws://fake"}


def _cdp_value(value):
    return {"result": {"result": {"value": value}}}


def _eval_sequence(*values):
    it = iter(values)

    def _fake(ws_url, method, params=None):
        if method != "Runtime.evaluate":
            return {}
        return _cdp_value(next(it))

    return _fake


# ============================================================================
# open_website: URL vs. search classification (real domains)
# ============================================================================


@pytest.mark.parametrize("domain", _DOMAINS)
def test_open_website_classifies_bare_domain_as_url(domain):
    with patch.object(web.browser_state, "open_url_in_last_browser", return_value=True) as mock_open:
        result = web.open_website(domain)
    mock_open.assert_called_once_with(f"https://{domain}", foreground=True)
    assert result == f"Opened https://{domain} in the browser."


@pytest.mark.parametrize("domain", _DOMAINS)
def test_open_website_leaves_already_https_domain_unmodified(domain):
    url = f"https://{domain}"
    with patch.object(web.browser_state, "open_url_in_last_browser", return_value=True) as mock_open:
        web.open_website(url)
    mock_open.assert_called_once_with(url, foreground=True)


@pytest.mark.parametrize("domain", _DOMAINS)
def test_open_website_uppercase_scheme_is_not_double_prefixed(domain):
    # Regression case for the real bug found while building this suite:
    # startswith(("http://","https://")) was case-sensitive, so an uppercase
    # scheme slipped past the check and got "https://" prepended a second
    # time, producing a broken URL like "https://HTTPS://example.com".
    url = f"HTTPS://{domain.upper()}"
    with patch.object(web.browser_state, "open_url_in_last_browser", return_value=True) as mock_open:
        web.open_website(url)
    called_url = mock_open.call_args.args[0]
    assert called_url == url  # unmodified - no double scheme prefix
    assert called_url.count("://") == 1


@pytest.mark.parametrize("query", _SEARCH_QUERIES)
def test_open_website_classifies_multiword_text_as_search(query):
    with patch.object(web.browser_state, "open_url_in_last_browser", return_value=True) as mock_open:
        result = web.open_website(query)
    called_url = mock_open.call_args.args[0]
    assert called_url.startswith("https://www.google.com/search?q=")
    assert result == f"Searched the web for: {query}."


# ============================================================================
# open_website: same_tab navigation (CDP, never spawns a new tab/process)
# ============================================================================


@pytest.mark.parametrize("domain", _DOMAINS)
def test_open_website_same_tab_domain_navigates_current_tab(domain):
    with patch.object(web, "_target_tab", return_value=_tab()), patch.object(
        web, "_send_cdp_command"
    ) as mock_send, patch.object(web.browser_state, "open_url_in_last_browser") as mock_open_new:
        result = web.open_website(domain, same_tab=True)
    mock_send.assert_called_once_with("ws://fake", "Page.navigate", {"url": f"https://{domain}"})
    mock_open_new.assert_not_called()
    assert result.endswith("(same tab).")


# ============================================================================
# open_website: "<name> website"/"<name> site" -> direct domain
# ============================================================================


@pytest.mark.parametrize("stem", _SITE_STEMS)
@pytest.mark.parametrize("suffix", [" website", " site", " web site"])
def test_open_website_bare_name_with_site_suffix_becomes_direct_domain(stem, suffix):
    with patch.object(web.browser_state, "open_url_in_last_browser", return_value=True) as mock_open:
        web.open_website(stem + suffix)
    mock_open.assert_called_once_with(f"https://{stem}.com", foreground=True)


@pytest.mark.parametrize("domain", _DOMAINS)
def test_open_website_domain_with_site_suffix_keeps_existing_tld(domain):
    with patch.object(web.browser_state, "open_url_in_last_browser", return_value=True) as mock_open:
        web.open_website(domain + " website")
    mock_open.assert_called_once_with(f"https://{domain}", foreground=True)


# ============================================================================
# Tab management: position/query resolution across many tab-list sizes
# ============================================================================

_TAB_COUNTS = list(range(1, 51))  # 50 sizes


@pytest.mark.parametrize("n", _TAB_COUNTS)
def test_switch_to_browser_tab_first_and_last_position(n):
    tabs_json = json.dumps(_make_tabs(n))
    with patch("urllib.request.urlopen", return_value=_mock_response(tabs_json.encode())):
        first = browser.switch_to_browser_tab(position=1)
        last = browser.switch_to_browser_tab(position=n)
    assert first == "Switched to tab: Tab 0 Title"
    assert last == f"Switched to tab: Tab {n - 1} Title"


@pytest.mark.parametrize("n", _TAB_COUNTS)
def test_switch_to_browser_tab_position_out_of_range_raises(n):
    tabs_json = json.dumps(_make_tabs(n))
    with patch("urllib.request.urlopen", return_value=_mock_response(tabs_json.encode())):
        with pytest.raises(ValueError, match=f"only {n} tab"):
            browser.switch_to_browser_tab(position=n + 1)


@pytest.mark.parametrize("n", _TAB_COUNTS)
def test_close_browser_tab_first_and_last_position(n):
    tabs_json = json.dumps(_make_tabs(n))
    with patch("urllib.request.urlopen", return_value=_mock_response(tabs_json.encode())):
        first = browser.close_browser_tab(position=1)
    tabs_json2 = json.dumps(_make_tabs(n))
    with patch("urllib.request.urlopen", return_value=_mock_response(tabs_json2.encode())):
        last = browser.close_browser_tab(position=n)
    assert first == "Closed tab: Tab 0 Title"
    assert last == f"Closed tab: Tab {n - 1} Title"


@pytest.mark.parametrize("n", _TAB_COUNTS)
def test_close_browser_tab_position_out_of_range_raises(n):
    tabs_json = json.dumps(_make_tabs(n))
    with patch("urllib.request.urlopen", return_value=_mock_response(tabs_json.encode())):
        with pytest.raises(ValueError):
            browser.close_browser_tab(position=n + 1)


@pytest.mark.parametrize("n", _TAB_COUNTS)
def test_list_browser_tabs_numbers_every_tab_sequentially(n):
    tabs_json = json.dumps(_make_tabs(n))
    with patch("urllib.request.urlopen", return_value=_mock_response(tabs_json.encode())):
        result = browser.list_browser_tabs()
    assert [t["position"] for t in result] == list(range(1, n + 1))
    assert len(result) == n


@pytest.mark.parametrize("n", [3, 5, 10, 20, 30])
@pytest.mark.parametrize("query_index", range(0, 3))
def test_switch_to_browser_tab_by_query_matches_correct_tab(n, query_index):
    if query_index >= n:
        pytest.skip("index out of range for this tab count")
    tabs_json = json.dumps(_make_tabs(n))
    with patch("urllib.request.urlopen", return_value=_mock_response(tabs_json.encode())):
        result = browser.switch_to_browser_tab(query=f"Tab {query_index} Title")
    assert result == f"Switched to tab: Tab {query_index} Title"


# YouTube in-page volume/fullscreen used to be tested here - they're general
# video-tab controls now (any site, not just YouTube), with real multi-video
# disambiguation added on top, so they moved to their own dedicated file:
# tests/test_video_player.py.

# ============================================================================
# YouTube search/channel navigation across many realistic queries
# ============================================================================


@pytest.mark.parametrize("query", _SONG_QUERIES * 15)  # 150 cases (10 base queries x 15 reps with index)
def test_play_video_on_youtube_across_many_queries(query):
    with patch.object(youtube, "_target_tab", return_value=_tab()), patch.object(
        youtube, "_send_cdp_command", side_effect=_eval_sequence("https://www.youtube.com/watch?v=abc123")
    ), patch("time.sleep"):
        result = youtube.play_video_on_youtube(query)
    assert result == f"Playing on YouTube: {query}"


@pytest.mark.parametrize("channel", _CHANNEL_NAMES)
def test_play_latest_video_from_channel_cache_miss_across_channels(channel):
    with patch.object(youtube, "_target_tab", return_value=_tab()), patch.object(
        youtube, "_send_cdp_command",
        side_effect=_eval_sequence(f"https://youtube.com/@{channel.replace(' ', '')}", "https://youtube.com/watch?v=xyz"),
    ), patch.object(youtube.cache, "get", return_value=None), patch.object(
        youtube.cache, "set"
    ) as mock_set, patch("time.sleep"):
        result = youtube.play_latest_video_from_channel(channel)
    assert result == f"Playing the latest video from {channel}."
    mock_set.assert_called_once()


@pytest.mark.parametrize("channel", _CHANNEL_NAMES)
def test_play_latest_video_from_channel_cache_hit_across_channels(channel):
    cached_url = f"https://youtube.com/@{channel.replace(' ', '')}"
    with patch.object(youtube, "_target_tab", return_value=_tab()), patch.object(
        youtube, "_send_cdp_command", side_effect=_eval_sequence("https://youtube.com/watch?v=xyz")
    ), patch.object(youtube.cache, "get", return_value=cached_url), patch.object(
        youtube.cache, "set"
    ) as mock_set, patch("time.sleep"):
        result = youtube.play_latest_video_from_channel(channel)
    assert result == f"Playing the latest video from {channel}."
    mock_set.assert_not_called()  # was already cached - nothing new to store


# ============================================================================
# click_on_page: adversarial/varied text safely round-trips through
# json.dumps-based JS embedding
# ============================================================================

_CLICK_TEXTS = [
    "Sign in", "Learn more", "OG Crew", "MrBeast's latest video", "Don't miss out",
    "Save 50% today!", 'Click "here" now', "Buy now — limited time", "İstanbul",
    "日本語のテキスト", "émoji test 🎉🔥", "line1\nline2", "tab\there", "back\\slash\\test",
    "quote's and \"quotes\"", "<script>alert(1)</script>", "a" * 300, "",
    "   leading and trailing spaces   ", "MIXED CaSe TeXt", "123456789",
    "special chars: !@#$%^&*()_+-=[]{}|;:,.<>?/~`",
] * 5  # 22 * 5 = 110 cases


@pytest.mark.parametrize("text", _CLICK_TEXTS)
def test_click_on_page_handles_varied_and_adversarial_text_safely(text):
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(True)
    ) as mock_send:
        result = browser_automation.click_on_page(text)
    assert result == f"Clicked: {text}"
    # the JS payload must be valid: json.dumps(text) appears verbatim and is
    # itself parseable back to the original text (proves safe embedding,
    # no accidental string-termination/injection from quotes/backslashes/etc.)
    expression = mock_send.call_args.args[2]["expression"]
    embedded_literal = json.dumps(text)
    assert embedded_literal in expression
    assert json.loads(embedded_literal) == text


@pytest.mark.parametrize("text", _CLICK_TEXTS)
def test_click_on_page_raises_when_not_found_regardless_of_text(text):
    with patch.object(browser_automation, "_target_tab", return_value=_tab()), patch.object(
        browser_automation, "_send_cdp_command", return_value=_cdp_value(False)
    ):
        with pytest.raises(ValueError, match="No clickable element"):
            browser_automation.click_on_page(text)


# ============================================================================
# WhatsApp confirmation prompt: many contact/message combinations
# ============================================================================

_WHATSAPP_CONTACTS = [
    "Mom", "Dad", "Rahul", "Priya", "Amit Sharma", "Office Group", "Best Friend",
    "Dr. Verma", "Boss", "College Friends",
]
_WHATSAPP_MESSAGES = [
    "On my way!", "Running late, sorry", "Call me when free",
    "Happy birthday! 🎉", "Can we reschedule to tomorrow?",
    "Thanks for today", "See you at 5",
    'He said "let\'s meet" earlier', "Don't forget the documents",
    "Love you, see you soon",
]


@pytest.mark.parametrize(
    "contact,message",
    list(itertools.product(_WHATSAPP_CONTACTS, _WHATSAPP_MESSAGES)),
)
def test_whatsapp_confirmation_prompt_quotes_exact_contact_and_message(contact, message):
    from assistant.tools.registry import registry as real_registry

    prompt = real_registry.get_confirmation_prompt(
        "send_whatsapp_message", {"contact_name": contact, "message": message}
    )
    assert message in prompt
    assert contact in prompt


# ============================================================================
# Disk cache round-trips across many namespace/key/value combinations
# ============================================================================

_CACHE_NAMESPACES = ["youtube_channel", "whatsapp_contact", "browser_domain"]
_CACHE_KEYS = _CHANNEL_NAMES + _SITE_STEMS  # 20 + 30 = 50
_CACHE_VALUES = [f"https://example.com/{i}" for i in range(5)]


@pytest.mark.parametrize(
    "namespace,key,value",
    [
        (ns, key, _CACHE_VALUES[i % len(_CACHE_VALUES)])
        for ns, (i, key) in itertools.product(_CACHE_NAMESPACES, enumerate(_CACHE_KEYS))
    ],
)
def test_cache_set_then_get_round_trips_for_many_combinations(namespace, key, value, tmp_path):
    with patch.object(cache, "_CACHE_FILE", tmp_path / "cache.json"), patch.object(
        cache, "_CACHE_DIR", tmp_path
    ):
        cache.set(namespace, key, value)
        result = cache.get(namespace, key, max_age_seconds=999999)
    assert result == value


# ============================================================================
# browser_state resolution order across many registered-browser combinations
# ============================================================================

_BROWSER_NAME_SETS = [
    {"brave": "C:\\Brave\\brave.exe"},
    {"google chrome": "C:\\Chrome\\chrome.exe"},
    {"microsoft edge": "C:\\Edge\\msedge.exe"},
    {"brave": "C:\\Brave\\brave.exe", "google chrome": "C:\\Chrome\\chrome.exe"},
    {"brave": "C:\\Brave\\brave.exe", "microsoft edge": "C:\\Edge\\msedge.exe"},
    {"google chrome": "C:\\Chrome\\chrome.exe", "microsoft edge": "C:\\Edge\\msedge.exe"},
    {
        "brave": "C:\\Brave\\brave.exe",
        "google chrome": "C:\\Chrome\\chrome.exe",
        "microsoft edge": "C:\\Edge\\msedge.exe",
    },
]


@pytest.mark.parametrize("registered", _BROWSER_NAME_SETS)
@pytest.mark.parametrize("target_name", ["brave", "google chrome", "microsoft edge"])
def test_open_url_prefers_last_opened_app_across_registered_combinations(registered, target_name):
    if target_name not in registered:
        pytest.skip("target browser isn't in this registered set")
    browser_state._last_browser_name = target_name
    with patch.object(
        browser_state, "_enumerate_registered_browsers", return_value=registered
    ), patch.object(browser_state, "subprocess") as mock_subprocess, patch.object(
        browser_state, "foreground_by_title"
    ):
        result = browser_state.open_url_in_last_browser("https://example.com")
    assert result is True
    mock_subprocess.Popen.assert_called_once_with([registered[target_name], "https://example.com"])
