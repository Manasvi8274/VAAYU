"""
YouTube control through YouTube's OWN player API (the #movie_player
element's playVideo/pauseVideo/setVolume/seekTo/... - the same methods its
buttons call), with every action verified by reading the player's state
back before anything is reported as done.

Why not the plain <video> element (video_player.py's generic approach)?
Found live: "set the volume to 75%" changed <video>.volume, the read-back
of that same property said 75%, Vaayu said "done" - while YouTube's own
volume (its slider, and what it re-applies) stayed at 90%. A false
confirmation. On YouTube the player API is the source of truth.

Requires the browser to have been opened by Vaayu (DevTools port 9222).
"""

import json
import time
from urllib.parse import quote_plus

from assistant.tools.browser import _list_tabs
from assistant.tools.registry import registry
from assistant.tools.video_player import _dispatch_key, _eval_js_on_tab

_P = "document.getElementById('movie_player')"
# getPlayerState(): -1 unstarted, 0 ended, 1 playing, 2 paused, 3 buffering, 5 cued
_PLAYING_STATES = (1, 3)
_QUALITY = {"144p": "tiny", "240p": "small", "360p": "medium", "480p": "large", "720p": "hd720",
            "1080p": "hd1080", "1440p": "hd1440", "2160p": "hd2160", "4k": "hd2160", "auto": "auto"}


# --- tab / state helpers -------------------------------------------------------

# The tab Vaayu last acted on. Found live (multi-tab Brave): a search ran in
# one YouTube tab and the next command picked a different one, so Vaayu
# reported on a tab it wasn't controlling. Sticking to the last-used tab
# keeps a run of commands on the same page.
_last_tab_id: str | None = None


def _is_watch(tab: dict) -> bool:
    url = tab.get("url") or ""
    return "/watch" in url or "/shorts/" in url


def remember(tab: dict | None) -> dict | None:
    global _last_tab_id
    if tab:
        _last_tab_id = tab.get("id")
    return tab


def youtube_tab(require_watch: bool = False) -> dict | None:
    """The YouTube tab to act on: the one Vaayu last used, else a watch page
    (most recent first), else any YouTube tab."""
    tabs = [t for t in _list_tabs() if "youtube.com" in (t.get("url") or "") and t.get("webSocketDebuggerUrl")]
    last = next((t for t in tabs if t.get("id") == _last_tab_id), None)
    if last and (_is_watch(last) or not require_watch):
        return last
    watch = [t for t in tabs if _is_watch(t)]
    if require_watch:
        return watch[0] if watch else None
    return (watch or tabs or [None])[0]


def _tab(require_watch: bool = True) -> dict:
    tab = remember(youtube_tab(require_watch))
    if tab is None:
        raise RuntimeError("No YouTube video is open in the browser I control. Say 'open youtube' or "
                           "'play <something> on youtube' first.")
    return tab


def _ev(tab, js):
    return _eval_js_on_tab(tab, js)


def _until(tab, js, ok, timeout: float = 2.0, every: float = 0.15):
    deadline = time.time() + timeout
    while True:
        value = _ev(tab, js)
        if ok(value) or time.time() >= deadline:
            return value
        time.sleep(every)


def _state(tab) -> int | None:
    return _ev(tab, f"{_P} && {_P}.getPlayerState ? {_P}.getPlayerState() : null")


def _require_player(tab) -> None:
    if _state(tab) is None:
        raise RuntimeError("The YouTube player isn't ready on that tab yet - try again in a moment.")


def is_playing(tab) -> bool:
    return _state(tab) in _PLAYING_STATES


def _fmt(seconds) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# --- actions ----------------------------------------------------------------------

def play(tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    if is_playing(tab):
        return "It's already playing."
    _ev(tab, f"{_P}.playVideo()")
    if _until(tab, f"{_P}.getPlayerState()", lambda s: s in _PLAYING_STATES) not in _PLAYING_STATES:
        _dispatch_key(tab, key="k", code="KeyK", vk=75)  # YouTube's own play/pause key - a trusted keypress
        if _until(tab, f"{_P}.getPlayerState()", lambda s: s in _PLAYING_STATES) not in _PLAYING_STATES:
            raise RuntimeError("I tried to play the video, but YouTube didn't start it.")
    return "Playing."


def pause(tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    if not is_playing(tab):
        return "It's already paused."
    _ev(tab, f"{_P}.pauseVideo()")
    if _until(tab, f"{_P}.getPlayerState()", lambda s: s == 2) != 2:
        # Found live: once in ~40 commands pauseVideo() didn't take (buffering) - YouTube's own 'k' key does.
        _dispatch_key(tab, key="k", code="KeyK", vk=75)
        if _until(tab, f"{_P}.getPlayerState()", lambda s: s == 2) != 2:
            raise RuntimeError("I tried to pause the video, but it's still playing.")
    return "Paused."


def set_volume(level: int | None = None, delta: int | None = None, tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    current = _ev(tab, f"{_P}.getVolume()")
    target = level if level is not None else (current or 0) + (delta or 0)
    target = max(0, min(100, int(round(target))))
    _ev(tab, f"{_P}.setVolume({target}); {_P}.unMute()")
    actual = _until(tab, f"{_P}.getVolume()", lambda v: v is not None and abs(v - target) <= 1)
    muted = _ev(tab, f"{_P}.isMuted()")
    if actual is None or abs(actual - target) > 1 or muted:
        raise RuntimeError(f"I tried to set the YouTube volume to {target}%, but it's still at {actual}%.")
    return f"YouTube volume is now {target}%."


def set_muted(muted: bool, tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    if _ev(tab, f"{_P}.isMuted()") == muted:
        return "It's already muted." if muted else "It's already unmuted."
    _ev(tab, f"{_P}.{'mute' if muted else 'unMute'}()")
    if _until(tab, f"{_P}.isMuted()", lambda m: m == muted) != muted:
        raise RuntimeError(f"I tried to {'mute' if muted else 'unmute'} the video, but it didn't change.")
    return "Muted the video." if muted else "Unmuted the video."


def seek(delta: int | None = None, to: int | None = None, tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    now = _ev(tab, f"{_P}.getCurrentTime()") or 0
    duration = _ev(tab, f"{_P}.getDuration()") or 0
    target = to if to is not None else now + (delta or 0)
    target = max(0, min(target, max(duration - 1, 0))) if duration else max(0, target)
    _ev(tab, f"{_P}.seekTo({target}, true)")
    actual = _until(tab, f"{_P}.getCurrentTime()", lambda t: t is not None and abs(t - target) < 2.5)
    if actual is None or abs(actual - target) >= 2.5:
        raise RuntimeError(f"I tried to jump to {_fmt(target)}, but the video is at {_fmt(actual)}.")
    if to is not None:
        return f"Jumped to {_fmt(target)}."
    return f"{'Skipped ahead' if (delta or 0) > 0 else 'Went back'} {abs(int(delta or 0))} seconds, now at {_fmt(target)}."


def _video_id(tab):
    return _ev(tab, f"{_P} && {_P}.getVideoData ? {_P}.getVideoData().video_id : null")


def next_video(tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    before = _video_id(tab)
    _ev(tab, f"{_P}.nextVideo ? {_P}.nextVideo() : document.querySelector('.ytp-next-button')?.click()")
    after = _until(tab, f"{_P}.getVideoData().video_id", lambda v: v and v != before, timeout=6)
    if not after or after == before:
        clicked = _ev(tab, "(() => { const b = document.querySelector('.ytp-next-button'); if (!b) return false; b.click(); return true; })()")
        after = _until(tab, f"{_P}.getVideoData().video_id", lambda v: v and v != before, timeout=6) if clicked else before
    if not after or after == before:
        raise RuntimeError("I tried to go to the next video, but YouTube stayed on the same one.")
    return f"Playing the next video: {_ev(tab, f'{_P}.getVideoData().title')}."


def previous_video(tab=None) -> str:
    tab = tab or _tab()
    before = _ev(tab, "location.href")
    _ev(tab, "history.back()")
    after = _until(tab, "location.href", lambda u: u and u != before, timeout=5)
    if after == before:
        raise RuntimeError("There's no previous video to go back to.")
    return "Went back to the previous video."


def set_speed(rate: float, tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    rates = _ev(tab, f"{_P}.getAvailablePlaybackRates()") or [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    rate = min(rates, key=lambda r: abs(r - rate))
    _ev(tab, f"{_P}.setPlaybackRate({rate})")
    actual = _until(tab, f"{_P}.getPlaybackRate()", lambda r: r == rate)
    if actual != rate:
        raise RuntimeError(f"I tried to set the speed to {rate}x, but it's at {actual}x.")
    return "Back to normal speed." if rate == 1 else f"Speed is now {rate:g}x."


def _toggle_button(tab, selector: str, attr: str, want: bool, what: str, missing: str) -> str:
    js_state = f"(() => {{ const b = document.querySelector({selector!r}); return b ? b.getAttribute({attr!r}) === 'true' : null; }})()"
    state = _ev(tab, js_state)
    if state is None:
        raise RuntimeError(missing)
    if state == want:
        return f"{what} {'are' if what.endswith('s') else 'is'} already {'on' if want else 'off'}."
    _ev(tab, f"document.querySelector({selector!r}).click()")
    if _until(tab, js_state, lambda s: s == want) != want:
        raise RuntimeError(f"I tried to turn {what.lower()} {'on' if want else 'off'}, but it didn't change.")
    return f"{what} {'on' if want else 'off'}."


def captions(on: bool, tab=None) -> str:
    tab = tab or _tab()
    # YouTube keeps the CC button on screen (disabled) for videos without captions - found live.
    if on and _ev(tab, "(() => { const b = document.querySelector('.ytp-subtitles-button'); return !!b && "
                       "(b.getAttribute('aria-disabled') === 'true' || /unavailable/i.test(b.getAttribute('title') || "
                       "b.getAttribute('data-title-no-tooltip') || '')); })()"):
        raise RuntimeError("This video doesn't have captions.")
    return _toggle_button(tab or _tab(), ".ytp-subtitles-button", "aria-pressed", on, "Captions",
                          "This video doesn't have captions.")


def autoplay(on: bool, tab=None) -> str:
    return _toggle_button(tab or _tab(), ".ytp-autonav-toggle-button", "aria-checked", on, "Autoplay",
                          "I can't find YouTube's autoplay switch on this page.")


def theater(on: bool, tab=None) -> str:
    tab = tab or _tab()
    js = "(() => { const f = document.querySelector('ytd-watch-flexy'); return f ? f.hasAttribute('theater') : null; })()"
    state = _ev(tab, js)
    if state is None:
        raise RuntimeError("Theater mode only works on a YouTube video page.")
    if state == on:
        return f"Theater mode is already {'on' if on else 'off'}."
    _dispatch_key(tab, key="t", code="KeyT", vk=84)
    if _until(tab, js, lambda s: s == on) != on:
        _ev(tab, "document.querySelector('.ytp-size-button')?.click()")
        if _until(tab, js, lambda s: s == on) != on:
            raise RuntimeError("I tried to change theater mode, but it didn't change.")
    return f"Theater mode {'on' if on else 'off'}."


def fullscreen(on: bool, tab=None) -> str:
    tab = tab or _tab()
    js = "!!document.fullscreenElement"
    if _ev(tab, js) == on:
        return "Already in fullscreen." if on else "Already out of fullscreen."
    _dispatch_key(tab, key="f", code="KeyF", vk=70)
    if _until(tab, js, lambda s: s == on, timeout=1.5) != on:
        raise RuntimeError("I tried to change fullscreen, but it didn't change.")
    return "Fullscreen on." if on else "Exited fullscreen."


def loop(on: bool, tab=None) -> str:
    tab = tab or _tab()
    _ev(tab, f"document.querySelector('#movie_player video').loop = {'true' if on else 'false'}")
    if _ev(tab, "document.querySelector('#movie_player video')?.loop") != on:
        raise RuntimeError("I couldn't change looping on this video.")
    return "This video will loop." if on else "Looping off."


def _pressed(tab, selector):
    return _ev(tab, f"(() => {{ const b = document.querySelector({selector!r}); return b ? b.getAttribute('aria-pressed') === 'true' : null; }})()")


def like(on: bool = True, dislike: bool = False, tab=None) -> str:
    tab = tab or _tab()
    sel = "dislike-button-view-model button" if dislike else "like-button-view-model button"
    word = "disliked" if dislike else "liked"
    state = _pressed(tab, sel)
    if state is None:
        raise RuntimeError("I can't find the like button on this page.")
    if state == on:
        return f"You've already {word} this video." if on else f"It isn't {word}."
    _ev(tab, f"document.querySelector({sel!r}).click()")
    if _until(tab, f"(() => {{ const b = document.querySelector({sel!r}); return b && b.getAttribute('aria-pressed') === 'true'; }})()",
              lambda s: s == on) != on:
        raise RuntimeError("YouTube didn't register that - you may need to be signed in to YouTube in this browser.")
    return f"{word.capitalize()} the video." if on else f"Removed the {word[:-1] if dislike else 'like'}."


_SUB_STATE_JS = ("(() => { const r = document.querySelector('#owner ytd-subscribe-button-renderer, ytd-subscribe-button-renderer');"
                 " if (!r) return null; const b = r.querySelector('button'); const t = ((b && b.innerText) || '').toLowerCase();"
                 " return r.hasAttribute('subscribed') || t.includes('subscribed'); })()")


def subscribe(tab=None) -> str:
    tab = tab or _tab()
    state = _ev(tab, _SUB_STATE_JS)
    if state is None:
        raise RuntimeError("I can't find the subscribe button on this page.")
    if state:
        return "You're already subscribed to this channel."
    _ev(tab, "(document.querySelector('#owner ytd-subscribe-button-renderer button') || "
             "document.querySelector('ytd-subscribe-button-renderer button')).click()")
    if not _until(tab, _SUB_STATE_JS, bool, timeout=3):
        raise RuntimeError("YouTube didn't register the subscription - you may need to be signed in.")
    return "Subscribed to the channel."


_AD_JS = "!!document.querySelector('#movie_player.ad-showing')"
_SKIP_BTN = ".ytp-skip-ad-button, .ytp-ad-skip-button, .ytp-ad-skip-button-modern, button[id^='skip-button']"


def skip_ad(tab=None) -> str:
    tab = tab or _tab()
    if not _ev(tab, _AD_JS):
        return "No ad is playing."
    clicked = _ev(tab, f"(() => {{ const b = document.querySelector({_SKIP_BTN!r}); if (!b || b.offsetParent === null) return false; b.click(); return true; }})()")
    if not clicked:
        return "This ad can't be skipped yet."
    if _until(tab, _AD_JS, lambda a: not a, timeout=3):
        raise RuntimeError("I clicked skip, but the ad is still showing.")
    return "Skipped the ad."


def set_quality(label: str, tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    q = _QUALITY.get(label.lower().replace(" ", ""))
    if not q:
        raise ValueError(f"I don't know the quality '{label}'. Try 1080p, 720p, 480p, 360p or auto.")
    available = _ev(tab, f"{_P}.getAvailableQualityLevels()") or []
    if q != "auto" and q not in available:
        names = [k for k, v in _QUALITY.items() if v in available and k != "4k"]
        raise RuntimeError(f"This video doesn't have {label}. It has: {', '.join(names) or 'only auto'}.")
    _ev(tab, f"{_P}.setPlaybackQualityRange({q!r}, {q!r})" if q != "auto" else f"{_P}.setPlaybackQualityRange('auto')")
    if q != "auto" and _until(tab, f"{_P}.getPlaybackQuality()", lambda v: v == q, timeout=4) != q:
        raise RuntimeError(f"I asked YouTube for {label}, but it's still playing at another quality.")
    return f"Quality set to {label}."


def now_playing(tab=None) -> str:
    tab = tab or _tab()
    _require_player(tab)
    # Found live: right after a video loads, getVideoData() is still empty ("'None' by None, 0:00").
    if not _until(tab, f"{_P}.getVideoData().title", bool, timeout=4):
        # Found live: answered "'None' by None, 0:00" instead of admitting it couldn't tell.
        raise RuntimeError("I can't read what's playing on YouTube right now.")
    # (Was half f-string / half plain string: the plain half kept a literal "}}",
    # the JS failed silently, and the reply was built from nothing - found live.)
    data = _ev(tab, "(() => { const p = " + _P + "; const d = p.getVideoData(); return {title: d.title, author: d.author, "
                    "t: p.getCurrentTime(), dur: p.getDuration(), state: p.getPlayerState(), vol: p.getVolume(), "
                    "muted: p.isMuted(), rate: p.getPlaybackRate()}; })()") or {}
    if not data.get("title"):
        raise RuntimeError("I can't read what's playing on YouTube right now.")
    status = "playing" if data.get("state") in _PLAYING_STATES else "paused"
    vol = "muted" if data.get("muted") else f"volume {data.get('vol')}%"
    speed = "" if data.get("rate") in (1, None) else f", at {data['rate']:g}x speed"
    return (f"'{data.get('title')}' by {data.get('author')} is {status}, {_fmt(data.get('t'))} of "
            f"{_fmt(data.get('dur'))}, {vol}{speed}.")


def _navigate(tab, url: str, wait_js: str, what: str) -> None:
    from assistant.tools.browser_automation import _send_cdp_command
    _send_cdp_command(tab["webSocketDebuggerUrl"], "Page.navigate", {"url": url})
    if not _until(tab, wait_js, bool, timeout=8, every=0.3):
        raise RuntimeError(f"YouTube didn't finish loading {what}.")


def search(query: str, tab=None) -> str:
    tab = tab or _tab(require_watch=False)
    remember(tab)
    _navigate(tab, f"https://www.youtube.com/results?search_query={quote_plus(query)}",
              "document.querySelectorAll('ytd-video-renderer a#video-title').length > 0", "the search results")
    titles = _ev(tab, "Array.from(document.querySelectorAll('ytd-video-renderer a#video-title')).slice(0, 3)"
                      ".map(a => a.title || a.innerText.trim())") or []
    top = "; ".join(f"{i + 1}: {t}" for i, t in enumerate(titles))
    return f"Here are YouTube results for {query}. Top ones - {top}." if top else f"Here are YouTube results for {query}."


def play_result(n: int, tab=None) -> str:
    tab = tab or _tab(require_watch=False)
    if _last_list and time.time() - _last_list[0] < _LIST_FRESH_SECONDS:  # "play the second one" after recommend
        videos = _last_list[1]
        if not 1 <= abs(n) <= len(videos):
            raise RuntimeError(f"I only suggested {len(videos)} - pick one of those.")
        return _play_video(tab, videos[n - 1 if n > 0 else n])
    href = _ev(tab, f"(() => {{ const a = Array.from(document.querySelectorAll('ytd-video-renderer a#video-title'))"
                    f".filter(a => a.href && !a.href.includes('/shorts/'))[{n - 1}]; return a ? a.href : null; }})()")
    if not href:
        raise RuntimeError(f"There's no result number {n} on this page - search for something first.")
    _navigate(tab, href, f"{_P} && {_P}.getPlayerState ? true : false", "the video")
    record_current(tab, background=True)
    return f"Playing result {n}: {_ev(tab, f'{_P}.getVideoData().title') or 'the video'}."


# --- videos on the page: the right-side recommendations, the home feed, results -------------------------

# The videos the user can see, in page order. YouTube's 2025+ layout uses
# yt-lockup-view-model (title, then channel, then views/age as text lines);
# the older renderers are kept for pages that still use them. On a watch
# page only the right-hand list counts (not the comments or description).
_CANDIDATES_JS = r"""(() => {
  const watch = location.pathname === '/watch';
  const scope = watch ? (document.querySelector('#related') || document.querySelector('#secondary')) : document;
  if (!scope) return [];
  const out = [], seen = new Set([new URLSearchParams(location.search).get('v')]);
  const boxes = scope.querySelectorAll('yt-lockup-view-model, ytd-compact-video-renderer, ytd-video-renderer, ytd-rich-grid-media');
  for (const b of boxes) {
    const a = b.querySelector('a[href*="/watch?v="]');
    if (!a || b.getBoundingClientRect().width === 0) continue;
    const id = new URL(a.href, location.origin).searchParams.get('v');
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const t = b.querySelector('h3, #video-title, .yt-lockup-metadata-view-model__title');
    const title = ((t && (t.getAttribute('title') || t.innerText)) || '').trim();
    const lines = (b.innerText || '').split('\n').map(s => s.trim()).filter(Boolean);
    const i = lines.indexOf(title);
    const rest = i >= 0 ? lines.slice(i + 1) : lines;
    if (!title) continue;
    out.push({id, title, channel: rest[0] || '', age: rest.find(s => /ago$/.test(s)) || '',
              href: new URL(a.getAttribute('href'), location.origin).href});
    if (out.length >= 30) break;
  }
  return out;
})()"""
_LIST_FRESH_SECONDS = 600  # "play the second one" refers to the last list Vaayu read out, for 10 minutes
_last_list: tuple[float, list[dict]] | None = None
_MATCH_SURE = 0.6  # calibration knob: share of the said words a video must contain to be picked


def _candidates(tab, timeout: float = 5.0) -> list[dict]:
    """Waits for the list to stop growing - live, right after a page loaded it had 4 of its 20."""
    deadline, last = time.time() + timeout, None
    while True:
        videos = _ev(tab, _CANDIDATES_JS) or []
        if (videos and last is not None and len(videos) == len(last) and len(videos) >= 8) or time.time() >= deadline:
            return videos
        last = videos
        time.sleep(0.5)


def record_current(tab, background: bool = False) -> None:
    """Log what's playing to the watch history (youtube_taste), once its data has loaded."""
    def run():
        from assistant.tools import youtube_taste
        try:
            data = _until(tab, f"(() => {{ const d = {_P} && {_P}.getVideoData ? {_P}.getVideoData() : null; "
                               f"return d && d.title ? {{id: d.video_id, title: d.title, author: d.author}} : null; }})()",
                          bool, timeout=6, every=0.5)
            if data:
                youtube_taste.record(data["id"], data["title"], data["author"])
        except Exception:
            pass  # history is a nice-to-have - never fail the command over it
    if background:
        import threading
        threading.Thread(target=run, daemon=True).start()
    else:
        run()


def _play_video(tab, video: dict) -> str:
    _navigate(tab, video["href"], f"(() => {{ const p = {_P}; return !!(p && p.getVideoData && "
                                  f"p.getVideoData().video_id === {json.dumps(video['id'])}); }})()", "the video")
    if _until(tab, f"{_P}.getPlayerState()", lambda s: s in _PLAYING_STATES, timeout=6) not in _PLAYING_STATES:
        _ev(tab, f"{_P}.playVideo()")
    record_current(tab, background=True)
    channel = f" by {video['channel']}" if video.get("channel") else ""
    return f"Playing '{video['title']}'{channel}."


def play_recommended(description: str, tab=None) -> str:
    """Play the video on the page that matches what was said - by its words (title, channel,
    "12 hours ago") or its place ("the third one"). Nothing close: search YouTube for it."""
    from assistant.tools import youtube_taste
    tab = tab or _tab(require_watch=False)
    videos = _candidates(tab)
    n = youtube_taste.ordinal(description)
    if n is not None:
        if not videos or not 1 <= abs(n) <= len(videos):
            raise RuntimeError(f"I can't see a video number {n} on this page.")
        return _play_video(tab, videos[n - 1 if n > 0 else n])
    best, sure = youtube_taste.match(description, videos)
    if best and sure >= _MATCH_SURE:
        return _play_video(tab, best)
    query = " ".join(youtube_taste._words(description)) or description
    from assistant.tools.youtube import play_video_on_youtube
    play_video_on_youtube(query)
    return f"I couldn't see that one on this page, so I searched YouTube: playing the top result for {query}."


def recommend(tab=None) -> str:
    """Suggest 3 videos for the user: YouTube's own suggestions on this page (or the home
    feed, which YouTube already personalises), re-ranked by what they watch."""
    global _last_list
    from assistant.tools import youtube_taste
    tab = tab or youtube_tab(require_watch=False)
    if tab is None:
        from assistant.tools.youtube import _resolve_target_tab
        tab = _resolve_target_tab(False)
    remember(tab)
    if "youtube.com" not in (_ev(tab, "location.hostname") or ""):
        go_home(tab)
    videos = _candidates(tab)
    if not videos and not _is_watch({"url": _ev(tab, "location.href") or ""}):
        go_home(tab)
        videos = _candidates(tab)
    if not videos:
        raise RuntimeError("I can't see any videos on YouTube to suggest right now.")
    top = youtube_taste.rank(videos)[:3]
    _last_list = (time.time(), top)
    because = " based on what you watch" if youtube_taste.history(1) else ""
    picks = "; ".join(f"{i + 1}, {v['title']}" + (f" by {v['channel']}" if v.get("channel") else "")
                      for i, v in enumerate(top))
    return f"Here are three{because}: {picks}. Say 'play the first one', or the second, or the third."


def open_channel(tab=None) -> str:
    tab = tab or _tab()
    # Waits: found live, right after "play the second video" the owner block hadn't rendered yet.
    href = _until(tab, "(() => { const a = document.querySelector(\"#owner a[href*='/@'], #owner a[href*='/channel/'], "
                       "ytd-video-owner-renderer a[href*='/@']\"); return a ? a.href : null; })()", bool, timeout=4)
    if not href:
        raise RuntimeError("I can't find this video's channel link.")
    _navigate(tab, href, "location.href.includes('/@') || location.href.includes('/channel/')", "the channel")
    return "Opened the channel."


def go_home(tab=None) -> str:
    tab = tab or _tab(require_watch=False)
    _navigate(tab, "https://www.youtube.com/", "location.pathname === '/'", "the home page")
    return "Opened the YouTube home page."


def next_short(tab=None) -> str:
    tab = tab or _tab()
    before = _ev(tab, "location.href")
    if "/shorts/" not in (before or ""):
        raise RuntimeError("That only works while watching a YouTube Short.")
    _dispatch_key(tab, key="ArrowDown", code="ArrowDown", vk=40)
    if _until(tab, "location.href", lambda u: u and u != before, timeout=4) == before:
        raise RuntimeError("I tried to go to the next Short, but it didn't move.")
    return "Next Short."


def copy_link(tab=None) -> str:
    import win32clipboard
    tab = tab or _tab()
    url = _ev(tab, "location.href")
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(url, win32clipboard.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()
    return "Copied the video link."


ACTIONS = {
    "play": lambda v: play(), "pause": lambda v: pause(),
    "toggle_play": lambda v: pause() if (t := youtube_tab(True)) and is_playing(t) else play(),
    "volume_set": lambda v: set_volume(level=int(float(v))), "volume_up": lambda v: set_volume(delta=int(float(v or 10))),
    "volume_down": lambda v: set_volume(delta=-int(float(v or 10))), "mute": lambda v: set_muted(True),
    "unmute": lambda v: set_muted(False), "seek_forward": lambda v: seek(delta=int(float(v or 10))),
    "seek_back": lambda v: seek(delta=-int(float(v or 10))), "seek_to": lambda v: seek(to=int(float(v or 0))),
    "restart": lambda v: seek(to=0), "next": lambda v: next_video(), "previous": lambda v: previous_video(),
    "speed": lambda v: set_speed(float(v or 1)), "captions_on": lambda v: captions(True),
    "captions_off": lambda v: captions(False), "theater_on": lambda v: theater(True), "theater_off": lambda v: theater(False),
    "fullscreen_on": lambda v: fullscreen(True), "fullscreen_off": lambda v: fullscreen(False),
    "loop_on": lambda v: loop(True), "loop_off": lambda v: loop(False), "like": lambda v: like(True),
    "unlike": lambda v: like(False), "dislike": lambda v: like(True, dislike=True), "subscribe": lambda v: subscribe(),
    "skip_ad": lambda v: skip_ad(), "autoplay_on": lambda v: autoplay(True), "autoplay_off": lambda v: autoplay(False),
    "quality": lambda v: set_quality(str(v or "auto")), "now_playing": lambda v: now_playing(),
    "search": lambda v: search(str(v)), "play_result": lambda v: play_result(int(float(v or 1))),
    "open_channel": lambda v: open_channel(), "home": lambda v: go_home(), "next_short": lambda v: next_short(),
    "copy_link": lambda v: copy_link(),
    "play_recommended": lambda v: play_recommended(str(v or "")), "recommend": lambda v: recommend(),
}


@registry.register(
    name="youtube_control",
    description=(
        "Control YouTube in the browser: play, pause, volume (volume_set with value 0-100, volume_up/"
        "volume_down by value points), mute/unmute, seek_forward/seek_back (value = seconds), seek_to "
        "(value = seconds), restart, next/previous video, speed (value like 1.5), captions_on/off, "
        "theater_on/off, fullscreen_on/off, loop_on/off, like/unlike/dislike, subscribe, skip_ad, "
        "autoplay_on/off, quality (value like 720p), now_playing, search (value = query, shows results), "
        "play_result (value = result number, or the number of a video Vaayu just suggested), "
        "play_recommended (value = what the user said about a video on the page - its title words, channel, "
        "age or position, e.g. 'stand-up by Pranav Sharma' or 'third'), recommend (suggest 3 videos for the "
        "user from YouTube's suggestions, ranked by their watch history), open_channel, home, next_short, "
        "copy_link. Every action is "
        "checked against YouTube's own player state before it's reported as done."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)},
            "value": {"type": "string", "description": "Number, query or quality, when the action needs one."},
        },
        "required": ["action"],
    },
)
def youtube_control(action: str, value: str | None = None) -> str:
    if action not in ACTIONS:
        raise ValueError(f"Unknown YouTube action '{action}'.")
    from assistant.tools import music
    if action in music.FROM_VIDEO and music.owns_controls():  # "skip to 1:10" while a speaker song plays
        return music.control(*music.FROM_VIDEO[action](value))
    return ACTIONS[action](value)
