"""
Answers from the internet without opening a browser: weather (Open-Meteo)
and general questions (DuckDuckGo). Both free, no account or API key.

A question asked again within 12 hours is answered instantly from the
answer cache (knowledge.py, "answers"). Offline, a question is answered from
what Vaayu read earlier (his research notes and past answers) when there's
something close enough, saying when he read it; otherwise a plain "couldn't
reach the internet" instead of guessing. Weather is never cached.
"""

import json
import re
from pathlib import Path
from datetime import datetime

import httpx

from assistant.tools.registry import registry

_TIMEOUT = 6  # seconds per request - a spoken answer can't wait longer
_OFFLINE = "I couldn't reach the internet to look that up."
_CACHE_SAME = 0.93  # calibration knob: this similar a question = the same question
_CACHE_HOURS = 12  # facts like "who is the PM" are re-checked after this
_OFFLINE_CLOSE = 0.75  # offline: how close something read earlier must be to answer from it

# WMO weather codes (Open-Meteo's weather_code) -> spoken words.
_WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "foggy", 48: "foggy",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow", 80: "rain showers", 81: "rain showers",
        82: "heavy rain showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
        96: "thunderstorms with hail", 99: "thunderstorms with hail"}


def _get_json(url: str, params: dict) -> dict:
    try:
        r = httpx.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        raise ConnectionError(_OFFLINE) from exc


def _ddg_text(query: str) -> list[dict]:
    """ddgs (the DuckDuckGo search library). Measured: its "bing" engine answers in
    ~3 s, "auto" (tries several engines in turn) in ~7 s - so bing first."""
    from ddgs import DDGS
    for backend in ("bing", "auto"):
        try:
            if results := DDGS(timeout=_TIMEOUT).text(query, max_results=5, backend=backend):
                return results
        except Exception:  # ddgs raises its own types for network / no-results / rate limits
            continue
    raise ConnectionError(_OFFLINE)


PLACE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "weather_place.txt"
_REGION_WORDS = {"india", "bharat", "haryana", "punjab", "delhi ncr", "ncr", "up", "uttar", "pradesh", "maharashtra",
                 "karnataka", "rajasthan", "gujarat", "state", "city", "district", "the", "in", "of"}


# Old names the geocoder puts elsewhere: "gurgaon" was a village in Maharashtra, "bangalore" a town in Pakistan.
_OLD_NAMES = {"gurgaon": "gurugram", "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata",
              "madras": "chennai", "poona": "pune", "baroda": "vadodara", "mysore": "mysuru", "trivandrum":
              "thiruvananthapuram", "benares": "varanasi", "banaras": "varanasi"}


def _locate(place: str) -> tuple[str, float, float]:
    if not place.strip():  # no place said: wherever this laptop is, from its IP
        here = _get_json("http://ip-api.com/json/", {"fields": "status,city,lat,lon"})
        if here.get("status") != "success":
            raise ValueError("I couldn't tell where you are - say the city, like: weather in Gurugram.")
        return here["city"], here["lat"], here["lon"]
    # Open-Meteo wants a bare place name - live, "gurgaon india" and "haryana, panipat" found nothing.
    # Try what was said, then without country/state words, then each word, Indian places first.
    words = [_OLD_NAMES.get(w, w) for w in re.split(r"[\s,]+", place.lower()) if w and w not in _REGION_WORDS]
    place = _OLD_NAMES.get(place.strip().lower(), place)
    for name in dict.fromkeys([place.strip(), " ".join(words), *reversed(words)]):
        if not name:
            continue
        found = _get_json("https://geocoding-api.open-meteo.com/v1/search",
                          {"name": name, "count": 5, "language": "en"}).get("results") or []
        found.sort(key=lambda r: (r.get("country_code") != "IN", -(r.get("population") or 0)))
        if found:
            return found[0]["name"], found[0]["latitude"], found[0]["longitude"]
    raise ValueError(f"I couldn't find a place called {place.strip()}.")


@registry.register(
    name="get_weather",
    description="Current weather and today's or tomorrow's forecast for a city, from the internet. "
                "Leave place empty for where the user is now.",
    parameters={
        "type": "object",
        "properties": {
            "place": {"type": "string", "description": "City or place name, e.g. 'Gurugram'. Empty = current location."},
            "day": {"type": "string", "enum": ["today", "tomorrow"]},
        },
    },
)
def get_weather(place: str = "", day: str = "today") -> str:
    # No city said: the one they asked about last - the IP lookup says "New Delhi" for a Gurugram user.
    if place.strip():
        try:
            PLACE_FILE.parent.mkdir(parents=True, exist_ok=True)
            PLACE_FILE.write_text(place.strip(), encoding="utf-8")
        except OSError:
            pass
    else:
        try:
            place = PLACE_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    name, lat, lon = _locate(place)
    data = _get_json("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 2,
        "current": "temperature_2m,apparent_temperature,weather_code",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    })
    d = data["daily"]
    i = 1 if day == "tomorrow" else 0
    hi, lo = round(d["temperature_2m_max"][i]), round(d["temperature_2m_min"][i])
    rain = d["precipitation_probability_max"][i]
    rain_part = f", {rain}% chance of rain" if rain is not None else ""
    if i:
        sky = _WMO.get(d["weather_code"][1], "mixed weather")
        return f"Tomorrow in {name}: {sky}, high {hi}, low {lo} degrees{rain_part}."
    c = data["current"]
    sky = _WMO.get(c["weather_code"], "mixed weather")
    return (f"In {name} it's {round(c['temperature_2m'])} degrees and {sky}, feels like "
            f"{round(c['apparent_temperature'])}. Today: high {hi}, low {lo}{rain_part}.")


@registry.register(
    name="answer_question",
    description="Look up a factual question on the internet (who is the PM of India, capital of France, "
                "news, sports, prices...) and get the top search results to answer from.",
    parameters={
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The question, in the user's words."}},
        "required": ["question"],
    },
)
def answer_question(question: str) -> dict | str:
    _log_asked(question)
    if cached := _cached_answer(question):
        return cached  # a str is spoken as-is (router.format_result) - no search, no LLM call
    try:
        return _look_up(question)
    except ConnectionError:
        if remembered := _from_memory(question):
            return remembered
        raise


def _look_up(question: str) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:  # both at once: the search alone takes 3-5 s
        # DuckDuckGo's instant answer: a one-line fact when it has one (often empty)
        instant_job = pool.submit(_get_json, "https://api.duckduckgo.com/",
                                  {"q": question, "format": "json", "no_html": 1, "skip_disambig": 1})
        search_job = pool.submit(_ddg_text, question)
    try:
        instant = instant_job.result()
    except ConnectionError:
        instant = {}
    answer = str(instant.get("Answer") or "")[:300]
    abstract = str(instant.get("AbstractText") or "")[:300]
    try:
        results = [{"title": r.get("title", ""), "snippet": (r.get("body") or "")[:220]} for r in search_job.result()]
    except ConnectionError:
        if not (answer or abstract):
            raise
        results = []
    # The abstract is a Wikipedia definition - live, "who is the prime minister of india" got "The prime
    # minister of India is the head of government..." and the reply never named anyone. Results first.
    return {"question": question, "results": results, "instant_answer": answer or ("" if results else abstract)}


# --- answer cache, offline answers, what was asked (research.py reads it) -------------------------

def _knowledge():
    from assistant import knowledge
    return knowledge


def _cached_answer(question: str) -> str | None:
    try:
        for score, row in _knowledge().search("answers", question, k=1, min_score=_CACHE_SAME):
            if _knowledge().age_hours(row) <= _CACHE_HOURS and row.get("meta", {}).get("answer"):
                return row["meta"]["answer"]
    except Exception:  # embedding model unavailable - just look it up
        pass
    return None


def remember_answer(question: str, answer: str) -> None:
    """Called after the spoken answer is phrased: the next time, it's instant."""
    if not answer or answer.startswith(("Sorry", "That didn't work")):
        return
    try:
        _knowledge().add("answers", question, {"answer": answer}, dedupe=_CACHE_SAME)
    except Exception:
        pass


def _from_memory(question: str) -> str | None:
    try:
        best = None
        for kind in ("notes", "answers"):
            for score, row in _knowledge().search(kind, question, k=1, min_score=_OFFLINE_CLOSE):
                if best is None or score > best[0]:
                    best = (score, kind, row)
    except Exception:
        return None
    if best is None:
        return None
    _score, kind, row = best
    text = row["meta"].get("answer") if kind == "answers" else row["text"]
    when = datetime.fromisoformat(row["ts"]).strftime("%d %B").lstrip("0")
    return f"I'm offline, but from what I read on {when}: {text}"


def _log_asked(question: str) -> None:
    from assistant import knowledge
    try:
        knowledge.DIR.mkdir(parents=True, exist_ok=True)
        with open(knowledge.DIR / "asked.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "q": question},
                               ensure_ascii=False) + "\n")
    except OSError:
        pass
