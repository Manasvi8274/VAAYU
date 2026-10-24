"""
Answers from the internet without opening a browser: weather (Open-Meteo)
and general questions (DuckDuckGo). Both free, no account or API key.

Offline, both fail with a plain "couldn't reach the internet" instead of
guessing - the assistant only knows what it looked up.
"""

import httpx

from assistant.tools.registry import registry

_TIMEOUT = 6  # seconds per request - a spoken answer can't wait longer
_OFFLINE = "I couldn't reach the internet to look that up."

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


def _locate(place: str) -> tuple[str, float, float]:
    if not place.strip():  # no place said: wherever this laptop is, from its IP
        here = _get_json("http://ip-api.com/json/", {"fields": "status,city,lat,lon"})
        if here.get("status") != "success":
            raise ValueError("I couldn't tell where you are - say the city, like: weather in Gurugram.")
        return here["city"], here["lat"], here["lon"]
    found = _get_json("https://geocoding-api.open-meteo.com/v1/search",
                      {"name": place.strip(), "count": 1, "language": "en"}).get("results")
    if not found:
        raise ValueError(f"I couldn't find a place called {place.strip()}.")
    return found[0]["name"], found[0]["latitude"], found[0]["longitude"]


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
def answer_question(question: str) -> dict:
    instant = {}
    try:  # DuckDuckGo's instant answer: a one-line fact when it has one (often empty)
        instant = _get_json("https://api.duckduckgo.com/", {"q": question, "format": "json", "no_html": 1,
                                                             "skip_disambig": 1})
    except ConnectionError:
        pass
    answer = str(instant.get("Answer") or "")[:300]
    abstract = str(instant.get("AbstractText") or "")[:300]
    try:
        results = [{"title": r.get("title", ""), "snippet": (r.get("body") or "")[:220]} for r in _ddg_text(question)]
    except ConnectionError:
        if not (answer or abstract):
            raise
        results = []
    # The abstract is a Wikipedia definition - live, "who is the prime minister of india" got "The prime
    # minister of India is the head of government..." and the reply never named anyone. Results first.
    return {"question": question, "results": results, "instant_answer": answer or ("" if results else abstract)}
