from unittest.mock import MagicMock

import pytest

import assistant.tools  # noqa: F401 - registers every tool
from assistant.llm import router
from assistant.llm.ollama_client import OllamaBrain
from assistant.tools import web_answer
from assistant.tools.registry import registry

_FORECAST = {
    "current": {"temperature_2m": 26.6, "apparent_temperature": 28.2, "weather_code": 2},
    "daily": {"weather_code": [2, 61], "temperature_2m_max": [31.4, 29.0], "temperature_2m_min": [19.2, 18.0],
              "precipitation_probability_max": [10, 70]},
}


def _fake_get_json(url, params):
    if "geocoding" in url:
        return {"results": [{"name": "Gurugram", "latitude": 28.46, "longitude": 77.03}]} if params["name"] != "zzz" else {}
    if "ip-api" in url:
        return {"status": "success", "city": "New Delhi", "lat": 28.6, "lon": 77.2}
    if "duckduckgo" in url:
        return {"Answer": "", "AbstractText": ""}
    return _FORECAST


def test_weather_now_for_a_named_city(monkeypatch):
    monkeypatch.setattr(web_answer, "_get_json", _fake_get_json)
    assert web_answer.get_weather("gurugram") == (
        "In Gurugram it's 27 degrees and partly cloudy, feels like 28. Today: high 31, low 19, 10% chance of rain.")


def test_weather_tomorrow_and_current_location(monkeypatch):
    monkeypatch.setattr(web_answer, "_get_json", _fake_get_json)
    assert web_answer.get_weather("", "tomorrow") == "Tomorrow in New Delhi: light rain, high 29, low 18 degrees, 70% chance of rain."


def test_unknown_place_says_so(monkeypatch):
    monkeypatch.setattr(web_answer, "_get_json", _fake_get_json)
    with pytest.raises(ValueError, match="couldn't find a place called zzz"):
        web_answer.get_weather("zzz")


def test_offline_says_it_could_not_reach_the_internet():
    result = registry.dispatch("get_weather", {"place": "Gurugram"})  # conftest: no network in tests
    assert not result["ok"] and "couldn't reach the internet" in result["error"]
    result = registry.dispatch("answer_question", {"question": "who is the president of india"})
    assert not result["ok"] and "couldn't reach the internet" in result["error"]


def test_answer_question_returns_search_snippets(monkeypatch):
    monkeypatch.setattr(web_answer, "_get_json", _fake_get_json)
    monkeypatch.setattr(web_answer, "_ddg_text", lambda q: [{"title": "Droupadi Murmu", "body": "15th President of India"}])
    out = web_answer.answer_question("who is the president of india")
    assert out["results"] == [{"title": "Droupadi Murmu", "snippet": "15th President of India"}]


def test_instant_answer_alone_is_enough(monkeypatch):
    monkeypatch.setattr(web_answer, "_get_json", lambda url, params: {"Answer": "Paris"})
    assert web_answer.answer_question("capital of france")["instant_answer"] == "Paris"  # search itself offline


@pytest.mark.parametrize("text, expected", [
    ("what is the weather in gurugram", ("get_weather", {"place": "gurugram"})),
    ("Vaayu, what's the weather like in Delhi today?", ("get_weather", {"place": "Delhi"})),
    ("gurugram ka mausam kaisa hai", ("get_weather", {"place": "gurugram"})),
    ("will it rain tomorrow", ("get_weather", {"place": "", "day": "tomorrow"})),
    ("how hot is it in jaipur", ("get_weather", {"place": "jaipur"})),
    ("aaj ka mausam", ("get_weather", {"place": ""})),
    ("who is the president of india", ("answer_question", {"question": "who is the president of india"})),
    ("who is the PM of india", ("answer_question", {"question": "who is the PM of india"})),
    ("who is the education minister of india", ("answer_question", {"question": "who is the education minister of india"})),
    ("india ka president kaun hai", ("answer_question", {"question": "india ka president kaun hai"})),
    ("tell me about the taj mahal", ("answer_question", {"question": "tell me about the taj mahal"})),
    ("when is diwali this year", ("answer_question", {"question": "when is diwali this year"})),
])
def test_questions_about_the_world_go_online(text, expected):
    assert router.fast_route(text) == ("tools", [expected])


@pytest.mark.parametrize("text", [
    "what should i do with my life", "what do you think about my plan", "what's your name", "who made you",
    "how old are you", "what's 25 times 4", "what's on this page", "what windows are open",
    "where is my budget spreadsheet", "tell me about yourself", "explain what an api is",
])
def test_personal_local_and_math_questions_stay_offline(text):
    assert router.fast_route(text) is None


@pytest.mark.parametrize("text, tool", [
    ("search for weather in delhi", "open_website"),  # an explicit search still opens the browser
    ("what is playing", "youtube_control"), ("what's my battery percentage", "get_system_status"),
    ("which wifi am I on", "get_network_status"), ("play rain songs on youtube", "play_music"),
])
def test_local_rules_still_win(text, tool):
    assert router.fast_route(text)[1][0][0] == tool


def test_weather_is_spoken_as_is_and_answers_are_phrased_by_the_llm(monkeypatch):
    monkeypatch.setattr(web_answer, "_get_json", _fake_get_json)
    brain = OllamaBrain(host="http://x", model="m")
    brain.client = MagicMock()
    reply, _ = brain.respond([{"role": "user", "content": "weather in gurugram"}], registry)
    assert reply.startswith("In Gurugram it's 27 degrees") and not brain.client.chat.called

    monkeypatch.setattr(web_answer, "_ddg_text", lambda q: [{"title": "Droupadi Murmu", "body": "15th President"}])
    brain.client.chat.return_value = {"message": {"content": "Droupadi Murmu is the President of India."}}
    reply, _ = brain.respond([{"role": "user", "content": "who is the president of india"}], registry)
    assert reply == "Droupadi Murmu is the President of India."
    sent = brain.client.chat.call_args.kwargs["messages"]
    assert "Droupadi Murmu" in sent[1]["content"] and "tools" not in brain.client.chat.call_args.kwargs


def test_place_said_with_country_or_old_name_still_geocodes(monkeypatch):
    # Real log: "weather in gurgaon india" / "haryana, panipat" -> "couldn't find a place".
    asked = []

    def fake_get(url, params):
        asked.append(params["name"])
        known = {"gurugram": [{"name": "Gurugram", "latitude": 28.46, "longitude": 77.03, "country_code": "IN"}],
                 "panipat": [{"name": "Panipat", "latitude": 29.39, "longitude": 76.97, "country_code": "IN"}]}
        return {"results": known.get(params["name"].lower(), [])}
    monkeypatch.setattr(web_answer, "_get_json", fake_get)
    assert web_answer._locate("gurgaon india")[0] == "Gurugram"
    assert web_answer._locate("haryana, panipat")[0] == "Panipat"


@pytest.mark.parametrize("text, place", [
    ("I am asking you about the weather in gurugram india.", "gurugram india"),  # real log: "no internet access"
    ("Why can't you check the weather.", ""), ("tell me the weather", ""),
])
def test_weather_with_personal_words_still_looks_it_up(text, place):
    assert router.fast_route(text) == ("tools", [("get_weather", {"place": place})])


def test_no_city_means_the_last_city_asked(monkeypatch, tmp_path):
    monkeypatch.setattr(web_answer, "PLACE_FILE", tmp_path / "place.txt")  # don't leak into other tests
    asked = []
    monkeypatch.setattr(web_answer, "_locate", lambda p: asked.append(p) or ("X", 1.0, 2.0))
    monkeypatch.setattr(web_answer, "_get_json", lambda url, params: {"current": {"temperature_2m": 25,
        "apparent_temperature": 26, "weather_code": 0}, "daily": {"weather_code": [0, 0], "temperature_2m_max": [30, 30],
        "temperature_2m_min": [20, 20], "precipitation_probability_max": [0, 0]}})
    web_answer.get_weather("Gurugram")
    web_answer.get_weather("")
    assert asked == ["Gurugram", "Gurugram"]
