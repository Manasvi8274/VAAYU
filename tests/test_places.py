"""Straight to the right place: Settings pages, folders, VS Code projects (assistant/tools/places.py)."""
import pytest

from assistant.llm import router
from assistant.tools import places


@pytest.mark.parametrize("said, call", [
    ("open bluetooth settings", ("open_settings_page", {"page": "bluetooth"})),
    ("open wifi settings", ("open_settings_page", {"page": "wifi"})),
    ("open downloads folder", ("open_folder", {"folder": "downloads"})),
    ("downloads kholo", ("open_folder", {"folder": "downloads"})),
    ("open the project test in vs code", ("open_in_vscode", {"project": "test"})),
])
def test_routes(said, call):
    assert router.fast_route(said) == ("tools", [call])


def test_unknown_settings_page_is_left_to_the_app_opener():
    assert router.fast_route("open settings") == ("tools", [("open_app", {"app_name": "settings"})])


def test_the_right_link_is_opened(monkeypatch):
    opened = []
    monkeypatch.setattr(places, "_open", opened.append)
    assert places.open_settings_page("Bluetooth") == "Opened Bluetooth settings."
    assert opened == ["ms-settings:bluetooth"]
    with pytest.raises(ValueError):
        places.open_settings_page("rocket")


def test_a_project_is_found_by_its_name_whatever_the_case(tmp_path, monkeypatch):
    (tmp_path / "My-App").mkdir()
    monkeypatch.setattr(places, "_PROJECT_ROOTS", [tmp_path])
    assert places.find_project("my app") == tmp_path / "My-App"
    assert places.find_project("nothing") is None
