from unittest.mock import patch

from assistant import stop


def test_stop_asks_vaayu_and_reports_when_he_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(stop, "STOP_FLAG", tmp_path / "stop_request")
    running = iter([[123], [123], []])  # running, still finishing, gone
    with patch("guardian.core._vaayu_pids", side_effect=lambda: next(running)):
        assert stop.request_stop(wait=5) == "Vaayu stopped."
    assert (tmp_path / "stop_request").exists()  # the request Vaayu reads (he deletes it himself)


def test_stop_says_use_kill_if_vaayu_ignores_it(tmp_path, monkeypatch):
    monkeypatch.setattr(stop, "STOP_FLAG", tmp_path / "stop_request")
    with patch("guardian.core._vaayu_pids", return_value=[123]):
        assert "Kill Vaayu" in stop.request_stop(wait=0.6)
    assert not (tmp_path / "stop_request").exists()  # no stale request left for the next start


def test_stop_when_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(stop, "STOP_FLAG", tmp_path / "stop_request")
    with patch("guardian.core._vaayu_pids", return_value=[]):
        assert stop.request_stop() == "Vaayu isn't running."
