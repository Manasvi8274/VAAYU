import json
from unittest.mock import patch

import pytest

from assistant.tools import devices, network_status


# --- devices.py ---


def test_list_connected_devices_parses_json_array():
    fake_output = json.dumps([{"FriendlyName": "USB Mouse", "Status": "OK"}])
    with patch.object(devices, "_run_ps", return_value=fake_output):
        result = devices.list_connected_devices("usb")
    assert result == [{"category": "usb", "name": "USB Mouse", "status": "OK"}]


def test_list_connected_devices_wraps_single_object_result():
    # PowerShell's ConvertTo-Json returns a bare object (not a list) when
    # there's exactly one result - must be handled, not crash.
    fake_output = json.dumps({"FriendlyName": "Only Printer", "Status": "OK"})
    with patch.object(devices, "_run_ps", return_value=fake_output):
        result = devices.list_connected_devices("printer")
    assert result == [{"category": "printer", "name": "Only Printer", "status": "OK"}]


def test_list_connected_devices_no_category_queries_every_category():
    # "what devices are connected" with no category named - found live that
    # the model wasn't reliably supplying one, so it's optional now and
    # defaults to every category rather than raising a missing-argument error.
    with patch.object(devices, "_run_ps", return_value=json.dumps([{"FriendlyName": "X", "Status": "OK"}])) as mock_ps:
        result = devices.list_connected_devices()
    assert mock_ps.call_count == len(devices._CATEGORY_CLASS_MAP)
    assert {d["category"] for d in result} == set(devices._CATEGORY_CLASS_MAP.keys())


def test_list_connected_devices_empty_output_returns_empty_list():
    with patch.object(devices, "_run_ps", return_value=""):
        assert devices.list_connected_devices("bluetooth") == []


def test_list_connected_devices_malformed_json_returns_empty_list():
    with patch.object(devices, "_run_ps", return_value="not json{{"):
        assert devices.list_connected_devices("display") == []


def test_list_connected_devices_rejects_unknown_category():
    with pytest.raises(ValueError, match="Unknown category"):
        devices.list_connected_devices("printer_farm")


def test_list_connected_devices_caps_at_20():
    fake_output = json.dumps([{"FriendlyName": f"Dev {i}", "Status": "OK"} for i in range(30)])
    with patch.object(devices, "_run_ps", return_value=fake_output):
        result = devices.list_connected_devices("usb")
    assert len(result) == 20


# --- network_status.py ---


_WIFI_CONNECTED_OUTPUT = """
    Name                   : Wi-Fi
    State                  : connected
    SSID                   : HomeNetwork
    Signal                 : 87%
"""

_WIFI_DISCONNECTED_OUTPUT = """
    Name                   : Wi-Fi
    State                  : disconnected
"""


def test_get_network_status_wifi_connected():
    with patch.object(network_status, "_run") as mock_run:
        mock_run.side_effect = [_WIFI_CONNECTED_OUTPUT, "[]"]
        result = network_status.get_network_status()
    assert result["wifi"] == {
        "available": True,
        "connected": True,
        "ssid": "HomeNetwork",
        "signal": "87%",
    }


def test_get_network_status_wifi_disconnected_has_no_ssid_or_signal():
    with patch.object(network_status, "_run") as mock_run:
        mock_run.side_effect = [_WIFI_DISCONNECTED_OUTPUT, "[]"]
        result = network_status.get_network_status()
    assert result["wifi"]["connected"] is False
    assert result["wifi"]["ssid"] is None
    assert result["wifi"]["signal"] is None


def test_get_network_status_no_wifi_adapter_reports_unavailable():
    with patch.object(network_status, "_run") as mock_run:
        mock_run.side_effect = ["", "[]"]
        result = network_status.get_network_status()
    assert result["wifi"] == {"available": False}


def test_get_network_status_bluetooth_enabled_when_any_device_ok():
    bt_output = json.dumps([{"FriendlyName": "Earbuds", "Status": "OK"}])
    with patch.object(network_status, "_run") as mock_run:
        mock_run.side_effect = [_WIFI_CONNECTED_OUTPUT, bt_output]
        result = network_status.get_network_status()
    assert result["bluetooth"]["enabled"] is True
    assert result["bluetooth"]["devices"] == [{"name": "Earbuds", "status": "OK"}]


def test_get_network_status_bluetooth_disabled_when_no_devices():
    with patch.object(network_status, "_run") as mock_run:
        mock_run.side_effect = [_WIFI_CONNECTED_OUTPUT, "[]"]
        result = network_status.get_network_status()
    assert result["bluetooth"] == {"enabled": False, "devices": []}
