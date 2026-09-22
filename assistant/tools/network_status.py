import json
import re
import subprocess

from assistant.tools.registry import registry


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return result.stdout


def _get_wifi_status() -> dict:
    output = _run(["netsh", "wlan", "show", "interfaces"])
    if not output.strip():
        return {"available": False}

    state_match = re.search(r"^\s*State\s*:\s*(.+)$", output, re.MULTILINE)
    ssid_match = re.search(r"^\s*SSID\s*:\s*(.+)$", output, re.MULTILINE)
    signal_match = re.search(r"^\s*Signal\s*:\s*(.+)$", output, re.MULTILINE)

    state = state_match.group(1).strip() if state_match else "unknown"
    connected = state.lower() == "connected"
    return {
        "available": True,
        "connected": connected,
        "ssid": ssid_match.group(1).strip() if ssid_match and connected else None,
        "signal": signal_match.group(1).strip() if signal_match and connected else None,
    }


def _get_bluetooth_status() -> dict:
    ps_cmd = "Get-PnpDevice -Class Bluetooth | Select-Object FriendlyName,Status | ConvertTo-Json -Compress"
    output = _run(["powershell", "-NoProfile", "-Command", ps_cmd])
    try:
        devices = json.loads(output) if output.strip() else []
    except json.JSONDecodeError:
        return {"enabled": False, "devices": []}
    if isinstance(devices, dict):
        devices = [devices]
    return {
        "enabled": any(d.get("Status") == "OK" for d in devices),
        "devices": [{"name": d.get("FriendlyName"), "status": d.get("Status")} for d in devices],
    }


@registry.register(
    name="get_network_status",
    description=(
        "Check whether Wi-Fi and Bluetooth are on, and what they're connected to "
        "(Wi-Fi network name/signal strength, paired Bluetooth devices)."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def get_network_status() -> dict:
    return {
        "wifi": _get_wifi_status(),
        "bluetooth": _get_bluetooth_status(),
    }
