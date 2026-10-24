import json
import re
import subprocess
import time

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
    # PnP Status "OK" only means the adapter is installed - it stays "OK" with
    # the radio switched off (found live testing set_radio). Ask the radio.
    return {
        "enabled": bool(devices) and _radio_ps("bluetooth", "Query") == "On",
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


# Windows.Devices.Radios (WinRT) - the same switch as the Action Center's
# Wi-Fi/Bluetooth tiles: works as a normal user, no admin needed (unlike
# `netsh interface set interface ... disable`). Windows PowerShell 5.1 can
# await WinRT calls via the AsTask bridge below.
_RADIO_PS = r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, $type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
[Windows.Devices.Radios.Radio,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
[Windows.Devices.Radios.RadioAccessStatus,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
[Windows.Devices.Radios.RadioState,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
Await ([Windows.Devices.Radios.Radio]::RequestAccessAsync()) ([Windows.Devices.Radios.RadioAccessStatus]) | Out-Null
$radios = Await ([Windows.Devices.Radios.Radio]::GetRadiosAsync()) ([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]])
$r = $radios | Where-Object { $_.Kind -eq '__KIND__' } | Select-Object -First 1
if (-not $r) { Write-Output 'NO_RADIO'; exit }
if ('__STATE__' -eq 'Query') { Write-Output $r.State; exit }
$s = Await ($r.SetStateAsync('__STATE__')) ([Windows.Devices.Radios.RadioAccessStatus])
Write-Output $s
"""
_RECONNECT_WAIT_SECONDS = 15
_last_wifi_ssid: str | None = None


def _radio_ps(kind: str, state: str) -> str:
    script = _RADIO_PS.replace("__KIND__", "WiFi" if kind == "wifi" else "Bluetooth").replace("__STATE__", state)
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                            capture_output=True, text=True, timeout=30)
    out = result.stdout.strip()
    return out.splitlines()[-1] if out else result.stderr.strip()


def _wifi_connected() -> bool:
    return bool(_get_wifi_status().get("connected"))


@registry.register(
    name="set_radio",
    description=(
        "Turn Wi-Fi or Bluetooth on or off. Hinglish: 'wifi band karo' = off, 'wifi on karo'/'chalu "
        "karo' = on. Turning Wi-Fi back on reconnects to the network it was on before."
    ),
    parameters={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["wifi", "bluetooth"]},
            "state": {"type": "string", "enum": ["on", "off"]},
        },
        "required": ["kind", "state"],
    },
)
def set_radio(kind: str, state: str) -> str:
    global _last_wifi_ssid
    kind, state = kind.lower(), state.lower()
    if kind not in ("wifi", "bluetooth") or state not in ("on", "off"):
        raise ValueError("kind must be wifi/bluetooth and state on/off")
    if kind == "wifi" and state == "off":
        _last_wifi_ssid = _get_wifi_status().get("ssid") or _last_wifi_ssid

    status = _radio_ps(kind, "On" if state == "on" else "Off")
    label = "Wi-Fi" if kind == "wifi" else "Bluetooth"
    if status == "NO_RADIO":
        raise RuntimeError(f"No {label} radio found on this computer.")
    if status != "Allowed":
        raise RuntimeError(f"Windows refused to change {label} ({status or 'no response'}).")

    if kind == "wifi" and state == "on":
        for _ in range(_RECONNECT_WAIT_SECONDS):
            if _wifi_connected():
                break
            time.sleep(1)
        else:
            if _last_wifi_ssid:  # Windows didn't auto-join - ask for the previous network explicitly
                _run(["netsh", "wlan", "connect", f"name={_last_wifi_ssid}"])
                time.sleep(5)
        ssid = _get_wifi_status().get("ssid")
        return f"Wi-Fi is on and connected to {ssid}." if ssid else "Wi-Fi is on, but not connected to a network yet."
    return f"{label} turned {state}."
