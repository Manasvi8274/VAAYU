import json
import subprocess

from assistant.tools.registry import registry

_CATEGORY_CLASS_MAP = {
    "usb": "USB",
    "bluetooth": "Bluetooth",
    "display": "Monitor",
    "printer": "Printer",
}


def _run_ps(cmd: str) -> str:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout


@registry.register(
    name="list_connected_devices",
    description=(
        "List connected hardware devices on this computer, optionally filtered by category: usb, "
        "bluetooth, display, or printer. Omit category for a generic 'what devices are connected' "
        "request - it then lists every category."
    ),
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(_CATEGORY_CLASS_MAP.keys()),
                "description": "Which category of device to list. Omit to list all categories.",
            }
        },
        "required": [],
    },
)
def list_connected_devices(category: str | None = None) -> list[dict]:
    # A bare "what devices are connected" has no natural single category, and
    # the model wasn't reliably supplying one anyway - found live, the exact
    # same phrase produced a missing-required-argument failure once and a
    # correctly-filled category on a separate run of the identical request.
    # Making it optional (defaulting to every category) sidesteps that
    # non-determinism instead of relying on the model to always guess right.
    categories = [category] if category else list(_CATEGORY_CLASS_MAP.keys())

    results: list[dict] = []
    for cat in categories:
        pnp_class = _CATEGORY_CLASS_MAP.get(cat)
        if pnp_class is None:
            raise ValueError(f"Unknown category '{cat}'. Valid categories: {list(_CATEGORY_CLASS_MAP)}")

        ps_cmd = (
            f"Get-PnpDevice -Class {pnp_class} | Where-Object {{$_.Status -eq 'OK'}} | "
            "Select-Object FriendlyName,Status | ConvertTo-Json -Compress"
        )
        output = _run_ps(ps_cmd)
        try:
            devices = json.loads(output) if output.strip() else []
        except json.JSONDecodeError:
            devices = []
        if isinstance(devices, dict):
            devices = [devices]
        results.extend({"category": cat, "name": d.get("FriendlyName"), "status": d.get("Status")} for d in devices)

    return results[:20]
