import psutil

from assistant.tools.registry import registry


@registry.register(
    name="get_system_status",
    description=(
        "Get the current system status: CPU usage, RAM usage, battery level, and free disk space. "
        "Use this for any battery/charge question, including Hinglish ('battery kitni hai', 'kitna "
        "charge hai', 'how much battery is left') - NOT get_audio_status, which has speaker/"
        "microphone volume, a different percentage that happens to look similar but is not battery."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def get_system_status() -> dict:
    battery = psutil.sensors_battery()
    disk = psutil.disk_usage("/")
    mem = psutil.virtual_memory()

    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "ram_percent": mem.percent,
        "ram_used_gb": round(mem.used / (1024**3), 1),
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "battery_percent": battery.percent if battery else None,
        "battery_plugged_in": battery.power_plugged if battery else None,
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "disk_total_gb": round(disk.total / (1024**3), 1),
    }
