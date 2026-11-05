"""
Straight to the right place (#11, smarter app control): a Settings page, a folder, a project in VS Code - opened
directly (ms-settings: links, the folder path, the `code` command), no clicking through windows, ~0.2 s.
"""

import os
import shutil
import subprocess
from pathlib import Path

from assistant.tools.registry import registry

SETTINGS = {
    "bluetooth": "bluetooth", "devices": "bluetooth", "wifi": "network-wifi", "wi fi": "network-wifi",
    "network": "network-status", "internet": "network-status", "display": "display", "screen": "display",
    "brightness": "display", "sound": "sound", "audio": "sound", "volume": "sound", "battery": "batterysaver",
    "power": "powersleep", "sleep": "powersleep", "notifications": "notifications", "apps": "appsfeatures",
    "installed apps": "appsfeatures", "default apps": "defaultapps", "startup": "startupapps",
    "startup apps": "startupapps", "storage": "storagesense", "update": "windowsupdate",
    "windows update": "windowsupdate", "updates": "windowsupdate", "privacy": "privacy",
    "microphone": "privacy-microphone", "mic": "privacy-microphone", "camera": "privacy-webcam",
    "mouse": "mousetouchpad", "touchpad": "devices-touchpad", "keyboard": "typing", "typing": "typing",
    "language": "regionlanguage", "time": "dateandtime", "date and time": "dateandtime", "date": "dateandtime",
    "personalization": "personalization", "wallpaper": "personalization-background",
    "background": "personalization-background", "theme": "themes", "themes": "themes", "colors": "colors",
    "dark mode": "colors", "night light": "nightlight", "vpn": "network-vpn", "hotspot": "network-mobilehotspot",
    "airplane mode": "network-airplanemode", "about": "about", "accounts": "yourinfo", "account": "yourinfo",
    "focus": "quiethours", "do not disturb": "quiethours", "printers": "printers", "printer": "printers",
    "multitasking": "multitasking", "taskbar": "taskbar", "lock screen": "lockscreen", "clipboard": "clipboard",
    "accessibility": "easeofaccess", "fonts": "fonts", "backup": "backup", "recovery": "recovery",
}
_HOME = Path.home()
FOLDERS = {"downloads": _HOME / "Downloads", "download": _HOME / "Downloads", "documents": _HOME / "Documents",
           "document": _HOME / "Documents", "desktop": _HOME / "Desktop", "pictures": _HOME / "Pictures",
           "photos": _HOME / "Pictures", "music": _HOME / "Music", "videos": _HOME / "Videos", "video": _HOME / "Videos",
           "screenshots": _HOME / "Pictures" / "Screenshots", "github": _HOME / "Documents" / "GitHub"}
_PROJECT_ROOTS = [_HOME / "Documents" / "GitHub", _HOME / "Documents", _HOME / "Desktop", _HOME / "source" / "repos",
                  _HOME / "projects", _HOME]


def _open(target: str) -> None:
    os.startfile(target)  # noqa: S606 - a fixed ms-settings: link or a known folder, never user text as a command


def _launch(cmd: list[str]) -> None:
    subprocess.Popen(cmd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


@registry.register(
    name="open_settings_page",
    description="Open a Windows Settings page directly: bluetooth, wifi, display, sound, battery, update, wallpaper...",
    parameters={"type": "object", "properties": {"page": {"type": "string"}}, "required": ["page"]},
)
def open_settings_page(page: str) -> str:
    key = page.lower().strip()
    if key not in SETTINGS:
        raise ValueError(f"I don't know a {page} settings page.")
    _open(f"ms-settings:{SETTINGS[key]}")
    return f"Opened {page} settings."


@registry.register(
    name="open_folder",
    description="Open a folder in File Explorer: downloads, documents, desktop, pictures, music, videos, screenshots.",
    parameters={"type": "object", "properties": {"folder": {"type": "string"}}, "required": ["folder"]},
)
def open_folder(folder: str) -> str:
    path = FOLDERS.get(folder.lower().strip())
    if path is None or not path.exists():
        raise ValueError(f"I can't find your {folder} folder.")
    _open(str(path))
    return f"Opened {folder}."


def find_project(name: str) -> Path | None:
    want = "".join(ch for ch in name.lower() if ch.isalnum())
    for root in _PROJECT_ROOTS:
        try:
            for child in root.iterdir():
                if child.is_dir() and "".join(ch for ch in child.name.lower() if ch.isalnum()) == want:
                    return child
        except OSError:
            continue
    return None


@registry.register(
    name="open_in_vscode",
    description="Open a project folder in VS Code by its name ('open the TEST project in VS Code').",
    parameters={"type": "object", "properties": {"project": {"type": "string"}}, "required": ["project"]},
)
def open_in_vscode(project: str) -> str:
    path = find_project(project)
    if path is None:
        raise ValueError(f"I couldn't find a project called {project}.")
    code = shutil.which("code")
    if not code:
        raise RuntimeError("VS Code's 'code' command isn't installed.")
    _launch([code, str(path)])
    return f"Opened {path.name} in VS Code."
