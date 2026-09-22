"""Importing this package registers every tool with the shared registry."""

from assistant.tools import (  # noqa: F401
    apps,
    audio_status,
    browser,
    browser_automation,
    devices,
    dictation,
    files,
    media,
    network_status,
    notes,
    orb_control,
    self_review_control,
    system_control,
    system_info,
    video_player,
    web,
    whatsapp,
    windows,
    youtube,
)
