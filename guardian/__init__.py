"""
Vaayu's guardian: a small supervisor process that is NOT part of Vaayu's own
code, so Vaayu (including its future self-upgrades) can't edit or disable it.

    python -m guardian          start Vaayu under the guardian
    python -m guardian kill     stop Vaayu and the guardian completely
    python -m guardian status   is it running?

Kill paths - all handled here, none depend on Vaayu's code still working:
"Vaayu die" (the guardian's own mic listener), Ctrl+Alt+Shift+F12, and the
"Kill Vaayu" desktop shortcut. "Sleep Vaayu" is Vaayu's own sleep, not this.
"""
