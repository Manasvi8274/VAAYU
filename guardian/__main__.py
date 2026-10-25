import sys

from guardian import core


def _say(text: str) -> None:
    if sys.stdout is not None:  # None under pythonw (the desktop shortcut)
        print(text)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        core.run()
    elif command == "kill":
        _say(core.kill_all())
    elif command == "status":
        _say(core.status())
    elif command in ("upgrades", "review", "show", "approve", "reject", "set-pin"):
        from guardian import upgrades
        sys.exit(upgrades.cli(sys.argv[1:]))
    else:
        sys.exit(f"Unknown command '{command}' - use run, kill, status, upgrades, show N, approve N, reject N "
                 "or set-pin.")
