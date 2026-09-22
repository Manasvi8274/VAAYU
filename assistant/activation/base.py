from abc import ABC, abstractmethod


class Activator(ABC):
    @abstractmethod
    def wait_for_trigger(self) -> None:
        """Blocks until the user triggers the assistant (hotkey press, or the wake word in M5)."""
