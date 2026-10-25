from abc import ABC, abstractmethod


class TTSEngine(ABC):
    @abstractmethod
    def speak(self, text: str, stop_event=None) -> None:
        """Speaks text out loud (blocking until finished, or until stop_event -
        a threading.Event - is set: the user said "stop")."""

    @abstractmethod
    def save_to_file(self, text: str, path: str) -> None:
        """Synthesizes text to a wav file instead of playing it - used for
        automated testing without needing a live speaker."""
