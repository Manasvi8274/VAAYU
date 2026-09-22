from abc import ABC, abstractmethod


class TTSEngine(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Speaks text out loud (blocking until finished)."""

    @abstractmethod
    def save_to_file(self, text: str, path: str) -> None:
        """Synthesizes text to a wav file instead of playing it - used for
        automated testing without needing a live speaker."""
