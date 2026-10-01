import threading
from collections.abc import Iterable

from tts.playback import PlaybackEngine


class StreamingSpeaker:
    def __init__(self, engine: PlaybackEngine | None = None) -> None:
        self.engine = engine or PlaybackEngine()

    def stop(self) -> None:
        self.engine.stop()

    def preload(self) -> None:
        self.engine.preload()

    def speak_stream(self, token_generator: Iterable[str], session_cancelled: threading.Event | None = None) -> str:
        return self.engine.speak(token_generator, session_cancelled)
