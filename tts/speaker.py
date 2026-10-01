import threading

from tts.playback import PlaybackEngine


class Speaker:
    def __init__(self, engine: PlaybackEngine | None = None) -> None:
        self.engine = engine or PlaybackEngine()

    def stop(self) -> None:
        self.engine.stop()

    def speak(self, text: str, session_cancelled: threading.Event | None = None) -> None:
        if text.strip():
            self.engine.speak([text], session_cancelled)
