import threading
from shutil import get_terminal_size
import pyautogui
from stt.stream_stt import StreamSTT
from config import STT_STABILIZE_PARTIALS


class WhisperFlowMode:
    def __init__(self, stream_stt: StreamSTT) -> None:
        self.stream_stt = stream_stt
        self.active: bool = False
        self._typed_words: int = 0
        self._last_partials: list[list[str]] = []
        self._type_lock: threading.Lock = threading.Lock()

    def toggle(self) -> None:
        if not self.active:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        self.active = True
        self._typed_words = 0
        self._last_partials = []
        self.stream_stt.set_realtime_callback(self._on_partial)
        self.stream_stt.start()

    def _on_partial(self, text: str) -> None:
        if not self.active:
            return

        cols = get_terminal_size().columns
        msg = f"🎤 {text}"
        print(f"\r{msg:<{cols}}", end="", flush=True)

        words = text.split()
        self._last_partials.append(words)
        if len(self._last_partials) > STT_STABILIZE_PARTIALS:
            self._last_partials.pop(0)

        if len(self._last_partials) < STT_STABILIZE_PARTIALS:
            return

        stable_count = self._get_stable_word_count()
        if stable_count > self._typed_words:
            new_words = words[self._typed_words:stable_count]
            to_type = (" " if self._typed_words > 0 else "") + " ".join(new_words)
            if to_type:
                with self._type_lock:
                    pyautogui.typewrite(to_type, interval=0.0)
                self._typed_words = stable_count

    def _get_stable_word_count(self) -> int:
        min_len = min(len(w) for w in self._last_partials)
        for i in range(min_len):
            first = self._last_partials[0][i]
            if not all(w[i] == first for w in self._last_partials):
                return i
        return min_len

    def _stop(self) -> None:
        self.active = False
        print()
        self.stream_stt.set_realtime_callback(None)
        self.stream_stt.stop()
        final_text = self.stream_stt.text()
        if not final_text:
            return

        final_words = final_text.split()
        if self._typed_words < len(final_words):
            remainder = " ".join(final_words[self._typed_words:])
            prefix = " " if self._typed_words > 0 else ""
            with self._type_lock:
                pyautogui.typewrite(prefix + remainder, interval=0.0)

        print(f"✅ Dictation complete ({len(final_words)} words)")
