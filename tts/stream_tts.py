import re
import threading
import queue
from collections.abc import Generator
from kokoro_onnx import Kokoro
import sounddevice as sd


class StreamingSpeaker:
    def __init__(self) -> None:
        self.kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
        self.voice = "af_bella"
        self.speed = 1.15
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
        sd.stop()

    def speak_stream(self, token_generator: Generator[str, None, None]) -> str:
        self._stop_event.clear()
        sent_queue: queue.Queue = queue.Queue(maxsize=2)
        _sentinel = object()
        full_text_parts: list[str] = []

        def _producer() -> None:
            buffer = ""
            for token in token_generator:
                if self._stop_event.is_set():
                    break
                if token is None:
                    continue
                full_text_parts.append(token)
                buffer += token
                while True:
                    m = re.search(r'(?<=[.!?])\s+', buffer)
                    if not m:
                        break
                    idx = m.end()
                    sentence = buffer[:idx].strip()
                    buffer = buffer[idx:].strip()
                    if sentence:
                        sent_queue.put(sentence)
            remaining = buffer.strip()
            if remaining and not self._stop_event.is_set():
                sent_queue.put(remaining)
            sent_queue.put(_sentinel)

        def _consumer() -> None:
            while not self._stop_event.is_set():
                try:
                    item = sent_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if item is _sentinel:
                    break
                samples, sample_rate = self.kokoro.create(
                    item, voice=self.voice, speed=self.speed
                )
                if self._stop_event.is_set():
                    break
                sd.play(samples, sample_rate)
                sd.wait()

        prod = threading.Thread(target=_producer, daemon=True)
        cons = threading.Thread(target=_consumer, daemon=True)
        prod.start()
        cons.start()
        prod.join()
        cons.join()
        return "".join(full_text_parts)
