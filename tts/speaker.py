import re
import threading
import queue
from kokoro_onnx import Kokoro
import sounddevice as sd


class Speaker:
    def __init__(self):
        self.kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
        self.voice = "af_bella"

    def speak(self, text: str):
        if not text.strip():
            return

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return

        audio_queue: queue.Queue = queue.Queue(maxsize=1)
        _sentinel = object()

        def _generator():
            for sentence in sentences:
                samples, sample_rate = self.kokoro.create(
                    sentence, voice=self.voice, speed=1.15
                )
                audio_queue.put((samples, sample_rate))
            audio_queue.put(_sentinel)

        def _player():
            while True:
                item = audio_queue.get()
                if item is _sentinel:
                    break
                samples, sample_rate = item
                sd.play(samples, sample_rate)
                sd.wait()

        gen = threading.Thread(target=_generator, daemon=True)
        ply = threading.Thread(target=_player, daemon=True)
        gen.start()
        ply.start()
        gen.join()
        ply.join()
