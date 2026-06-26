import re
import time
import threading
import queue
from kokoro_onnx import Kokoro
import sounddevice as sd


class Speaker:
    def __init__(self):
        self.kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
        self.voice = "af_bella"
        self.speed = 1.15

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
            for i, sentence in enumerate(sentences):
                t0 = time.time()
                samples, sample_rate = self.kokoro.create(
                    sentence, voice=self.voice, speed=self.speed
                )
                gen_time = time.time() - t0
                audio_queue.put((samples, sample_rate, gen_time))
            audio_queue.put(_sentinel)

        def _player():
            while True:
                item = audio_queue.get()
                if item is _sentinel:
                    break
                samples, sample_rate, gen_time = item
                t0 = time.time()
                sd.play(samples, sample_rate)
                sd.wait()
                play_time = time.time() - t0
                print(f"  ⏱  TTS: gen={gen_time:.2f}s play={play_time:.2f}s total={gen_time+play_time:.2f}s")

        gen = threading.Thread(target=_generator, daemon=True)
        ply = threading.Thread(target=_player, daemon=True)
        gen.start()
        ply.start()
        gen.join()
        ply.join()
