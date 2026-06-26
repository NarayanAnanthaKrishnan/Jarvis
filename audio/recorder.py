import sounddevice as sd
import numpy as np


class Recorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.recording: list[np.ndarray] = []
        self.is_recording = False
        self.stream: sd.InputStream | None = None

    def start(self):
        if self.is_recording:
            return
        self.recording = []
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._callback
        )
        self.stream.start()
        import time
        time.sleep(0.03)
        print("🎙 Listening...")

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        if self.is_recording:
            self.recording.append(indata.copy())

    def get_partial(self, since: int) -> tuple[np.ndarray, int]:
        chunks = self.recording[since:]
        if len(chunks) == 0:
            return np.array([]), since
        return np.concatenate(chunks, axis=0).flatten(), len(self.recording)

    def stop(self) -> np.ndarray:
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.recording:
            return np.array([])
        return np.concatenate(self.recording, axis=0).flatten()
