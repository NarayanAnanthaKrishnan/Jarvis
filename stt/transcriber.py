from faster_whisper import WhisperModel
import numpy as np


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", vad_filter: bool = True):
        print(f"Loading Whisper ({model_size}) on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.vad_filter = vad_filter

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) < sample_rate * 0.5:
            return ""
        segments, _ = self.model.transcribe(audio, beam_size=1, vad_filter=self.vad_filter)
        text = " ".join([s.text for s in segments]).strip()
        return text
