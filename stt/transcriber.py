import os
from faster_whisper import WhisperModel
import numpy as np
from config import STT_MODEL, STT_DEVICE


def _add_cuda_dll_dirs() -> None:
    import sys
    site = sys.prefix + "/Lib/site-packages"
    for rel in ["nvidia/cublas/bin", "nvidia/cuda_runtime/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"]:
        d = os.path.join(site, rel)
        if os.path.isdir(d):
            os.add_dll_directory(d)


class Transcriber:
    def __init__(self, model_size: str = STT_MODEL, device: str = STT_DEVICE, vad_filter: bool = True):
        compute = "float16" if device == "cuda" else "int8"
        print(f"Loading Whisper ({model_size}) on {device} ({compute})...")
        if device == "cuda":
            _add_cuda_dll_dirs()
        self.model = WhisperModel(model_size, device=device, compute_type=compute)
        self.vad_filter = vad_filter

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) < sample_rate * 0.5:
            return ""
        segments, _ = self.model.transcribe(audio, beam_size=1, vad_filter=self.vad_filter)
        text = " ".join([s.text for s in segments]).strip()
        return text
