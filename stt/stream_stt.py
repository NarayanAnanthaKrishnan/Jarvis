import os
import logging
from typing import Optional, Callable
from RealtimeSTT import AudioToTextRecorder
from config import STT_MODEL, STT_DEVICE, STT_PREROLL_SECONDS


def _add_cuda_dll_dirs() -> None:
    import sys
    site = sys.prefix + "/Lib/site-packages"
    for rel in ["nvidia/cublas/bin", "nvidia/cuda_runtime/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"]:
        d = os.path.join(site, rel)
        if os.path.isdir(d) and d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


class StreamSTT:
    def __init__(self) -> None:
        _add_cuda_dll_dirs()
        self._realtime_callback: Optional[Callable[[str], None]] = None
        self._recorder = AudioToTextRecorder(
            model=STT_MODEL,
            language="en",
            device=STT_DEVICE,
            compute_type="float16",
            silero_sensitivity=1.0,
            post_speech_silence_duration=9999,
            min_length_of_recording=0,
            min_gap_between_recordings=0,
            pre_recording_buffer_duration=STT_PREROLL_SECONDS,
            enable_realtime_transcription=True,
            use_main_model_for_realtime=True,
            on_realtime_transcription_update=self._on_realtime_update,
            spinner=False,
            level=logging.WARNING,
        )

    def set_realtime_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        self._realtime_callback = callback

    def _on_realtime_update(self, text: str) -> None:
        if self._realtime_callback is not None:
            self._realtime_callback(text)

    def start(self) -> None:
        self._recorder.start()
        print("🎙 Listening...")

    def stop(self) -> None:
        self._recorder.stop()

    def text(self) -> str:
        return self._recorder.text()
