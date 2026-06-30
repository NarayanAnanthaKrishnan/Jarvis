import os
import sys
import time
import threading
import logging
from typing import Optional, Callable
from RealtimeSTT import AudioToTextRecorder
from config import STT_MODEL, STT_DEVICE, STT_PREROLL_SECONDS, SESSION_SILENCE_SECONDS, SESSION_STT_MODEL


def _add_cuda_dll_dirs() -> None:
    site = sys.prefix + "/Lib/site-packages"
    for rel in ["nvidia/cublas/bin", "nvidia/cuda_runtime/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"]:
        d = os.path.join(site, rel)
        if os.path.isdir(d):
            os.add_dll_directory(d)


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

    def start_session(self, on_turn_callback: Callable[[str], None]) -> None:
        self._on_turn_callback = on_turn_callback
        self._session_active = True
        self._session_paused = False
        self._session_recorder = AudioToTextRecorder(
            model=SESSION_STT_MODEL,
            language="en",
            device=STT_DEVICE,
            compute_type="float16" if STT_DEVICE == "cuda" else "int8",
            silero_sensitivity=0.8,
            post_speech_silence_duration=SESSION_SILENCE_SECONDS,
            min_length_of_recording=0.5,
            min_gap_between_recordings=0.3,
            pre_recording_buffer_duration=STT_PREROLL_SECONDS,
            enable_realtime_transcription=True,
            use_main_model_for_realtime=True,
            on_realtime_transcription_update=self._on_realtime_update,
            spinner=False,
            level=logging.WARNING,
        )
        threading.Thread(target=self._session_loop, daemon=True).start()

    def pause(self) -> None:
        self._session_paused = True
        if self._session_recorder:
            self._session_recorder.stop()

    def resume(self) -> None:
        self._session_paused = False

    def stop_session(self) -> None:
        self._session_active = False
        if self._session_recorder:
            self._session_recorder.stop()
            self._session_recorder = None
        self._on_turn_callback = None

    def _session_loop(self) -> None:
        while self._session_active:
            if self._session_paused:
                time.sleep(0.1)
                continue
            self._session_recorder.start()
            text = self._session_recorder.text()
            if not self._session_active:
                break
            if self._session_paused:
                continue
            if text and text.strip():
                self._on_turn_callback(text.strip())
