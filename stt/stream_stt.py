import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from typing import Any

from config import STT_MODEL, STT_DEVICE, STT_PREROLL_SECONDS, SESSION_SILENCE_SECONDS, SESSION_STT_MODEL
from ops.tracer import trace


_DLL_HANDLES: list[Any] = []
_LOG = logging.getLogger("jarvis.audio")


def _add_cuda_dll_dirs() -> None:
    if _DLL_HANDLES or not hasattr(os, "add_dll_directory"):
        return
    for relative in ["nvidia/cublas/bin", "nvidia/cuda_runtime/bin", "nvidia/cudnn/bin", "nvidia/cuda_nvrtc/bin"]:
        path = os.path.join(sys.prefix, "Lib", "site-packages", relative)
        if os.path.isdir(path):
            _DLL_HANDLES.append(os.add_dll_directory(path))


class StreamSTT:
    def __init__(self, recorder_factory: Callable[..., Any] | None = None, use_microphone: bool = True) -> None:
        if STT_DEVICE == "cuda":
            _add_cuda_dll_dirs()
        self._factory = recorder_factory
        self._use_microphone = use_microphone
        self._realtime_callback: Callable[[str], None] | None = None
        self._recorder: Any = None
        self._owner: str | None = None
        self._lock = threading.RLock()
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._session_thread: threading.Thread | None = None
        self._generation = 0
        self._log_handlers: list[logging.Handler] = []
        self._input_stream: Any = None

    @property
    def owner(self) -> str | None:
        with self._lock:
            return self._owner

    def _create(self, owner: str, cancelled: threading.Event | None = None) -> Any:
        with self._lock:
            if self._owner is not None:
                raise RuntimeError(f"Microphone is already owned by {self._owner}")
            self._owner = owner
            self._stop = threading.Event()
            self._paused.set()
            self._generation += 1
            generation = self._generation
        started = time.monotonic()
        recorder = None
        before_handlers = set(logging.getLogger("realtimestt").handlers)
        print(f"Loading speech recognition for {owner}... Toggle again to cancel.", flush=True)
        _LOG.info("Recorder initializing: mode=%s model=%s", owner, SESSION_STT_MODEL if owner == "jarvis" else STT_MODEL)
        try:
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Activation cancelled")
            factory = self._factory
            if factory is None:
                from stt.recorder_runtime import create_recorder
                factory = lambda **kwargs: create_recorder(cancelled, self._stop, **kwargs)
            recorder = factory(model=SESSION_STT_MODEL if owner == "jarvis" else STT_MODEL, language="en",
                               device=STT_DEVICE, compute_type="float16" if STT_DEVICE == "cuda" else "int8",
                               use_microphone=False,
                               silero_sensitivity=0.8 if owner == "jarvis" else 1.0,
                               post_speech_silence_duration=SESSION_SILENCE_SECONDS if owner == "jarvis" else 9999,
                               min_length_of_recording=0.5 if owner == "jarvis" else 0, min_gap_between_recordings=0.3 if owner == "jarvis" else 0,
                               pre_recording_buffer_duration=STT_PREROLL_SECONDS, enable_realtime_transcription=True,
                               use_main_model_for_realtime=True,
                               on_realtime_transcription_update=lambda text: self._on_realtime_update(text, generation),
                               spinner=False, level=logging.WARNING, no_log_file=True, beam_size=1)
            self._log_handlers = [h for h in logging.getLogger("realtimestt").handlers if h not in before_handlers]
            with self._lock:
                self._recorder = recorder
            recorder.set_microphone(False)
            self._clear_audio(recorder)
            if self._stop.is_set() or (cancelled is not None and cancelled.is_set()):
                raise InterruptedError("Activation cancelled")
            _LOG.info("Recorder ready: mode=%s elapsed_s=%.2f", owner, time.monotonic() - started)
            return recorder
        except BaseException:
            if recorder is not None:
                self._dispose(recorder)
            else:
                for handler in set(logging.getLogger("realtimestt").handlers) - before_handlers:
                    logging.getLogger("realtimestt").removeHandler(handler)
                    handler.close()
                with self._lock:
                    self._owner = None
            raise

    def _start_capture(self, recorder: Any) -> None:
        if not self._use_microphone:
            return
        import sounddevice as sd

        def capture(data: Any, frames: int, time_info: Any, status: Any) -> None:
            if recorder is self._recorder and not self._paused.is_set() and not self._stop.is_set():
                recorder.feed_audio(bytes(data))

        self._input_stream = sd.RawInputStream(samplerate=16000, blocksize=512, channels=1, dtype="int16", callback=capture)
        self._input_stream.start()

    @staticmethod
    def _clear_audio(recorder: Any) -> None:
        recorder.clear_audio_queue()
        for name in ("frames", "last_frames", "last_words_buffer"):
            value = getattr(recorder, name, None)
            if isinstance(value, list) or hasattr(value, "maxlen"):
                value.clear()
        completed = getattr(recorder, "recorded_audio_queue", None)
        if isinstance(completed, queue.Queue):
            while True:
                try:
                    completed.get_nowait()
                except queue.Empty:
                    break

    def set_realtime_callback(self, callback: Callable[[str], None] | None) -> None:
        self._realtime_callback = callback

    def _on_realtime_update(self, text: str, generation: int | None = None) -> None:
        callback = self._realtime_callback
        if callback is not None and generation in (None, self._generation) and not self._paused.is_set() and not self._stop.is_set():
            callback(text)

    def start(self, cancelled: threading.Event | None = None) -> None:
        recorder = self._create("dictation", cancelled)
        try:
            if self._stop.is_set() or (cancelled is not None and cancelled.is_set()):
                raise InterruptedError("Activation cancelled")
            recorder.start()
            self._paused.clear()
            self._start_capture(recorder)
        except BaseException:
            self._dispose(recorder)
            raise
        print("Listening for dictation. CTRL+SHIFT+K finishes.", flush=True)

    def stop(self) -> None:
        with self._lock:
            self._paused.set()
            if self._recorder is not None:
                self._recorder.set_microphone(False)
                self._recorder.stop()

    def text(self) -> str:
        recorder = self._recorder
        if recorder is None:
            return ""
        try:
            pending = recorder.has_pending_recordings()
            if pending is False:
                return ""
            return recorder.text()
        finally:
            self._dispose(recorder)

    def start_session(self, on_turn_callback: Callable[[str], None], cancelled: threading.Event | None = None) -> None:
        recorder = self._create("jarvis", cancelled)
        if self._stop.is_set() or (cancelled is not None and cancelled.is_set()):
            self._dispose(recorder)
            raise InterruptedError("Activation cancelled")
        try:
            self._paused.clear()
            self._start_capture(recorder)
            self._session_thread = threading.Thread(target=self._session_loop, args=(recorder, on_turn_callback, self._stop), name="session-recorder", daemon=True)
            self._session_thread.start()
        except BaseException:
            self._dispose(recorder)
            raise

    def pause(self) -> None:
        with self._lock:
            self._paused.set()
            recorder = self._recorder
            if recorder is not None and self._owner == "jarvis":
                recorder.set_microphone(False)
                recorder.start_recording_on_voice_activity = False
                recorder.stop_recording_on_voice_deactivity = False
                recorder.continuous_listening = False
                if recorder.is_recording is True:
                    recorder.stop()

    def resume(self) -> None:
        with self._lock:
            if self._owner != "jarvis" or self._stop.is_set() or self._recorder is None:
                return
            self._clear_audio(self._recorder)
            self._paused.clear()

    def stop_session(self) -> None:
        self._stop.set()
        self._paused.set()
        self._realtime_callback = None
        recorder = self._recorder
        if recorder is not None:
            self._dispose(recorder)

    def shutdown(self) -> None:
        self.stop_session()

    def _dispose(self, recorder: Any) -> None:
        with self._lock:
            if self._recorder is not recorder:
                return
            self._owner = "stopping"
            self._recorder = None
            self._stop.set()
            self._paused.set()
            session_thread = self._session_thread
            input_stream, self._input_stream = self._input_stream, None
        started = time.monotonic()
        processes = [p for name in ("reader_process", "transcript_process") if isinstance(p := getattr(recorder, name, None), BaseProcess)]
        try:
            if input_stream is not None:
                try:
                    input_stream.stop()
                except Exception:
                    _LOG.exception("Microphone stop failed")
                try:
                    input_stream.close()
                except Exception:
                    _LOG.exception("Microphone close failed")
            recorder.interrupt_stop_event.set()
            recorder.shutdown()
        finally:
            for process in processes:
                process.join(timeout=2)
                if process.is_alive():
                    _LOG.warning("Terminating recorder child that did not stop: pid=%s", process.pid)
                    process.terminate()
                    process.join(timeout=2)
            stdout_thread = getattr(recorder, "stdout_thread", None)
            if isinstance(stdout_thread, threading.Thread):
                stdout_thread.join(timeout=1)
            stdout_pipe = getattr(recorder, "parent_stdout_pipe", None)
            if stdout_pipe is not None:
                stdout_pipe.close()
            if session_thread is not None and session_thread is not threading.current_thread():
                session_thread.join(timeout=3)
            for handler in self._log_handlers:
                logging.getLogger("realtimestt").removeHandler(handler)
                handler.close()
            self._log_handlers = []
            with self._lock:
                self._owner = None
            _LOG.info("Recorder stopped: elapsed_s=%.2f", time.monotonic() - started)

    def _session_loop(self, recorder: Any, callback: Callable[[str], None], stopped: threading.Event) -> None:
        try:
            while not stopped.is_set():
                if self._paused.is_set():
                    stopped.wait(0.05)
                    continue
                text = recorder.text()
                if stopped.is_set():
                    break
                if text and text.strip() and not self._paused.is_set():
                    self.pause()
                    callback(text.strip())
        except Exception as exc:
            if not stopped.is_set():
                _LOG.exception("Session recorder failed")
                trace("error", where="session_recorder", error_type=type(exc).__name__)
                print(f"STT session stopped: {type(exc).__name__}: {exc}. Toggle the session to restart.", flush=True)
        finally:
            self._dispose(recorder)
