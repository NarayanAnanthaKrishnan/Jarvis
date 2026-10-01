import logging
import threading
import time
from typing import Any


_LOG = logging.getLogger("jarvis.audio")


def _signal_stop(recorder: Any) -> None:
    recorder.is_running = False
    recorder.is_recording = False
    for name in ("shutdown_event", "interrupt_stop_event", "main_transcription_ready_event", "start_recording_event", "stop_recording_event"):
        event = getattr(recorder, name, None)
        if event is not None:
            event.set()


def _cleanup_failed_start(recorder: Any) -> None:
    _signal_stop(recorder)
    recorder.is_shut_down = True
    for name in ("reader_process", "transcript_process"):
        process = getattr(recorder, name, None)
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
    for name in ("recording_thread", "realtime_thread", "stdout_thread"):
        thread = getattr(recorder, name, None)
        if thread is not None and thread.ident is not None:
            thread.join(timeout=2)
    for name in ("parent_transcription_pipe", "parent_stdout_pipe"):
        pipe = getattr(recorder, name, None)
        if pipe is not None:
            pipe.close()


def create_recorder(cancelled: threading.Event | None, stopped: threading.Event, timeout: float = 60,
                    recorder_type: type | None = None, **kwargs: Any) -> Any:
    if recorder_type is None:
        from RealtimeSTT import AudioToTextRecorder
        recorder_type = AudioToTextRecorder

    recorder = recorder_type.__new__(recorder_type)
    finished = threading.Event()
    failures: list[Exception] = []
    deadline = time.monotonic() + timeout

    def supervise() -> None:
        while not finished.wait(.05):
            if not failures:
                if stopped.is_set() or (cancelled is not None and cancelled.is_set()):
                    failures.append(InterruptedError("Activation cancelled"))
                elif time.monotonic() >= deadline:
                    failures.append(TimeoutError(f"Speech recognition did not start within {timeout:g} seconds. Check .logs/jarvis.log and the model files."))
                else:
                    process = getattr(recorder, "transcript_process", None)
                    exit_code = process.exitcode if process is not None else None
                    if exit_code is not None:
                        failures.append(RuntimeError(f"Speech recognition worker exited during startup (code 0x{exit_code & 0xffffffff:08X})."))
            if failures:
                _signal_stop(recorder)

    watchdog = threading.Thread(target=supervise, name="recorder-startup-watchdog", daemon=True)
    watchdog.start()
    try:
        recorder_type.__init__(recorder, **kwargs)
        if failures:
            raise failures[0]
        if stopped.is_set() or (cancelled is not None and cancelled.is_set()):
            raise InterruptedError("Activation cancelled")
        return recorder
    except BaseException:
        finished.set()
        watchdog.join(timeout=1)
        try:
            _cleanup_failed_start(recorder)
        except Exception:
            _LOG.exception("Failed to clean up recorder initialization")
        raise
    finally:
        finished.set()
        watchdog.join(timeout=1)
