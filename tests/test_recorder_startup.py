import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from stt.recorder_runtime import create_recorder
from stt.stream_stt import StreamSTT


def test_windows_spawn_does_not_import_application_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    code = "import runpy, sys; runpy.run_path('main.py', run_name='__mp_main__'); assert not ({'modes.jarvis', 'tools.registry', 'winocr', 'keyboard'} & sys.modules.keys())"
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, timeout=15)


def test_loading_agent_tools_does_not_load_windows_ocr() -> None:
    root = Path(__file__).resolve().parents[1]
    code = "import sys; import modes.jarvis; assert 'winocr' not in sys.modules; import onnxruntime"
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, timeout=15)


@pytest.mark.parametrize("reason", ["worker_crash", "cancel", "timeout"])
def test_startup_wait_exits_on_worker_crash_cancellation_or_timeout(reason: str) -> None:
    cancelled, stopped = threading.Event(), threading.Event()
    process = Mock(pid=123, exitcode=0xC0000005 if reason == "worker_crash" else None)
    process.is_alive.return_value = reason != "worker_crash"
    ready, shutdown = threading.Event(), threading.Event()

    class WaitingRecorder:
        def __init__(self, **kwargs: object) -> None:
            self.transcript_process = process
            self.main_transcription_ready_event = ready
            self.shutdown_event = shutdown
            if reason == "cancel":
                cancelled.set()
            assert ready.wait(2), "Constructor remained stuck waiting for the worker"

    error = {"worker_crash": RuntimeError, "cancel": InterruptedError, "timeout": TimeoutError}[reason]
    with pytest.raises(error):
        create_recorder(cancelled, stopped, timeout=.15, recorder_type=WaitingRecorder)
    assert shutdown.is_set()
    process.join.assert_called_once()
    assert process.terminate.call_count == (reason != "worker_crash")


def test_microphone_capture_starts_after_model_and_drops_paused_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    import sounddevice

    released = threading.Event()
    recorder, stream = Mock(), Mock()
    recorder.text.side_effect = lambda: (released.wait(2), "")[1]
    recorder.shutdown.side_effect = released.set
    input_stream = Mock(return_value=stream)
    monkeypatch.setattr(sounddevice, "RawInputStream", input_stream)

    def factory(**kwargs: object) -> Mock:
        assert kwargs["use_microphone"] is False
        input_stream.assert_not_called()
        return recorder

    stt = StreamSTT(recorder_factory=factory)
    stt.start_session(lambda text: None)
    callback = input_stream.call_args.kwargs["callback"]
    data = bytes(1024)
    try:
        stream.start.assert_called_once()
        callback(data, 512, None, None)
        assert recorder.feed_audio.call_count == 1
        stt.pause()
        callback(data, 512, None, None)
        assert recorder.feed_audio.call_count == 1
        stt.resume()
        callback(data, 512, None, None)
        assert recorder.feed_audio.call_count == 2
    finally:
        stt.shutdown()
    callback(data, 512, None, None)
    assert recorder.feed_audio.call_count == 2
    stream.stop.assert_called_once()
    stream.close.assert_called_once()


def test_microphone_open_failure_releases_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    import sounddevice

    monkeypatch.setattr(sounddevice, "RawInputStream", Mock(side_effect=RuntimeError("device unavailable")))
    recorder = Mock()
    stt = StreamSTT(recorder_factory=Mock(return_value=recorder))
    with pytest.raises(RuntimeError, match="device unavailable"):
        stt.start()
    assert stt.owner is None
    recorder.shutdown.assert_called_once()
