import faulthandler
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ops.logging_setup import configure_logging
from stt.stream_stt import StreamSTT


def wait_until(predicate: Callable[[], bool], timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("Audio check timed out")
        time.sleep(.02)


def feed(recorder: Any, samples: Any) -> None:
    for offset in range(0, len(samples), 512):
        recorder.feed_audio(samples[offset:offset + 512])
        time.sleep(512 / 16000)


def check() -> None:
    import numpy as np
    import soundfile as sf
    from importlib.util import find_spec

    configure_logging()
    spec = find_spec("RealtimeSTT")
    sample, rate = sf.read(Path(spec.origin).parent / "assets" / "warmup_audio.wav", dtype="float32")
    if rate != 16000 or sample.ndim != 1:
        raise RuntimeError("Expected a mono 16 kHz sample")
    audio = (np.concatenate([np.zeros(4800, dtype=np.float32), sample, np.zeros(32000, dtype=np.float32)]) * 32767).astype(np.int16)
    stt = StreamSTT(use_microphone=False)
    children = []

    def track() -> Any:
        recorder = stt._recorder
        children.append(recorder.transcript_process)
        return recorder

    started = time.monotonic()
    try:
        stt.start()
        recorder = track()
        feed(recorder, audio)
        stt.stop()
        text = stt.text()
        if not text.strip():
            raise AssertionError("Dictation returned no text")
        print(json.dumps({"check": "dictation", "text": text, "elapsed_s": round(time.monotonic() - started, 2)}), flush=True)

        turns: list[str] = []
        stt.start_session(turns.append)
        recorder = track()
        for number in (1, 2):
            if number == 2:
                stt.resume()
            wait_until(lambda: recorder.state == "listening")
            feed(recorder, audio)
            wait_until(lambda: len(turns) == number)
            if not stt._paused.is_set() or not turns[-1].strip():
                raise AssertionError("Session did not pause after transcription")
            print(json.dumps({"check": "jarvis_turn", "number": number, "text": turns[-1]}), flush=True)
        stt.stop_session()
        if stt.owner is not None or stt._session_thread.is_alive():
            raise AssertionError("Paused session failed to stop")

        stt.start_session(lambda text: None)
        track()
        wait_until(lambda: stt._recorder.state == "listening")
        stt.stop_session()
        print(json.dumps({"check": "stop_while_listening", "passed": stt.owner is None}), flush=True)

        stt.start()
        track()
        stt.stop()
        if stt.text():
            raise AssertionError("Empty dictation unexpectedly transcribed")
        if any(process.is_alive() for process in children):
            raise AssertionError("Recorder child process was left running")
        print(json.dumps({"check": "repeated_mode_switch_and_cleanup", "passed": True, "elapsed_s": round(time.monotonic() - started, 2)}), flush=True)
    finally:
        stt.shutdown()


def check_microphone() -> None:
    stt = StreamSTT()
    children = []
    try:
        for mode in ("jarvis", "dictation", "jarvis"):
            started = time.monotonic()
            recorder = stt._create(mode)
            children.append(recorder.transcript_process)
            chunks = []
            recorder.on_recorded_chunk = lambda data: chunks.append(len(data))
            stt._paused.clear()
            stt._start_capture(recorder)
            wait_until(lambda: len(chunks) >= 10, timeout=5)
            stt.shutdown()
            if stt.owner is not None or any(child.is_alive() for child in children):
                raise AssertionError("Audio resources were left running")
            print(json.dumps({"check": "microphone_startup_and_cleanup", "mode": mode, "chunks": len(chunks),
                              "elapsed_s": round(time.monotonic() - started, 2), "passed": True}), flush=True)
    finally:
        stt.shutdown()


if __name__ == "__main__":
    faulthandler.dump_traceback_later(90)
    try:
        check()
    finally:
        faulthandler.cancel_dump_traceback_later()
