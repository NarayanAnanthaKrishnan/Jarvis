import logging
import queue
import re
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any

import config
from ops.tracer import trace


_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"})


def _boundary(text: str, final: bool) -> int:
    for match in re.finditer(r"[.!?](?=\s|$)", text):
        end = match.end()
        if end == len(text) and not final:
            continue
        word = text[:end - 1].rsplit(maxsplit=1)[-1].lower() if text[:end - 1].strip() else ""
        if text[end - 1] == "." and (word in _ABBREVIATIONS or len(word) == 1 and word.isalpha()):
            continue
        if end <= 160:
            return end
        break
    if len(text) >= 80:
        clauses = [match.end() for match in re.finditer(r"[,;:](?=\s)", text[:160]) if match.end() >= 40]
        if clauses:
            return clauses[0]
    if len(text) > 160:
        spaces = list(re.finditer(r"\s+", text[:161]))
        if spaces:
            return spaces[-1].start()
        match = re.search(r"\s+", text[160:])
        if match:
            return 160 + match.start()
    return len(text) if final else 0


def speech_chunks(tokens: Iterable[str]) -> Iterator[str]:
    buffer = ""
    for token in tokens:
        buffer += token or ""
        while (end := _boundary(buffer, False)) > 0:
            chunk, buffer = buffer[:end].strip(), buffer[end:].lstrip()
            if chunk:
                yield chunk
    while buffer.strip():
        end = _boundary(buffer, True)
        chunk, buffer = buffer[:end].strip(), buffer[end:].lstrip()
        if chunk:
            yield chunk


class PlaybackEngine:
    def __init__(self, model: Any = None, audio: Any = None) -> None:
        self.model = model
        self.audio = audio
        self._run_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._ready: Future | None = None
        self._generation = 0
        self._cancel = threading.Event()
        self._output_stream: Any = None
        self._producer: threading.Thread | None = None

    def preload(self) -> Future:
        with self._load_lock:
            if self._ready is not None:
                return self._ready
            ready: Future = Future()
            self._ready = ready
            if self.model is not None:
                ready.set_result(self.model)
                return ready

            def load() -> None:
                started = time.monotonic()
                try:
                    from kokoro_onnx import Kokoro
                    root = Path(__file__).resolve().parent.parent
                    path = root / config.TTS_MODEL
                    if not path.is_file():
                        path = root / config.TTS_FALLBACK_MODEL
                    model = Kokoro(str(path), str(root / "voices-v1.0.bin"))
                    model.create("Ready.", voice="af_bella", speed=1.15)
                    self.model = model
                    ready.set_result(model)
                    trace("tts_init", model=path.name, elapsed_s=round(time.monotonic() - started, 3))
                except Exception as exc:
                    ready.set_exception(exc)
                    logging.getLogger("jarvis.audio").error("Speech initialization failed: %s", type(exc).__name__)

            threading.Thread(target=load, name="tts-preload", daemon=True).start()
            return ready

    def stop(self) -> None:
        with self._state_lock:
            self._generation += 1
            self._cancel.set()
            if self._output_stream is not None:
                self._output_stream.abort()

    def speak(self, tokens: Iterable[str], session_cancelled: threading.Event | None = None) -> str:
        session_cancelled = session_cancelled or threading.Event()
        with self._state_lock:
            generation = self._generation
        with self._run_lock:
            with self._state_lock:
                if generation != self._generation or session_cancelled.is_set():
                    return ""
                cancelled = threading.Event()
                self._cancel = cancelled
            started = time.monotonic()
            chunks: queue.Queue = queue.Queue(maxsize=config.TTS_QUEUE_CHUNKS)
            full: list[str] = []
            done = object()

            def stopped() -> bool:
                return cancelled.is_set() or session_cancelled.is_set()

            def put(item: Any) -> None:
                while not stopped():
                    try:
                        chunks.put(item, timeout=.05)
                        return
                    except queue.Full:
                        pass

            def record_tokens() -> Iterator[str]:
                for token in tokens:
                    if stopped():
                        break
                    if token:
                        full.append(token)
                        yield token

            def produce() -> None:
                try:
                    ready = self.preload()
                    while not stopped():
                        try:
                            model = ready.result(timeout=.05)
                            break
                        except FutureTimeout:
                            pass
                    else:
                        return
                    for index, sentence in enumerate(speech_chunks(record_tokens())):
                        with self._model_lock:
                            if stopped():
                                return
                            before = time.monotonic()
                            samples, rate = model.create(sentence, voice="af_bella", speed=1.15)
                        trace("tts_synthesis", step=index, elapsed_s=round(time.monotonic() - before, 3))
                        if stopped():
                            return
                        put((samples, rate))
                except Exception as exc:
                    put(exc)
                finally:
                    put(done)

            self._producer = threading.Thread(target=produce, name="tts-synthesis", daemon=True)
            self._producer.start()
            stream = None
            sample_rate = None
            try:
                while not stopped():
                    try:
                        item = chunks.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if item is done:
                        break
                    if isinstance(item, Exception):
                        raise item
                    samples, rate = item
                    if stopped():
                        break
                    if stream is None:
                        if self.audio is None:
                            import sounddevice
                            self.audio = sounddevice
                        stream = self.audio.OutputStream(samplerate=rate, channels=1, dtype="float32")
                        with self._state_lock:
                            if stopped() or generation != self._generation:
                                break
                            self._output_stream = stream
                            stream.start()
                        sample_rate = rate
                        trace("tts_first_audio", elapsed_s=round(time.monotonic() - started, 3))
                    elif rate != sample_rate:
                        raise RuntimeError("Speech sample rate changed during playback")
                    if stream.write(samples) is True:
                        trace("tts_underrun", status="underflow")
            except Exception:
                if not stopped():
                    raise
            finally:
                was_stopped = stopped()
                cancelled.set()
                if stream is not None:
                    try:
                        if was_stopped:
                            stream.abort()
                        else:
                            stream.stop()
                    finally:
                        with self._state_lock:
                            if self._output_stream is stream:
                                self._output_stream = None
                        stream.close()
                trace("tts_end", status="cancelled" if was_stopped else "complete", elapsed_s=round(time.monotonic() - started, 3))
            return "".join(full)
