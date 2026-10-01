import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModeActions:
    start: Callable[[threading.Event], None]
    stop: Callable[[], None]
    cancel_on_stop: bool = True


class ModeController:
    def __init__(self, actions: dict[str, ModeActions], transition_lock: threading.RLock) -> None:
        self.actions = actions
        self.transition_lock = transition_lock
        self._lock = threading.Lock()
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._desired: str | None = None
        self._active: str | None = None
        self._cancelled = threading.Event()
        self._loading = False
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="mode-controller", daemon=True)
        self._thread.start()

    def request_toggle(self, mode: str) -> None:
        with self._lock:
            if self._closed or mode not in self.actions:
                return
            if self._desired is not None and self._desired != mode:
                self._commands.put("busy")
                return
            if self._desired == mode:
                self._desired = None
                if self._loading or self.actions[mode].cancel_on_stop:
                    self._cancelled.set()
            else:
                self._desired = mode
                self._cancelled = threading.Event()
                self._loading = True
            self._commands.put("update")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._desired = None
            self._cancelled.set()
            self._commands.put("update")
        self._thread.join(timeout=20)
        if self._thread.is_alive():
            logging.getLogger("jarvis.audio").error("Mode cleanup is still waiting for recorder initialization")

    def _run(self) -> None:
        while True:
            command = self._commands.get()
            if command == "busy":
                print("Microphone is busy. End the current mode before starting the other one.", flush=True)
                continue
            while True:
                with self._lock:
                    desired, cancelled = self._desired, self._cancelled
                    active, closed = self._active, self._closed
                if active == desired:
                    if closed:
                        return
                    break
                try:
                    with self.transition_lock:
                        if active is not None:
                            self.actions[active].stop()
                            with self._lock:
                                self._active = None
                            continue
                        if desired is not None and not cancelled.is_set():
                            self.actions[desired].start(cancelled)
                            with self._lock:
                                self._active = desired
                                if self._cancelled is cancelled:
                                    self._loading = False
                except Exception as exc:
                    if isinstance(exc, InterruptedError):
                        print("Mode activation cancelled.", flush=True)
                    else:
                        logging.getLogger("jarvis.audio").exception("Mode %s failed", desired or active)
                        print(f"Mode failed: {type(exc).__name__}: {exc}. Details: .logs/jarvis.log", flush=True)
                    with self._lock:
                        self._active = None
                        if self._cancelled is cancelled:
                            self._desired = None
                            self._loading = False
