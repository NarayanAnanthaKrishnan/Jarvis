import threading
import uuid
from shutil import get_terminal_size
from typing import Any

from agent.context import TurnContext
from agent.loop import run_agent
from config import STREAMING_ENABLED, AUTO_EXTRACT, TTS_PRELOAD
from email_agent.models import EmailError
from email_agent.service import confirmation_number
from email_agent.workflow import EmailWorkflow
from ops.tracer import trace
from tts.text import speech_safe_text


class JarvisMode:
    def __init__(self, stream_stt: Any, llm: Any, speaker: Any, streaming_speaker: Any) -> None:
        self.stream_stt = stream_stt
        self.llm = llm
        self.speaker = speaker
        self.streaming_speaker = streaming_speaker
        self.session_active = False
        self._history: list[dict] = []
        self._lock = threading.RLock()
        self._turn_lock = threading.Lock()
        self.transition_lock = threading.RLock()
        self._session_id = ""
        self._cancelled = threading.Event()
        self._active_draft_id: int | None = None
        self._email_service: Any = None
        self._email_workflow = EmailWorkflow()
        self._saved = False
        self._save_threads: list[threading.Thread] = []
        self._turn_thread: threading.Thread | None = None

    @property
    def _conversation_history(self) -> list[str]:
        with self._lock:
            return [f"{item['role'].title()}: {item['content']}" for item in self._history]

    def toggle_session(self) -> None:
        if self.session_active:
            self._end_session()
        else:
            self._start_session()

    def _start_session(self, cancelled: threading.Event | None = None) -> None:
        with self._lock:
            if self.stream_stt.owner is not None:
                raise RuntimeError("Microphone is busy. End dictation or wait for session cleanup.")
            self._history = []
            self._saved = False
            self._session_id = uuid.uuid4().hex
            self._cancelled = cancelled or threading.Event()
            cancelled = self._cancelled
            self._active_draft_id = None
            self._email_workflow = EmailWorkflow()
            self._turn_lock = threading.Lock()
            self.session_active = True
        try:
            self.stream_stt.set_realtime_callback(lambda text: self._on_partial(text) if not cancelled.is_set() else None)
            self.stream_stt.start_session(lambda text: self._queue_turn(text, cancelled), cancelled=cancelled)
            if cancelled.is_set():
                self.stream_stt.stop_session()
                raise InterruptedError("Activation cancelled")
            if TTS_PRELOAD:
                self.streaming_speaker.preload()
            print("\n🗣 Session started. Press CTRL+SHIFT+J to end.")
        except Exception:
            self.session_active = False
            self._cancelled.set()
            raise

    def _on_partial(self, text: str) -> None:
        if self.session_active:
            print(f"\r{'🎤 ' + text:<{get_terminal_size().columns}}", end="", flush=True)

    def _interrupt(self) -> None:
        self.streaming_speaker.stop()
        self.speaker.stop()

    def _append(self, text: str, reply: str, sensitive: bool, session_id: str) -> None:
        with self._lock:
            if self.session_active and session_id == self._session_id and not self._cancelled.is_set():
                self._history.extend([{"role": "user", "content": text, "sensitive": sensitive}, {"role": "assistant", "content": reply, "sensitive": sensitive}])

    def _dispatch_turn(self, text: str, session_id: str, cancelled: threading.Event) -> None:
        with self._lock:
            if cancelled.is_set() or session_id != self._session_id or not self.session_active:
                return
            self._turn_thread = threading.Thread(target=self._on_turn, args=(text, session_id, cancelled), name="jarvis-turn", daemon=True)
            self._turn_thread.start()

    def _queue_turn(self, text: str, cancelled: threading.Event) -> None:
        with self._lock:
            if cancelled is self._cancelled and not cancelled.is_set():
                self._dispatch_turn(text, self._session_id, cancelled)

    def _on_turn(self, text: str, session_id: str | None = None, cancelled: threading.Event | None = None) -> None:
        with self._lock:
            session_id = session_id or self._session_id
            cancelled = cancelled or self._cancelled
            if not text.strip() or not self.session_active or cancelled.is_set() or session_id != self._session_id:
                return
        with self._turn_lock:
            if cancelled.is_set() or session_id != self._session_id:
                return
            if self._email_service and hasattr(self._email_service.provider, "connection_current") and not self._email_service.provider.connection_current():
                self._email_service = None
                self._active_draft_id = None
                self._email_workflow = EmailWorkflow()
            context = TurnContext(session_id=session_id, cancelled=cancelled,
                                  active_draft_id=self._active_draft_id, email_service=self._email_service,
                                  email_workflow=EmailWorkflow(**self._email_workflow.snapshot()))
            if context.cancelled.is_set():
                return
            self.stream_stt.pause()
            print(f"\n🗣 You: {text}")
            reply = ""
            try:
                normalized = text.lower().strip().rstrip(".!?")
                if normalized in ("start fresh", "new session", "clear context", "forget everything"):
                    self._save_session()
                    if self._email_service:
                        self._email_service.end_session(self._session_id)
                    with self._lock:
                        context.check_active()
                        self._history = []
                        self.llm.reset_history()
                        self._active_draft_id = None
                        context.active_draft_id = None
                        context.email_workflow = EmailWorkflow()
                        self._email_workflow = EmailWorkflow()
                        self._session_id = uuid.uuid4().hex
                        context.session_id = self._session_id
                        self._saved = False
                    reply = "Started a fresh session."
                    self._speak(reply, context.cancelled)
                    return
                number = confirmation_number(text)
                if number is not None or normalized.startswith("confirm email"):
                    context.email_touched = True
                    if number is None or self._email_service is None:
                        reply = "Request an email preview first, then say confirm email followed by its action number."
                    else:
                        result = self._email_service.confirm(number, context.session_id, context.cancelled)
                        reply = result.message
                        context.email_workflow = EmailWorkflow()
                    context.check_active()
                    self._speak(reply, context.cancelled)
                else:
                    result = run_agent(text, self._conversation_history, self.llm, context)
                    context.check_active()
                    if result.get("preview"):
                        print(f"\n{result['preview']}\n")
                    if result.get("confirmation_id") is not None:
                        context.email_service.mark_presented(result["confirmation_id"], context.session_id)
                    reply = result["output"]
                    if result.get("stream") is not None:
                        if STREAMING_ENABLED:
                            reply = self.streaming_speaker.speak_stream(self._active_tokens(result["stream"], context), context.cancelled)
                        else:
                            reply = "".join(self._active_tokens(result["stream"], context))
                            context.check_active()
                            self._speak(reply, context.cancelled)
                    else:
                        if result.get("delivery", "speak") in ("speak", "both") and reply:
                            self._speak(reply, context.cancelled)
                        context.check_active()
                        if result.get("delivery") in ("paste", "both") and reply and not context.email_touched:
                            from tools.registry import execute_tool
                            execute_tool("paste_at_cursor", {"text": reply}, context=context)
                context.check_active()
            except InterruptedError:
                return
            except EmailError as exc:
                reply = str(exc)
                if not context.cancelled.is_set():
                    self._speak(reply, context.cancelled)
            except Exception as exc:
                trace("error", where="jarvis_turn", error_type=type(exc).__name__)
                if context.cancelled.is_set():
                    return
                if context.email_touched:
                    reply = "The email request could not complete. Check email status before retrying."
                else:
                    try:
                        messages = [{"role": "system", "content": "Answer the user briefly. You have no tools in this fallback; do not claim actions were performed."}]
                        messages += [{"role": m["role"], "content": m["content"]} for m in self._history[-8:]]
                        messages.append({"role": "user", "content": text})
                        response = self.llm.call_raw(messages, max_tokens=300)
                        reply = response["message"]["content"] if response else "The model is unavailable right now."
                    except Exception:
                        reply = "The request could not complete right now."
                if not context.cancelled.is_set():
                    self._speak(reply, context.cancelled)
            finally:
                with self._lock:
                    if not context.cancelled.is_set() and self.session_active and context.session_id == self._session_id:
                        self._email_service = context.email_service
                        self._active_draft_id = context.active_draft_id
                        self._email_workflow = context.email_workflow
                        self.stream_stt.resume()
            if reply and not context.cancelled.is_set():
                print(f"🤖 Jarvis: {reply}")
                self._append(text, reply, context.email_touched, context.session_id)
                if AUTO_EXTRACT and not context.email_touched:
                    from memory.extractor import extract_and_store
                    threading.Thread(target=extract_and_store, args=(text, self.llm), daemon=True).start()

    @staticmethod
    def _active_tokens(tokens: Any, context: TurnContext) -> Any:
        for token in tokens:
            context.check_active()
            yield token

    def _speak(self, text: str, cancelled: threading.Event | None = None) -> None:
        cancelled = cancelled or self._cancelled
        if not self.session_active or cancelled.is_set() or cancelled is not self._cancelled:
            return
        text = speech_safe_text(text)
        if STREAMING_ENABLED:
            self.streaming_speaker.speak_stream(iter([text]), cancelled)
        else:
            self.speaker.speak(text, cancelled)

    def announce(self, text: str) -> bool:
        if not self.transition_lock.acquire(blocking=False):
            return False
        if self.stream_stt.owner in ("dictation", "stopping") or not self._turn_lock.acquire(blocking=False):
            self.transition_lock.release()
            return False
        try:
            active = self.session_active
            if active:
                self.stream_stt.pause()
            self.speaker.speak(text)
            if active and self.session_active:
                self.stream_stt.resume()
            return True
        finally:
            self._turn_lock.release()
            self.transition_lock.release()

    def _save_session(self) -> None:
        with self._lock:
            if self._saved:
                return
            self._saved = True
            history = [{"role": m["role"], "content": m["content"]} for m in self._history if not m.get("sensitive")]
        if history:
            def save() -> None:
                try:
                    from memory.session import save_session
                    save_session(self.llm, history)
                except Exception as exc:
                    trace("error", where="session_save", error_type=type(exc).__name__)
            thread = threading.Thread(target=save, name="session-save", daemon=True)
            self._save_threads = [t for t in self._save_threads if t.is_alive()]
            self._save_threads.append(thread)
            thread.start()

    def _end_session(self) -> None:
        with self._lock:
            if not self.session_active:
                return
            self.session_active = False
            self._cancelled.set()
        self._interrupt()
        self.stream_stt.stop_session()
        if self._email_service:
            self._email_service.end_session(self._session_id)
        self._save_session()
        print("🗣 Session ended.")

    def shutdown(self) -> None:
        self._end_session()
        self._save_session()
        self.stream_stt.shutdown()
        for thread in self._save_threads:
            thread.join(timeout=1)
