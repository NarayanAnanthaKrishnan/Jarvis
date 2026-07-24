import threading
import time
from shutil import get_terminal_size
from stt.stream_stt import StreamSTT
from llm.client import LLMClient
from tts.speaker import Speaker
from tts.stream_tts import StreamingSpeaker
from agent.loop import run_agent
from config import STREAMING_ENABLED, AUTO_EXTRACT
from memory.retrieval_gate import needs_memory
from ops.tracer import trace


class JarvisMode:
    def __init__(self, stream_stt: StreamSTT, llm: LLMClient, speaker: Speaker, streaming_speaker: StreamingSpeaker) -> None:
        self.stream_stt = stream_stt
        self.llm = llm
        self.speaker = speaker
        self.streaming_speaker = streaming_speaker
        self.session_active: bool = False
        self._conversation_history: list[str] = []
        self._lock: threading.Lock = threading.Lock()

    def toggle_session(self) -> None:
        if not self.session_active:
            self._start_session()
        else:
            self._end_session()

    def _start_session(self) -> None:
        self.session_active = True
        self._conversation_history = []
        print("\n🗣  Session started. Press CTRL+SHIFT+J to end.")
        self.stream_stt.set_realtime_callback(self._on_partial)
        self.stream_stt.start_session(self._on_turn)

    def _on_partial(self, text: str) -> None:
        cols = get_terminal_size().columns
        msg = f"🎤 {text}"
        print(f"\r{msg:<{cols}}", end="", flush=True)

    def _on_turn(self, text: str) -> None:
        cols = get_terminal_size().columns
        print(f"\r{'':<{cols}}", end="", flush=True)
        print(f"🗣  You: {text}")

        if not text.strip():
            return

        self.stream_stt.pause()
        start_ts = time.time()

        try:
            if text.lower() in ("start fresh", "new session", "clear context", "forget everything"):
                self.llm.rotate_session()
                reply = "Started a fresh session."
                if STREAMING_ENABLED:
                    def _gen():
                        yield reply
                    self.streaming_speaker.speak_stream(_gen())
                else:
                    self.speaker.speak(reply)
                with self._lock:
                    self._conversation_history.append(f"User: {text}")
                    self._conversation_history.append(f"Assistant: {reply}")
                    self._conversation_history = self._conversation_history[-8:]
                trace("final", delivery="speak", output_preview="start fresh")
                trace("turn_end", steps=0, elapsed_s=round(time.time() - start_ts, 2))
                self.stream_stt.resume()
                return

            result = run_agent(text, self._conversation_history, self.llm)
            reply = result["output"]
            delivery = result.get("delivery", "speak")
            stream = result.get("stream")

            if stream is not None:
                reply = self.streaming_speaker.speak_stream(stream)

            elif delivery in ("speak", "both"):
                if STREAMING_ENABLED:
                    def _gen():
                        yield reply
                    self.streaming_speaker.speak_stream(_gen())
                else:
                    self.speaker.speak(reply)

            if delivery in ("paste", "both") and reply:
                from tools.registry import execute_tool
                execute_tool("paste_at_cursor", {"text": reply})

            print(f"🤖 Jarvis: {reply}")
            with self._lock:
                self._conversation_history.append(f"User: {text}")
                self._conversation_history.append(f"Assistant: {reply}")
                self._conversation_history = self._conversation_history[-8:]

        except Exception as e:
            print(f"  [X] Agent loop error: {e}, falling back to direct chat")
            trace("error", where="jarvis_loop", message=str(e))
            try:
                if needs_memory(text, self.llm):
                    self.llm.refresh_memories(text)
                reply = self.llm.chat(text)
                if reply.strip():
                    if STREAMING_ENABLED:
                        def _gen():
                            yield reply
                        self.streaming_speaker.speak_stream(_gen())
                    else:
                        self.speaker.speak(reply)
                print(f"🤖 Jarvis: {reply}")
                trace("final", delivery="speak", output_preview=reply[:200])
                trace("turn_end", steps=0, elapsed_s=round(time.time() - start_ts, 2))
                with self._lock:
                    self._conversation_history.append(f"User: {text}")
                    self._conversation_history.append(f"Assistant: {reply}")
                    self._conversation_history = self._conversation_history[-8:]
            except Exception as e2:
                print(f"  [X] Fallback also failed: {e2}")
                trace("error", where="jarvis_fallback", message=str(e2))

        if AUTO_EXTRACT and text.strip():
            from memory.extractor import extract_and_store
            threading.Thread(
                target=extract_and_store,
                args=(text, self.llm),
                daemon=True
            ).start()

        self.stream_stt.resume()

    def _end_session(self) -> None:
        self.session_active = False
        self.stream_stt.set_realtime_callback(None)
        self.stream_stt.stop_session()
        self.streaming_speaker.stop()
        self.speaker.stop()
        try:
            from memory.session import save_session
            saved = save_session(self.llm, self.llm.history)
            if saved:
                print("  💾 Session saved to long-term memory.")
        except Exception:
            pass
        print("🗣  Session ended.")
