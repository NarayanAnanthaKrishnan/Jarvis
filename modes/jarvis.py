import threading
from shutil import get_terminal_size
from stt.stream_stt import StreamSTT
from llm.client import LLMClient
from tts.speaker import Speaker
from tts.stream_tts import StreamingSpeaker
from agent.planner import create_plan
from agent.executor import execute_plan
from agent.summarizer import summarize, stream_summarize
from config import STREAMING_ENABLED


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

        steps: list[dict] = []

        try:
            print("  🤔 Planning...")
            plan = create_plan(text, self.llm, conversation_history=self._conversation_history)

            if plan.get("intent") == "new_session":
                self.llm.rotate_session()
                reply = "Started a fresh session."
                if STREAMING_ENABLED:
                    def _gen():
                        yield reply
                    self.streaming_speaker.speak_stream(_gen())
                else:
                    self.speaker.speak(reply)

            elif plan.get("intent") == "chat":
                reply = plan.get("message", "")
                if not reply:
                    self.llm.refresh_memories(text)
                    reply = self.llm.chat(text)
                if STREAMING_ENABLED:
                    def _gen():
                        yield reply
                    self.streaming_speaker.speak_stream(_gen())
                else:
                    self.speaker.speak(reply)
                reply_display = reply
                print(f"🤖 Jarvis: {reply_display}")
                with self._lock:
                    self._conversation_history.append(text)
                    if len(self._conversation_history) > 8:
                        self._conversation_history = self._conversation_history[-8:]

            else:
                steps = plan.get("plan", [])
                if steps:
                    print(f"  📋 Plan: {[s['tool'] for s in steps]}")
                    results = execute_plan(steps)
                    error_ctx = "; ".join(r["result"] for r in results if "Error" in r.get("result", ""))
                    if error_ctx:
                        print(f"  🔄 Tool error: {error_ctx}. Re-planning...")
                        plan = create_plan(f"{text} (note: previous tool attempt had errors: {error_ctx})", self.llm)
                        new_steps = plan.get("plan", [])
                        if new_steps:
                            print(f"  📋 Re-plan: {[s['tool'] for s in new_steps]}")
                            results = execute_plan(new_steps)
                    print("  📝 Summarizing...")

                    if STREAMING_ENABLED:
                        token_gen = stream_summarize(text, results, self.llm)
                        reply = self.streaming_speaker.speak_stream(token_gen)
                    else:
                        reply = summarize(text, results, self.llm)
                        if not reply.strip():
                            reply = "I could not find an answer to that."
                        reply = self._truncate_response(reply, max_chars=400)
                        self.speaker.speak(reply)

                    print(f"🤖 Jarvis: {reply}")
                    with self._lock:
                        self._conversation_history.append(text)
                        if len(self._conversation_history) > 8:
                            self._conversation_history = self._conversation_history[-8:]
                else:
                    self.llm.refresh_memories(text)
                    reply = self.llm.chat(text)
                    if STREAMING_ENABLED:
                        def _gen():
                            yield reply
                        self.streaming_speaker.speak_stream(_gen())
                    else:
                        self.speaker.speak(reply)
                    print(f"🤖 Jarvis: {reply}")
                    with self._lock:
                        self._conversation_history.append(text)
                        if len(self._conversation_history) > 8:
                            self._conversation_history = self._conversation_history[-8:]

        except Exception as e:
            print(f"  [X] Streaming error: {e}, falling back to blocking")
            try:
                reply = summarize(text, [], self.llm) if steps else self.llm.chat(text)
                if reply.strip():
                    self.speaker.speak(reply)
            except Exception as e2:
                print(f"  [X] Fallback also failed: {e2}")

        self.stream_stt.resume()

    def _truncate_response(self, text: str, max_chars: int = 400) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        for sep in (". ", "! ", "? "):
            idx = truncated.rfind(sep)
            if idx > max_chars // 2:
                return truncated[:idx + 1]
        return truncated[:max_chars - 3] + "..."

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
