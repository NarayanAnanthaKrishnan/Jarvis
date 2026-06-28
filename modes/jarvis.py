import threading
import time
from stt.stream_stt import StreamSTT
from llm.client import LLMClient
from tts.speaker import Speaker
from agent.planner import create_plan
from agent.executor import execute_plan
from agent.summarizer import summarize


class JarvisMode:
    def __init__(self, stream_stt: StreamSTT, llm: LLMClient, speaker: Speaker):
        self.stream_stt = stream_stt
        self.llm = llm
        self.speaker = speaker
        self._is_recording = False
        self._lock = threading.Lock()
        self.conversation_history: list[str] = []

    def on_activate(self):
        if self._is_recording:
            return
        self.speaker.stop()
        self._is_recording = True
        self.stream_stt.start()

    def on_release(self):
        if not self._is_recording:
            return
        self._is_recording = False
        self.stream_stt.stop()
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _truncate_response(self, text: str, max_chars: int = 400) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        for sep in (". ", "! ", "? "):
            idx = truncated.rfind(sep)
            if idx > max_chars // 2:
                return truncated[:idx + 1]
        return truncated[:max_chars - 3] + "..."

    def _run_pipeline(self):
        try:
            t0 = time.time()

            text = self.stream_stt.text()
            if not text:
                print("⚠ Could not understand audio.")
                return
            t1 = time.time()

            print(f"\n🗣 You: {text}")
            print("  🤔 Planning...")

            plan = create_plan(text, self.llm, conversation_history=self.conversation_history)
            t2 = time.time()

            if plan.get("intent") == "new_session":
                summary = self.llm.rotate_session()
                reply = "Started a fresh session."
                if summary:
                    reply = f"Started a fresh session."
                llm_end = time.time()

            elif plan.get("intent") == "chat":
                reply = plan.get("message", "")
                if not reply:
                    self.llm.refresh_memories(text)
                    reply = self.llm.chat(text)
                llm_end = time.time()
            else:
                steps = plan.get("plan", [])
                if steps:
                    print(f"  📋 Plan: {[s['tool'] for s in steps]}")
                    results = execute_plan(steps)
                    t3 = time.time()
                    print("  📝 Summarizing...")
                    reply = summarize(text, results, self.llm)
                    llm_end = time.time()
                    print(f"  ⏱  Execute: {t3-t2:.2f}s | Summarize: {llm_end-t3:.2f}s")
                else:
                    self.llm.refresh_memories(text)
                    reply = self.llm.chat(text)
                    llm_end = time.time()

            if not reply.strip():
                reply = "I could not find an answer to that."

            reply = self._truncate_response(reply, max_chars=400)
            with self._lock:
                self.conversation_history.append(text)
                if len(self.conversation_history) > 2:
                    self.conversation_history = self.conversation_history[-2:]

            print(f"🤖 Jarvis: {reply}\n")
            self.speaker.speak(reply)
            t4 = time.time()

            print(f"  ⏱  STT: {t1-t0:.2f}s | Plan: {t2-t1:.2f}s | LLM: {llm_end-t2:.2f}s | TTS: {t4-llm_end:.2f}s | TOTAL: {t4-t0:.2f}s")
        except Exception as e:
            print(f"[X] Pipeline error: {e}")
