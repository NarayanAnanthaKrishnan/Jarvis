import threading
import time
from audio.recorder import Recorder
from stt.transcriber import Transcriber
from llm.client import LLMClient
from tts.speaker import Speaker
from agent.planner import create_plan
from agent.executor import execute_plan
from agent.summarizer import summarize


class JarvisMode:
    def __init__(self, recorder: Recorder, transcriber: Transcriber, llm: LLMClient, speaker: Speaker):
        self.recorder = recorder
        self.transcriber = transcriber
        self.llm = llm
        self.speaker = speaker

    def on_activate(self):
        self.recorder.start()

    def on_release(self):
        if self.recorder.is_recording:
            self.recorder.stop()
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            t0 = time.time()

            audio = self.recorder.stop()
            if len(audio) == 0:
                return
            t1 = time.time()

            text = self.transcriber.transcribe(audio)
            if not text:
                print("⚠ Could not understand audio.")
                return
            t2 = time.time()

            print(f"\n🗣 You: {text}")
            print("  🤔 Planning...")

            plan = create_plan(text, self.llm)
            t3 = time.time()

            if plan.get("intent") == "chat":
                reply = plan.get("message", "")
                if not reply:
                    reply = self.llm.chat(text)
                llm_end = time.time()
            else:
                steps = plan.get("plan", [])
                if steps:
                    print(f"  📋 Plan: {[s['tool'] for s in steps]}")
                    results = execute_plan(steps)
                    t4 = time.time()
                    print("  📝 Summarizing...")
                    reply = summarize(text, results, self.llm)
                    llm_end = time.time()
                    print(f"  ⏱  Execute: {t4-t3:.2f}s | Summarize: {llm_end-t4:.2f}s")
                else:
                    reply = self.llm.chat(text)
                    llm_end = time.time()

            if not reply.strip():
                reply = "I could not find an answer to that."

            print(f"🤖 Jarvis: {reply}\n")
            self.speaker.speak(reply)
            t5 = time.time()

            print(f"  ⏱  Record: {t1-t0:.2f}s | STT: {t2-t1:.2f}s | Plan: {t3-t2:.2f}s | LLM: {llm_end-t3:.2f}s | TTS: {t5-llm_end:.2f}s | TOTAL: {t5-t0:.2f}s")
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
        finally:
            if self.recorder.is_recording:
                self.recorder.stop()
