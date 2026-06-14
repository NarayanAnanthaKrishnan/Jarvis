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
        audio = self.recorder.stop()
        if len(audio) == 0:
            return

        text = self.transcriber.transcribe(audio)
        if not text:
            print("⚠ Could not understand audio.")
            return

        print(f"\n🗣 You: {text}")
        print("  🤔 Planning...")

        plan = create_plan(text, self.llm)

        if plan.get("intent") == "chat":
            reply = plan.get("message", "")
            if not reply:
                reply = self.llm.chat(text)
        else:
            steps = plan.get("plan", [])
            if steps:
                print(f"  📋 Plan: {[s['tool'] for s in steps]}")
                results = execute_plan(steps)
                print("  📝 Summarizing...")
                reply = summarize(text, results, self.llm)
            else:
                reply = self.llm.chat(text)

        if not reply.strip():
            reply = "I could not find an answer to that."

        print(f"🤖 Jarvis: {reply}\n")
        self.speaker.speak(reply)
