import threading
import time
import numpy as np
import pyautogui
import pyperclip
from audio.recorder import Recorder
from stt.transcriber import Transcriber
from llm.client import LLMClient
from llm.prompts import ULTRA_SYSTEM_PROMPT
from tools.screen_ocr import capture_and_ocr
from tools.web_search import search_web
from tools.profile_loader import load_profile


class UltraMode:
    def __init__(self, recorder: Recorder, transcriber: Transcriber, llm: LLMClient):
        self.recorder = recorder
        self.transcriber = transcriber
        self.llm = llm

    def on_activate(self):
        self.recorder.start()

    def on_release(self):
        audio: np.ndarray = np.array([])
        if self.recorder.is_recording:
            audio = self.recorder.stop()
        if len(audio) >= self.recorder.sample_rate * 0.5:
            threading.Thread(target=self._run_pipeline, args=(audio,), daemon=True).start()

    def _run_pipeline(self, audio: np.ndarray):
        try:
            t0 = time.time()

            user_text = self.transcriber.transcribe(audio)
            if not user_text:
                print("⚠ Could not understand audio.")
                return
            t1 = time.time()
            print(f"\n🗣 You: {user_text}")

            print("  📸 OCR-ing screen...")
            ocr_text = capture_and_ocr()
            t2 = time.time()
            print(f"  📄 Screen text ({len(ocr_text)} chars)")

            company = self._extract_company(user_text, ocr_text)
            search_result = None
            if company:
                print(f"  🔍 Searching for: {company}")
                search_result = search_web(f"{company} company overview", num_results=3)
                t3 = time.time()
                print(f"  ✅ Search done ({len(search_result)} chars)")

            print("  ✍️ Generating...")
            messages = self._build_messages(user_text, ocr_text, search_result)
            response = self.llm.call_raw(messages, temp=0.3, max_tokens=2000)
            t4 = time.time()

            if response is None:
                print("[X] LLM call failed.")
                return

            reply = response["message"]["content"].strip()
            if not reply:
                print("⚠ Empty response from LLM.")
                return

            pyperclip.copy(reply)
            pyautogui.hotkey("ctrl", "v")
            t5 = time.time()

            print(f"✅ Pasted ({len(reply)} chars)")
            print(f"  ⏱  STT: {t1-t0:.2f}s | OCR: {t2-t1:.2f}s | "
                  f"Search: {t3-t2 if company else 0:.2f}s | Gen: {t4-t3 if company else t4-t2:.2f}s | "
                  f"Paste: {t5-t4:.2f}s | TOTAL: {t5-t0:.2f}s")

        except Exception as e:
            print(f"[X] Ultra mode error: {e}")

    def _extract_company(self, user_text: str, ocr_text: str) -> str | None:
        keywords = [" at ", " for ", " company ", " corp ", " inc "]
        for kw in keywords:
            if kw in user_text.lower():
                words = user_text.split()
                for i, w in enumerate(words):
                    if w.lower().strip() == kw.strip():
                        if i + 1 < len(words):
                            return words[i + 1].strip(".,!?")
        return None

    def _build_messages(self, user_text: str, ocr_text: str, search_result: str | None) -> list[dict]:
        parts = []
        profile = load_profile()
        if profile:
            parts.append(f"About the user (use for personal tasks like cover letters, emails, applications; ignore for generic tasks):\n{profile}\n")
        parts.append(f"Screen content:\n{ocr_text}\n")
        if search_result:
            parts.append(f"Company info from web:\n{search_result}\n")
        parts.append(f"User instruction: {user_text}")
        content = "\n".join(parts)
        return [
            {"role": "system", "content": ULTRA_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
