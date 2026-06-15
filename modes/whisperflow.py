import threading
import pyautogui
import pyperclip
from audio.recorder import Recorder
from stt.transcriber import Transcriber


class WhisperFlowMode:
    def __init__(self, recorder: Recorder, transcriber: Transcriber):
        self.recorder = recorder
        self.transcriber = transcriber

    def on_activate(self):
        self.recorder.start()

    def on_release(self):
        if self.recorder.is_recording:
            self.recorder.stop()
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            audio = self.recorder.stop()
            if len(audio) < self.recorder.sample_rate * 0.5:
                return

            full_text = self.transcriber.transcribe(audio)
            if not full_text:
                return

            pyperclip.copy(full_text)
            pyautogui.hotkey("ctrl", "v")
        except Exception as e:
            print(f"❌ WhisperFlow error: {e}")
        finally:
            if self.recorder.is_recording:
                self.recorder.stop()
