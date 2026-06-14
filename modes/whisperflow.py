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
        audio = self.recorder.stop()
        if len(audio) < self.recorder.sample_rate * 0.5:
            return

        full_text = self.transcriber.transcribe(audio)
        if not full_text:
            return

        pyperclip.copy(full_text)
        pyautogui.hotkey("ctrl", "v")
