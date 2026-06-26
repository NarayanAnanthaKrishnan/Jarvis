import atexit
import keyboard
from audio.recorder import Recorder
from stt.transcriber import Transcriber
from llm.client import LLMClient
from tts.speaker import Speaker
from config import HOTKEY, WHISPERFLOW_HOTKEY, ULTRA_HOTKEY, SAMPLE_RATE
from modes.jarvis import JarvisMode
from modes.whisperflow import WhisperFlowMode
from modes.ultra import UltraMode

recorder = Recorder(sample_rate=SAMPLE_RATE)
transcriber = Transcriber()
llm = LLMClient()
speaker = Speaker()

jarvis = JarvisMode(recorder, transcriber, llm, speaker)
whisperflow = WhisperFlowMode(recorder, transcriber)
ultra = UltraMode(recorder, transcriber, llm)


@atexit.register
def _save_session():
    try:
        from memory.session import save_session
        saved = save_session(llm, llm.history)
        if saved:
            print(f"  💾 Session saved to long-term memory.")
    except Exception:
        pass


print("✅ Jarvis running. Hold CTRL+SHIFT+J to chat, CTRL+SHIFT+K to dictate, CTRL+SHIFT+U for Ultra mode.")

keyboard.add_hotkey(HOTKEY, jarvis.on_activate, suppress=False, trigger_on_release=False)
keyboard.add_hotkey(HOTKEY, jarvis.on_release, suppress=True, trigger_on_release=True)

keyboard.add_hotkey(WHISPERFLOW_HOTKEY, whisperflow.on_activate, suppress=False, trigger_on_release=False)
keyboard.add_hotkey(WHISPERFLOW_HOTKEY, whisperflow.on_release, suppress=True, trigger_on_release=True)

keyboard.add_hotkey(ULTRA_HOTKEY, ultra.on_activate, suppress=False, trigger_on_release=False)
keyboard.add_hotkey(ULTRA_HOTKEY, ultra.on_release, suppress=True, trigger_on_release=True)

keyboard.wait()
