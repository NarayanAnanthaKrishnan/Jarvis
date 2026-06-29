import atexit
import keyboard
from stt.stream_stt import StreamSTT
from llm.client import LLMClient
from tts.speaker import Speaker
from tts.stream_tts import StreamingSpeaker
from config import HOTKEY, WHISPERFLOW_HOTKEY, ULTRA_HOTKEY
from modes.jarvis import JarvisMode
from modes.whisperflow import WhisperFlowMode
from modes.ultra import UltraMode


def main() -> None:
    stream_stt = StreamSTT()
    llm = LLMClient()
    speaker = Speaker()
    streaming_speaker = StreamingSpeaker()

    jarvis = JarvisMode(stream_stt, llm, speaker, streaming_speaker)
    whisperflow = WhisperFlowMode(stream_stt)
    ultra = UltraMode(stream_stt, llm)

    def _save_session() -> None:
        try:
            from memory.session import save_session
            saved = save_session(llm, llm.history)
            if saved:
                print("  💾 Session saved to long-term memory.")
        except Exception:
            pass

    atexit.register(_save_session)

    print("✅ Jarvis running. Press CTRL+SHIFT+J to toggle session, press CTRL+SHIFT+K to toggle dictation, CTRL+SHIFT+U for Ultra mode.")

    keyboard.add_hotkey(HOTKEY, jarvis.toggle_session, suppress=True)

    keyboard.add_hotkey(WHISPERFLOW_HOTKEY, whisperflow.toggle, suppress=True)

    keyboard.add_hotkey(ULTRA_HOTKEY, ultra.on_activate, suppress=False, trigger_on_release=False)
    keyboard.add_hotkey(ULTRA_HOTKEY, ultra.on_release, suppress=True, trigger_on_release=True)

    keyboard.wait()


if __name__ == "__main__":
    main()
