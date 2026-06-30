import atexit
import threading
import time

import keyboard
from stt.stream_stt import StreamSTT
from llm.client import LLMClient
from tts.speaker import Speaker
from tts.stream_tts import StreamingSpeaker
from config import HOTKEY, WHISPERFLOW_HOTKEY, ULTRA_HOTKEY, REMINDER_CHECK_SECONDS
from modes.jarvis import JarvisMode
from modes.whisperflow import WhisperFlowMode
from modes.ultra import UltraMode


def reminder_loop(speaker: Speaker) -> None:
    from memory.reminders import get_due_reminders, mark_fired
    while True:
        try:
            for r in get_due_reminders():
                speaker.speak(f"Reminder: {r['message']}")
                try:
                    from plyer import notification
                    notification.notify(
                        title="Jarvis Reminder",
                        message=r["message"],
                        timeout=10
                    )
                except Exception:
                    pass
                mark_fired(r["id"])
        except Exception:
            pass
        time.sleep(REMINDER_CHECK_SECONDS)


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

    threading.Thread(target=reminder_loop, args=(speaker,), daemon=True).start()

    print("✅ Jarvis running. Press CTRL+SHIFT+J to toggle session, press CTRL+SHIFT+K to toggle dictation, CTRL+SHIFT+U for Ultra mode.")

    keyboard.add_hotkey(HOTKEY, jarvis.toggle_session, suppress=True)

    keyboard.add_hotkey(WHISPERFLOW_HOTKEY, whisperflow.toggle, suppress=True)

    keyboard.add_hotkey(ULTRA_HOTKEY, ultra.on_activate, suppress=False, trigger_on_release=False)
    keyboard.add_hotkey(ULTRA_HOTKEY, ultra.on_release, suppress=True, trigger_on_release=True)

    keyboard.wait()


if __name__ == "__main__":
    main()
