import argparse
import faulthandler
import threading
from collections.abc import Callable


def reminder_loop(announce: Callable[[str], bool], stop: threading.Event) -> None:
    from config import REMINDER_CHECK_SECONDS
    from memory.reminders import get_due_reminders, mark_fired
    from ops.tracer import trace
    while not stop.is_set():
        try:
            for reminder in get_due_reminders():
                if stop.is_set() or not announce(f"Reminder: {reminder['message']}"):
                    break
                try:
                    from plyer import notification
                    notification.notify(title="Jarvis Reminder", message=reminder["message"], timeout=10)
                except Exception:
                    pass
                mark_fired(reminder["id"])
        except Exception as exc:
            trace("error", where="reminder_loop", error_type=type(exc).__name__)
        stop.wait(REMINDER_CHECK_SECONDS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Jarvis voice assistant")
    parser.add_argument("--check-audio", action="store_true", help="Check speech worker startup and microphone capture in both modes")
    args = parser.parse_args(argv)

    import keyboard

    from config import HOTKEY, WHISPERFLOW_HOTKEY
    from llm.client import LLMClient
    from modes.controller import ModeActions, ModeController
    from modes.jarvis import JarvisMode
    from modes.whisperflow import WhisperFlowMode
    from ops.logging_setup import configure_logging
    from stt.stream_stt import StreamSTT
    from tts.playback import PlaybackEngine
    from tts.speaker import Speaker
    from tts.stream_tts import StreamingSpeaker

    configure_logging()
    if args.check_audio:
        from ops.audio_check import check_microphone
        check_microphone()
        return
    stream_stt = StreamSTT()
    llm = LLMClient()
    engine = PlaybackEngine()
    speaker, streaming_speaker = Speaker(engine), StreamingSpeaker(engine)
    jarvis = JarvisMode(stream_stt, llm, speaker, streaming_speaker)
    whisperflow = WhisperFlowMode(stream_stt)
    stop = threading.Event()
    controller = ModeController({
        "jarvis": ModeActions(jarvis._start_session, jarvis._end_session),
        "dictation": ModeActions(whisperflow._start, whisperflow._stop, cancel_on_stop=False),
    }, jarvis.transition_lock)

    reminder_thread = threading.Thread(target=reminder_loop, args=(jarvis.announce, stop), daemon=True)
    reminder_thread.start()
    print("✅ Jarvis running. CTRL+SHIFT+J toggles a session; CTRL+SHIFT+K toggles dictation.")
    keyboard.add_hotkey(HOTKEY, lambda: controller.request_toggle("jarvis"), suppress=True, trigger_on_release=True)
    keyboard.add_hotkey(WHISPERFLOW_HOTKEY, lambda: controller.request_toggle("dictation"), suppress=True, trigger_on_release=True)
    try:
        keyboard.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        keyboard.unhook_all_hotkeys()
        engine.stop()
        controller.close()
        jarvis.shutdown()
        reminder_thread.join(timeout=2)


if __name__ in {"__main__", "__mp_main__"}:
    faulthandler.enable()

if __name__ == "__main__":
    main()
