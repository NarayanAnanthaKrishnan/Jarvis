import threading
import time
from unittest.mock import Mock

import pytest

import modes.jarvis as mode
from modes.controller import ModeActions, ModeController
from stt.stream_stt import StreamSTT


def test_hotkey_request_returns_while_startup_is_blocked_and_can_cancel() -> None:
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    activated = []

    def start(cancelled: threading.Event) -> None:
        entered.set()
        assert release.wait(2)
        try:
            if cancelled.is_set():
                raise InterruptedError()
            activated.append(True)
        finally:
            finished.set()

    controller = ModeController({"jarvis": ModeActions(start, Mock())}, threading.RLock())
    try:
        before = time.monotonic()
        controller.request_toggle("jarvis")
        assert time.monotonic() - before < .1
        assert entered.wait(1)
        before = time.monotonic()
        controller.request_toggle("jarvis")
        assert time.monotonic() - before < .1
        release.set()
        assert finished.wait(1)
        assert not activated
    finally:
        release.set()
        controller.close()


def test_hotkey_can_queue_next_mode_during_slow_stop() -> None:
    started, stopping, release, second_started = [threading.Event() for _ in range(4)]

    def stop() -> None:
        stopping.set()
        assert release.wait(2)

    controller = ModeController({
        "jarvis": ModeActions(lambda cancelled: started.set(), stop),
        "dictation": ModeActions(lambda cancelled: second_started.set(), Mock(), False),
    }, threading.RLock())
    try:
        controller.request_toggle("jarvis")
        assert started.wait(1)
        controller.request_toggle("jarvis")
        assert stopping.wait(1)
        before = time.monotonic()
        controller.request_toggle("dictation")
        assert time.monotonic() - before < .1
        assert not second_started.is_set()
        release.set()
        assert second_started.wait(1)
    finally:
        release.set()
        controller.close()


def test_cancelled_recorder_initialization_never_enables_microphone() -> None:
    entered, release = threading.Event(), threading.Event()
    cancelled = threading.Event()
    recorder = Mock()
    errors = []

    def factory(**kwargs: object) -> Mock:
        entered.set()
        assert release.wait(2)
        return recorder

    stt = StreamSTT(recorder_factory=factory, use_microphone=False)

    def start() -> None:
        try:
            stt.start(cancelled)
        except InterruptedError:
            errors.append("cancelled")

    thread = threading.Thread(target=start)
    thread.start()
    assert entered.wait(1)
    cancelled.set()
    release.set()
    thread.join(1)
    assert not thread.is_alive()
    assert errors == ["cancelled"]
    assert stt.owner is None
    recorder.start.assert_not_called()
    recorder.shutdown.assert_called_once()
    assert all(call.args == (False,) for call in recorder.set_microphone.call_args_list)


def test_stopping_paused_session_does_not_call_blocking_abort() -> None:
    recorder = Mock()
    recorder.text.return_value = "sample"
    delivered = threading.Event()
    stt = StreamSTT(recorder_factory=Mock(return_value=recorder), use_microphone=False)
    stt.start_session(lambda text: delivered.set())
    assert delivered.wait(1)
    assert stt._paused.is_set()
    stt.stop_session()
    assert not stt._session_thread.is_alive()
    assert stt.owner is None
    recorder.abort.assert_not_called()
    recorder.shutdown.assert_called_once()


def test_empty_dictation_does_not_start_waiting_for_new_speech() -> None:
    recorder = Mock()
    recorder.has_pending_recordings.return_value = False
    stt = StreamSTT(recorder_factory=Mock(return_value=recorder), use_microphone=False)
    stt.start()
    stt.stop()
    assert stt.text() == ""
    recorder.text.assert_not_called()
    recorder.shutdown.assert_called_once()


def test_failed_claim_does_not_clear_cancellation_of_existing_recorder() -> None:
    recorder = Mock()
    stt = StreamSTT(recorder_factory=Mock(return_value=recorder), use_microphone=False)
    stt.start()
    stt._stop.set()
    with pytest.raises(RuntimeError):
        stt.start()
    assert stt._stop.is_set()
    stt.shutdown()


def test_old_session_response_cannot_speak_or_resume_new_session(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()
    stt, llm, speaker, streaming = [Mock() for _ in range(4)]
    stt.owner = None
    jarvis = mode.JarvisMode(stt, llm, speaker, streaming)
    monkeypatch.setattr(mode, "AUTO_EXTRACT", False)

    def run(*args: object) -> dict:
        entered.set()
        assert release.wait(2)
        return {"output": "old answer", "delivery": "speak"}

    monkeypatch.setattr(mode, "run_agent", run)
    jarvis._start_session()
    old_id, old_cancelled = jarvis._session_id, jarvis._cancelled
    jarvis._dispatch_turn("old question", old_id, old_cancelled)
    old_thread = jarvis._turn_thread
    assert entered.wait(1)
    jarvis._end_session()
    jarvis._start_session()
    stt.resume.reset_mock()
    release.set()
    old_thread.join(1)
    assert not old_thread.is_alive()
    streaming.speak_stream.assert_not_called()
    stt.resume.assert_not_called()
    assert jarvis._history == []
    jarvis._dispatch_turn("stale callback", old_id, old_cancelled)
    assert jarvis._turn_thread is old_thread
    jarvis.shutdown()


def test_session_summary_does_not_block_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    import memory.session as session
    entered, release = threading.Event(), threading.Event()
    stt = Mock()
    stt.owner = None
    jarvis = mode.JarvisMode(stt, Mock(), Mock(), Mock())
    jarvis._start_session()
    jarvis._history = [{"role": "user", "content": "synthetic", "sensitive": False}]

    def save(*args: object) -> None:
        entered.set()
        release.wait(2)

    monkeypatch.setattr(session, "save_session", save)
    before = time.monotonic()
    jarvis._end_session()
    assert time.monotonic() - before < .1
    assert entered.wait(1)
    release.set()
    jarvis.shutdown()


def test_recorder_callback_still_works_after_start_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    stt = Mock()
    stt.owner = None
    jarvis = mode.JarvisMode(stt, Mock(), Mock(), Mock())
    monkeypatch.setattr(mode, "AUTO_EXTRACT", False)
    run = Mock(return_value={"output": "new answer", "delivery": "speak"})
    monkeypatch.setattr(mode, "run_agent", run)
    jarvis._start_session()
    callback = stt.start_session.call_args.args[0]
    jarvis._on_turn("start fresh")
    callback("new question")
    jarvis._turn_thread.join(1)
    run.assert_called_once()
    assert jarvis._history[-1]["content"] == "new answer"
    monkeypatch.setattr(jarvis, "_save_session", Mock())
    jarvis.shutdown()


def test_new_session_does_not_wait_for_cancelled_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release, answered = threading.Event(), threading.Event(), threading.Event()
    stt, streaming = Mock(), Mock()
    stt.owner = None
    jarvis = mode.JarvisMode(stt, Mock(), Mock(), streaming)
    monkeypatch.setattr(mode, "AUTO_EXTRACT", False)
    monkeypatch.setattr(jarvis, "_save_session", Mock())

    def run(text: str, *args: object) -> dict:
        if text == "old":
            entered.set()
            release.wait(2)
        else:
            answered.set()
        return {"output": text, "delivery": "speak"}

    monkeypatch.setattr(mode, "run_agent", run)
    jarvis._start_session()
    jarvis._queue_turn("old", jarvis._cancelled)
    old_thread = jarvis._turn_thread
    assert entered.wait(1)
    jarvis._end_session()
    jarvis._start_session()
    jarvis._queue_turn("new", jarvis._cancelled)
    assert answered.wait(1)
    jarvis._turn_thread.join(1)
    release.set()
    old_thread.join(1)
    assert streaming.speak_stream.call_count == 1
    assert jarvis._history[-1]["content"] == "new"
    jarvis.shutdown()


def test_dictation_finishes_text_and_ignores_old_partials(monkeypatch: pytest.MonkeyPatch) -> None:
    import modes.whisperflow as dictation
    stt = Mock()
    stt.owner = None
    typed = Mock()
    monkeypatch.setattr(dictation.pyautogui, "typewrite", typed)
    whisperflow = dictation.WhisperFlowMode(stt)
    whisperflow._start()
    callback = stt.set_realtime_callback.call_args.args[0]
    callback("hello")
    callback("hello")
    stt.text.return_value = "hello world"
    whisperflow._stop()
    assert "".join(call.args[0] for call in typed.call_args_list) == "hello world"
    whisperflow._start()
    typed.reset_mock()
    callback("old words")
    callback("old words")
    typed.assert_not_called()
    whisperflow._cancelled.set()
    whisperflow._stop()
    stt.shutdown.assert_called_once()


def test_main_registers_hotkeys_on_release(monkeypatch: pytest.MonkeyPatch) -> None:
    import main
    import keyboard
    import llm.client
    import modes.controller
    import ops.logging_setup
    import stt.stream_stt as stt_module
    controller = Mock()
    stt = Mock()
    stt.owner = None
    monkeypatch.setattr(stt_module, "StreamSTT", lambda: stt)
    monkeypatch.setattr(llm.client, "LLMClient", Mock())
    monkeypatch.setattr(modes.controller, "ModeController", Mock(return_value=controller))
    monkeypatch.setattr(ops.logging_setup, "configure_logging", Mock())
    monkeypatch.setattr(main, "reminder_loop", Mock())
    add_hotkey = Mock()
    monkeypatch.setattr(keyboard, "add_hotkey", add_hotkey)
    monkeypatch.setattr(keyboard, "unhook_all_hotkeys", Mock())
    monkeypatch.setattr(keyboard, "wait", Mock(side_effect=KeyboardInterrupt))
    main.main([])
    assert add_hotkey.call_count == 2
    for call in add_hotkey.call_args_list:
        assert call.kwargs == {"suppress": True, "trigger_on_release": True}
        call.args[1]()
    assert [call.args[0] for call in controller.request_toggle.call_args_list] == ["jarvis", "dictation"]
    controller.close.assert_called_once()
