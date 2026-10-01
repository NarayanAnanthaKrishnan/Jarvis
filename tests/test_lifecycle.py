import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

import modes.jarvis as mode
import memory.session as session
from email_agent.models import EmailResult
from stt.stream_stt import StreamSTT
from tools import notes
from tts.playback import PlaybackEngine


def test_playback_cancel_prevents_later_sentences_without_threads() -> None:
    entered, release = threading.Event(), threading.Event()
    model, audio = Mock(), Mock()
    def create(*args: object, **kwargs: object) -> tuple:
        entered.set()
        release.wait(2)
        return [0.0], 24000
    model.create.side_effect = create
    engine = PlaybackEngine(model, audio)
    thread = threading.Thread(target=engine.speak, args=(["One. Two. Three. Four."],))
    thread.start()
    assert entered.wait(1)
    engine.stop()
    release.set()
    thread.join(1)
    assert not thread.is_alive()
    audio.play.assert_not_called()
    assert model.create.call_count == 1


def test_recorder_ownership_and_cleanup() -> None:
    recorder = Mock()
    released = threading.Event()
    recorder.text.side_effect = lambda: (released.wait(2), "")[1]
    recorder.shutdown.side_effect = released.set
    stt = StreamSTT(recorder_factory=Mock(return_value=recorder), use_microphone=False)
    stt.start_session(Mock())
    with pytest.raises(RuntimeError):
        stt.start()
    stt.stop_session()
    stt._session_thread.join(2)
    assert stt.owner is None
    recorder.shutdown.assert_called_once()
    recorder.abort.assert_not_called()


def test_playback_rejects_ended_session_even_after_stop() -> None:
    model, audio = Mock(), Mock()
    engine = PlaybackEngine(model, audio)
    cancelled = threading.Event()
    cancelled.set()
    engine.stop()
    assert engine.speak(["Stale response."], cancelled) == ""
    model.create.assert_not_called()
    audio.play.assert_not_called()


def test_recorder_start_failure_releases_ownership() -> None:
    recorder = Mock()
    recorder.start.side_effect = RuntimeError("device unavailable")
    stt = StreamSTT(recorder_factory=Mock(return_value=recorder), use_microphone=False)
    with pytest.raises(RuntimeError):
        stt.start()
    assert stt.owner is None
    recorder.shutdown.assert_called_once()


def make_mode(monkeypatch: pytest.MonkeyPatch) -> tuple:
    monkeypatch.setattr(mode, "AUTO_EXTRACT", False)
    stt, llm, speaker, streaming = Mock(), Mock(), Mock(), Mock()
    stt.owner = None
    jarvis = mode.JarvisMode(stt, llm, speaker, streaming)
    jarvis._start_session()
    return jarvis, stt, llm, streaming


def test_mic_resumes_after_playback_and_actual_history_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    jarvis, stt, _, streaming = make_mode(monkeypatch)
    saved = Mock(return_value="summary")
    monkeypatch.setattr(session, "save_session", saved)
    monkeypatch.setattr(mode, "run_agent", lambda *a: {"output": "answer", "delivery": "speak"})
    events = []
    stt.pause.side_effect = lambda: events.append("pause")
    streaming.speak_stream.side_effect = lambda *a: events.append("speech_finished")
    stt.resume.side_effect = lambda: events.append("resume")
    jarvis._on_turn("question")
    assert events == ["pause", "speech_finished", "resume"]
    jarvis._end_session()
    jarvis.shutdown()
    saved.assert_called_once()
    assert saved.call_args.args[1] == [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]


def test_start_fresh_clears_both_histories(monkeypatch: pytest.MonkeyPatch) -> None:
    jarvis, _, llm, _ = make_mode(monkeypatch)
    monkeypatch.setattr(session, "save_session", Mock())
    jarvis._history = [{"role": "user", "content": "old"}]
    old_id = jarvis._session_id
    jarvis._on_turn("Start fresh.")
    assert not jarvis._history
    assert jarvis._session_id != old_id
    llm.reset_history.assert_called_once()


def test_confirmation_bypasses_models_and_uses_session(monkeypatch: pytest.MonkeyPatch) -> None:
    jarvis, _, _, _ = make_mode(monkeypatch)
    run = Mock()
    monkeypatch.setattr(mode, "run_agent", run)
    service = Mock()
    service.confirm.return_value = EmailResult("sent", "Accepted by Gmail")
    jarvis._email_service = service
    jarvis._on_turn("Confirm email seven.")
    run.assert_not_called()
    assert service.confirm.call_args.args[0:2] == (7, jarvis._session_id)
    assert all(m["sensitive"] for m in jarvis._history)


def test_displayed_note_ids_address_correct_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("\n".join(f"note {i}" for i in range(1, 8)), encoding="utf-8")
    monkeypatch.setattr(notes, "NOTES_PATH", path)
    assert notes.read_notes(5).startswith("3. note 3")
    notes.update_note(3, "updated")
    assert path.read_text(encoding="utf-8").splitlines()[2].endswith("updated")
