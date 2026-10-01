import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
from memory.retrieval_gate import needs_memory
from tts.playback import PlaybackEngine, speech_chunks


def test_chunking_preserves_addresses_numbers_and_abbreviations() -> None:
    text = "Dr. Lee sent a note to alex@example.com at 11:07 AM. The amount is 16.5 dollars. " + "Some longer words " * 25
    chunks = list(speech_chunks(text[index:index + 3] for index in range(0, len(text), 3)))
    assert " ".join(chunks) == text.strip()
    assert any("Dr. Lee" in chunk for chunk in chunks)
    assert any("alex@example.com" in chunk for chunk in chunks)
    assert any("16.5" in chunk for chunk in chunks)
    assert all(len(chunk) <= 160 for chunk in chunks)


def test_synthesis_overlaps_playback_and_uses_one_stream() -> None:
    second_ready = threading.Event()
    model, audio, stream = Mock(), Mock(), Mock()
    samples = []

    def create(text: str, **kwargs: object) -> tuple:
        if text == "Second sentence.":
            second_ready.set()
        return [text], 24000

    def write(data: list) -> bool:
        samples.extend(data)
        if len(samples) == 1:
            assert second_ready.wait(1), "Synthesis was blocked behind playback"
        return False

    model.create.side_effect = create
    audio.OutputStream.return_value = stream
    stream.write.side_effect = write
    engine = PlaybackEngine(model, audio)
    text = "First sentence. Second sentence. Third sentence."
    assert engine.speak([text]) == text
    assert samples == ["First sentence.", "Second sentence.", "Third sentence."]
    audio.OutputStream.assert_called_once()
    stream.start.assert_called_once()
    stream.stop.assert_called_once()
    stream.close.assert_called_once()


def test_cancel_aborts_playback_discards_queued_audio_and_allows_next_turn() -> None:
    playing, release = threading.Event(), threading.Event()
    model, audio = Mock(), Mock()
    first_stream, second_stream = Mock(), Mock()
    audio.OutputStream.side_effect = [first_stream, second_stream]
    model.create.side_effect = lambda text, **kwargs: ([text], 24000)

    def write(data: list) -> bool:
        playing.set()
        assert release.wait(2)
        return False

    first_stream.write.side_effect = write
    first_stream.abort.side_effect = release.set
    second_stream.write.return_value = False
    engine = PlaybackEngine(model, audio)
    old = threading.Thread(target=engine.speak, args=(["First. Second. Third. Fourth. Fifth."],))
    old.start()
    assert playing.wait(1)
    producer = engine._producer
    engine.stop()
    old.join(1)
    producer.join(1)
    assert not old.is_alive()
    assert not producer.is_alive()
    first_stream.abort.assert_called()
    assert first_stream.write.call_count == 1
    engine.speak(["Fresh response."])
    assert second_stream.write.call_args.args[0] == ["Fresh response."]


def test_synthesis_failure_reaches_caller_and_closes_stream() -> None:
    model, audio, stream = Mock(), Mock(), Mock()
    audio.OutputStream.return_value = stream
    model.create.side_effect = [([0.0], 24000), RuntimeError("synthesis failed")]
    stream.write.return_value = False
    engine = PlaybackEngine(model, audio)
    with pytest.raises(RuntimeError, match="synthesis failed"):
        engine.speak(["One sentence. Another sentence."])
    stream.close.assert_called_once()


def test_preload_is_shared_and_warms_only_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()
    model = Mock()
    model.create.side_effect = lambda *args, **kwargs: (entered.set(), release.wait(2), ([0.0], 24000))[-1]
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", SimpleNamespace(Kokoro=factory))
    model_file = tmp_path / "model.onnx"
    model_file.touch()
    monkeypatch.setattr(config, "TTS_MODEL", str(model_file))
    engine = PlaybackEngine()
    first = engine.preload()
    assert entered.wait(1)
    assert engine.preload() is first
    release.set()
    assert first.result(timeout=1) is model
    factory.assert_called_once()
    model.create.assert_called_once_with("Ready.", voice="af_bella", speed=1.15)


@pytest.mark.parametrize("query", ["What time is it?", "Hello, hello, what time is it right now and how's the weather outside?", "What is the date today?"])
def test_stateless_lookups_skip_memory_model(query: str) -> None:
    llm = Mock()
    assert needs_memory(query, llm) is False
    llm.call_raw.assert_not_called()


@pytest.mark.parametrize("query", ["What time is my meeting?", "What's the weather where I'm going?", "What did we discuss yesterday?", "Draft a cover letter"])
def test_other_requests_keep_memory_gate(query: str) -> None:
    llm = Mock()
    llm.call_raw.return_value = {"message": {"content": "YES"}}
    assert needs_memory(query, llm) is True
    llm.call_raw.assert_called_once()
