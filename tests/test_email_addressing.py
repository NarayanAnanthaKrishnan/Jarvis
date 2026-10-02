import pytest

from email_agent.addressing import normalize_spoken_address, speak_address
from tts.text import speech_safe_text


@pytest.mark.parametrize("spoken, expected", [
    ("Narayan at Gmail dot com", "Narayan@gmail.com"),
    ("narayan at the rate gmail dot com", "narayan@gmail.com"),
    ("narayan at sign g mail dot com", "narayan@gmail.com"),
    ("narayan ad gmail dot com", "narayan@gmail.com"),
    ("narayan adgmail.com", "narayan@gmail.com"),
    ("narayan add the regmail dot com", "narayan@gmail.com"),
    ("narayan at the red outlook dot com", "narayan@outlook.com"),
    ("narayan at example dot org", "narayan@example.org"),
])
def test_spoken_address_normalization(spoken: str, expected: str) -> None:
    assert normalize_spoken_address(spoken) == (expected, True)


def test_canonical_address_is_unchanged_and_spoken_names_get_confirmable_gmail_defaults() -> None:
    assert normalize_spoken_address("narayan@gmail.com") == ("narayan@gmail.com", False)
    assert normalize_spoken_address("Narayan") == ("narayan@gmail.com", True)
    assert normalize_spoken_address("Alex Smith") == ("alex.smith@gmail.com", True)
    assert normalize_spoken_address("please") is None


def test_speech_formats_email_punctuation_for_tts() -> None:
    assert speak_address("narayan.k@gmail.com") == "narayan dot k at gmail dot com"
    assert speech_safe_text("I heard narayan@gmail.com.") == "I heard narayan at gmail dot com."
