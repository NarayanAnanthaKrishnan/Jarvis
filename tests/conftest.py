import pytest

from ops import tracer


@pytest.fixture(autouse=True)
def disable_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracer, "TRACING", False)
