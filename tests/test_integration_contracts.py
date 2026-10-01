import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient, RetryPolicy

import config
from agent import loop
from agent.router import RouteDecision, route_turn
from email_agent.gmail import GmailProvider
from email_agent.models import ProviderError
from email_agent.timing import parse_send_time
from email_agent.worker import WorkerLock
from llm.client import LLMClient
from ops import tracer


def test_real_jev_sdk_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "JEV_ROUTING_ENABLED", True)
    seen = []
    def respond(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json={"model": "jev-latest", "answers": {"route": {"type": "choice", "choice": "email", "confidence": .95, "probabilities": {"email": .98, "general": .01, "mixed": .01, "unclear": 0}}, "memory": {"type": "noul", "noul": .1}}, "usage": {"input_tokens": 20, "output_tokens": 0}})
    with TypeSafeClient(api_key="synthetic", transport=httpx2.MockTransport(respond), retry=RetryPolicy(max_retries=0)) as client:
        result = route_turn("draft email", [], {}, client)
    assert result.agent_id == "email"
    assert seen[0]["questions"]["route"]["type"] == "choice"
    assert seen[0]["questions"]["memory"]["type"] == "noul"


@pytest.mark.parametrize("status, uncertain, retryable", [(503, True, False), (429, False, True), (401, False, False), (404, False, False)])
def test_gmail_send_error_contract(status: int, uncertain: bool, retryable: bool) -> None:
    import threading
    import httplib2
    from googleapiclient.errors import HttpError
    provider = object.__new__(GmailProvider)
    provider._lock = threading.RLock()
    request = Mock()
    request.execute.side_effect = HttpError(httplib2.Response({"status": status}), b'{}')
    with patch.object(provider, "connection_current", return_value=True):
        with pytest.raises(ProviderError) as error:
            provider._execute(request, write=True)
    assert error.value.uncertain is uncertain
    assert error.value.retryable is retryable
    request.execute.assert_called_once_with(num_retries=0)


def test_relative_duration_crosses_dst_as_elapsed_time() -> None:
    now = datetime(2026, 3, 7, 17, tzinfo=timezone.utc)
    due = parse_send_time("in 24 hours", "America/New_York", now)
    assert (due - now).total_seconds() == 86400
    with pytest.raises(Exception):
        parse_send_time("tomorrow at 9am PST", "America/New_York", now)


def test_single_worker_lock(tmp_path: Path) -> None:
    path = tmp_path / "worker.lock"
    with WorkerLock(path):
        with pytest.raises(RuntimeError):
            with WorkerLock(path):
                pass
    with WorkerLock(path):
        pass


def test_trace_never_records_body_or_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracer, "TRACING", True)
    monkeypatch.setattr(tracer, "TRACE_DIR", tmp_path)
    tracer.trace("email", status="sent", job_id=7, body="private content", args={"token": "private token"}, user_input="private utterance")
    content = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "private" not in content
    assert json.loads(content)["job_id"] == 7
    tracer._HANDLER.close()
    tracer._HANDLER = None


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_stateless_temperature_forwarded(provider: str) -> None:
    llm = LLMClient()
    sdk = Mock()
    with patch.object(llm, "_get_provider_order", return_value=[provider]), patch.object(llm, "_get_client", return_value=sdk), patch.object(llm, "_normalize_response", return_value={"message": {"content": "OK"}}):
        llm.call_raw([{"role": "user", "content": "synthetic"}], temp=.1)
    call = sdk.chat.completions.create if provider == "openai" else sdk.messages.create
    assert call.call_args.kwargs["temperature"] == .1


def test_email_success_cannot_be_invented_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "route_turn", lambda *a: RouteDecision(memory_needed=False))
    monkeypatch.setattr(loop, "think", lambda *a: {"done": True, "answer": "I sent your email."})
    result = loop.run_agent("send an email", [], Mock())
    assert "No email action" in result["output"]
