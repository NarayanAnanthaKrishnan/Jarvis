from concurrent.futures import ThreadPoolExecutor
import json
from unittest.mock import Mock

import pytest

import config
import llm.client as client_module
from email_agent.contracts import DECISION_SCHEMA, DRAFT_SCHEMA
from llm.client import LLMClient, ModelRequestError


def test_email_profile_and_schema_are_isolated_from_general_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = Mock()
    llm = LLMClient()
    monkeypatch.setattr(client_module, "GEMINI_API_KEY", "synthetic")
    monkeypatch.setattr(llm, "_get_client", Mock(return_value=sdk))
    monkeypatch.setattr(llm, "_get_provider_order", lambda: ["gemini"])
    monkeypatch.setattr(llm, "_normalize_response", lambda *args: {"message": {"content": '{"subject":"Test","body":"Hello"}'}})
    messages = [{"role": "user", "content": "synthetic"}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        email = pool.submit(llm.call_raw, messages, profile="email", response_schema=DRAFT_SCHEMA)
        general = pool.submit(llm.call_raw, messages, max_tokens=300, temp=.2)
        assert email.result()
        assert general.result()
    calls = {call.kwargs["model"]: call.kwargs["config"] for call in sdk.models.generate_content.call_args_list}
    email_config = calls[config.EMAIL_MODEL]
    general_config = calls[client_module.GEMINI_MODEL]
    assert email_config.response_json_schema == DRAFT_SCHEMA
    assert email_config.thinking_config.thinking_level.value == "LOW"
    assert email_config.max_output_tokens == 4096
    assert email_config.http_options.timeout == 20000
    assert email_config.http_options.retry_options.attempts == 1
    assert general_config.response_json_schema is None
    assert general_config.thinking_config is None
    assert general_config.max_output_tokens == 300
    assert general_config.temperature == .2


@pytest.mark.parametrize("code, expected", [(429, "quota"), (403, "access"), (404, "access"), (504, "timed out"), (500, "connection")])
def test_email_model_errors_are_actionable_without_fallback(code: int, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = Mock()
    failure = RuntimeError("private provider response")
    failure.code = code
    sdk.models.generate_content.side_effect = failure
    llm = LLMClient()
    monkeypatch.setattr(client_module, "GEMINI_API_KEY", "synthetic")
    client = Mock(return_value=sdk)
    monkeypatch.setattr(llm, "_get_client", client)
    with pytest.raises(ModelRequestError, match=expected) as exc:
        llm.call_raw([{"role": "user", "content": "synthetic"}], profile="email", response_schema=DECISION_SCHEMA)
    assert "private" not in str(exc.value)
    client.assert_called_once_with("gemini")
    sdk.models.generate_content.assert_called_once()


def test_email_decisions_request_email_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.context import TurnContext
    from agent import loop
    llm = Mock()
    llm.call_raw.return_value = {"message": {"content": '{"done":true,"answer":"What time should I use?","workflow":{"action":"schedule","when":null,"timezone":null,"awaiting":"time","new_draft":false}}'}}
    monkeypatch.setattr(loop, "load_profile", lambda: "")
    loop.think("schedule", [], [], llm, context=TurnContext(agent_id="email"))
    assert llm.call_raw.call_args.kwargs == {"profile": "email", "response_schema": DECISION_SCHEMA}


@pytest.mark.parametrize("agent, profile", [("calendar", "calendar"), ("mixed", "mixed")])
def test_calendar_profiles_request_structured_action_schema(agent: str, profile: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.context import TurnContext
    from agent import loop

    llm = Mock()
    decision = {"done": True, "answer": "When should the meeting start?", "calendar_workflow": {
        "action": "create", "title": "Review", "when": None, "timezone": None, "duration_minutes": 30,
        "attendees": [], "description": "", "location": "", "awaiting": "time", "event_id": None}}
    if agent == "mixed":
        decision["email_workflow"] = {"action": "draft", "when": None, "timezone": None, "awaiting": None, "new_draft": False}
    llm.call_raw.return_value = {"message": {"content": json.dumps(decision)}}
    monkeypatch.setattr(loop, "load_profile", lambda: "")
    loop.think("schedule a meeting", [], [], llm, context=TurnContext(agent_id=agent))
    kwargs = llm.call_raw.call_args.kwargs
    assert kwargs["profile"] == profile
    assert "anyOf" in kwargs["response_schema"]


def test_email_and_meeting_confirmation_phrases_are_distinct() -> None:
    from email_agent.service import confirmation_number

    assert confirmation_number("Confirm email two") == 2
    assert confirmation_number("Confirm meeting twenty one", "meeting") == 21
    assert confirmation_number("Confirm meeting two") is None
    assert confirmation_number("Confirm email two", "meeting") is None
