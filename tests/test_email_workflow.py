import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

import config
from agent import loop
from agent.context import TurnContext
from agent.json_utils import extract_object
from agent.router import route_turn
from email_agent.models import EmailError
from email_agent.runtime import execute_email
from email_agent.service import EmailService
from email_agent.store import EmailStore
from email_agent.workflow import EmailWorkflow
from llm.client import ModelRequestError
from test_email import FakeProvider


def decision(tool: str, args: dict, action: str = "schedule", when: str | None = None, new: bool = False) -> dict:
    return {"tool": tool, "args": args, "workflow": {"action": action, "when": when,
            "timezone": None, "awaiting": None, "new_draft": new}}


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    monkeypatch.setattr(config, "JEV_ROUTING_ENABLED", False)
    monkeypatch.setattr(config, "RETRIEVAL_GATE", True)
    monkeypatch.setattr(loop, "load_profile", lambda: "")
    provider = FakeProvider()
    service = EmailService(EmailStore(tmp_path / "workflow.db"), provider)
    llm = Mock()
    context = TurnContext(session_id="synthetic-session", email_service=service)
    return provider, service, llm, context


def responses(llm: Mock, *values: dict) -> None:
    llm.call_raw.side_effect = [{"message": {"content": json.dumps(value)}} for value in values]


def next_context(previous: TurnContext) -> TurnContext:
    return TurnContext(session_id=previous.session_id, email_service=previous.email_service,
                       active_draft_id=previous.active_draft_id, email_workflow=previous.email_workflow)


def test_complete_progressive_conversation_keeps_one_draft(harness: tuple) -> None:
    provider, service, llm, context = harness
    responses(llm, decision("email_draft", {"instruction": "Ask Alex about an opening"}),
              {"subject": "Software role", "body": "Hello Alex, are you hiring?"})
    first = loop.run_agent("Schedule an email asking Alex about a software role", [], llm, context)
    assert "saved in Gmail" in first["output"] and "address" in first["output"]
    assert context.email_workflow.awaiting == "recipient"
    original = provider.drafts["1"]
    history = ["User: Schedule an email asking Alex about a software role", "Assistant: " + first["output"]]
    responses(llm, decision("email_recipients", {"draft_id": 1, "to": ["alex@example.com"]}))
    context = next_context(context)
    second = loop.run_agent("alex@example.com", history, llm, context)
    assert "When" in second["output"] and context.email_workflow.awaiting == "time"
    assert provider.drafts["1"].body == original.body
    assert provider.drafts["1"].subject == original.subject
    history.extend(["User: alex@example.com", "Assistant: " + second["output"]])
    responses(llm, decision("email_prepare", {"draft_id": 1, "action": "schedule", "when": "tomorrow at 9am"}, when="tomorrow at 9am"))
    context = next_context(context)
    third = loop.run_agent("tomorrow at 9am", history, llm, context)
    assert third["confirmation_id"] == 1
    assert service.store.job(1)["status"] == "awaiting_confirmation"
    assert len(provider.drafts) == 1 and not provider.sent
    assert llm.call_raw.call_count == 3


def test_pending_schedule_time_followup_is_parsed_without_an_llm_decision(harness: tuple) -> None:
    provider, service, llm, context = harness
    service.clock = lambda: datetime(2026, 10, 1, 20, 0, tzinfo=timezone.utc).timestamp()
    responses(llm, {"subject": "Software role", "body": "Hello Alex, are you hiring?"})
    draft = service.draft("Ask Alex about an opening", context, llm, to=["alex@example.com"])
    context.active_draft_id = draft.draft_id
    context.email_workflow = EmailWorkflow(action="schedule", awaiting="time")
    followup_context = next_context(context)
    before = llm.call_raw.call_count
    result = loop.run_agent("later today 6pm", [], llm, followup_context)
    assert llm.call_raw.call_count == before
    assert result["confirmation_id"] == 1
    assert service.store.job(1)["status"] == "awaiting_confirmation"
    assert service.store.job(1)["due_at"] == pytest.approx(datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc).timestamp())
    assert not provider.sent


def test_unparseable_schedule_followup_does_not_create_a_job(harness: tuple) -> None:
    _, service, llm, context = harness
    context.agent_id = "email"
    context.active_draft_id = 1
    context.email_workflow = EmailWorkflow(action="schedule", awaiting="time")
    llm.call_raw.return_value = {"message": {"content": json.dumps({"done": True, "answer": "What day should I schedule it for?", "workflow": {"action": "schedule", "when": None, "timezone": None, "awaiting": "time", "new_draft": False}})}}
    result = loop.run_agent("at 99pm", [], llm, context)
    assert "couldn't read that as a schedule time" in result["output"]
    with service.store.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_recipient_name_is_preserved_for_draft_writing_and_address_is_requested(harness: tuple) -> None:
    provider, _, llm, context = harness
    responses(llm, {"subject": "Role question", "body": "Hello Alex, are you hiring?"})
    result = execute_email("email_draft", {"instruction": "Write to Alex about a software developer role; use Alex in the greeting"}, context, llm)
    generation_request = llm.call_raw.call_args.args[0][1]["content"]
    assert "Alex" in generation_request
    assert provider.drafts["1"].to == []
    assert "address" in result.message.lower()
    assert context.email_workflow.awaiting == "recipient"


def test_spoken_recipient_requires_readback_before_it_is_added(harness: tuple) -> None:
    provider, _, llm, context = harness
    responses(llm, {"subject": "Role question", "body": "Hello Narayan, are you hiring?"})
    result = execute_email("email_draft", {"instruction": "Ask about a software developer role", "to": ["Narayan add the regmail dot com"]}, context, llm)
    assert result.status == "draft"
    assert context.email_workflow.awaiting == "recipient_confirmation"
    assert context.email_workflow.pending_recipient == {"field": "to", "address": "Narayan@gmail.com", "draft_id": 1, "mode": "replace"}
    assert provider.drafts["1"].to == []
    assert "Narayan at gmail dot com" in result.message
    confirmed = loop.run_agent("yes", [], llm, context)
    assert provider.drafts["1"].to == ["Narayan@gmail.com"]
    assert context.email_workflow.pending_recipient is None
    assert "Draft" in confirmed["output"]
    assert llm.call_raw.call_count == 1


def test_initial_schedule_extracts_time_even_when_model_omits_it(harness: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, service, llm, context = harness
    service.clock = lambda: datetime(2026, 10, 1, 20, 0, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(loop, "needs_memory", lambda *args: False)
    responses(llm, decision("email_draft", {"instruction": "Ask about a project review", "to": ["alex@example.com"]}, action="draft"),
              {"subject": "Project review", "body": "Could we discuss the review?"})
    result = loop.run_agent("Schedule an email to alex@example.com for later today 6pm asking about a project review", [], llm, context)
    assert result["confirmation_id"] == 1
    assert service.store.job(1)["due_at"] == pytest.approx(datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc).timestamp())
    assert service.store.job(1)["status"] == "awaiting_confirmation"
    assert not provider.sent


def test_two_schedule_time_choices_are_clarified_without_creating_job(harness: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    _, service, llm, context = harness
    service.clock = lambda: datetime(2026, 10, 1, 20, 0, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(loop, "needs_memory", lambda *args: False)
    responses(llm, decision("email_draft", {"instruction": "Ask about a project review", "to": ["alex@example.com"]}, action="schedule", when="6pm"),
              {"subject": "Project review", "body": "Could we discuss the review?"})
    result = loop.run_agent("Schedule an email to alex@example.com for later today 6pm or 7pm asking about a project review", [], llm, context)
    assert "more than one possible time" in result["output"].lower()
    with service.store.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_complete_request_prepares_after_draft_without_extra_model_decision(harness: tuple) -> None:
    provider, service, llm, context = harness
    responses(llm, decision("email_draft", {"instruction": "Ask to meet", "to": ["alex@example.com"]}, when="in 10 minutes"),
              {"subject": "Meeting", "body": "Can we meet?"})
    result = loop.run_agent("Schedule an email to alex@example.com in 10 minutes asking to meet", [], llm, context)
    assert result["confirmation_id"] == 1
    assert llm.call_raw.call_count == 2
    assert service.store.job(1)["presented"] == 0
    assert not provider.sent


def test_bad_recipient_saves_work_but_blocks_prepare_until_corrected(harness: tuple) -> None:
    provider, service, llm, context = harness
    context.email_workflow = EmailWorkflow(action="send")
    responses(llm, {"subject": "Meeting", "body": "Can we meet?"})
    result = execute_email("email_draft", {"instruction": "Meeting", "to": ["alex@example.com"], "cc": ["Robin at company"]}, context, llm)
    assert result.status == "draft" and "address" in result.message
    assert provider.drafts["1"].to == ["alex@example.com"]
    assert context.email_workflow.unresolved_recipients == ["cc"]
    result = execute_email("email_prepare", {"draft_id": 1, "action": "send"}, context, llm)
    assert result.status == "needs_details"
    result = execute_email("email_recipients", {"draft_id": 1, "cc": ["robin@example.com"]}, context, llm)
    assert result.confirmation_id == 1
    assert not provider.sent
    llm.call_raw.assert_called_once()


def test_revision_without_id_reuses_draft_and_explicit_new_request_creates_another(harness: tuple) -> None:
    provider, _, llm, context = harness
    responses(llm, decision("email_draft", {"instruction": "Meeting"}, "draft"), {"subject": "First", "body": "Hello"})
    loop.run_agent("Draft a meeting email", [], llm, context)
    context = next_context(context)
    responses(llm, decision("email_draft", {"instruction": "Shorter"}, "draft"), {"subject": "First", "body": "Hi"})
    loop.run_agent("Make it shorter", [], llm, context)
    assert len(provider.drafts) == 1
    context = next_context(context)
    responses(llm, decision("email_draft", {"instruction": "Another meeting"}, "draft", new=True), {"subject": "Second", "body": "Good morning"})
    loop.run_agent("Draft another email", [], llm, context)
    assert len(provider.drafts) == 2 and context.active_draft_id == 2


@pytest.mark.parametrize("text, workflow, expected", [
    ("draft an email asking about a role", {}, "email"),
    ("alex@example.com", {"awaiting": "recipient", "active_draft_id": 1}, "email"),
    ("tomorrow at nine am", {"awaiting": "time", "active_draft_id": 1}, "email"),
    ("in half an hour", {"awaiting": "time", "active_draft_id": 1}, "email"),
    ("about a software role", {"awaiting": "purpose"}, "email"),
    ("what's the weather outside?", {"awaiting": "time", "active_draft_id": 1}, "general"),
    ("research the company and email a summary", {}, "general"),
    ("what time is it?", {"awaiting": "recipient"}, "general"),
    ("tell me a joke", {"awaiting": "time", "active_draft_id": 1}, "general"),
])
def test_email_routing_and_topic_switches(text: str, workflow: dict, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "JEV_ROUTING_ENABLED", False)
    assert route_turn(text, [], workflow).agent_id == expected


def test_invalid_decision_repaired_once_before_any_operation(harness: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _, llm, context = harness
    tool = Mock()
    monkeypatch.setattr(loop, "_execute_timed", tool)
    llm.call_raw.side_effect = [
        {"message": {"content": '{"tool":"email_draft","args":{"instruction":"truncated"}'}},
        {"message": {"content": json.dumps(decision("email_draft", {"instruction": "Meeting"}, "draft"))}},
    ]
    context.agent_id = "email"
    result = loop.think("Draft a meeting email", [], [], llm, context=context)
    assert result["tool"] == "email_draft"
    assert context.remaining_steps == config.MAX_STEPS - 1
    tool.assert_not_called()
    assert not provider.drafts


def test_invalid_decision_retry_exhaustion_is_actionable(harness: tuple) -> None:
    _, _, llm, context = harness
    llm.call_raw.return_value = {"message": {"content": "invalid"}}
    with pytest.raises(ModelRequestError, match="after retrying"):
        loop.think("Draft a meeting email", [], [], llm, context=context)
    assert llm.call_raw.call_count == 2


@pytest.mark.parametrize("value", ['{"broken": {"tool":"email_draft","args":{}}', '{"done":true} {"tool":"email_draft"}', '[]'])
def test_partial_or_multiple_objects_cannot_become_executable_decisions(value: str) -> None:
    with pytest.raises(ValueError):
        extract_object(value)


def test_ambiguous_schedule_keeps_draft_and_requests_time(harness: tuple) -> None:
    provider, service, llm, context = harness
    context.email_workflow = EmailWorkflow(action="schedule", when="tomorrow at nine")
    responses(llm, {"subject": "Meeting", "body": "Can we meet?"})
    result = execute_email("email_draft", {"instruction": "Meeting", "to": ["alex@example.com"]}, context, llm)
    assert result.status == "draft" and "unambiguous" in result.message
    assert context.email_workflow.awaiting == "time"
    assert len(provider.drafts) == 1 and not provider.sent
    with service.store.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_relative_schedule_is_anchored_before_recipient_followup(harness: tuple) -> None:
    _, service, llm, context = harness
    now = [service.clock()]
    service.clock = lambda: now[0]
    original = now[0]
    context.email_workflow = EmailWorkflow(action="schedule", when="in 10 minutes")
    responses(llm, {"subject": "Meeting", "body": "Can we meet?"})
    execute_email("email_draft", {"instruction": "Meeting"}, context, llm)
    now[0] += 120
    context = next_context(context)
    result = execute_email("email_recipients", {"draft_id": 1, "to": ["alex@example.com"], "mode": "add"}, context, llm)
    assert result.confirmation_id
    assert service.store.job(result.job_id)["due_at"] == pytest.approx(original + 600, abs=0.01, rel=0)


def test_add_missing_recipient_keeps_already_known_address(harness: tuple) -> None:
    provider, _, llm, context = harness
    responses(llm, {"subject": "Meeting", "body": "Can we meet?"})
    result = execute_email("email_draft", {"instruction": "Meeting", "to": ["alex@example.com", "Robin"]}, context, llm)
    assert result.status == "draft" and "address" in result.message
    assert provider.drafts["1"].to == ["alex@example.com"]
    execute_email("email_recipients", {"draft_id": 1, "to": ["robin@example.com"], "mode": "add"}, context, llm)
    assert provider.drafts["1"].to == ["alex@example.com", "robin@example.com"]
    assert context.email_workflow.unresolved_recipients == []
    execute_email("email_recipients", {"draft_id": 1, "to": ["alex@example.com"], "mode": "remove"}, context, llm)
    assert provider.drafts["1"].to == ["robin@example.com"]
    assert provider.drafts["1"].body.rstrip("\n") == "Can we meet?"


def test_session_retains_draft_progress_after_recoverable_error(harness: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    import modes.jarvis as mode
    provider, service, llm, _ = harness
    monkeypatch.setattr(mode, "AUTO_EXTRACT", False)
    stt = Mock()
    stt.owner = None
    jarvis = mode.JarvisMode(stt, llm, Mock(), Mock())
    jarvis._start_session()

    def run(text: str, history: list[str], client: Mock, context: TurnContext) -> dict:
        context.email_service = service
        context.email_touched = True
        context.active_draft_id = 7
        context.email_workflow = EmailWorkflow(action="schedule", awaiting="time")
        raise EmailError("Please specify a time")

    monkeypatch.setattr(mode, "run_agent", run)
    jarvis._on_turn("Schedule my email")
    assert jarvis._active_draft_id == 7
    assert jarvis._email_service is service
    assert jarvis._email_workflow.awaiting == "time"
    assert jarvis._history[-1]["sensitive"]
    stt.resume.assert_called_once()


def test_stale_session_cannot_replace_new_workflow(harness: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import modes.jarvis as mode
    _, _, llm, _ = harness
    stt = Mock()
    stt.owner = None
    jarvis = mode.JarvisMode(stt, llm, Mock(), Mock())
    jarvis._start_session()

    def run(text: str, history: list[str], client: Mock, context: TurnContext) -> dict:
        context.active_draft_id = 7
        context.email_workflow.awaiting = "time"
        context.cancelled.set()
        jarvis._session_id = "new-session"
        jarvis._cancelled = threading.Event()
        jarvis._active_draft_id = 9
        jarvis._email_workflow = EmailWorkflow(awaiting="recipient")
        raise InterruptedError("Old session ended")

    monkeypatch.setattr(mode, "run_agent", run)
    jarvis._on_turn("Schedule email")
    assert jarvis._active_draft_id == 9
    assert jarvis._email_workflow.awaiting == "recipient"
    stt.resume.assert_not_called()


def test_unknown_draft_has_actionable_error(harness: tuple) -> None:
    _, service, _, _ = harness
    with pytest.raises(EmailError, match="list your email drafts"):
        service.get(999)
