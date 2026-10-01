import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from agent.context import TurnContext
from email_agent.models import EmailError, EmailPayload, ProviderError
from email_agent.service import EmailService, confirmation_number
from email_agent.store import EmailStore
from email_agent.timing import parse_send_time


class FakeProvider:
    account = "owner@example.com"

    def __init__(self) -> None:
        self.drafts: dict[str, EmailPayload] = {}
        self.sent: list[EmailPayload] = []
        self.error: Exception | None = None
        self.on_send: Any = None

    def create_draft(self, payload: EmailPayload) -> str:
        key = str(len(self.drafts) + 1)
        self.drafts[key] = payload
        return key

    def update_draft(self, draft_id: str, payload: EmailPayload) -> None:
        self.drafts[draft_id] = payload

    def get_draft(self, draft_id: str) -> EmailPayload:
        if draft_id not in self.drafts:
            raise ProviderError("draft_missing")
        return self.drafts[draft_id]

    def send_draft(self, draft_id: str, payload: EmailPayload) -> str:
        if self.on_send:
            self.on_send()
        if self.error:
            raise self.error
        self.sent.append(payload)
        del self.drafts[draft_id]
        return "sent-" + str(len(self.sent))


@pytest.fixture
def setup(tmp_path: Path) -> tuple:
    now = [datetime(2026, 9, 30, 16, tzinfo=timezone.utc).timestamp()]
    provider = FakeProvider()
    store = EmailStore(tmp_path / "email.db")
    service = EmailService(store, provider, lambda: now[0])
    context = TurnContext(session_id="session", email_service=service)
    llm = Mock()
    llm.call_raw.return_value = {"message": {"content": json.dumps({"subject": "Meeting", "body": "Hi José,\nPlease bring {notes}.\nThank you."})}}
    return service, provider, context, llm, now


def create(setup: tuple) -> int:
    service, _, context, llm, _ = setup
    return service.draft("Write a meeting email", context, llm, to=["friend@example.com"]).draft_id


def approve(setup: tuple, action: str = "send") -> int:
    service, _, context, _, _ = setup
    draft_id = create(setup)
    result = service.prepare(draft_id, action, context, when="in 2 minutes" if action == "schedule" else None)
    service.mark_presented(result.job_id, context.session_id)
    service.confirm(result.job_id, context.session_id)
    return result.job_id


def test_draft_does_not_send_and_preserves_content(setup: tuple) -> None:
    service, provider, context, llm, _ = setup
    draft_id = create(setup)
    original = provider.drafts["1"]
    assert not provider.sent
    assert EmailPayload.from_raw(original.raw()) == original
    context.turn_id = "edit"
    llm.call_raw.return_value = {"message": {"content": '{"subject":"New subject","body":"Shorter."}'}}
    service.draft("Make it shorter", context, llm, draft_id=draft_id)
    assert provider.drafts["1"].to == ["friend@example.com"]
    assert len(provider.drafts) == 1


@pytest.mark.parametrize("args", [
    {}, {"instruction": ""}, {"instruction": "test", "subject": "unexpected"},
    {"instruction": "test", "to": "test@example.com"},
    {"instruction": "test", "draft_id": True}, {"instruction": "test", "draft_id": 0},
    {"instruction": "test", "bcc": [7]},
])
def test_email_arguments_rejected_before_provider_or_model(args: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    from email_agent import runtime
    from tools.registry import execute_tool
    service = Mock()
    monkeypatch.setattr(runtime, "get_service", service)
    llm = Mock()
    context = TurnContext(agent_id="email")
    result = execute_tool("email_draft", args, llm, context)
    assert result.status == "invalid_arguments"
    assert result.data["validation_errors"]
    service.assert_not_called()
    llm.call_raw.assert_not_called()
    assert not context.mutation_results


@pytest.mark.parametrize("content", ['null', '{"subject":7,"body":"Hello"}', '{"subject":"Test","body":" "}', '{"body":"Hello"}', '{"subject":"Test","body":"Hello","extra":1}'])
def test_invalid_writer_output_does_not_create_draft(setup: tuple, content: str) -> None:
    service, provider, context, llm, _ = setup
    llm.call_raw.return_value = {"message": {"content": content}}
    with pytest.raises(EmailError, match="invalid draft"):
        service.draft("test", context, llm, to=["test@example.com"])
    assert not provider.drafts
    with service.store.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 0


def test_argument_correction_then_clarification_preserves_saved_draft(setup: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import loop
    from agent.router import RouteDecision
    service, provider, context, llm, _ = setup
    monkeypatch.setattr(loop, "route_turn", lambda *args: RouteDecision(agent_id="email", memory_needed=False))
    context.email_workflow.action = "schedule"
    decisions = iter([
        {"tool": "email_draft", "args": {"instruction": "Meeting", "subject": "unsupported", "to": ["test@example.com"]}},
        {"tool": "email_draft", "args": {"instruction": "Meeting subject", "to": ["test@example.com"]}},
        {"done": True, "answer": "The draft is saved. When would you like it scheduled?"},
    ])
    monkeypatch.setattr(loop, "think", lambda *args: next(decisions))
    result = loop.run_agent("Draft and schedule a meeting email", [], llm, context)
    assert len(provider.drafts) == 1
    assert not provider.sent
    assert context.active_draft_id == 1
    assert "saved in Gmail" in result["output"]
    assert "When would" in result["output"]
    assert result["preview"]
    llm.call_raw.assert_called_once()
    assert llm.call_raw.call_args.kwargs["profile"] == "email"


def test_argument_correction_is_limited_to_one_attempt(setup: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import loop
    from agent.router import RouteDecision
    _, provider, context, llm, _ = setup
    monkeypatch.setattr(loop, "route_turn", lambda *args: RouteDecision(agent_id="email", memory_needed=False))
    decide = Mock(return_value={"tool": "email_draft", "args": {"subject": "test"}})
    monkeypatch.setattr(loop, "think", decide)
    result = loop.run_agent("Draft", [], llm, context)
    assert decide.call_count == 2
    assert "repeat" in result["output"]
    assert not provider.drafts


def test_schedule_without_time_keeps_existing_draft(setup: tuple) -> None:
    from tools.registry import execute_tool
    service, provider, context, llm, _ = setup
    draft_id = create(setup)
    context.agent_id = "email"
    result = execute_tool("email_prepare", {"draft_id": draft_id, "action": "schedule"}, llm, context)
    assert "When" in result.message
    assert service.store.draft(draft_id)["status"] == "draft"
    assert len(provider.drafts) == 1
    assert not provider.sent


def test_unexpected_email_error_logs_metadata_only(monkeypatch: pytest.MonkeyPatch, caplog: Any) -> None:
    from email_agent import runtime
    from tools.registry import execute_tool
    service = Mock()
    service.draft.side_effect = TypeError("private body and secret token")
    monkeypatch.setattr(runtime, "get_service", Mock(return_value=service))
    result = execute_tool("email_draft", {"instruction": "private body", "to": ["private@example.com"]}, Mock(), TurnContext(agent_id="email"))
    assert result.status == "error"
    assert "stage=email_draft" in caplog.text
    assert "TypeError" in caplog.text
    assert "private" not in caplog.text
    assert "secret" not in caplog.text


def test_expired_worker_claim_cannot_send(setup: tuple) -> None:
    service, provider, _, _, now = setup
    job_id = approve(setup, "schedule")
    now[0] += 120
    original_get = provider.get_draft

    def delayed_get(draft_id: str) -> EmailPayload:
        payload = original_get(draft_id)
        with service.store.transaction() as conn:
            conn.execute("UPDATE jobs SET lease_until=? WHERE id=?", (now[0] - 1, job_id))
        return payload

    provider.get_draft = delayed_get
    service.deliver_one()
    assert not provider.sent


def test_voice_draft_preview_confirmation_flow(setup: tuple, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    import agent.loop as loop
    import modes.jarvis as mode
    from agent.router import RouteDecision

    service, provider, _, llm, _ = setup
    monkeypatch.setattr(mode, "AUTO_EXTRACT", False)
    monkeypatch.setattr(loop, "route_turn", lambda *a: RouteDecision(memory_needed=False))
    monkeypatch.setattr(loop, "load_profile", lambda: "")
    decisions = [
        {"tool": "handoff_email", "args": {"instruction": "Draft and send a meeting email to friend@example.com"}},
        {"tool": "email_draft", "args": {"instruction": "Ask to meet tomorrow", "to": ["friend@example.com"]},
         "workflow": {"action": "send", "when": None, "timezone": None, "awaiting": None, "new_draft": False}},
        {"subject": "Meeting", "body": "Can we meet tomorrow?"},
    ]
    llm.call_raw.side_effect = [{"message": {"content": json.dumps(d)}} for d in decisions]
    stt = Mock()
    stt.owner = None
    jarvis = mode.JarvisMode(stt, llm, Mock(), Mock())
    jarvis._email_service = service
    jarvis._start_session()
    jarvis._on_turn("Draft and send a meeting email to friend@example.com")
    assert "Subject: Meeting" in capsys.readouterr().out
    assert service.store.job(1)["presented"] == 1
    assert not provider.sent
    assert jarvis._active_draft_id == 1
    jarvis._on_turn("Confirm email one.")
    assert len(provider.sent) == 1
    assert llm.call_raw.call_count == 3
    assert all(item["sensitive"] for item in jarvis._history)


def test_confirmation_requires_display_session_and_freshness(setup: tuple) -> None:
    service, provider, context, _, now = setup
    result = service.prepare(create(setup), "send", context)
    with pytest.raises(EmailError):
        service.confirm(result.job_id, context.session_id)
    service.mark_presented(result.job_id, context.session_id)
    with pytest.raises(EmailError):
        service.confirm(result.job_id, "another-session")
    now[0] += 601
    with pytest.raises(EmailError):
        service.confirm(result.job_id, context.session_id)
    assert not provider.sent


def test_confirm_sends_once_and_replay_rejected(setup: tuple) -> None:
    service, provider, context, _, _ = setup
    job_id = approve(setup)
    assert len(provider.sent) == 1
    assert service.store.job(job_id)["provider_message_id"] == "sent-1"
    with pytest.raises(EmailError):
        service.confirm(job_id, context.session_id)
    service.deliver_one()
    assert len(provider.sent) == 1


def test_schedule_survives_restart_and_never_sends_early(setup: tuple) -> None:
    service, provider, _, _, now = setup
    job_id = approve(setup, "schedule")
    restarted = EmailService(EmailStore(service.store.path), provider, lambda: now[0])
    restarted.deliver_one()
    assert not provider.sent
    now[0] += 120
    restarted.deliver_one()
    assert restarted.store.job(job_id)["status"] == "sent"
    assert len(provider.sent) == 1


def test_missed_deadline_does_not_send_late(setup: tuple) -> None:
    service, provider, _, _, now = setup
    job_id = approve(setup, "schedule")
    now[0] += 181
    service.deliver_one()
    assert service.store.job(job_id)["status"] == "missed"
    assert not provider.sent


def test_changed_gmail_draft_is_held(setup: tuple) -> None:
    service, provider, _, _, now = setup
    job_id = approve(setup, "schedule")
    provider.drafts["1"] = EmailPayload(provider.account, ["different@example.com"], "Changed", "Changed")
    now[0] += 120
    service.deliver_one()
    assert service.store.job(job_id)["status"] == "needs_review"
    assert not provider.sent


def test_cancel_and_reschedule_require_new_approval(setup: tuple) -> None:
    service, provider, context, _, now = setup
    job_id = approve(setup, "schedule")
    result = service.prepare(1, "schedule", context, when="in 5 minutes")
    assert service.store.job(job_id)["status"] == "needs_review"
    service.cancel(result.job_id)
    now[0] += 300
    service.deliver_one()
    assert not provider.sent and provider.drafts


def test_session_end_revokes_pending_but_not_confirmed_schedules(setup: tuple) -> None:
    service, _, context, _, _ = setup
    job_id = approve(setup, "schedule")
    service.end_session(context.session_id)
    assert service.store.job(job_id)["status"] == "scheduled"
    pending = service.prepare(1, "send", context)
    service.end_session(context.session_id)
    assert service.store.job(pending.job_id)["status"] == "expired"


@pytest.mark.parametrize("error", [TimeoutError(), ProviderError("server_error", uncertain=True)])
def test_uncertain_delivery_never_retried(setup: tuple, error: Exception) -> None:
    service, provider, _, _, now = setup
    provider.error = error
    job_id = approve(setup)
    assert service.store.job(job_id)["status"] == "delivery_unknown"
    provider.error = None
    now[0] += 10
    service.deliver_one()
    assert not provider.sent


def test_concurrent_workers_and_cancel_after_claim(setup: tuple) -> None:
    service, provider, _, _, now = setup
    job_id = approve(setup, "schedule")
    now[0] += 120
    entered, release = threading.Event(), threading.Event()
    provider.on_send = lambda: (entered.set(), release.wait(3))
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(service.deliver_one)
        assert entered.wait(2)
        second = pool.submit(service.deliver_one)
        assert second.result() is None
        with pytest.raises(EmailError):
            service.cancel(job_id)
        release.set()
        first.result()
    assert len(provider.sent) == 1


def test_interrupted_send_recovery_is_unknown(setup: tuple) -> None:
    service, _, _, _, now = setup
    job_id = approve(setup, "schedule")
    with service.store.transaction() as conn:
        conn.execute("UPDATE jobs SET status='sending',lease_until=? WHERE id=?", (now[0] - 1, job_id))
    service.maintenance()
    assert service.store.job(job_id)["status"] == "delivery_unknown"


def test_mutation_repetition_and_invalid_recipients(setup: tuple) -> None:
    service, provider, context, llm, _ = setup
    assert create(setup) == create(setup)
    assert len(provider.drafts) == 1
    with pytest.raises(EmailError):
        service.draft("Another email", context, llm, to=["Bob"])
    with pytest.raises(EmailError):
        EmailPayload(provider.account, ["ok@example.com\nBcc: bad@example.com"], "s", "b").raw()


@pytest.mark.parametrize("value", ["tomorrow", "at 9", "yesterday at 9am", "10/11 at 9am", "later today 6pm or 7pm"])
def test_ambiguous_or_past_time_rejected(value: str) -> None:
    with pytest.raises(EmailError):
        parse_send_time(value, "America/New_York", datetime(2026, 9, 30, 16, tzinfo=timezone.utc))


def test_dst_and_timezone() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for value in ["2026-03-08T02:30:00", "2026-11-01T01:30:00"]:
        with pytest.raises(EmailError):
            parse_send_time(value, "America/New_York", now)
    assert parse_send_time("2026-11-01T01:30:00-04:00", "America/New_York", now).hour == 5


@pytest.mark.parametrize("value, expected_hour", [("later today 6pm", 22), ("later today 7pm", 23), ("schedule it for later today 6pm", 22), ("today at six pm", 22)])
def test_spoken_later_today_times(value: str, expected_hour: int) -> None:
    now = datetime(2026, 10, 1, 20, tzinfo=timezone.utc)
    due = parse_send_time(value, "America/New_York", now)
    assert (due.hour, due.minute) == (expected_hour, 0)


@pytest.mark.parametrize("value, expected_minutes", [("in half an hour", 30), ("in an hour", 60), ("in a quarter hour", 15)])
def test_spoken_relative_durations(value: str, expected_minutes: int) -> None:
    now = datetime(2026, 10, 1, 20, tzinfo=timezone.utc)
    due = parse_send_time(value, "America/New_York", now)
    assert (due - now).total_seconds() == expected_minutes * 60


@pytest.mark.parametrize("text, expected", [("Confirm email seven.", 7), ("confirm email 12", 12), ("confirm email twenty three", 23), ("yes", None), ("The email says confirm email 7", None)])
def test_exact_confirmation(text: str, expected: int | None) -> None:
    assert confirmation_number(text) == expected
