from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any
from zoneinfo import ZoneInfo
import uuid
from unittest.mock import Mock

import pytest

import config
from agent.context import TurnContext
from calendar_agent.models import CalendarResult
from calendar_agent.service import CalendarService, _parse_range_boundary
from email_agent.models import EmailError, ProviderError
from email_agent.store import EmailStore


class FakeCalendar:
    account = "calendar@example.com"

    def __init__(self) -> None:
        self.events: dict[str, dict] = {}
        self.insert_calls: list[tuple[str, dict]] = []
        self.update_calls: list[tuple[str, dict]] = []
        self.delete_calls: list[str] = []

    def insert(self, event_id: str, body: dict) -> dict:
        self.insert_calls.append((event_id, body))
        value = {**body, "id": event_id}
        self.events[event_id] = value
        return value

    def update(self, event_id: str, body: dict) -> dict:
        self.update_calls.append((event_id, body))
        self.events[event_id] = {**self.events[event_id], **body, "id": event_id}
        return self.events[event_id]

    def delete(self, event_id: str) -> None:
        self.delete_calls.append(event_id)
        self.events.pop(event_id, None)

    def get(self, event_id: str) -> dict | None:
        return self.events.get(event_id)

    def list_events(self, start: str, end: str, limit: int = 20) -> list[dict]:
        return list(self.events.values())[:limit]

    def connection_current(self) -> bool:
        return True


class Context:
    session_id = "calendar-test-session"
    calendar_workflow = type("Workflow", (), {"awaiting": None})()

    def check_active(self) -> None:
        return None


@pytest.fixture
def calendar_harness(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(config, "EMAIL_CONFIRM_SECONDS", 600)
    provider = FakeCalendar()
    now = datetime(2030, 1, 1, 12, tzinfo=timezone.utc).timestamp()
    db_path = Path(__file__).with_name(f"calendar-test-{uuid.uuid4().hex}.db")
    service = CalendarService(EmailStore(db_path), provider, clock=lambda: now)
    try:
        yield service, provider, Context()
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(db_path) + suffix).unlink()
            except FileNotFoundError:
                pass


def prepare(service: CalendarService, context: Context, **overrides: Any) -> CalendarResult:
    values = {"action": "create", "title": "Project review", "when": "2030-01-03T15:00:00+00:00",
              "attendees": ["alex@example.com"]}
    values.update(overrides)
    return service.prepare(context=context, **values)


def approve(service: CalendarService, result: CalendarResult, context: Context) -> CalendarResult:
    assert result.confirmation_id is not None
    service.mark_presented(result.confirmation_id, context.session_id)
    return service.confirm(result.confirmation_id, context.session_id)


def test_create_requires_displayed_exact_confirmation_before_google_write(calendar_harness: tuple) -> None:
    service, provider, context = calendar_harness
    preview = prepare(service, context)
    assert preview.status == "awaiting_confirmation"
    assert "Confirm meeting 1" in preview.preview
    assert not provider.insert_calls

    result = approve(service, preview, context)
    assert result.status == "completed"
    assert result.managed_id == 1
    assert len(provider.insert_calls) == 1
    assert provider.insert_calls[0][1]["attendees"] == [{"email": "alex@example.com"}]
    assert service.store is not None
    with pytest.raises(EmailError, match="no matching|displayed"):
        service.confirm(preview.confirmation_id, context.session_id)


def test_update_and_cancel_each_require_their_own_preview(calendar_harness: tuple) -> None:
    service, provider, context = calendar_harness
    created = approve(service, prepare(service, context), context)

    update_preview = service.prepare("update", context, managed_id=created.managed_id,
                                     when="2030-01-04T16:00:00+00:00")
    assert update_preview.confirmation_id != created.confirmation_id
    assert len(provider.update_calls) == 0
    updated = approve(service, update_preview, context)
    assert updated.status == "completed"
    assert len(provider.update_calls) == 1

    cancel_preview = service.prepare("cancel", context, managed_id=created.managed_id)
    assert "Cancel Jarvis meeting 1" in cancel_preview.preview
    assert not provider.delete_calls
    cancelled = approve(service, cancel_preview, context)
    assert cancelled.status == "cancelled"
    assert len(provider.delete_calls) == 1


def test_voice_cancel_request_is_not_rewritten_to_create(calendar_harness: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from agent import loop
    from agent.context import TurnContext

    service, provider, _ = calendar_harness
    initial_context = Context()
    created = approve(service, prepare(service, initial_context), initial_context)
    decision = {"tool": "calendar_prepare", "args": {"action": "cancel", "event_id": created.managed_id},
                "calendar_workflow": {"action": "cancel", "title": None, "when": None, "timezone": None,
                                      "duration_minutes": 30, "attendees": [], "description": "", "location": "",
                                      "awaiting": None, "event_id": created.managed_id}}
    llm = Mock()
    llm.call_raw.return_value = {"message": {"content": json.dumps(decision)}}
    monkeypatch.setattr(config, "JEV_ROUTING_ENABLED", False)
    monkeypatch.setattr(loop, "load_profile", lambda: "")
    context = TurnContext(session_id="calendar-test-session", calendar_service=service)

    result = loop.run_agent("Cancel meeting one on my calendar", [], llm, context)

    assert result["previews"][0]["id"] is not None
    assert "Action: Cancel Jarvis meeting 1" in result["previews"][0]["text"]
    assert len(provider.insert_calls) == 1
    assert not provider.delete_calls


def test_uncertain_insert_reconciles_by_stable_event_id_without_duplicate(calendar_harness: tuple) -> None:
    service, provider, context = calendar_harness
    original_insert = provider.insert

    def accepted_then_timeout(event_id: str, body: dict) -> dict:
        original_insert(event_id, body)
        raise ProviderError("calendar_connection_failed", uncertain=True)

    provider.insert = accepted_then_timeout
    preview = prepare(service, context)
    result = approve(service, preview, context)
    assert result.status == "completed"
    assert len(provider.insert_calls) == 1
    assert service.store is not None


def test_external_change_after_preview_stops_an_update_for_review(calendar_harness: tuple) -> None:
    service, provider, context = calendar_harness
    created = approve(service, prepare(service, context), context)
    preview = service.prepare("update", context, managed_id=created.managed_id,
                              when="2030-01-04T16:00:00+00:00")
    event_id = next(iter(provider.events))
    provider.events[event_id]["summary"] = "Changed in Google Calendar"
    result = approve(service, preview, context)
    assert result.status == "needs_review"
    assert not provider.update_calls


def test_restart_marks_inflight_calendar_write_uncertain(calendar_harness: tuple) -> None:
    service, provider, context = calendar_harness
    preview = prepare(service, context)
    with service.store.transaction() as conn:
        conn.execute("UPDATE calendar_actions SET status='processing' WHERE id=?", (preview.action_id,))
    restarted = CalendarService(service.store, provider, clock=service.clock)
    with restarted.store.transaction() as conn:
        row = conn.execute("SELECT status,error_code FROM calendar_actions WHERE id=?", (preview.action_id,)).fetchone()
    assert tuple(row) == ("delivery_unknown", "process_interrupted")


def test_new_service_expires_an_unconfirmed_preview_from_old_session(calendar_harness: tuple) -> None:
    service, provider, context = calendar_harness
    preview = prepare(service, context)
    CalendarService(service.store, provider, clock=service.clock)
    with service.store.transaction() as conn:
        row = conn.execute("SELECT status FROM calendar_actions WHERE id=?", (preview.action_id,)).fetchone()
    assert row[0] == "expired"


def test_range_parser_accepts_past_dates_and_date_only_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EMAIL_TIMEZONE", "America/New_York")
    base = datetime(2026, 10, 2, 14, tzinfo=timezone.utc)
    yesterday_start = _parse_range_boundary("yesterday", base, ZoneInfo("America/New_York"), False)
    yesterday_end = _parse_range_boundary("yesterday", base, ZoneInfo("America/New_York"), True)
    assert yesterday_start < yesterday_end
    assert (yesterday_end - yesterday_start).total_seconds() == pytest.approx(86399.999999, abs=1)


def test_calendar_insert_notifies_guests() -> None:
    from unittest.mock import Mock
    from calendar_agent.google_calendar import GoogleCalendarProvider

    provider = object.__new__(GoogleCalendarProvider)
    provider._execute = Mock(return_value={"id": "event"})
    provider.api = Mock()
    provider.insert("event", {"summary": "Review"})
    assert provider.api.events.return_value.insert.call_args.kwargs["sendUpdates"] == "all"
    assert provider._execute.call_args.kwargs["write"] is True


def test_v1_email_database_migrates_additively_to_calendar_schema() -> None:
    db_path = Path(__file__).with_name(f"calendar-migration-{uuid.uuid4().hex}.db")
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE drafts (id INTEGER PRIMARY KEY, account TEXT NOT NULL, remote_id TEXT, payload TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL, operation_key TEXT UNIQUE NOT NULL, updated_at REAL NOT NULL)")
            conn.execute("INSERT INTO drafts(id,account,payload,status,operation_key,updated_at) VALUES(1,'calendar@example.com','{}','draft','existing-draft',0)")
            conn.execute("PRAGMA user_version=1")
            conn.commit()
        finally:
            conn.close()
        store = EmailStore(db_path)
        with store.transaction() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
            assert conn.execute("SELECT operation_key FROM drafts WHERE id=1").fetchone()[0] == "existing-draft"
            assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='calendar_events'").fetchone()
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(db_path) + suffix).unlink()
            except FileNotFoundError:
                pass
