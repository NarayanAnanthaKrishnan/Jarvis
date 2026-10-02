import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import config
import dateparser
from calendar_agent.models import CalendarEvent, CalendarResult
from email_agent.models import EmailError, ProviderError
from email_agent.store import EmailStore
from email_agent.timing import parse_send_time
from ops.tracer import trace


class CalendarService:
    def __init__(self, store: EmailStore, provider: Any, clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.provider = provider
        self.clock = clock
        self._recover_interrupted_actions()

    def _recover_interrupted_actions(self) -> None:
        with self.store.transaction() as conn:
            now = self.clock()
            conn.execute("UPDATE calendar_actions SET status='delivery_unknown',error_code='process_interrupted',updated_at=? WHERE account=? AND status='processing'",
                         (now, self.provider.account))
            conn.execute("UPDATE calendar_actions SET status='expired',updated_at=? WHERE account=? AND status='awaiting_confirmation'",
                         (now, self.provider.account))

    def prepare(self, action: str, context: Any, title: str | None = None, when: str | None = None,
                timezone_name: str | None = None, duration_minutes: int | None = None,
                attendees: list[str] | None = None, description: str | None = None,
                location: str | None = None, managed_id: int | None = None) -> CalendarResult:
        context.check_active()
        if action not in {"create", "update", "cancel"}:
            raise EmailError("Calendar action must be create, update, or cancel")
        zone_name = timezone_name or config.EMAIL_TIMEZONE
        try:
            zone = ZoneInfo(zone_name)
        except Exception as exc:
            raise EmailError("Use a valid IANA timezone such as America/New_York") from exc
        old_row = None
        old = None
        if action != "create":
            if type(managed_id) is not int or managed_id < 1:
                context.calendar_workflow.awaiting = "event"
                return CalendarResult("needs_details", "Which Jarvis meeting ID should I change? Ask me to list your calendar events.")
            old_row = self._managed(managed_id)
            old = CalendarEvent.deserialize(old_row["payload"])
        if action == "cancel":
            return self._save_action(action, context, managed_id, old_row["event_id"], old, zone_name)
        if action == "update" and old:
            title = title or old.summary
            when = when or old.start
            zone_name = timezone_name or old.timezone
            duration_minutes = duration_minutes or int((datetime.fromisoformat(old.end) - datetime.fromisoformat(old.start)).total_seconds() // 60)
            attendees = attendees if attendees is not None else old.attendees
            description = description if description is not None else old.description
            location = location if location is not None else old.location
        title = title.strip() if isinstance(title, str) else ""
        if not title:
            context.calendar_workflow.awaiting = "title"
            return CalendarResult("needs_details", "What should I call the meeting?")
        if not when:
            context.calendar_workflow.awaiting = "time"
            return CalendarResult("needs_details", "When should the meeting start? Please give one future date and time.")
        if not attendees:
            context.calendar_workflow.awaiting = "attendee"
            return CalendarResult("needs_details", "What is the complete attendee email address?")
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        start = parse_send_time(when, zone_name, now)
        duration = 30 if duration_minutes is None else duration_minutes
        if type(duration) is not int or duration < 5 or duration > 720:
            raise EmailError("Meeting duration must be between 5 minutes and 12 hours")
        local_start = start.astimezone(zone)
        local_end = (start + timedelta(minutes=duration)).astimezone(zone)
        event = CalendarEvent(title, local_start.isoformat(), local_end.isoformat(), zone_name,
                              attendees, description or "", location or "").normalized()
        return self._save_action(action, context, managed_id, old_row["event_id"] if old_row else uuid.uuid4().hex,
                                 event, zone_name)

    def _save_action(self, action: str, context: Any, managed_id: int | None, event_id: str,
                     event: CalendarEvent | None, zone_name: str) -> CalendarResult:
        now = self.clock()
        serialized = event.serialize() if event else "{}"
        key = hashlib.sha256(json.dumps([context.session_id, action, managed_id, serialized], sort_keys=True).encode()).hexdigest()
        with self.store.transaction() as conn:
            old = conn.execute("SELECT * FROM calendar_actions WHERE request_key=?", (key,)).fetchone()
            if old:
                row = dict(old)
                if row["status"] == "awaiting_confirmation":
                    return self._preview(row, event)
                return CalendarResult(row["status"], "This meeting action was already attempted. Ask me to check its status.", row["id"], managed_id)
            action_id = conn.execute("INSERT INTO calendar_actions(account,session_id,action,managed_id,event_id,payload,request_key,expires_at,status,updated_at) VALUES(?,?,?,?,?,?,?,?, 'awaiting_confirmation',?)",
                                     (self.provider.account, context.session_id, action, managed_id, event_id,
                                      serialized, key, now + config.EMAIL_CONFIRM_SECONDS, now)).lastrowid
            row = dict(conn.execute("SELECT * FROM calendar_actions WHERE id=?", (action_id,)).fetchone())
        result = self._preview(row, event)
        trace("calendar", action_id=action_id, managed_id=managed_id, status="awaiting_confirmation")
        return result

    def _preview(self, row: dict, event: CalendarEvent | None = None) -> CalendarResult:
        action_id = row["id"]
        event = event or (CalendarEvent.deserialize(row["payload"]) if row["payload"] != "{}" else None)
        if row["action"] == "cancel":
            assert event is not None
            preview = "\n".join((f"Action: Cancel Jarvis meeting {row['managed_id']}", f"Title: {event.summary}",
                                  f"When: {event.start} to {event.end} ({event.timezone})",
                                  f"Attendees: {', '.join(event.attendees)}", "Calendar: primary"))
        else:
            assert event is not None
            preview = "\n".join((f"Action: {row['action'].title()} meeting", f"Title: {event.summary}",
                                  f"When: {event.start} to {event.end} ({event.timezone})",
                                  f"Attendees: {', '.join(event.attendees)}", f"Location: {event.location or '(none)'}",
                                  f"Description: {event.description or '(none)'}"))
        preview += f"\n\nTo approve, say: Confirm meeting {action_id}.\nConfirmation expires in ten minutes or when this session ends."
        return CalendarResult("awaiting_confirmation", f"Review meeting action {action_id}, then say confirm meeting {action_id}.",
                              action_id, row["managed_id"], preview, action_id)

    def mark_presented(self, action_id: int, session_id: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE calendar_actions SET presented=1 WHERE id=? AND session_id=? AND status='awaiting_confirmation' AND expires_at>?",
                         (action_id, session_id, self.clock()))

    def end_session(self, session_id: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE calendar_actions SET status='expired',updated_at=? WHERE session_id=? AND status='awaiting_confirmation'",
                         (self.clock(), session_id))

    def confirm(self, action_id: int, session_id: str, cancelled: Any = None) -> CalendarResult:
        now = self.clock()
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM calendar_actions WHERE id=?", (action_id,)).fetchone()
            if row is None or row["session_id"] != session_id or row["status"] != "awaiting_confirmation" or not row["presented"]:
                raise EmailError("There is no matching, displayed meeting confirmation in this session")
            if row["expires_at"] <= now:
                conn.execute("UPDATE calendar_actions SET status='expired',updated_at=? WHERE id=?", (now, action_id))
                raise EmailError("Meeting confirmation expired; request a new preview")
            if row["account"].lower() != self.provider.account.lower():
                raise EmailError("The connected Google account changed")
            if cancelled is not None and cancelled.is_set():
                raise EmailError("Session ended before confirmation")
            conn.execute("UPDATE calendar_actions SET status='processing',updated_at=? WHERE id=?", (now, action_id))
            action = row["action"]
            event_id = row["event_id"]
            payload = row["payload"]
            managed_id = row["managed_id"]
        try:
            if action == "create":
                event = CalendarEvent.deserialize(payload)
                value = self.provider.insert(event_id, self._api_body(event))
                managed_id = self._save_event(event_id, event)
            elif action == "update":
                event = CalendarEvent.deserialize(payload)
                current_row = self._managed(managed_id)
                current_event = CalendarEvent.deserialize(current_row["payload"])
                current_remote = self.provider.get(event_id)
                if not current_remote or not self._matches(current_remote, current_event):
                    self._set_action_status(action_id, "needs_review", "event_changed")
                    return CalendarResult("needs_review", f"Meeting {managed_id} changed in Google Calendar since Jarvis last checked it. Review it in Calendar before requesting another update.", action_id, managed_id)
                value = self.provider.update(event_id, self._api_body(event))
                managed_id = self._save_event(event_id, event, managed_id)
            else:
                current_row = self._managed(managed_id)
                current_event = CalendarEvent.deserialize(current_row["payload"])
                current_remote = self.provider.get(event_id)
                if not current_remote or not self._matches(current_remote, current_event):
                    self._set_action_status(action_id, "needs_review", "event_changed")
                    return CalendarResult("needs_review", f"Meeting {managed_id} changed in Google Calendar since Jarvis last checked it. Review it in Calendar before requesting cancellation.", action_id, managed_id)
                self.provider.delete(event_id)
                self._set_event_status(managed_id, "cancelled")
                value = {}
        except ProviderError as exc:
            if exc.uncertain:
                try:
                    found = self.provider.get(event_id)
                except ProviderError:
                    found = None
                if found and action in ("create", "update"):
                    event = CalendarEvent.deserialize(payload)
                    if self._matches(found, event):
                        managed_id = self._save_event(event_id, event, managed_id)
                        value = found
                    else:
                        self._set_action_status(action_id, "delivery_unknown", exc.code)
                        return CalendarResult("delivery_unknown", f"Meeting action {action_id} has an uncertain outcome. Check Google Calendar before retrying.", action_id, managed_id)
                elif not found and action == "cancel":
                    self._set_event_status(managed_id, "cancelled")
                    value = {}
                else:
                    self._set_action_status(action_id, "delivery_unknown", exc.code)
                    return CalendarResult("delivery_unknown", f"Meeting action {action_id} has an uncertain outcome. Check Google Calendar before retrying.", action_id, managed_id)
            else:
                self._set_action_status(action_id, "failed", exc.code)
                return CalendarResult("failed", f"Google Calendar rejected meeting action {action_id}. Check the connection and status before retrying.", action_id, managed_id)
        except Exception as exc:
            self._set_action_status(action_id, "delivery_unknown", type(exc).__name__)
            return CalendarResult("delivery_unknown", f"Meeting action {action_id} has an uncertain outcome. Check Google Calendar before retrying.", action_id, managed_id)
        self._set_action_status(action_id, "completed")
        trace("calendar", action_id=action_id, managed_id=managed_id, status="completed")
        if action == "cancel":
            return CalendarResult("cancelled", f"Meeting {managed_id} was cancelled and guests were notified.", action_id, managed_id)
        start = value.get("start", {}).get("dateTime", "")
        message = f"Meeting {managed_id} was added to Google Calendar for {start}. Guest invitations were sent." if action == "create" else f"Meeting {managed_id} was updated and guests were notified."
        return CalendarResult("completed", message, action_id, managed_id, data={"html_link": value.get("htmlLink")})

    def _save_event(self, event_id: str, event: CalendarEvent, managed_id: int | None = None) -> int:
        with self.store.transaction() as conn:
            if managed_id is None:
                conn.execute("INSERT INTO calendar_events(account,event_id,payload,status,updated_at) VALUES(?,?,?,'active',?)",
                             (self.provider.account, event_id, event.serialize(), self.clock()))
                return int(conn.execute("SELECT id FROM calendar_events WHERE account=? AND event_id=?", (self.provider.account, event_id)).fetchone()[0])
            conn.execute("UPDATE calendar_events SET payload=?,revision=revision+1,status='active',updated_at=? WHERE id=? AND account=?",
                         (event.serialize(), self.clock(), managed_id, self.provider.account))
        return managed_id

    def _managed(self, managed_id: int) -> dict:
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM calendar_events WHERE id=? AND account=? AND status='active'", (managed_id, self.provider.account)).fetchone()
        if row is None:
            raise EmailError("That Jarvis meeting ID was not found. Ask me to list managed meetings.")
        return dict(row)

    def _set_event_status(self, managed_id: int | None, status: str) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE calendar_events SET status=?,updated_at=? WHERE id=? AND account=?", (status, self.clock(), managed_id, self.provider.account))

    def _set_action_status(self, action_id: int, status: str, error_code: str | None = None) -> None:
        with self.store.transaction() as conn:
            conn.execute("UPDATE calendar_actions SET status=?,error_code=?,updated_at=? WHERE id=?", (status, error_code, self.clock(), action_id))

    def _api_body(self, event: CalendarEvent) -> dict:
        return {"summary": event.summary, "description": event.description, "location": event.location,
                "start": {"dateTime": event.start, "timeZone": event.timezone},
                "end": {"dateTime": event.end, "timeZone": event.timezone},
                "attendees": [{"email": address} for address in event.attendees]}

    @staticmethod
    def _matches(remote: dict, event: CalendarEvent) -> bool:
        try:
            remote_start = datetime.fromisoformat(remote.get("start", {}).get("dateTime", ""))
            remote_end = datetime.fromisoformat(remote.get("end", {}).get("dateTime", ""))
            return (remote.get("summary") == event.summary
                    and remote_start == datetime.fromisoformat(event.start)
                    and remote_end == datetime.fromisoformat(event.end)
                    and remote.get("description", "") == event.description
                    and remote.get("location", "") == event.location
                    and sorted(item.get("email", "").lower() for item in remote.get("attendees", [])) == sorted(value.lower() for value in event.attendees))
        except (AttributeError, TypeError, ValueError):
            return False

    def list_events(self, start: str | None = None, end: str | None = None) -> CalendarResult:
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        zone_name = config.EMAIL_TIMEZONE
        try:
            zone = ZoneInfo(zone_name)
        except Exception as exc:
            raise EmailError("Use a valid IANA timezone such as America/New_York") from exc
        start_at = _parse_range_boundary(start, now, zone, is_end=False) if start else now
        end_at = _parse_range_boundary(end, start_at, zone, is_end=True) if end else start_at + timedelta(days=7)
        if end_at <= start_at:
            raise EmailError("Calendar range end must be after its start")
        items = self.provider.list_events(start_at.isoformat(), end_at.isoformat(), 20)
        managed: dict[str, int] = {}
        with self.store.transaction() as conn:
            rows = conn.execute("SELECT event_id,id FROM calendar_events WHERE account=? AND status='active'", (self.provider.account,)).fetchall()
            managed = {row["event_id"]: row["id"] for row in rows}
        lines = []
        for item in items:
            event_start = item.get("start", {}).get("dateTime", item.get("start", {}).get("date", "time unavailable"))
            local_id = managed.get(item.get("id", ""))
            title = str(item.get("summary", "Untitled meeting"))[:200]
            lines.append(f"{local_id if local_id is not None else 'external'}: {title} — {event_start}")
        return CalendarResult("listed", "\n".join(lines) if lines else "No events found in that date range.", data={"count": len(lines)})


def _parse_range_boundary(value: str, base: datetime, zone: ZoneInfo, is_end: bool) -> datetime:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise EmailError("Provide a readable calendar date or time range")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = dateparser.parse(text, languages=["en"], settings={
            "RELATIVE_BASE": base.astimezone(zone).replace(tzinfo=None),
            "PREFER_DATES_FROM": "past" if is_end else "future",
            "RETURN_AS_TIMEZONE_AWARE": False,
        })
    if parsed is None:
        raise EmailError("Could not read that calendar range. Give a date such as yesterday or next Friday.")
    if parsed.tzinfo is None:
        if not re.search(r"\b(?:\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))\b", text, re.I):
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999) if is_end else parsed.replace(hour=0, minute=0, second=0, microsecond=0)
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(timezone.utc)
