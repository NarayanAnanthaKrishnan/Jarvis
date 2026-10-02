from datetime import datetime, timezone
from typing import Any

import config
from calendar_agent.contracts import TOOL_SCHEMAS
from calendar_agent.google_calendar import GoogleCalendarProvider
from calendar_agent.models import CalendarResult
from calendar_agent.service import CalendarService
from email_agent.addressing import normalize_spoken_address, speak_address
from email_agent.contracts import validation_errors
from email_agent.gmail import GmailProvider
from email_agent.models import EmailError, addresses
from email_agent.store import EmailStore
from email_agent.timing import extract_time_phrase, parse_send_time


def get_service() -> CalendarService:
    if not config.EMAIL_ENABLED:
        raise EmailError("Google Workspace actions are disabled. Enable EMAIL_ENABLED and connect the account first.")
    gmail = GmailProvider()
    provider = GoogleCalendarProvider(gmail.credentials, gmail.account)
    provider.account = gmail.account
    return CalendarService(EmailStore(config.EMAIL_DB_PATH), provider)


def execute_calendar(name: str, args: dict, context: Any) -> CalendarResult:
    context.check_active()
    context.calendar_touched = True
    if name not in TOOL_SCHEMAS:
        raise EmailError("Unknown calendar operation")
    errors = validation_errors(args, TOOL_SCHEMAS[name])
    if errors:
        return CalendarResult("invalid_arguments", "I couldn't interpret the meeting details. Please repeat the action, date and attendee address.")
    if context.calendar_service is None:
        context.calendar_service = get_service()
    service = context.calendar_service
    if name == "calendar_list":
        return service.list_events(args.get("start_after"), args.get("end_before"))
    values = dict(args)
    workflow = context.calendar_workflow
    workflow.update({key: value for key, value in values.items() if value is not None})
    workflow.action = values["action"]
    if values.get("event_id") is not None:
        workflow.event_id = values["event_id"]
    if values.get("when"):
        try:
            phrase = extract_time_phrase(values["when"]) or values["when"]
            due = parse_send_time(phrase, values.get("timezone") or workflow.timezone or config.EMAIL_TIMEZONE,
                                  datetime.fromtimestamp(service.clock(), timezone.utc))
        except EmailError as exc:
            workflow.awaiting = "time"
            return CalendarResult("needs_details", str(exc))
        workflow.when = due.isoformat()
        workflow.timezone = values.get("timezone") or workflow.timezone or config.EMAIL_TIMEZONE
    if values.get("attendees"):
        normalized = []
        for address in values["attendees"]:
            candidate = normalize_spoken_address(address)
            if candidate is None:
                normalized.extend(addresses([address]))
            elif candidate[1]:
                workflow.pending_attendee = {"address": candidate[0]}
                workflow.awaiting = "attendee_confirmation"
                return CalendarResult("needs_details", f"I heard {speak_address(candidate[0])}. Is that correct? Say yes to add it, or say no and repeat the full address.")
            else:
                normalized.append(candidate[0])
        if normalized:
            workflow.attendees = addresses(normalized)
    try:
        result = service.prepare(workflow.action, context, title=workflow.title, when=workflow.when,
                                 timezone_name=workflow.timezone, duration_minutes=workflow.duration_minutes,
                                 attendees=workflow.attendees or None, description=workflow.description,
                                 location=workflow.location, managed_id=workflow.event_id)
    except EmailError as exc:
        workflow.awaiting = "attendee" if "attendee" in str(exc).lower() else "time" if workflow.action == "create" else "details"
        return CalendarResult("needs_details", str(exc))
    if result.status != "needs_details":
        workflow.awaiting = None
    if result.status == "needs_details" and result.message.startswith("Which Jarvis meeting ID"):
        workflow.awaiting = "event"
    return result
