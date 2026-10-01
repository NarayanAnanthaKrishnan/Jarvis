from datetime import datetime, timezone
from typing import Any

import config
from email_agent.addressing import normalize_spoken_address, speak_address
from email_agent.contracts import TOOL_SCHEMAS, validation_errors
from email_agent.models import EmailError, EmailResult, addresses
from email_agent.timing import parse_send_time


def get_service() -> Any:
    if not config.EMAIL_ENABLED:
        raise EmailError("Email is disabled. Connect Gmail and set EMAIL_ENABLED=true in .env first.")
    from email_agent.gmail import GmailProvider
    from email_agent.service import EmailService
    from email_agent.store import EmailStore
    return EmailService(EmailStore(config.EMAIL_DB_PATH), GmailProvider())


def execute_email(name: str, args: dict, context: Any, llm: Any) -> EmailResult:
    context.check_active()
    context.email_touched = True
    context.email_stage = "argument_validation"
    if name not in TOOL_SCHEMAS:
        raise EmailError("Unknown email operation")
    errors = validation_errors(args, TOOL_SCHEMAS[name])
    if errors:
        return EmailResult("invalid_arguments", "I could not interpret the email details. Please repeat the recipient and requested action.", data={"validation_errors": errors})
    held_recipient = None
    if name in ("email_draft", "email_recipients"):
        args = dict(args)
        for field in ("to", "cc", "bcc"):
            if args.get(field) is not None:
                normalized = []
                for value in args[field]:
                    candidate = normalize_spoken_address(value)
                    if candidate is None:
                        normalized.append(value)
                    elif candidate[1]:
                        if context.email_workflow.pending_recipient is not None:
                            return EmailResult("needs_details", "Please confirm the previously read-back address first.", context.active_draft_id)
                        held_recipient = {"field": field, "address": candidate[0],
                                          "draft_id": args.get("draft_id") or context.active_draft_id,
                                          "mode": args.get("mode") or ("add" if context.email_workflow.awaiting == "recipient" else "replace")}
                        context.email_workflow.pending_recipient = held_recipient
                        if field not in context.email_workflow.unresolved_recipients:
                            context.email_workflow.unresolved_recipients.append(field)
                    else:
                        normalized.append(candidate[0])
                if normalized:
                    args[field] = normalized
                else:
                    args.pop(field, None)
        if held_recipient and name == "email_recipients":
            if held_recipient["draft_id"] is None:
                raise EmailError("Retrieve the draft before changing its recipient")
            context.email_workflow.awaiting = "recipient_confirmation"
            prompt = f"I heard {speak_address(held_recipient['address'])}. Is that correct? Say yes to add it, or say no and spell the address."
            return EmailResult("needs_details", prompt, held_recipient["draft_id"])
        if held_recipient and name == "email_draft" and any(args.get(field) for field in ("to", "cc", "bcc")):
            held_recipient["mode"] = "add"
        for field in ("to", "cc", "bcc"):
            if args.get(field) is not None:
                try:
                    addresses(args[field])
                except EmailError:
                    if field not in context.email_workflow.unresolved_recipients:
                        context.email_workflow.unresolved_recipients.append(field)
                    if name == "email_recipients":
                        context.email_workflow.awaiting = "recipient"
                        raise EmailError("What is the complete recipient email address?")
                    args = dict(args)
                    valid = []
                    for address in args[field]:
                        try:
                            valid.extend(addresses([address]))
                        except EmailError:
                            continue
                    if valid:
                        args[field] = valid
                    else:
                        args.pop(field)
                    context.email_workflow.awaiting = "recipient"
                else:
                    if field in context.email_workflow.unresolved_recipients:
                        context.email_workflow.unresolved_recipients.remove(field)
    context.email_stage = "service_initialization"
    if context.email_service is None:
        context.email_service = get_service()
    service = context.email_service
    context.email_stage = name
    _anchor_schedule(context)
    if name == "email_draft":
        values = dict(args)
        if values.get("draft_id") is None and context.active_draft_id is not None:
            values["draft_id"] = context.active_draft_id
        result = _advance(service.draft(context=context, llm=llm, **values), context)
        if held_recipient:
            held_recipient["draft_id"] = result.draft_id
            context.email_workflow.pending_recipient = held_recipient
            context.email_workflow.awaiting = "recipient_confirmation"
            result.message += f" I heard {speak_address(held_recipient['address'])}. Is that correct? Say yes to add it, or say no and spell the address."
        return result
    if name == "email_recipients":
        values = dict(args)
        if "mode" not in values:
            values["mode"] = "add" if context.email_workflow.awaiting == "recipient" else "replace"
        return _advance(service.recipients(context=context, **values), context)
    if name == "email_get":
        result = service.get(**args)
        context.active_draft_id = result.draft_id
        return result
    if name == "email_prepare":
        values = dict(args)
        workflow = context.email_workflow
        workflow.action = values["action"]
        workflow.update({"when": values.get("when") or workflow.when, "timezone": values.get("timezone") or workflow.timezone})
        _anchor_schedule(context)
        if workflow.unresolved_recipients:
            workflow.awaiting = "recipient"
            return EmailResult("needs_details", "What is the complete address for the remaining recipient?", values["draft_id"])
        if workflow.action == "schedule" and not workflow.when:
            draft = service.get(values["draft_id"])
            context.active_draft_id = draft.draft_id
            workflow.awaiting = "time"
            return EmailResult("needs_details", "When would you like this draft to be sent? Please include a date and time.", values["draft_id"], preview=draft.preview)
        try:
            result = service.prepare(values["draft_id"], workflow.action, context, when=workflow.resolved_when or workflow.when, timezone_name=workflow.timezone)
        except EmailError as exc:
            workflow.awaiting = "recipient" if "recipient" in str(exc).lower() else "time" if workflow.action == "schedule" else "details"
            return EmailResult("needs_details", str(exc), values["draft_id"])
        workflow.awaiting = None
        return result
    if name == "email_list":
        return service.list(**args)
    if name == "email_cancel":
        result = service.cancel(**args)
        context.email_workflow.action = "draft"
        context.email_workflow.awaiting = None
        context.email_workflow.when = None
        context.email_workflow.resolved_when = None
        return result
    raise EmailError("Unknown email operation")


def _anchor_schedule(context: Any) -> None:
    workflow = context.email_workflow
    if workflow.action != "schedule" or not workflow.when or workflow.resolved_when:
        return
    try:
        due = parse_send_time(workflow.when, workflow.timezone or config.EMAIL_TIMEZONE,
                              datetime.fromtimestamp(context.email_service.clock(), timezone.utc))
    except EmailError:
        return
    workflow.resolved_when = due.isoformat()


def _advance(result: EmailResult, context: Any) -> EmailResult:
    if result.status != "draft" or not isinstance(result.data, dict):
        return result
    workflow = context.email_workflow
    if result.data.get("missing_recipient") or workflow.unresolved_recipients:
        workflow.awaiting = "recipient"
        result.message += " What is the recipient's email address?"
    elif workflow.action == "schedule" and not workflow.when:
        workflow.awaiting = "time"
        result.message += " When would you like it sent? Please include a date and time."
    elif workflow.action in ("send", "schedule"):
        try:
            prepared = context.email_service.prepare(result.draft_id, workflow.action, context,
                                                       when=workflow.resolved_when or workflow.when, timezone_name=workflow.timezone)
        except EmailError as exc:
            workflow.awaiting = "time" if workflow.action == "schedule" else "details"
            result.message += " " + str(exc)
        else:
            workflow.awaiting = None
            return prepared
    else:
        workflow.awaiting = None
    result.data["workflow_complete"] = True
    return result
