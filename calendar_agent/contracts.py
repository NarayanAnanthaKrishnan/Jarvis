import json
from typing import Any

from email_agent.contracts import _ID, _OPTIONAL_TEXT, _TEXT, _object


_ATTENDEES = {"type": ["array", "null"], "items": _TEXT}
_DURATION = {"type": ["integer", "null"], "minimum": 5, "maximum": 720}

TOOL_SCHEMAS = {
    "calendar_prepare": _object({
        "action": {"type": "string", "enum": ["create", "update", "cancel"]},
        "event_id": {"type": ["integer", "null"], "minimum": 1},
        "title": _OPTIONAL_TEXT,
        "when": _OPTIONAL_TEXT,
        "timezone": _OPTIONAL_TEXT,
        "duration_minutes": _DURATION,
        "attendees": _ATTENDEES,
        "description": _OPTIONAL_TEXT,
        "location": _OPTIONAL_TEXT,
    }, ["action"]),
    "calendar_list": _object({"start_after": _OPTIONAL_TEXT, "end_before": _OPTIONAL_TEXT}, []),
}

DESCRIPTIONS = {
    "calendar_prepare": "Prepare a create, update, or cancel preview for a Google Calendar event. Infer a concise title from the meeting purpose. Create requires one clear future start time and at least one complete attendee email; missing details are asked after other work is saved. Duration defaults to 30 minutes and timezone defaults to the user's configured local timezone. Never invent attendee addresses. Every change requires a separate exact user confirmation.",
    "calendar_list": "List events in the primary calendar, bounded to the next seven days by default. Jarvis can change or cancel only events it manages.",
}

WORKFLOW_SCHEMA = _object({
    "action": {"type": "string", "enum": ["create", "update", "cancel", "list"]},
    "title": _OPTIONAL_TEXT,
    "when": _OPTIONAL_TEXT,
    "timezone": _OPTIONAL_TEXT,
    "duration_minutes": _DURATION,
    "attendees": _ATTENDEES,
    "description": _OPTIONAL_TEXT,
    "location": _OPTIONAL_TEXT,
    "awaiting": {"type": ["string", "null"], "enum": ["title", "time", "attendee", "event", "details", None]},
    "event_id": {"type": ["integer", "null"], "minimum": 1},
}, ["action", "title", "when", "timezone", "duration_minutes", "attendees", "description", "location", "awaiting", "event_id"])


def decision_schema(email_schemas: dict[str, Any], email_workflow_schema: dict[str, Any]) -> dict:
    return {"anyOf": [
        _object({"done": {"type": "boolean", "enum": [True]}, "answer": {"type": "string"},
                 "delivery": {"type": "string", "enum": ["speak"]}, "email_workflow": email_workflow_schema,
                 "calendar_workflow": WORKFLOW_SCHEMA}, ["done", "answer"]),
        *[_object({"tool": {"type": "string", "enum": [name]}, "args": schema,
                  "email_workflow": email_workflow_schema, "calendar_workflow": WORKFLOW_SCHEMA}, ["tool", "args"])
          for name, schema in {**email_schemas, **TOOL_SCHEMAS}.items()],
    ]}


def calendar_decision_schema() -> dict:
    return {"anyOf": [
        _object({"done": {"type": "boolean", "enum": [True]}, "answer": {"type": "string"},
                 "delivery": {"type": "string", "enum": ["speak"]}, "calendar_workflow": WORKFLOW_SCHEMA}, ["done", "answer"]),
        *[_object({"tool": {"type": "string", "enum": [name]}, "args": schema,
                  "calendar_workflow": WORKFLOW_SCHEMA}, ["tool", "args"])
          for name, schema in TOOL_SCHEMAS.items()],
    ]}


def mixed_decision_schema(email_schemas: dict[str, Any], email_workflow_schema: dict[str, Any]) -> dict:
    return {"anyOf": [
        _object({"done": {"type": "boolean", "enum": [True]}, "answer": {"type": "string"},
                 "delivery": {"type": "string", "enum": ["speak"]}, "email_workflow": email_workflow_schema,
                 "calendar_workflow": WORKFLOW_SCHEMA}, ["done", "answer"]),
        *[_object({"tool": {"type": "string", "enum": [name]}, "args": schema,
                  "email_workflow": email_workflow_schema, "calendar_workflow": WORKFLOW_SCHEMA}, ["tool", "args"])
          for name, schema in {**email_schemas, **TOOL_SCHEMAS}.items()],
    ]}


def tool_descriptions() -> str:
    return "\n".join(f"- {name}: {DESCRIPTIONS[name]} Arguments: {json.dumps(schema)}" for name, schema in TOOL_SCHEMAS.items())
