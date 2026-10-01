import json
import re
from typing import Any


_TEXT = {"type": "string", "minLength": 1}
_OPTIONAL_TEXT = {"type": ["string", "null"]}
_ID = {"type": "integer", "minimum": 1}
_RECIPIENTS = {"type": ["array", "null"], "items": _TEXT}


def _object(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


TOOL_SCHEMAS = {
    "email_draft": _object({"instruction": _TEXT, "draft_id": {"type": ["integer", "null"], "minimum": 1},
                            "to": _RECIPIENTS, "cc": _RECIPIENTS, "bcc": _RECIPIENTS}, ["instruction"]),
    "email_get": _object({"draft_id": _ID}, ["draft_id"]),
    "email_recipients": _object({"draft_id": _ID, "to": _RECIPIENTS, "cc": _RECIPIENTS, "bcc": _RECIPIENTS,
                                 "mode": {"type": "string", "enum": ["replace", "add", "remove"]}}, ["draft_id"]),
    "email_prepare": _object({"draft_id": _ID, "action": {"type": "string", "enum": ["send", "schedule"]},
                              "when": _OPTIONAL_TEXT, "timezone": _OPTIONAL_TEXT}, ["draft_id", "action"]),
    "email_list": _object({"status": _OPTIONAL_TEXT}, []),
    "email_cancel": _object({"job_id": _ID}, ["job_id"]),
    "get_datetime": _object({}, []),
}

_DESCRIPTIONS = {
    "email_draft": "Write/revise a plain-text Gmail draft now, even without recipients or a delivery time. Put subject/body requirements in instruction. Omit unknown or uncertain addresses; never guess. Recipients are complete address arrays. Omitted recipients are preserved on edits. Use the active draft_id for revisions. The application asks for missing details after saving, and prepares delivery when the workflow is complete.",
    "email_get": "Read a Jarvis-managed Gmail draft by local numeric ID.",
    "email_recipients": "Update only recipients without rewriting subject/body. Use mode=add for adding an address or filling a missing recipient, replace for changing recipients, remove for removing addresses. Default mode is replace. Supply complete address arrays. Omitted fields are preserved. Empty arrays with replace clear that field. The application continues the pending workflow after updating.",
    "email_prepare": "Prepare a complete send/schedule preview. Scheduling requires a natural-language time and optional IANA timezone. Ask for missing times. A separate user confirmation is required; this tool does not send.",
    "email_list": "List Jarvis drafts and delivery jobs, optionally filtered by status.",
    "email_cancel": "Cancel an email action by job_id. Its Gmail draft is retained.",
    "get_datetime": "Get current date and time.",
}

DRAFT_SCHEMA = _object({"subject": {"type": "string"}, "body": _TEXT}, ["subject", "body"])
WORKFLOW_SCHEMA = _object({
    "action": {"type": "string", "enum": ["draft", "send", "schedule"]},
    "when": _OPTIONAL_TEXT, "timezone": _OPTIONAL_TEXT,
    "awaiting": {"type": ["string", "null"], "enum": ["purpose", "recipient", "recipient_confirmation", "time", "details", None]},
    "new_draft": {"type": "boolean"},
}, ["action", "when", "timezone", "awaiting", "new_draft"])
DECISION_SCHEMA = {"anyOf": [
    _object({"done": {"type": "boolean", "enum": [True]}, "answer": {"type": "string"},
             "delivery": {"type": "string", "enum": ["speak"]}, "workflow": WORKFLOW_SCHEMA}, ["done", "answer", "workflow"]),
    *[_object({"tool": {"type": "string", "enum": [name]}, "args": schema, "workflow": WORKFLOW_SCHEMA}, ["tool", "args", "workflow"])
      for name, schema in TOOL_SCHEMAS.items()],
]}


def tool_descriptions() -> str:
    return "\n".join(f"- {name}: {_DESCRIPTIONS[name]} Arguments: {json.dumps(schema)}" for name, schema in TOOL_SCHEMAS.items())


def safe_key(value: Any) -> str:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", value) else "<invalid-key>"


def validation_errors(value: Any, schema: dict, path: str = "args") -> list[str]:
    kind = "null" if value is None else {str: "string", int: "integer", bool: "boolean", dict: "object", list: "array"}.get(type(value))
    allowed = schema["type"]
    allowed = [allowed] if isinstance(allowed, str) else allowed
    if kind not in allowed:
        return [f"{path} must be {' or '.join(allowed)}"]
    if value is None:
        return []
    errors = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
    if kind == "string" and schema.get("minLength") and not value.strip():
        errors.append(f"{path} must not be empty")
    if kind == "integer" and value < schema.get("minimum", value):
        errors.append(f"{path} must be a positive integer")
    if kind == "array":
        for item in value:
            errors.extend(validation_errors(item, schema["items"], path + "[]"))
    if kind == "object":
        properties = schema["properties"]
        errors.extend(f"{path}.{name} is required" for name in schema["required"] if name not in value)
        for name, item in value.items():
            if name not in properties:
                errors.append(f"{path}.{safe_key(name)} is not supported")
            else:
                errors.extend(validation_errors(item, properties[name], path + "." + name))
    return errors
