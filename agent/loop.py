import json
import re
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from agent.context import TurnContext
from agent.json_utils import extract_object
from agent.prompts import SYSTEM_PROMPT, USER_PROMPT_FORMAT, REFLECTION_PROMPT
from agent.router import RouteDecision, route_turn
from agent.specs import AGENTS, descriptions
import config
from config import MAX_STEPS, PARALLEL_WORKERS, REFLECTION_ENABLED
from email_agent.models import EmailError, EmailResult, addresses
from email_agent.addressing import is_address_confirmation, is_address_rejection, normalize_spoken_address, speak_address
from email_agent.contracts import DECISION_SCHEMA, WORKFLOW_SCHEMA, validation_errors
from email_agent.timing import extract_time_phrase, parse_send_time, ScheduleTimeError
from email_agent.prompts import SYSTEM_PROMPT as EMAIL_SYSTEM_PROMPT
from email_agent.workflow import EmailWorkflow
from calendar_agent.contracts import TOOL_SCHEMAS as CALENDAR_TOOL_SCHEMAS, WORKFLOW_SCHEMA as CALENDAR_WORKFLOW_SCHEMA, calendar_decision_schema, mixed_decision_schema
from calendar_agent.models import CalendarResult
from calendar_agent.prompts import SYSTEM_PROMPT as CALENDAR_SYSTEM_PROMPT, MIXED_SYSTEM_PROMPT
from calendar_agent.workflow import CalendarWorkflow
from llm.client import ModelRequestError
from memory.retrieval_gate import needs_memory
from ops.tracer import trace
from tools.profile_loader import load_profile
from tools.registry import READ_ONLY_TOOLS, execute_tool


def _extract_json(text: str) -> str:
    return json.dumps(extract_object(text), ensure_ascii=False)


def _normalize_args(args: dict) -> dict:
    return dict(args)


def _clarification(answer: str) -> str:
    question = re.split(r"(?<=[.!?])\s+", answer.strip())[-1]
    claims = r"\b(?:(?:I|we)\s+(?:have\s+)?|(?:email|draft|it)\s+(?:is|was|has been)\s+)(?:already\s+|successfully\s+)?(?:sent|scheduled|saved|drafted)\b"
    return question if question.endswith("?") and not re.search(claims, question, re.I) else ""


def _format_steps(steps: list[dict]) -> str:
    return "\n".join(f"{i}. {step['tool']}({step['args']}) -> {step['result']}" for i, step in enumerate(steps, 1)) or "(none)"


def _execute_timed(name: str, args: dict, llm: Any, context: TurnContext, parallel: bool = False) -> Any:
    started = time.monotonic()
    status = "error"
    try:
        value = execute_tool(name, args, llm, context, parallel)
        status = value.status if isinstance(value, (EmailResult, CalendarResult)) else "error" if str(value).startswith("Error:") else "complete"
        return value
    finally:
        trace("tool", name=name, turn_id=context.turn_id, status=status, elapsed_s=round(time.monotonic() - started, 3))


def validate_decision(decision: Any) -> dict:
    if not isinstance(decision, dict):
        raise ValueError("Decision must be an object")
    if "done" in decision and not isinstance(decision["done"], bool):
        raise ValueError("done must be boolean")
    if "workflow" in decision and validation_errors(decision["workflow"], WORKFLOW_SCHEMA, "workflow"):
        raise ValueError("Invalid email workflow")
    if "email_workflow" in decision and validation_errors(decision["email_workflow"], WORKFLOW_SCHEMA, "email_workflow"):
        raise ValueError("Invalid email workflow")
    if "calendar_workflow" in decision and validation_errors(decision["calendar_workflow"], CALENDAR_WORKFLOW_SCHEMA, "calendar_workflow"):
        raise ValueError("Invalid calendar workflow")
    if decision.get("done"):
        if "tool" in decision or "parallel" in decision:
            raise ValueError("Decision mixes a final answer and tools")
        if not isinstance(decision.get("answer"), str) or decision.get("delivery", "speak") not in ("speak", "paste", "both"):
            raise ValueError("Invalid final response")
        return decision
    tools = decision.get("parallel", [decision])
    if not isinstance(tools, list) or not tools or len(tools) > MAX_STEPS:
        raise ValueError("Invalid tool batch")
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("tool"), str) or not tool["tool"] or not isinstance(tool.get("args", {}), dict):
            raise ValueError("Invalid tool arguments")
    return decision


def think(user_input: str, steps: list[dict], conversation_history: list[str], llm: Any, memories: str = "(none)", context: TurnContext | None = None) -> dict:
    context = context or TurnContext()
    spec = AGENTS[context.agent_id]
    template = {"email": EMAIL_SYSTEM_PROMPT, "calendar": CALENDAR_SYSTEM_PROMPT,
                "mixed": MIXED_SYSTEM_PROMPT}.get(context.agent_id, SYSTEM_PROMPT)
    system = (template.replace("{TOOL_DESCRIPTIONS}", descriptions(context.agent_id))
              .replace("{PROFILE}", load_profile() or "(none)").replace("{MEMORIES}", memories)
              .replace("{CURRENT_DATE}", datetime.now().astimezone().isoformat())
              .replace("{CONVERSATION_HISTORY}", "\n".join(conversation_history[-8:]) or "(none)")
              .replace("{CALENDAR_WORKFLOW}", json.dumps(context.calendar_workflow.snapshot()))
              .replace("{EMAIL_WORKFLOW}", json.dumps(context.email_workflow.snapshot())))
    system += f"\n\n{spec.system_prompt}\nActive local draft ID: {context.active_draft_id}."
    system += "\nPending email workflow: " + json.dumps(context.email_workflow.snapshot())
    system += "\nPending calendar workflow: " + json.dumps(context.calendar_workflow.snapshot())
    if context.agent_id == "email":
        schema, profile = DECISION_SCHEMA, "email"
    elif context.agent_id == "calendar":
        schema, profile = calendar_decision_schema(), "calendar"
    elif context.agent_id == "mixed":
        from email_agent.contracts import TOOL_SCHEMAS
        schema, profile = mixed_decision_schema(TOOL_SCHEMAS, WORKFLOW_SCHEMA), "mixed"
    else:
        schema, profile = None, "general"
    options = {"profile": profile, "response_schema": schema} if schema else {"temp": 0.1, "max_tokens": 2048}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": USER_PROMPT_FORMAT.replace("{GOAL}", user_input).replace("{STEPS}", _format_steps(steps))}]
    for attempt in range(2):
        context.check_active()
        response = llm.call_raw(messages, **options)
        if response is None:
            raise ModelRequestError("The model is unavailable; please retry. Any saved draft is retained." if context.agent_id == "email" else "The model is unavailable; please retry.")
        try:
            if response.get("finish_reason") in ("MAX_TOKENS", "length", "max_tokens"):
                raise ValueError("Decision was truncated")
            decision = validate_decision(extract_object(response["message"]["content"]))
            if context.agent_id == "email" and "workflow" not in decision:
                raise ValueError("Email decision omitted workflow")
            if context.agent_id == "calendar" and "calendar_workflow" not in decision:
                raise ValueError("Calendar decision omitted workflow")
            if context.agent_id == "mixed" and not {"email_workflow", "calendar_workflow"}.issubset(decision):
                raise ValueError("Mixed decision omitted a workflow")
            names = [entry.get("tool") for entry in decision.get("parallel", [decision])] if not decision.get("done") else []
            if any(name not in spec.allowed_tools for name in names):
                raise ValueError("Decision used a tool outside this agent")
            return decision
        except (KeyError, TypeError, ValueError) as exc:
            content = response.get("message", {}).get("content", "")
            trace("error", where="decision", turn_id=context.turn_id, agent_id=context.agent_id,
                  error_type=type(exc).__name__, error_code="invalid_decision", attempt=attempt + 1,
                  output_chars=len(content) if isinstance(content, str) else 0, finish_reason=response.get("finish_reason"))
            if attempt or context.remaining_steps < 2:
                message = "I couldn't interpret the model's response after retrying. Please try again."
                if context.agent_id == "email":
                    message += " Any saved draft is retained."
                raise ModelRequestError(message) from exc
            context.remaining_steps -= 1
            messages.append({"role": "user", "content": "Your previous response was not a complete valid decision. No tool was executed for it. Return one complete JSON object using only the available tools and the required schema. Do not return a schema, markdown, commentary or multiple decisions."})
    raise ModelRequestError("No usable decision")


def _reflect(user_input: str, steps: list[dict], answer: str, llm: Any) -> dict:
    prompt = (REFLECTION_PROMPT.replace("{CURRENT_DATE}", datetime.now().isoformat()).replace("{GOAL}", user_input)
              .replace("{STEPS}", _format_steps(steps)).replace("{ANSWER}", answer))
    try:
        response = llm.call_raw([{"role": "system", "content": "Verify the answer. Return valid JSON."}, {"role": "user", "content": prompt}], temp=0.1)
        return extract_object(response["message"]["content"]) if response else {"correct": True}
    except (ValueError, TypeError, KeyError):
        return {"correct": True}


def _stream_summarize(user_input: str, steps: list[dict], llm: Any) -> Generator[str, None, None]:
    yield from llm.stream_chat([{"role": "system", "content": "Answer concisely from tool results. State what is missing. Never claim actions occurred without successful tool results."}, {"role": "user", "content": f"Question: {user_input}\nResults:\n{_format_steps(steps)}"}], temp=0.1)


def run_agent(user_input: str, conversation_history: list[str], llm: Any, context: TurnContext | None = None) -> dict:
    context = context or TurnContext()
    context.check_active()
    started = time.monotonic()
    trace("turn_start", turn_id=context.turn_id)
    pending_address = context.email_workflow.pending_recipient
    pending_attendee = context.calendar_workflow.pending_attendee
    route_state = {"active_draft_id": context.active_draft_id, **context.email_workflow.snapshot(),
                   "calendar_workflow": context.calendar_workflow.snapshot()}
    route = (RouteDecision("email", 1.0, False, "local", "recipient_confirmation") if pending_address else
             RouteDecision("calendar", 1.0, False, "local", "attendee_confirmation") if pending_attendee else
             route_turn(user_input, conversation_history, route_state))
    context.agent_id = route.agent_id
    context.email_touched = context.agent_id in ("email", "mixed") or bool(
        (context.active_draft_id is not None or re.search(r"\b(email|e-mail|gmail)\b", user_input, re.I))
        and re.search(r"\b(draft|write|send|schedule|reschedule|cancel|edit|revise|shorter|longer|formal)\b", user_input, re.I))
    context.calendar_touched = context.agent_id in ("calendar", "mixed") or bool(
        re.search(r"\b(calendar|appointment)\b|\b(schedule|create|book|set up|move|reschedule|cancel)\s+(?:(?:a|the|my)\s+)?(?:calendar\s+)?(?:meeting|event)\b", user_input, re.I))
    gate = route.memory_needed if route.memory_needed is not None else needs_memory(user_input, llm)
    context.check_active()
    memories = "(none)"
    if gate:
        try:
            from memory.store import memory_store
            values = memory_store.query("semantic", user_input, n=5) + memory_store.query("episodic", user_input, n=3)
            memories = "\n".join(f"- {value}" for value in values) or "(none)"
        except Exception as exc:
            trace("error", where="memory_retrieval", error_type=type(exc).__name__)
    context.memories = memories
    steps: list[dict] = []
    seen: set[str] = set()
    last_email: EmailResult | None = None
    last_calendar: CalendarResult | None = None
    pending_previews: list[dict[str, Any]] = []
    argument_failures = 0
    schedule_requested = context.agent_id == "email" and bool(re.search(r"\b(schedule|reschedule|send\s+(?:it\s+)?later)\b", user_input, re.I))
    schedule_due: datetime | None = None
    schedule_error: str | None = None
    if schedule_requested:
        try:
            phrase = extract_time_phrase(user_input)
            if phrase:
                now = datetime.fromtimestamp(context.email_service.clock(), timezone.utc) if context.email_service else datetime.now(timezone.utc)
                schedule_due = parse_send_time(phrase, context.email_workflow.timezone or config.EMAIL_TIMEZONE, now)
        except EmailError as exc:
            schedule_error = str(exc)
            trace("email_time_followup", turn_id=context.turn_id, time_present=True, time_parse="rejected", time_source="request",
                  time_error_code=getattr(exc, "code", "time_error"))
        else:
            if schedule_due:
                trace("email_time_followup", turn_id=context.turn_id, time_present=True, time_parse="accepted", time_source="request")

    calendar_requested = context.agent_id == "calendar" and bool(re.search(r"\b(schedule|create|book|set up|move|reschedule|cancel|meeting|appointment)\b", user_input, re.I))
    if re.search(r"\b(cancel|delete)\b", user_input, re.I):
        calendar_action_hint = "cancel"
    elif re.search(r"\b(move|reschedule|update|change|edit)\b", user_input, re.I):
        calendar_action_hint = "update"
    else:
        calendar_action_hint = "create"
    calendar_due: datetime | None = None
    calendar_error: str | None = None
    if calendar_requested and calendar_action_hint in ("create", "update"):
        try:
            phrase = extract_time_phrase(user_input)
            if phrase:
                clock = context.calendar_service.clock() if context.calendar_service else time.time()
                calendar_due = parse_send_time(phrase, context.calendar_workflow.timezone or config.EMAIL_TIMEZONE,
                                               datetime.fromtimestamp(clock, timezone.utc))
        except EmailError as exc:
            calendar_error = str(exc)
            context.calendar_workflow.action = "create"
            context.calendar_workflow.when = None
            context.calendar_workflow.awaiting = "time"
            trace("calendar_time", turn_id=context.turn_id, time_present=True, time_parse="rejected",
                  time_source="request", time_error_code=getattr(exc, "code", "time_error"))
        else:
            if calendar_due:
                context.calendar_workflow.when = calendar_due.isoformat()
                context.calendar_workflow.timezone = context.calendar_workflow.timezone or config.EMAIL_TIMEZONE
                trace("calendar_time", turn_id=context.turn_id, time_present=True, time_parse="accepted", time_source="request")

    def finish(output: str, delivery: str = "speak", stream: Any = None) -> dict:
        context.check_active()
        trace("turn_end", turn_id=context.turn_id, steps=MAX_STEPS - context.remaining_steps, elapsed_s=round(time.monotonic() - started, 2))
        if pending_previews:
            message_parts = [str(item.get("message", "")) for item in pending_previews if item.get("message")]
            if output and output not in message_parts:
                message_parts.append(output)
            output = " ".join(message_parts)
        return {"output": output, "delivery": delivery, "stream": stream, "gate_needed": gate,
                "email_touched": context.email_touched, "email_result": last_email,
                "calendar_touched": context.calendar_touched,
                "preview": last_email.preview if last_email else None,
                "confirmation_id": last_email.confirmation_id if last_email else None,
                "previews": [{"kind": item["kind"], "id": item["id"], "text": item["preview"]}
                             for item in pending_previews if item.get("preview") and item.get("id") is not None]}

    if (pending_address is not None):
        if is_address_confirmation(user_input):
            context.email_workflow.pending_recipient = None
            field = pending_address["field"]
            context.email_workflow.unresolved_recipients = [name for name in context.email_workflow.unresolved_recipients if name != field]
            context.email_workflow.awaiting = "recipient"
            args = {"draft_id": pending_address["draft_id"], field: [pending_address["address"]], "mode": pending_address["mode"]}
            context.remaining_steps -= 1
            last_email = _execute_timed("email_recipients", args, llm, context)
            return finish(last_email.message if isinstance(last_email, EmailResult) else str(last_email))
        if is_address_rejection(user_input):
            context.email_workflow.pending_recipient = None
            context.email_workflow.awaiting = "recipient"
            return finish("Okay. Please say the complete address again, spelling any unclear part.")
        corrected = normalize_spoken_address(re.sub(r"^\s*(?:actually|no[, ]+)\s*", "", user_input, flags=re.I))
        if corrected:
            pending_address["address"] = corrected[0]
            context.email_workflow.pending_recipient = pending_address
            return finish(f"I heard {speak_address(corrected[0])}. Is that correct? Say yes to add it, or say no and spell the address.")
        return finish(f"I heard {speak_address(pending_address['address'])}. Say yes if that is correct, or say no and spell the address.")

    if pending_attendee is not None:
        if is_address_confirmation(user_input):
            context.calendar_workflow.pending_attendee = None
            context.calendar_workflow.attendees = list(dict.fromkeys(context.calendar_workflow.attendees + [pending_attendee["address"]]))
            context.calendar_workflow.awaiting = None
            context.remaining_steps -= 1
            values = context.calendar_workflow.snapshot()
            args = {key: values.get(key) for key in ("action", "event_id", "title", "when", "timezone", "duration_minutes", "attendees", "description", "location")}
            last_calendar = _execute_timed("calendar_prepare", args, llm, context)
            if isinstance(last_calendar, CalendarResult):
                if last_calendar.confirmation_id is not None:
                    pending_previews.append({"kind": "calendar", "id": last_calendar.confirmation_id,
                                             "preview": last_calendar.preview, "message": last_calendar.message})
                return finish(last_calendar.message)
        if is_address_rejection(user_input):
            context.calendar_workflow.pending_attendee = None
            context.calendar_workflow.awaiting = "attendee"
            return finish("Okay. Please say the complete attendee address again, spelling any unclear part.")
        corrected = normalize_spoken_address(re.sub(r"^\s*(?:actually|no[, ]+)\s*", "", user_input, flags=re.I))
        if corrected:
            pending_attendee["address"] = corrected[0]
            context.calendar_workflow.pending_attendee = pending_attendee
            return finish(f"I heard {speak_address(corrected[0])}. Is that correct? Say yes to add it, or say no and spell the address.")
        return finish(f"I heard {speak_address(pending_attendee['address'])}. Say yes if that is correct, or say no and spell the address.")

    if (context.agent_id == "email" and context.email_workflow.action == "schedule"
            and context.email_workflow.awaiting == "time" and context.active_draft_id is not None):
        try:
            now = datetime.fromtimestamp(context.email_service.clock(), timezone.utc) if context.email_service else datetime.now(timezone.utc)
            phrase = extract_time_phrase(user_input) or user_input
            due = parse_send_time(phrase, context.email_workflow.timezone or config.EMAIL_TIMEZONE, now)
        except EmailError as exc:
            trace("email_time_followup", turn_id=context.turn_id, time_present=True, time_parse="rejected", time_source="deterministic",
                  time_error_code=getattr(exc, "code", "time_error"))
            return finish(str(exc) if isinstance(exc, ScheduleTimeError) else "I couldn't read that as a schedule time. Try one time, such as later today at 6 PM.")
        else:
            trace("email_time_followup", turn_id=context.turn_id, time_present=True, time_parse="accepted", time_source="deterministic")
            context.email_workflow.when = due.isoformat()
            context.email_workflow.resolved_when = None
            context.remaining_steps -= 1
            last_email = _execute_timed("email_prepare", {"draft_id": context.active_draft_id,
                                        "action": "schedule", "when": due.isoformat(),
                                        "timezone": context.email_workflow.timezone}, llm, context)
            if isinstance(last_email, EmailResult):
                return finish(last_email.message)
            return finish("I couldn't prepare the scheduling preview. The draft is still saved.")

    if context.agent_id == "calendar" and context.calendar_workflow.awaiting == "time":
        try:
            phrase = extract_time_phrase(user_input) or user_input
            clock = context.calendar_service.clock() if context.calendar_service else time.time()
            due = parse_send_time(phrase, context.calendar_workflow.timezone or config.EMAIL_TIMEZONE,
                                  datetime.fromtimestamp(clock, timezone.utc))
        except EmailError as exc:
            trace("calendar_time", turn_id=context.turn_id, time_present=True, time_parse="rejected",
                  time_source="followup", time_error_code=getattr(exc, "code", "time_error"))
            return finish(str(exc) if isinstance(exc, ScheduleTimeError) else "I couldn't read that meeting time. Try one future time, such as later today at 6 PM.")
        context.calendar_workflow.when = due.isoformat()
        context.calendar_workflow.timezone = context.calendar_workflow.timezone or config.EMAIL_TIMEZONE
        context.calendar_workflow.awaiting = None
        trace("calendar_time", turn_id=context.turn_id, time_present=True, time_parse="accepted", time_source="followup")
        context.remaining_steps -= 1
        values = context.calendar_workflow.snapshot()
        args = {key: values.get(key) for key in ("action", "event_id", "title", "when", "timezone", "duration_minutes", "attendees", "description", "location")}
        last_calendar = _execute_timed("calendar_prepare", args, llm, context)
        if isinstance(last_calendar, CalendarResult):
            if last_calendar.confirmation_id is not None:
                pending_previews.append({"kind": "calendar", "id": last_calendar.confirmation_id,
                                         "preview": last_calendar.preview, "message": last_calendar.message})
            return finish(last_calendar.message)

    if context.agent_id == "calendar" and context.calendar_workflow.awaiting == "attendee":
        try:
            candidate = normalize_spoken_address(user_input)
            if candidate is None:
                candidate = (addresses([user_input])[0], False)
        except EmailError:
            return finish("What is the complete attendee email address?")
        if candidate[1]:
            context.calendar_workflow.pending_attendee = {"address": candidate[0]}
            context.calendar_workflow.awaiting = "attendee_confirmation"
            return finish(f"I heard {speak_address(candidate[0])}. Is that correct? Say yes to add it, or say no and repeat the full address.")
        context.calendar_workflow.attendees = list(dict.fromkeys(context.calendar_workflow.attendees + [candidate[0]]))
        context.calendar_workflow.awaiting = None
        values = context.calendar_workflow.snapshot()
        args = {key: values.get(key) for key in ("action", "event_id", "title", "when", "timezone", "duration_minutes", "attendees", "description", "location")}
        context.remaining_steps -= 1
        last_calendar = _execute_timed("calendar_prepare", args, llm, context)
        if isinstance(last_calendar, CalendarResult):
            if last_calendar.confirmation_id is not None:
                pending_previews.append({"kind": "calendar", "id": last_calendar.confirmation_id,
                                         "preview": last_calendar.preview, "message": last_calendar.message})
            return finish(last_calendar.message)

    while context.remaining_steps > 0:
        context.check_active()
        try:
            decision = think(user_input, steps, conversation_history, llm, memories, context)
        except ModelRequestError as exc:
            return finish(" ".join(part for part in (last_email.message if last_email else "", str(exc)) if part))
        context.remaining_steps -= 1
        context.check_active()
        email_state = decision.get("workflow", decision.get("email_workflow"))
        if context.agent_id in ("email", "mixed") and email_state is not None:
            state = email_state
            if state.get("new_draft"):
                context.active_draft_id = None
                context.email_workflow = EmailWorkflow()
            context.email_workflow.update(state)
            if schedule_requested and context.agent_id == "email":
                context.email_workflow.action = "schedule"
                if schedule_due:
                    context.email_workflow.when = schedule_due.isoformat()
                    context.email_workflow.timezone = context.email_workflow.timezone or config.EMAIL_TIMEZONE
                    context.email_workflow.resolved_when = schedule_due.isoformat()
                    if context.email_workflow.awaiting == "time":
                        context.email_workflow.awaiting = None
                elif schedule_error:
                    context.email_workflow.when = None
                    context.email_workflow.resolved_when = None
                    if context.email_workflow.awaiting not in ("recipient", "purpose"):
                        context.email_workflow.awaiting = "time"
        calendar_state = decision.get("calendar_workflow")
        if context.agent_id in ("calendar", "mixed") and calendar_state is not None:
            context.calendar_workflow.update(calendar_state)
            if calendar_requested and context.agent_id == "calendar" and calendar_action_hint != "create":
                context.calendar_workflow.action = calendar_action_hint
            if calendar_requested and context.agent_id == "calendar" and calendar_action_hint in ("create", "update"):
                if calendar_due:
                    context.calendar_workflow.when = calendar_due.isoformat()
                    context.calendar_workflow.timezone = context.calendar_workflow.timezone or config.EMAIL_TIMEZONE
                    if context.calendar_workflow.awaiting == "time":
                        context.calendar_workflow.awaiting = None
                elif calendar_error:
                    context.calendar_workflow.when = None
                    context.calendar_workflow.awaiting = "time"
        if decision.get("done"):
            answer = decision["answer"]
            workflow_questions = {"purpose": "What should the email be about?", "recipient": "What is the complete recipient email address?",
                                 "time": "When should I schedule it? Please include a date and time.", "details": "What detail should I change?"}
            calendar_questions = {"title": "What should I call the meeting?", "time": "When should the meeting start? Please give one future date and time.",
                                  "attendee": "What is the complete attendee email address?", "event": "Which Jarvis meeting ID should I change? Ask me to list your calendar events.",
                                  "details": "What meeting detail should I change?"}
            if pending_previews or last_email or last_calendar:
                parts = []
                if last_email and last_email.status == "needs_details":
                    parts.append(last_email.message)
                if last_calendar and last_calendar.status == "needs_details":
                    parts.append(last_calendar.message)
                if pending_previews:
                    if context.agent_id == "mixed":
                        email_question = workflow_questions.get(context.email_workflow.awaiting)
                        calendar_question = calendar_questions.get(context.calendar_workflow.awaiting)
                        for question in (email_question, calendar_question):
                            if question and question not in parts:
                                parts.append(question)
                    return finish(" ".join(parts))
                if parts:
                    return finish(" ".join(parts))
                if last_calendar and last_calendar.status == "listed":
                    return finish(answer)
                if last_email and last_email.status == "listed":
                    return finish(last_email.message)
            if context.agent_id == "mixed":
                questions = [workflow_questions.get(context.email_workflow.awaiting),
                             calendar_questions.get(context.calendar_workflow.awaiting)]
                questions = list(dict.fromkeys(question for question in questions if question))
                if questions:
                    return finish(" ".join(questions))
                return finish(answer)
            if context.email_touched:
                if schedule_error and context.email_workflow.awaiting == "time":
                    return finish(schedule_error)
                question = workflow_questions.get(context.email_workflow.awaiting, _clarification(answer))
                if question:
                    return finish(question)
                return finish("No email action has been completed. Please provide the recipient or draft ID and the action you want.")
            if context.calendar_touched:
                if calendar_error and context.calendar_workflow.awaiting == "time":
                    return finish(calendar_error)
                question = calendar_questions.get(context.calendar_workflow.awaiting, _clarification(answer))
                if question:
                    return finish(question)
                return finish("No meeting action has been completed. Ask me to create, list, change or cancel a meeting.")
            if REFLECTION_ENABLED and steps and context.remaining_steps > 0:
                verdict = _reflect(user_input, steps, answer, llm)
                if verdict.get("correct") is False:
                    steps.append({"tool": "reflection", "args": {}, "result": verdict})
                    continue
            return finish(answer, decision.get("delivery", "speak"))
        batch = decision.get("parallel", [decision])
        if "parallel" in decision:
            if any(item["tool"] not in READ_ONLY_TOOLS or item["tool"] not in AGENTS[context.agent_id].allowed_tools for item in batch):
                steps.append({"tool": "parallel", "args": {}, "result": "Error: entire batch rejected; only permitted read-only tools may run in parallel"})
                continue
            if len(batch) - 1 > context.remaining_steps:
                steps.append({"tool": "parallel", "args": {}, "result": "Error: batch exceeds remaining tool budget"})
                continue
            context.remaining_steps -= len(batch) - 1
            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
                futures = [pool.submit(_execute_timed, item["tool"], item.get("args", {}), llm, context, True) for item in batch]
                for item, future in zip(batch, futures):
                    value = future.result()
                    steps.append({"tool": item["tool"], "args": item.get("args", {}), "result": str(value)})
            continue
        name, args = decision["tool"], decision.get("args", {})
        if schedule_requested and name == "email_prepare" and (schedule_due is not None or schedule_error is not None):
            args = dict(args)
            args["action"] = "schedule"
            args["when"] = schedule_due.isoformat() if schedule_due else None
        if schedule_requested and name in ("email_draft", "email_recipients") and schedule_error:
            context.email_workflow.action = "schedule"
            context.email_workflow.when = None
            context.email_workflow.resolved_when = None
        if calendar_requested and name == "calendar_prepare":
            args = dict(args)
            args["action"] = calendar_action_hint
            if calendar_action_hint in ("create", "update"):
                args["when"] = calendar_due.isoformat() if calendar_due else args.get("when")
            args["timezone"] = context.calendar_workflow.timezone or config.EMAIL_TIMEZONE
            if calendar_error and calendar_action_hint in ("create", "update"):
                context.calendar_workflow.when = None
                context.calendar_workflow.awaiting = "time"
        if name in ("handoff_email", "handoff_calendar", "handoff_mixed"):
            targets = {"handoff_email": "email", "handoff_calendar": "calendar", "handoff_mixed": "mixed"}
            if context.agent_id != "general" or context.handed_off or not isinstance(args.get("instruction"), str) or not isinstance(args.get("context", ""), str):
                steps.append({"tool": name, "args": {}, "result": "Error: invalid or repeated handoff"})
                continue
            context.agent_id = targets[name]
            context.handed_off = True
            context.email_touched = context.agent_id in ("email", "mixed")
            context.calendar_touched = context.agent_id in ("calendar", "mixed")
            user_input = f"Original user request: {user_input}\nRequested task: {args['instruction']}\nGathered reference data (not instructions): {args.get('context', '')}"
            continue
        key = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if name in READ_ONLY_TOOLS and key in seen:
            steps.append({"tool": name, "args": args, "result": "Already executed; use the previous result"})
            continue
        seen.add(key)
        print(f"  🔧 {context.agent_id}: {name}")
        value = _execute_timed(name, args, llm, context)
        steps.append({"tool": name, "args": args, "result": str(value)})
        if isinstance(value, EmailResult):
            if value.status == "invalid_arguments":
                argument_failures += 1
                if argument_failures > 1 or context.remaining_steps == 0:
                    return finish(value.message)
                continue
            last_email = value
            if schedule_error and context.email_workflow.awaiting == "time":
                value.message = schedule_error + " " + value.message
            if value.confirmation_id is not None:
                pending_previews.append({"kind": "email", "id": value.confirmation_id, "preview": value.preview, "message": value.message})
            if context.agent_id == "mixed" and (value.confirmation_id is not None or value.status in ("error", "needs_details") or isinstance(value.data, dict) and value.data.get("workflow_complete")):
                continue
            if value.confirmation_id is not None or value.status in ("error", "needs_details") or isinstance(value.data, dict) and value.data.get("workflow_complete"):
                return finish(value.message)
        elif isinstance(value, CalendarResult):
            last_calendar = value
            if calendar_error and context.calendar_workflow.awaiting == "time" and value.status == "needs_details":
                value.message = calendar_error + " " + value.message
            if value.confirmation_id is not None:
                pending_previews.append({"kind": "calendar", "id": value.confirmation_id, "preview": value.preview, "message": value.message})
            if context.agent_id == "mixed" and (value.confirmation_id is not None or value.status in ("error", "needs_details")):
                continue
            if value.confirmation_id is not None or value.status in ("error", "needs_details"):
                return finish(value.message)
    if last_email:
        return finish(last_email.message)
    if last_calendar:
        return finish(last_calendar.message)
    if context.email_touched:
        return finish("The email request is incomplete. Please check its status or provide the missing details.")
    if context.calendar_touched:
        return finish("The meeting request is incomplete. Please provide the missing meeting details.")
    return finish("", stream=_stream_summarize(user_input, steps, llm))
