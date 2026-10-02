import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import MAX_STEPS
from calendar_agent.workflow import CalendarWorkflow
from email_agent.workflow import EmailWorkflow


@dataclass
class TurnContext:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancelled: threading.Event = field(default_factory=threading.Event)
    remaining_steps: int = MAX_STEPS
    agent_id: str = "general"
    handed_off: bool = False
    email_touched: bool = False
    active_draft_id: int | None = None
    email_service: Any = None
    email_workflow: EmailWorkflow = field(default_factory=EmailWorkflow)
    calendar_service: Any = None
    calendar_workflow: CalendarWorkflow = field(default_factory=CalendarWorkflow)
    calendar_touched: bool = False
    email_stage: str = "dispatch"
    memories: str = "(none)"
    mutation_results: dict[str, Any] = field(default_factory=dict)

    def check_active(self) -> None:
        if self.cancelled.is_set():
            raise InterruptedError("Session ended")
