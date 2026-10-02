import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from email_agent.models import EmailError, addresses


@dataclass(frozen=True)
class CalendarEvent:
    summary: str
    start: str
    end: str
    timezone: str
    attendees: list[str]
    description: str = ""
    location: str = ""

    def normalized(self) -> "CalendarEvent":
        if not isinstance(self.summary, str) or not self.summary.strip() or len(self.summary) > 200:
            raise EmailError("A meeting title is required")
        if not isinstance(self.timezone, str) or len(self.timezone) > 128:
            raise EmailError("Use a valid named timezone")
        try:
            start = datetime.fromisoformat(self.start)
            end = datetime.fromisoformat(self.end)
        except (TypeError, ValueError) as exc:
            raise EmailError("A valid meeting start and end time are required") from exc
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise EmailError("Meeting end time must be after its start time")
        if not isinstance(self.attendees, list) or not self.attendees:
            raise EmailError("Provide at least one complete attendee email address")
        attendee_values = addresses(self.attendees)
        if not isinstance(self.description, str) or len(self.description) > 8000:
            raise EmailError("Meeting description is too long")
        if not isinstance(self.location, str) or len(self.location) > 1024:
            raise EmailError("Meeting location is too long")
        return CalendarEvent(self.summary.strip(), start.isoformat(), end.isoformat(), self.timezone,
                             attendee_values, self.description.strip(), self.location.strip())

    def serialize(self) -> str:
        return json.dumps(asdict(self.normalized()), sort_keys=True, ensure_ascii=False)

    @classmethod
    def deserialize(cls, value: str) -> "CalendarEvent":
        return cls(**json.loads(value)).normalized()


@dataclass
class CalendarResult:
    status: str
    message: str
    action_id: int | None = None
    managed_id: int | None = None
    preview: str | None = None
    confirmation_id: int | None = None
    data: Any = None

    def __str__(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
