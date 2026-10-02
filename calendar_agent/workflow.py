from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CalendarWorkflow:
    action: str = "create"
    title: str | None = None
    when: str | None = None
    timezone: str | None = None
    duration_minutes: int | None = 30
    attendees: list[str] = field(default_factory=list)
    description: str = ""
    location: str = ""
    awaiting: str | None = None
    event_id: int | None = None
    pending_attendee: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, values: dict[str, Any]) -> None:
        for key in self.__dataclass_fields__:
            if key in values and values[key] is not None:
                setattr(self, key, values[key])
