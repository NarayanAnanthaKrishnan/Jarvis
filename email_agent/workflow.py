from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EmailWorkflow:
    action: str = "draft"
    when: str | None = None
    timezone: str | None = None
    awaiting: str | None = None
    unresolved_recipients: list[str] = field(default_factory=list)
    resolved_when: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, values: dict[str, Any]) -> None:
        if any(name in values and values[name] != getattr(self, name) for name in ("when", "timezone")):
            self.resolved_when = None
        for name in ("action", "when", "timezone", "awaiting"):
            if name in values:
                setattr(self, name, values[name])
