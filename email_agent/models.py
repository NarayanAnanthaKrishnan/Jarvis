import base64
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses
from typing import Any, Protocol


class EmailError(Exception):
    pass


class ProviderError(EmailError):
    def __init__(self, code: str, uncertain: bool = False, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.uncertain = uncertain
        self.retryable = retryable


def addresses(values: list[str]) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise EmailError("Recipients must be a list of email addresses")
    result = []
    for value in values:
        if any(c in value for c in "\r\n"):
            raise EmailError("Invalid address header")
        parsed = getaddresses([value])
        if len(parsed) != 1 or not re.fullmatch(r"[^\s<>@,;]+@[^\s<>@,;]+\.[^\s<>@,;]+", parsed[0][1]):
            raise EmailError("Please provide a complete recipient email address")
        address = parsed[0][1]
        if address not in result:
            result.append(address)
    return result


@dataclass(frozen=True)
class EmailPayload:
    sender: str
    to: list[str]
    subject: str
    body: str
    cc: list[str] = field(default_factory=list)
    bcc: list[str] = field(default_factory=list)

    def normalized(self) -> "EmailPayload":
        if not isinstance(self.subject, str) or any(c in self.subject for c in "\r\n"):
            raise EmailError("Invalid subject")
        if not isinstance(self.body, str) or not self.body.strip():
            raise EmailError("Email body is empty")
        return EmailPayload(addresses([self.sender])[0], addresses(self.to), self.subject,
                            self.body.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n",
                            addresses(self.cc), addresses(self.bcc))

    def serialize(self) -> str:
        return json.dumps(asdict(self.normalized()), sort_keys=True, ensure_ascii=False)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.serialize().encode()).hexdigest()

    def raw(self) -> str:
        value = self.normalized()
        message = EmailMessage(policy=policy.SMTP)
        message["From"] = value.sender
        for name, entries in (("To", value.to), ("Cc", value.cc), ("Bcc", value.bcc)):
            if entries:
                message[name] = ", ".join(entries)
        message["Subject"] = value.subject
        message.set_content(value.body)
        return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

    @classmethod
    def deserialize(cls, value: str) -> "EmailPayload":
        return cls(**json.loads(value)).normalized()

    @classmethod
    def from_raw(cls, raw: str) -> "EmailPayload":
        message = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        if message.is_multipart() or message.get_content_type() != "text/plain" or message.get_filename():
            raise EmailError("Draft contains formatting or attachments unsupported in this version")
        for header in ("From", "To", "Cc", "Bcc", "Subject"):
            if len(message.get_all(header, [])) > 1:
                raise EmailError("Duplicate email headers require review")
        def entries(name: str) -> list[str]:
            return [address for _, address in getaddresses(message.get_all(name, []))]
        senders = entries("From")
        if len(senders) != 1:
            raise EmailError("Invalid sender")
        return cls(senders[0], entries("To"), str(message.get("Subject", "")), message.get_content(), entries("Cc"), entries("Bcc")).normalized()


@dataclass
class EmailResult:
    status: str
    message: str
    draft_id: int | None = None
    job_id: int | None = None
    preview: str | None = None
    confirmation_id: int | None = None
    data: Any = None

    def __str__(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class EmailProvider(Protocol):
    account: str

    def create_draft(self, payload: EmailPayload) -> str: ...
    def update_draft(self, draft_id: str, payload: EmailPayload) -> None: ...
    def get_draft(self, draft_id: str) -> EmailPayload: ...
    def send_draft(self, draft_id: str, payload: EmailPayload) -> str: ...
