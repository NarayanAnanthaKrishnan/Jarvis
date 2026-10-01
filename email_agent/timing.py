import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser

from email_agent.models import EmailError


def parse_send_time(value: str, zone: str, now: datetime) -> datetime:
    try:
        tz = ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise EmailError("Use a named timezone such as America/New_York") from exc
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise EmailError("Provide a date and time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if "T" not in value and " " not in value:
            raise ValueError("Missing time")
    except ValueError:
        duration = re.fullmatch(r"\s*in\s+(\d+)\s+(seconds?|minutes?|hours?)\s*", value, re.I)
        if duration:
            seconds = int(duration[1]) * (3600 if duration[2].lower().startswith("hour") else 60 if duration[2].lower().startswith("minute") else 1)
            if seconds <= 0:
                raise EmailError("The scheduled time must be in the future")
            return now.astimezone(timezone.utc) + timedelta(seconds=seconds)
        if re.search(r"\b(?:UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|CET|BST|IST)\b|[A-Za-z]+/[A-Za-z_]+", value, re.I):
            raise EmailError("Provide the timezone separately using an IANA name, or use an ISO timestamp with a UTC offset")
        relative = bool(re.search(r"\bin\s+\d+\s+(seconds?|minutes?|hours?)\b", value, re.I))
        clock = bool(re.search(r"\b\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?\b|\b(?:1[3-9]|2[0-3]):[0-5]\d\b|\bnoon\b|\bmidnight\b", value, re.I))
        if not relative and not clock:
            raise EmailError("Specify an unambiguous time, for example tomorrow at 9am")
        if re.search(r"\d{1,2}/\d{1,2}", value):
            raise EmailError("Use a month name or an ISO date to avoid ambiguous dates")
        parsed = dateparser.parse(value, languages=["en"], settings={"RELATIVE_BASE": now.astimezone(tz).replace(tzinfo=None), "PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False})
        if parsed is None:
            raise EmailError("Could not parse the scheduled time")
    if parsed.tzinfo is None:
        candidates = [parsed.replace(tzinfo=tz, fold=fold) for fold in (0, 1)]
        valid = [d for d in candidates if d.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) == parsed]
        if not valid:
            raise EmailError("That local time does not exist because of daylight saving time")
        if len({d.utcoffset() for d in valid}) > 1:
            raise EmailError("That time occurs twice; supply an ISO time with an explicit UTC offset")
        parsed = valid[0]
    result = parsed.astimezone(timezone.utc)
    if result <= now.astimezone(timezone.utc):
        raise EmailError("The scheduled time must be in the future")
    return result
