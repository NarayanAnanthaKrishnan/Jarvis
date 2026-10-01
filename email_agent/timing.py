import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import dateparser

from email_agent.models import EmailError


_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                 "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}
_CLOCK = re.compile(r"\b(?:\d{1,2}(?::\d{2})?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(?:a\.?m\.?|p\.?m\.?)\b|\b(?:noon|midnight)\b", re.I)
_RELATIVE = re.compile(r"\bin\s+(?:\d+|one|two|three|four|five|ten|fifteen|twenty|thirty|forty|forty-five|sixty|half\s+(?:an?\s+)?|an?\s+)(?:seconds?|minutes?|hours?)\b", re.I)


def normalize_time_phrase(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^(?:please\s+)?(?:schedule|send|deliver)(?:\s+(?:it|this|the email|my email))?(?:\s+(?:for|at))?\s+", "", text, flags=re.I)
    text = re.sub(r"\blater\s+(today|tonight)\b", r"\1", text, flags=re.I)
    for word, digit in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}(?=\s*(?:a\.?m\.?|p\.?m\.?))", digit, text, flags=re.I)
    text = re.sub(r"\bin\s+half\s+(?:an?\s+)?hour\b", "in 30 minutes", text, flags=re.I)
    text = re.sub(r"\bin\s+(?:an?|one)\s+hour\b", "in 60 minutes", text, flags=re.I)
    text = re.sub(r"\bin\s+a\s+quarter\s+hour\b", "in 15 minutes", text, flags=re.I)
    text = re.sub(r"\b(today|tomorrow|tonight)\s+(\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?)\b", r"\1 at \2", text, flags=re.I)
    return text


def extract_time_phrase(value: str) -> str | None:
    text = normalize_time_phrase(value)
    relative = list(_RELATIVE.finditer(text))
    clocks = list(_CLOCK.finditer(text))
    if len(relative) + len(clocks) > 1:
        raise EmailError("I heard more than one possible time. Please give me just one time to schedule.")
    if relative:
        return relative[0].group(0)
    if not clocks:
        return None
    clock = clocks[0]
    date_pattern = re.compile(
        r"\b(?:today|tomorrow|tonight|(?:next|this)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\b"
        r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)?", re.I)
    dates = [match for match in date_pattern.finditer(text, 0, clock.start())]
    start = dates[-1].start() if dates else clock.start()
    phrase = text[start:clock.end()].strip(" ,.;")
    if re.search(r"\bor\b", phrase, re.I) and len(_CLOCK.findall(text)) > 1:
        raise EmailError("I heard more than one possible time. Please give me just one time to schedule.")
    return phrase


def parse_send_time(value: str, zone: str, now: datetime) -> datetime:
    try:
        tz = ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise EmailError("Use a named timezone such as America/New_York") from exc
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise EmailError("Provide a date and time")
    value = normalize_time_phrase(value)
    if len(_CLOCK.findall(value)) > 1 and re.search(r"\bor\b", value, re.I):
        raise EmailError("I heard more than one possible time. Please give me just one time to schedule.")
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
        clock = bool(_CLOCK.search(value) or re.search(r"\b(?:1[3-9]|2[0-3]):[0-5]\d\b", value, re.I))
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
