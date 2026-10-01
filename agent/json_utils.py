import json
import re
from typing import Any


def extract_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("Response must be text")
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.S | re.I)
    if fenced:
        value = fenced[1]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Response must be one complete JSON object")
    return parsed
