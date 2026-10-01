import re

from email_agent.addressing import speak_address


_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+")


def speech_safe_text(value: str) -> str:
    return _EMAIL.sub(lambda match: speak_address(match.group()), value)
