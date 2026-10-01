import re


_ADDRESS = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
_AT_PHRASE = re.compile(r"\b(?:at\s+the\s+rate|at\s+rate|at\s+sign|at\s+symbol|add\s+the|ad\s+the|at|ad|add)(?=\s|$)", re.I)
_DOMAIN_ALIASES = (
    (re.compile(r"\b(?:g\s*mail|gee\s*mail|gemail|regmail|red\s+gmail)\b", re.I), "gmail"),
    (re.compile(r"\b(?:out\s*look|red\s+outlook)\b", re.I), "outlook"),
    (re.compile(r"\bhot\s*mail\b", re.I), "hotmail"),
    (re.compile(r"\byahoo\s*mail\b", re.I), "yahoo"),
    (re.compile(r"\bi\s*cloud\b", re.I), "icloud"),
)


def normalize_spoken_address(value: str) -> tuple[str, bool] | None:
    if not isinstance(value, str) or not value.strip() or len(value) > 320:
        return None
    original = value.strip().strip(" ,;<>\"'")
    if _ADDRESS.fullmatch(original):
        return original, False
    candidate = original
    candidate = re.sub(r"^(?:email\s+address\s+is|the\s+address\s+is|it\s+is|it's)\s+", "", candidate, flags=re.I)
    candidate = re.sub(r"(?<=\w)\s+(?:ad|add)(?=gmail(?:\s*(?:dot|\.)|\.?)com\b)", "@", candidate, flags=re.I)
    candidate = _AT_PHRASE.sub("@", candidate)
    candidate = re.sub(r"\b(?:dot|period)\b", ".", candidate, flags=re.I)
    candidate = re.sub(r"\s*@\s*", "@", candidate)
    candidate = re.sub(r"\s*\.\s*", ".", candidate)
    for pattern, domain in _DOMAIN_ALIASES:
        candidate = pattern.sub(domain, candidate)
    candidate = re.sub(r"(?<=@)the\s*(?=(?:gmail|outlook|hotmail|yahoo|icloud))", "", candidate, flags=re.I)
    candidate = re.sub(r"(?<=@)(?:red|ad|add)\s*(?=(?:gmail|outlook))", "", candidate, flags=re.I)
    candidate = candidate.replace(" ", "")
    if not _ADDRESS.fullmatch(candidate):
        return None
    return candidate, candidate != original


def speak_address(value: str) -> str:
    return re.sub(r"[.@]", lambda match: " at " if match.group() == "@" else " dot ", value)


def is_address_confirmation(value: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:yes|yeah|yep|correct|that's right|that is right|sounds right|confirm)\s*[.!]?\s*", value, re.I))


def is_address_rejection(value: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:no|nope|wrong|that's wrong|not right|incorrect|cancel)\s*[.!]?\s*", value, re.I))
