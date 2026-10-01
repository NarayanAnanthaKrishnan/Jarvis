import re
import time
from typing import Any

from config import RETRIEVAL_GATE
from ops.tracer import trace

GATE_PROMPT = """Does answering this need memory of the user's facts, preferences, plans, or past conversations? Answer only YES or NO.

Examples:
'what is 2+2' -> NO
'when am I meeting Alex' -> YES
'what's the weather' -> NO
'what did we discuss yesterday' -> YES
'write my cover letter' -> YES

Message: {user_input}
Answer:"""


def is_stateless_lookup(text: str) -> bool:
    value = text.lower().strip().replace("’", "'")
    value = re.sub(r"^(?:(?:hello|hi|hey|jarvis|please)[,\s]+)+", "", value)
    clauses = re.split(r"\s+(?:and|also)\s+", value)
    if not 1 <= len(clauses) <= 2:
        return False
    patterns = (
        r"(?:what time is it|what(?:'s| is) (?:the )?(?:current )?(?:time|date)|what day is (?:it|today)|tell me (?:the )?(?:time|date))(?: right now| now| today)?",
        r"(?:what(?:'s| is)|how(?:'s| is)|tell me) (?:the )?weather(?: like)?(?: outside| today| right now)?",
    )
    return all(any(re.fullmatch(pattern, clause.strip(" ,.!?")) for pattern in patterns) for clause in clauses)


def needs_memory(user_input: str, llm: Any) -> bool:
    started = time.monotonic()
    if not RETRIEVAL_GATE:
        return True
    if is_stateless_lookup(user_input):
        trace("memory_gate", source="stateless_lookup", needs_memory=False, elapsed_s=0)
        return False

    try:
        prompt = GATE_PROMPT.replace("{user_input}", user_input)
        messages = [
            {"role": "system", "content": "You decide if a query needs memory. Answer only YES or NO."},
            {"role": "user", "content": prompt}
        ]
        response = llm.call_raw(messages, temp=0.0, max_tokens=5)
        if response is None:
            return True
        answer = response["message"]["content"].strip().lower()
        return "yes" in answer
    except Exception:
        return True
    finally:
        trace("memory_gate", source="model", elapsed_s=round(time.monotonic() - started, 3))
