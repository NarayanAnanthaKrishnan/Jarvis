from config import RETRIEVAL_GATE

GATE_PROMPT = """Does answering this need memory of the user's facts, preferences, plans, or past conversations? Answer only YES or NO.

Examples:
'what is 2+2' -> NO
'when am I meeting Alex' -> YES
'what's the weather' -> NO
'what did we discuss yesterday' -> YES
'write my cover letter' -> YES

Message: {user_input}
Answer:"""


def needs_memory(user_input: str, llm) -> bool:
    if not RETRIEVAL_GATE:
        return True

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
