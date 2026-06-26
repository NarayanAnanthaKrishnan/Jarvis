from memory.store import memory_store

SESSION_PROMPT = """Summarize this conversation in 1-2 sentences for future retrieval.
Focus on key topics, decisions, and user preferences mentioned.

Conversation:
{HISTORY}

Summary:"""


def summarize(llm, history: list[dict]) -> str:
    user_msgs = [m["content"] for m in history if m["role"] in ("user", "assistant")]
    text = "\n".join(user_msgs[-10:])
    if not text.strip():
        return ""
    prompt = SESSION_PROMPT.replace("{HISTORY}", text)
    messages = [
        {"role": "system", "content": "You produce 1-2 sentence conversation summaries."},
        {"role": "user", "content": prompt}
    ]
    response = llm.call_raw(messages, temp=0.1)
    if response is None:
        return ""
    return response["message"]["content"].strip()


def save_session(llm, history: list[dict]):
    summary = summarize(llm, history)
    if summary:
        memory_store.add("episodic", summary, metadata={"type": "session"})
        return summary
    return ""
