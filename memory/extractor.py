import json
import threading

from config import AUTO_EXTRACT

EXTRACTION_PROMPT = """Analyze the user's message. If it contains a DURABLE fact about the user worth remembering long-term (preference, plan, personal fact, commitment), return JSON: {"store": true, "fact": "<concise fact>", "type": "preference|fact|plan"}
If it's a question, casual chat, or transient, return {"store": false}.
Only store facts ABOUT THE USER. Be conservative — when unsure, don't store.

User message: {MESSAGE}
JSON:"""


def extract_and_store(user_text: str, llm) -> None:
    if not AUTO_EXTRACT:
        return
    if len(user_text.strip()) < 10:
        return

    try:
        messages = [
            {"role": "system", "content": "You extract durable facts about the user. Return ONLY valid JSON."},
            {"role": "user", "content": EXTRACTION_PROMPT.replace("{MESSAGE}", user_text.strip())}
        ]
        response = llm.call_raw(messages, temp=0.1, max_tokens=150)
        if response is None:
            return

        raw = response["message"]["content"]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        brace = cleaned.find("{")
        end = cleaned.rfind("}")
        if brace >= 0 and end > brace:
            cleaned = cleaned[brace:end+1]

        data = json.loads(cleaned)
        if not data.get("store"):
            return

        fact = data.get("fact", "").strip()
        fact_type = data.get("type", "fact")
        if not fact:
            return

        from memory.store import memory_store
        existing = memory_store.query_with_scores("semantic", fact, n=1)
        if existing:
            doc, dist = existing[0]
            if dist is not None and dist < 0.3:
                return

        memory_store.add("semantic", fact, metadata={"type": fact_type, "source": "auto"})

    except Exception:
        pass
