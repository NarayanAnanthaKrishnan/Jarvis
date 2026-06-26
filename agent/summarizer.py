from llm.client import LLMClient
from tools.profile_loader import load_profile

SUMMARIZER_PROMPT = """You are a voice assistant response formatter. Given the user's question and data from tools, produce a short spoken response.

Rules:
- ONE short sentence max (under 200 characters total).
- Use EXACT numbers from the data. Never change them.
- Natural speech. No markdown, no lists, no formatting, no JSON.
- Never say "according to the data" or "the tool returned" or "the results show".
- For weather: "In [location] it's [condition], [temperature] with [wind]."
- NEVER say "I found information about" or "According to search results" or any meta-description. State the actual answer directly.
- If search results are partial or truncated, extract the single most useful specific fact available and state it. Never describe the data.
- For comparison questions (X vs Y), always state one concrete difference or recommendation, never just acknowledge both exist.
- If the data doesn't fully answer the question, report what you found anyway.
- JUST THE ANSWER. Nothing else.

User question: {USER_INPUT}

Data:
{TOOL_RESULTS}

Your spoken response:"""


def summarize(user_input: str, tool_results: list[dict], llm: LLMClient) -> str:
    if not tool_results:
        return ""

    formatted = "\n".join(
        f"[{r['tool']}] {r['result']}" for r in tool_results
    )

    if not formatted.strip():
        return ""

    prompt = SUMMARIZER_PROMPT.replace("{USER_INPUT}", user_input).replace("{TOOL_RESULTS}", formatted)

    profile = load_profile()
    from memory.store import memory_store
    semantic = memory_store.query("semantic", user_input, n=3)
    episodic = memory_store.query("episodic", user_input, n=2)
    system_msg = "You produce concise spoken responses from tool data. Use exact numbers. Never say you couldn't find something if data exists — report what was found."
    if profile:
        system_msg += f"\nUser context:\n{profile}"
    if semantic or episodic:
        items = [f"- {m}" for m in semantic + episodic]
        system_msg += "\nRelevant memories:\n" + "\n".join(items)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]

    response = llm.call_raw(messages, temp=0.1)

    if response is None:
        return "I could not summarize the results right now."

    reply = response["message"]["content"].strip()
    if not reply:
        return "I could not find an answer to that."
    return reply
