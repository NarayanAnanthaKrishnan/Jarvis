from llm.client import LLMClient

SUMMARIZER_PROMPT = """You are a voice assistant response formatter. Given the user's question and data from tools, produce a short spoken response.

Rules:
- ONE to TWO short sentences.
- Use EXACT numbers from the data. Never change them.
- Natural speech. No markdown, no lists, no formatting, no JSON.
- Never say "according to the data" or "the tool returned" or "the results show".
- For weather: "In [location] it's [condition], [temperature] with [wind]."
- For search: "I found that [key fact]."
- If there's an error: "Sorry, I couldn't find that."
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

    messages = [
        {"role": "system", "content": "You produce concise spoken responses. Use exact numbers."},
        {"role": "user", "content": prompt}
    ]

    response = llm.call_raw(messages, temp=0.1)

    if response is None:
        return "I could not summarize the results right now."

    reply = response["message"]["content"].strip()
    if not reply:
        return "I could not find an answer to that."
    return reply
