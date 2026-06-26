import json
from llm.client import LLMClient
from tools.profile_loader import load_profile

TOOL_DESCRIPTIONS = """
- search_web(query): Search the web for current information, news, facts
- get_city_info(ip): Look up city and region from IP address (use "auto" for current location)
- get_weather(city): Get current weather conditions for a city
- get_datetime(): Get the current date and time
- calculate(expression): Safely evaluate a math expression, e.g. calculate("(2+3)*4")
- open_app(app_name): Launch a desktop application (chrome, vscode, notepad, explorer, spotify, terminal, cmd, calculator, brave)
- take_note(note): Save a note to the user's notes file
- read_notes(last_n): Read the user's saved notes (default 5)
- update_note(index, content): Update or correct an existing note by its index number
- delete_note(index): Delete a note by its index number
- get_system_info(): Get CPU, RAM, and disk usage on this machine
- open_url(url): Open a URL or bookmark (github, gmail, youtube) in the browser
- read_clipboard(): Read the current clipboard contents
- get_news(topic): Fetch news headlines (general, tech, science, us)
- media_control(action): Control media playback (play, pause, next, previous, volume up, volume down, mute)
"""

PLANNER_PROMPT = """You are Jarvis's intent router. Analyze the user's request and create a plan.
User profile (for personalization):
{PROFILE}

Relevant memories:
{MEMORIES}

Recent conversation (for context — use this to resolve pronouns like "he", "she", "it", "they", "that"):
{CONVERSATION_HISTORY}

Available tools:
{TOOL_DESCRIPTIONS}

Rules:
- If the user is just greeting or chatting, output "chat" intent with a brief friendly message.
- If the user needs information, output "tool_request" intent with a plan.
- If the user says "remember that" followed by a fact, store it in memory via store_memory.
- If the user says "start fresh", "new session", "clear context", or "forget everything", output "new_session" intent.
- For weather: if a city is named, use get_weather directly. If no city is named but there is a default weather city in memories, use that city. If no city anywhere, use get_city_info first.
- Use conversation history to resolve pronouns — "he", "she", "it", "they", "that" refer to the subject of the previous user message.
- If the user corrects or contradicts a fact that was previously noted, use update_note to fix the existing note.
- For general knowledge, news, facts: use search_web.
- Only include tools DIRECTLY needed. Do NOT include get_city_info unless the query needs location context (weather, local info, "near me").
- Output ONLY valid JSON. No markdown, no code blocks, no backticks, no formatting.

Output format for chat:
{"intent": "chat", "plan": [], "message": "short friendly reply"}

Output format for tool requests:
{"intent": "tool_request", "plan": [{"tool": "tool_name", "args": {"param": "value"}}], "message": ""}

Output format for new session:
{"intent": "new_session", "plan": [], "message": ""}

Examples:
User: hello
Assistant: {"intent": "chat", "plan": [], "message": "Hello! How can I help you today?"}

User: what is the weather in London
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_weather", "args": {"city": "London"}}], "message": ""}

User: what is the weather like here
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_weather", "args": {"city": "auto"}}], "message": ""}

User: search for AI news
Assistant: {"intent": "tool_request", "plan": [{"tool": "search_web", "args": {"query": "AI news 2026"}}], "message": ""}

User: what time is it
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_datetime", "args": {}}], "message": ""}

User: what is the date
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_datetime", "args": {}}], "message": ""}

User: calculate 15 percent of 200
Assistant: {"intent": "tool_request", "plan": [{"tool": "calculate", "args": {"expression": "0.15*200"}}], "message": ""}

User: take a note
Assistant: {"intent": "tool_request", "plan": [{"tool": "take_note", "args": {"note": "note text"}}], "message": ""}

User: read my notes
Assistant: {"intent": "tool_request", "plan": [{"tool": "read_notes", "args": {"last_n": 5}}], "message": ""}

User: correct note 2 to say Ronaldo scored twice instead
Assistant: {"intent": "tool_request", "plan": [{"tool": "update_note", "args": {"index": 2, "content": "Cristiano Ronaldo scored two goals today."}}], "message": ""}

User: delete note 3
Assistant: {"intent": "tool_request", "plan": [{"tool": "delete_note", "args": {"index": 3}}], "message": ""}

User: what is my system status
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_system_info", "args": {}}], "message": ""}

User: open github in browser
Assistant: {"intent": "tool_request", "plan": [{"tool": "open_url", "args": {"url": "github"}}], "message": ""}

User: what is on my clipboard
Assistant: {"intent": "tool_request", "plan": [{"tool": "read_clipboard", "args": {}}], "message": ""}

User: give me the latest news
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_news", "args": {"topic": "general"}}], "message": ""}

User: play music
Assistant: {"intent": "tool_request", "plan": [{"tool": "media_control", "args": {"action": "play"}}], "message": ""}

User: next song
Assistant: {"intent": "tool_request", "plan": [{"tool": "media_control", "args": {"action": "next"}}], "message": ""}

User: remember that I like pepperoni pizza
Assistant: {"intent": "tool_request", "plan": [{"tool": "store_memory", "args": {"content": "User likes pepperoni pizza"}}], "message": ""}

User: start fresh
Assistant: {"intent": "new_session", "plan": [], "message": ""}

User: Actually it was Ronaldo who scored, not me
Assistant: {"intent": "tool_request", "plan": [{"tool": "read_notes", "args": {"last_n": 5}}, {"tool": "update_note", "args": {"index": 2, "content": "Cristiano Ronaldo scored two goals today."}}], "message": ""}"""


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    brace = text.find("{")
    if brace >= 0:
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace:i+1]
    return text


def create_plan(user_input: str, llm: LLMClient, conversation_history: list[str] | None = None) -> dict:
    profile = load_profile()
    from memory.store import memory_store
    semantic = memory_store.query("semantic", user_input, n=5)
    episodic = memory_store.query("episodic", user_input, n=3)
    
    weather_memories = memory_store.query("semantic", "weather location default city", n=2)
    weather_ctx = ""
    if weather_memories:
        weather_ctx = "Default weather location from memories:\n" + "\n".join(f"- {m}" for m in weather_memories)

    memories = ""
    all_mem = (semantic or []) + (episodic or [])
    if all_mem:
        items = [f"- {m}" for m in all_mem]
        memories = "\n".join(items)
    if weather_ctx:
        memories = weather_ctx + "\n" + memories if memories else weather_ctx

    conv_history = ""
    if conversation_history:
        lines = [f"Previous: {utterance}" for utterance in conversation_history[-2:]]
        conv_history = "\n".join(lines)

    prompt = (PLANNER_PROMPT.replace("{PROFILE}", profile or "(none)")
                           .replace("{MEMORIES}", memories or "(none)")
                           .replace("{CONVERSATION_HISTORY}", conv_history or "(none)")
                           .replace("{TOOL_DESCRIPTIONS}", TOOL_DESCRIPTIONS))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]

    response = llm.call_raw(messages, temp=0.1)

    if response is None:
        return {
            "intent": "chat",
            "plan": [],
            "message": "I could not process that request right now."
        }

    raw = response["message"]["content"]
    cleaned = _extract_json(raw)

    try:
        plan = json.loads(cleaned)
        if not isinstance(plan, dict) or "intent" not in plan:
            raise ValueError("Missing intent field")
        return plan
    except (json.JSONDecodeError, ValueError):
        print(f"⚠ Planner parse failed: {cleaned[:200]}")
        return {
            "intent": "chat",
            "plan": [],
            "message": cleaned
        }
