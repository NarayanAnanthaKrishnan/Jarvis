import json
from datetime import datetime
from llm.client import LLMClient
from tools.profile_loader import load_profile

TOOL_DESCRIPTIONS = """
- search_web(query, num_results=8): Search the web for current information, news, facts. Be specific with your query.
- fetch_url(url): Fetch and extract readable text from a URL. Use when search results give a promising link but not enough detail.
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
- store_memory(content): Remember a fact or preference about the user
- set_reminder(message, when): Set a reminder with natural language time like '3pm tomorrow', 'in 20 minutes', '8pm'
- list_reminders(): List all pending reminders
"""

PLANNER_PROMPT = """Today's date: {CURRENT_DATE}

You are Jarvis's intent router. Analyze the user's request and create a plan.
User profile (for personalization):
{PROFILE}

Relevant memories:
{MEMORIES}

Recent conversation (includes both user and assistant messages — use the assistant's previous reply to resolve references like "the three matches", "their timings", "those teams"):
{CONVERSATION_HISTORY}

Available tools:
{TOOL_DESCRIPTIONS}

Rules:
- If the user is just greeting or chatting, output "chat" intent with a brief friendly message.
- If the user needs information, output "tool_request" intent with a plan.
- If the user says "remember that" followed by a fact, store it in memory via store_memory.
- If the user says "start fresh", "new session", "clear context", or "forget everything", output "new_session" intent.
- For weather: if a city is named, use get_weather directly. If no city is named but there is a default weather city in memories, use that city. If no city anywhere, use get_city_info first.
- Use conversation history to resolve references. "User:" and "Assistant:" labels show who spoke. "the three matches", "their timings", "those teams" refer to specific entities mentioned in the ASSISTANT's last reply.
- For follow-up questions about specific items from a previous answer (e.g. "timings for the three matches"), extract the named entities from the assistant's last response and construct ONE precise search query containing all of them. Do NOT split into multiple searches.
- If the user corrects or contradicts a fact that was previously noted, use read_notes first to find the index, then update_note.
- For general knowledge, news, facts: use search_web first. If results are too vague, use fetch_url on the most promising link.
- Multi-step: You can chain tools. For example, search_web for something, then fetch_url on the best result to get full details.
- Include current year or context in search queries for better results (e.g. "2026 World Cup" not just "World Cup").
- Include "confidence" (0.0 to 1.0) in your output. Set confidence < 0.6 if you're unsure about the plan or need clarification.
- Only include tools DIRECTLY needed. Do NOT include get_city_info unless the query needs location context (weather, local info, "near me").
- Output ONLY valid JSON. No markdown, no code blocks, no backticks, no formatting.

Output format for chat:
{"intent": "chat", "plan": [], "message": "short friendly reply", "confidence": 1.0}

Output format for tool requests:
{"intent": "tool_request", "plan": [{"tool": "tool_name", "args": {"param": "value"}}], "message": "", "confidence": 1.0}

Output format for new session:
{"intent": "new_session", "plan": [], "message": "", "confidence": 1.0}

Examples:
User: hello
Assistant: {"intent": "chat", "plan": [], "message": "Hello! How can I help you today?", "confidence": 1.0}

User: what is the weather in London
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_weather", "args": {"city": "London"}}], "message": "", "confidence": 1.0}

User: what is the weather like here
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_weather", "args": {"city": "auto"}}], "message": "", "confidence": 1.0}

User: who won the latest election and what did they promise
Assistant: {"intent": "tool_request", "plan": [{"tool": "search_web", "args": {"query": "2026 latest election results and promises"}}], "message": "", "confidence": 0.9}

User: what does the article say about AI regulation
Assistant: {"intent": "tool_request", "plan": [{"tool": "search_web", "args": {"query": "AI regulation 2026 latest developments"}}, {"tool": "fetch_url", "args": {"url": "(best result URL from previous step)"}}], "message": "", "confidence": 0.8}

User: what time is it
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_datetime", "args": {}}], "message": "", "confidence": 1.0}

User: what is the date
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_datetime", "args": {}}], "message": "", "confidence": 1.0}

User: what are the timings for the three matches
Assistant: {"intent": "tool_request", "plan": [{"tool": "search_web", "args": {"query": "Ivory Coast vs Norway Mexico vs Ecuador France vs Sweden June 30 2026 World Cup match times kickoff"}}], "message": "", "confidence": 0.9}

User: calculate 15 percent of 200
Assistant: {"intent": "tool_request", "plan": [{"tool": "calculate", "args": {"expression": "0.15*200"}}], "message": "", "confidence": 1.0}

User: take a note
Assistant: {"intent": "tool_request", "plan": [{"tool": "take_note", "args": {"note": "note text"}}], "message": "", "confidence": 1.0}

User: read my notes
Assistant: {"intent": "tool_request", "plan": [{"tool": "read_notes", "args": {"last_n": 5}}], "message": "", "confidence": 1.0}

User: correct note 2 to say Ronaldo scored twice instead
Assistant: {"intent": "tool_request", "plan": [{"tool": "read_notes", "args": {"last_n": 5}}, {"tool": "update_note", "args": {"index": 2, "content": "Cristiano Ronaldo scored two goals today."}}], "message": "", "confidence": 0.9}

User: delete note 3
Assistant: {"intent": "tool_request", "plan": [{"tool": "delete_note", "args": {"index": 3}}], "message": "", "confidence": 1.0}

User: what is my system status
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_system_info", "args": {}}], "message": "", "confidence": 1.0}

User: open github in browser
Assistant: {"intent": "tool_request", "plan": [{"tool": "open_url", "args": {"url": "github"}}], "message": "", "confidence": 1.0}

User: what is on my clipboard
Assistant: {"intent": "tool_request", "plan": [{"tool": "read_clipboard", "args": {}}], "message": "", "confidence": 1.0}

User: give me the latest news
Assistant: {"intent": "tool_request", "plan": [{"tool": "get_news", "args": {"topic": "general"}}], "message": "", "confidence": 1.0}

User: play music
Assistant: {"intent": "tool_request", "plan": [{"tool": "media_control", "args": {"action": "play"}}], "message": "", "confidence": 1.0}

User: next song
Assistant: {"intent": "tool_request", "plan": [{"tool": "media_control", "args": {"action": "next"}}], "message": "", "confidence": 1.0}

User: remember that I like pepperoni pizza
Assistant: {"intent": "tool_request", "plan": [{"tool": "store_memory", "args": {"content": "User likes pepperoni pizza"}}], "message": "", "confidence": 1.0}

User: Actually it was Ronaldo who scored, not me
Assistant: {"intent": "tool_request", "plan": [{"tool": "read_notes", "args": {"last_n": 5}}, {"tool": "update_note", "args": {"index": 2, "content": "Cristiano Ronaldo scored two goals today."}}], "message": "", "confidence": 0.9}

User: start fresh
Assistant: {"intent": "new_session", "plan": [], "message": "", "confidence": 1.0}

User: remind me to push the code at 8pm
Assistant: {"intent": "tool_request", "plan": [{"tool": "set_reminder", "args": {"message": "push the code", "when": "8pm today"}}], "message": "", "confidence": 1.0}

User: remind me in 20 minutes to take a break
Assistant: {"intent": "tool_request", "plan": [{"tool": "set_reminder", "args": {"message": "take a break", "when": "in 20 minutes"}}], "message": "", "confidence": 1.0}

User: what are my reminders
Assistant: {"intent": "tool_request", "plan": [{"tool": "list_reminders", "args": {}}], "message": "", "confidence": 1.0}"""


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
        conv_history = "\n".join(conversation_history[-8:])

    current_date = datetime.now().strftime("%A, %B %d, %Y")

    prompt = (PLANNER_PROMPT.replace("{CURRENT_DATE}", current_date)
                           .replace("{PROFILE}", profile or "(none)")
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
            "message": "I could not process that request right now.",
            "confidence": 0.0
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
            "message": cleaned,
            "confidence": 0.3
        }
