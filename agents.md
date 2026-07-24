# Jarvis — AI Agent Reference

## Project Overview
- **What:** Personal AI assistant — press hotkey, speak, get spoken response
- **Hardware:** GTX 1650 (4GB VRAM), 8GB RAM, Windows
- **LLM:** Cloud BYOLLM (Gemini / OpenAI / Anthropic) via SDK. No local models.
- **STT:** faster-whisper (base, CUDA float16 — no VRAM conflict since LLM is cloud)
- **TTS:** Kokoro ONNX int8 (CPU, speed=1.0)

## Dev Rules
- STT can run on GPU now (no local LLM competing for VRAM)
- Terminal must run as Administrator (keyboard lib requirement)
- Python 3.12
- Project lives in `D:\Jarvis\` directly (no subfolder)

---

## Current Implementation (Phase D — ReAct Agent Loop)

### Architecture
```
[CTRL+SHIFT+J] press
    -> Session starts, mic listens with VAD
    -> User speaks → pause 1.5s silence → VAD auto-submits turn
    -> RealtimeSTT transcribes (base, CUDA float16)
    -> run_agent() → ReAct loop (up to 6 steps):
       1. THINK: LLM decides next tool or signals done
       2. OBSERVE: execute_tool() runs the tool
       3. REPEAT until done or MAX_STEPS reached
    -> StreamingSpeaker: accumulates tokens → splits on sentence boundaries
       → Kokoro TTS (int8, speed=1.15) → sounddevice plays sentence by sentence
    -> Delivery: "speak" (TTS), "paste" (clipboard + paste at cursor), or "both"
[CTRL+SHIFT+J] press again
    -> Session ends, mic stops, TTS stops
    -> Full session saved as one episodic memory
```

### ReAct Agent Pipeline

```
User Input
    │
    ▼
┌──────────────────┐
│  run_agent()     │  MAX_STEPS=6 hard cap
│  ┌────────────┐  │
│  │  think()   │──│── LLM call (temp=0.1) → JSON decision
│  └─────┬──────┘  │     {"tool": "...", "args": {...}}
│        │         │     or {"done": True, "answer": "...", "delivery": "speak"}
│  ┌─────▼──────┐  │
│  │exec_tool() │  │  tools.registry.execute_tool(name, args, llm)
│  └─────┬──────┘  │
│        │         │
│  ┌─────▼──────┐  │
│  │ observe    │  │  Step result appended → fed back to next think()
│  └────────────┘  │
└──────────────────┘
        │
        ▼
   Streaming TTS → spoken response
```

### Delivery Modes
- **speak** — TTS via StreamingSpeaker
- **paste** — clipboard + paste at cursor via paste_at_cursor tool
- **both** — speak and paste simultaneously

### File Reference

#### `config.py`
- `HOTKEY = "ctrl+shift+j"` — Jarvis agent hotkey
- `WHISPERFLOW_HOTKEY = "ctrl+shift+k"` — Pure STT mode hotkey
- `MAX_STEPS = 6` — maximum ReAct loop iterations per turn
- `STT_MODEL = "base"`
- `STT_DEVICE = "cuda"` — GPU accelerated Whisper (no VRAM conflict)
- `SAMPLE_RATE = 16000`
- `CHANNELS = 1`
- `PROVIDER = "gemini"` — LLM provider: "openai" | "gemini" | "anthropic"
- `OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY` — API keys for each provider (loaded from .env via `os.getenv`)
- `OPENAI_MODEL / GEMINI_MODEL / ANTHROPIC_MODEL` — per-provider model selection
- `AUTO_EXTRACT = True` — toggle async auto fact extraction after each turn
- `REMINDER_CHECK_SECONDS = 30` — how often the background thread checks due reminders
- `RETRIEVAL_GATE = True` — toggle the cheap LLM gate that decides if memories are needed before querying ChromaDB
- `TRACING = True` — toggle always-on JSONL tracing to `.traces/<date>.jsonl`

#### `audio/recorder.py`
- Class: `Recorder`
- `__init__(sample_rate=16000)` — stores sample rate, init empty lists
- `start()` — guards against re-entry via `if self.is_recording: return`, then creates `sd.InputStream`, starts callback-based recording
- `_callback(indata, frames, time, status)` — appends `indata.copy()` if recording
- `stop() -> np.ndarray` — stops stream, closes it, concatenates chunks, flattens
- `get_partial(since: int) -> tuple[np.ndarray, int]` — thread-safe read of audio chunks from an index checkpoint

#### `audio/player.py`
- Class: `Player`
- `play(samples, sample_rate)` — `sd.play()`, `sd.wait()`

#### `stt/transcriber.py`
- `_add_cuda_dll_dirs()` — adds CUDA DLL paths via `os.add_dll_directory()` so ctranslate2 can find `cublas64_12.dll`, `cudart64_12.dll`, `cudnn64_9.dll`
- Class: `Transcriber`
- `__init__(model_size, device)` — calls `_add_cuda_dll_dirs()` when CUDA, loads WhisperModel with `compute_type="float16"` on CUDA or `"int8"` on CPU
- `transcribe(audio)` — ignores < 0.5s, transcribes with `beam_size=1, vad_filter=True` for speed

#### `llm/prompts.py`
- `SYSTEM_PROMPT` — strict English, 1-2 sentences, no formatting, no "how can I help". Contains `{PROFILE}` placeholder replaced at init.
- `ULTRA_SYSTEM_PROMPT` — content generation prompt for generate_content tool
- `AGENT_SYSTEM_PROMPT` — used by agent/loop.py think() for ReAct decision-making

#### `llm/client.py`
- Class: `LLMClient`
- **BYOLLM architecture** — supports OpenAI, Gemini, Anthropic via unified internal interface
- `__init__()` — initializes session, history with system prompt, client cache. Loads user profile and injects into system prompt.
- `_get_client(provider)` — lazy-loads the right SDK client (OpenAI / gemini / Anthropic). Uses `threading.Lock` for safe concurrent access from async extraction thread.
- `_convert_messages(messages, provider)` — translates internal format to each provider's message schema:
  - **OpenAI**: pass-through with tool_call_id
  - **Gemini**: `parts` array for user/model, `functionCall`/`functionResponse` for tools
  - **Anthropic**: content blocks (`text`, `tool_use`, `tool_result`)
- `_convert_tools(tools, provider)` — translates OpenAI-format tool defs to:
  - **Gemini**: `FunctionDeclaration` wrapped in `function_declarations`
  - **Anthropic**: `input_schema` (renamed from `parameters`)
- `_normalize_response(response, provider)` — converts each provider's response back to unified `{"message": {"content", "tool_calls": [...]}}` format
- `_chat(tools=None)` — internal helper, dispatches to active provider. Uses `max_tokens=300`.
- `call_raw(messages, tools=None, temp=0.3, max_tokens=1000)` — stateless LLM call. Used by think() and generate_content.
- `stream_chat(user_input) -> Generator[str, None, None]` — streaming chat, used as best-effort fallback when agent loop fails.
- `chat(user_input)` — simple conversational chat. Trims history to last 20 exchanges to prevent unbounded growth.
- `chat_with_tools(user_input, tools)` — tool-calling loop (up to 5 rounds). Stores assistant `tool_calls` in history for OpenAI/Anthropic protocol compliance.
- `refresh_memories(query)` — queries ChromaDB, injects relevant memories into system prompt
- `rotate_session()` — summarizes current session, stores as episode, resets history
- On model failure: returns error string, pops user message from history
- Provider fallback chain: primary (config.PROVIDER) → fallback to providers with API keys

#### `tts/speaker.py`
- Class: `Speaker`
- `__init__()` — loads `Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")`, voice `"af_bella"`
- `speak(text: str)` — creates audio at speed=1.0, `sd.play()`, `sd.wait()`

#### `agent/loop.py`
- `run_agent(user_input, conversation_history, llm) -> dict` — main ReAct loop
  - Calls `needs_memory()` once per turn to decide if ChromaDB retrieval is needed
  - If gate says yes: queries top-5 semantic + top-3 episodic memories
  - If gate says no: passes `"(none)"` — skips retrieval entirely
  - Up to `MAX_STEPS` iterations
  - Each iteration: think() → execute_tool() → append step
  - On tool error: prints warning, error visible in steps context for agent to self-correct
  - When done: returns `{"output": str, "delivery": "speak"|"paste"|"both", "stream": Generator|None, "gate_needed": bool}`
  - On JSON parse failure: returns safe fallback message, logs raw text
- `think(user_input, steps, conversation_history, llm, memories="(none)") -> dict` — LLM decision step
  - Builds prompt from `SYSTEM_PROMPT`, `USER_PROMPT_FORMAT`, `TOOL_DESCRIPTIONS`
  - Injects: user profile, passed memories string, current date, last 8 conversation turns, current step context
  - Calls `llm.call_raw()` with system + user split (Gemini requirement)
  - Extracts JSON via `_extract_json()` with markdown code block stripping
  - Returns `{"tool": name, "args": {...}, "reason": "..."}` or `{"done": True, "answer": "...", "delivery": "..."}`
- `_extract_json(raw: str) -> str` — strips ```json fences, extracts JSON block
- `_format_steps(steps: list[dict]) -> str` — formats tool step context for prompt
- `_stream_summarize(...) -> Generator` — streaming best-effort fallback via `llm.stream_chat()`

#### `agent/prompts.py`
- `SYSTEM_PROMPT` — base prompt with rules, tool list format, delivery mode instructions, profile, memories, conversation history placeholders
- `USER_PROMPT_FORMAT` — step template for each tool call in the ReAct loop
- `TOOL_DESCRIPTIONS` — formatted list of all 22 tools with names, param schemas, and descriptions

#### `tools/registry.py`
- `TOOL_DEFINITIONS` — list of OpenAI-compatible function schemas (22 tools)
- `TOOL_MAP` — dict mapping tool name string → callable function
- `execute_tool(name, args, llm=None) -> str` — dispatches to the registered function, returns result string or error. Special-cases `generate_content` to pass `llm`.
- New tools:
  - **read_screen()** — wraps `screen_ocr.capture_and_ocr()`, returns screen text
  - **generate_content(instruction, screen_context="", llm=None)** — generates content (emails, code, letters) using `ULTRA_SYSTEM_PROMPT` + profile + semantic memories. Has `None` guard.
  - **paste_at_cursor(text)** — copies text to clipboard via `pyperclip`, pastes at cursor via `pyautogui.hotkey("ctrl", "v")`

#### `tools/geoip.py`
- `get_city_info(ip="auto")` — if "auto", resolves current public IP via `api.ipify.org`, then looks up city/region/country from `data/GeoLite2-City.mmdb` via `geoip2`
- Returns string like `"Syracuse, New York, United States"`

#### `tools/weather.py`
- `get_weather(city)` — queries `wttr.in` (no API key needed) with format: condition, temperature, wind, humidity. If `city="auto"`, resolves location via GeoIP first.
- Returns string like `"London: Partly cloudy, +20°C, ↘22km/h, 46% humidity"`

#### `tools/web_search.py`
- `search_web(query, num_results=8)` — uses `ddgs.DDGS`, returns formatted list of title/body/URL
- Returns full-length entries (no truncation — cloud LLMs handle long text)
- `fetch_url(url)` — downloads page via `requests`, strips HTML via `BeautifulSoup`, returns first ~4000 chars of readable text
- Used when search results have promising links but not enough detail

#### `tools/datetime_tool.py`
- `get_datetime()` — returns current date/time as natural string like `"Monday, June 15 2026, 10:34 AM"`
- No parameters, no external dependencies

#### `tools/calculator.py`
- `calculate(expression)` — safely evaluates math expressions using `ast` module node whitelist
- Only allows basic arithmetic operators. Returns result string or error message.
- No external dependencies

#### `tools/app_launcher.py`
- `open_app(app_name)` — launches a Windows application via `subprocess.Popen()`
- Maintains `APP_REGISTRY` dict mapping names to executable paths
- Resolves `%USERNAME%` via `os.environ`

#### `tools/notes.py`
- `take_note(note)` — appends timestamped note to `notes.txt` in project root
- `read_notes(last_n=5)` — reads last N notes, returns numbered list

#### `tools/system_info.py`
- `get_system_info()` — uses `psutil` to return CPU%, RAM used/total, Disk C: usage
- Returns a single readable string

#### `tools/browser.py`
- `open_url(url)` — opens a URL or named bookmark (github, gmail, youtube, etc.) in the browser
- Normalizes URL, checks `BOOKMARK_REGISTRY` first

#### `tools/clipboard_tool.py`
- `read_clipboard()` — uses `pyperclip.paste()`, returns clipboard content (truncated at 500 chars)

#### `tools/news.py`
- `get_news(topic="general")` — fetches top 5 RSS headlines. Topics: general, tech, science, us.
- Uses `feedparser`, no API key needed

#### `tools/media.py`
- `media_control(action)` — simulates media key presses via `keyboard.send()`
- Actions: play, pause, next, previous, volume up, volume down, mute

#### `tools/screen_ocr.py`
- `capture_and_ocr()` — takes full screenshot via `pyautogui`, OCRs via `winocr` (Windows native)
- Returns extracted text string or `"[No text detected on screen]"` / `"[OCR error: ...]"`
- Used by the `read_screen` tool in the ReAct loop

#### `tools/profile_loader.py`
- `load_profile()` — reads `profile/profile.md` from project root
- Returns the file content as a string, or empty string if file doesn't exist
- Used by LLMClient, agent loop (think), and generate_content tool

#### `modes/jarvis.py`
- Class: `JarvisMode`
- `__init__(stream_stt, llm, speaker, streaming_speaker)` — stores shared instances
- `toggle_session()` — toggles session on/off
- Session mode: press CTRL+SHIFT+J to start, press again to end (no hold)
- VAD auto-submit: after 1.5s silence, turn is auto-submitted to pipeline
- Mic pauses during processing + TTS, resumes after (no feedback loop)
- Streaming TTS via `StreamingSpeaker` (token-by-token → sentence TTS)
- Pipeline per turn:
  1. Check for "start fresh"/"new session" keywords → `llm.rotate_session()` → resume mic
  2. `agent.loop.run_agent(text, conversation_history, llm)` → result dict
  3. If `delivery` is "speak" or "both": stream or speak the output
  4. If `delivery` is "paste" or "both": `execute_tool("paste_at_cursor", ...)`
   5. If agent loop errors: fallback to `llm.chat()` with print + history save (fallback also gated by `needs_memory()`)
- After response spoken: fires async `extract_and_store(text, llm)` in daemon thread (zero latency impact)
- Stores conversation history as labeled `"User: ..."` / `"Assistant: ..."` pairs (not just user text) for accurate follow-up resolution
- Session end: saves full session history as one episodic memory

#### `modes/whisperflow.py`
- Class: `WhisperFlowMode`
- `__init__(recorder, transcriber)` — stores shared instances
- `on_activate()` — starts recording (no streaming, no terminal output)
- `on_release()` — stops recording, transcribes full audio, copies to clipboard, `ctrl+a` + `ctrl+v` at cursor
- No LLM, no TTS — press CTRL+SHIFT+K, speak, release → text appears
- Streaming was attempted (SendInput KEYEVENTF_UNICODE, PostMessage WM_CHAR) but all methods fail during hotkey hold due to `WH_KEYBOARD_LL` hook interception. Final-paste-only is the current approach.

#### `memory/store.py`
- `MemoryStore` class — ChromaDB wrapper with two collections (`semantic`, `episodic`)
- `add(collection, content, metadata)` — stores content with timestamp-based ID
- `query(collection, text, n)` — semantic search, returns content strings
- `query_with_scores(collection, text, n)` — returns `[(doc, distance)]` tuples for dedup checking
- `count(collection)` — returns document count (handles empty collection gracefully)
- Module-level singleton `memory_store` for use by tools and agent loop

#### `memory/session.py`
- `summarize(llm, history)` — summarizes last 10 exchanges via LLM call
- `save_session(llm, history)` — calls `summarize()` + stores result in episodic collection

#### `memory/extractor.py`
- `EXTRACTION_PROMPT` — LLM judge prompt that determines if a user message contains a durable fact
- `extract_and_store(user_text, llm)` — runs in background daemon thread after each turn:
  1. Calls LLM with extraction prompt (temp=0.1, max_tokens=150)
  2. Parses `{"store": true/false, "fact": "...", "type": "..."}` from response
  3. Dedup check via `memory_store.query_with_scores("semantic", fact, n=1)` with threshold < 0.3
  4. If not duplicate: stores in semantic collection with `"source": "auto"` metadata
- Silent on all errors (background thread must never crash)

#### `memory/reminders.py`
- SQLite reminder database (`reminders.db` in project root)
- `init_db()` — creates table on import
- `parse_when(natural)` — uses `dateparser` for natural language times like "3pm tomorrow", "in 20 minutes"
- `add_reminder(message, fire_at)` — inserts reminder, returns confirmation string with formatted time
- `get_due_reminders()` — queries `fire_at <= now AND fired = 0`
- `mark_fired(reminder_id)` — marks reminder as fired
- `list_pending()` / `format_pending()` — lists all pending reminders sorted by time

#### `memory/retrieval_gate.py` — NEW (Phase E)
- `needs_memory(user_input, llm) -> bool` — cheap fast classifier (one LLM call, temp=0, max_tokens=5)
- Prompt: 5 examples (2+2→NO, meeting→YES, weather→NO, yesterday→YES, cover letter→YES)
- Returns `True` if "yes" in response, `False` otherwise. `except` → `True` (safe default)
- Config guard: skip if `RETRIEVAL_GATE = False` (always retrieve)

#### `ops/tracer.py` — NEW (Phase E)
- Always-on JSONL tracing to `.traces/<YYYY-MM-DD>.jsonl`
- `trace(event, **data)` — appends one JSON line per call, wrapped in try/except (never crashes)
- Events: `turn_start`, `gate`, `think`, `tool`, `final`, `turn_end`, `error`
- Config guard: skip if `TRACING = False`

#### `main.py`
- Global instances: `Recorder`, `Transcriber`, `LLMClient`, `Speaker`
- Two mode instances: `JarvisMode`, `WhisperFlowMode` — shared `Recorder`, `Transcriber`, `LLMClient`
- Four hotkey registrations (press + release for each of two modes), all with `suppress=True`
- Starts `reminder_loop` daemon thread that checks `get_due_reminders()` every 30s, fires via TTS + plyer desktop notification
- Call `keyboard.wait()` at end

---

## Evolution Plan

### Phase A — Dual-Mode Architecture ✅ (Complete)

#### What was done
- Split current pipeline into two modes: Jarvis (agent) and WhisperFlow (dictation)
- New files: `modes/__init__.py`, `modes/jarvis.py`, `modes/whisperflow.py`
- `main.py` refactored as router with two hotkey pairs
- `config.py`: added `WHISPERFLOW_HOTKEY = "ctrl+shift+k"`
- Dependency: `pyautogui` for `Ctrl+V` paste simulation
- `README.md` created with setup/usage instructions

#### Mode Overview
| Mode | Hotkey | Pipeline | Purpose |
|---|---|---|---|
| **Jarvis** | `CTRL+SHIFT+J` | Hotkey → Record → STT → ReAct Agent Loop → TTS (or paste) | Full assistant with tools |
| **WhisperFlow** (Type mode) | `CTRL+SHIFT+K` | Hotkey → Record → STT → Paste at cursor | Dictate text anywhere |

---

### Phase B — Tool Calling + Web Search ✅ (Complete)

#### What was done
- New `tools/` package: `registry.py`, `geoip.py`, `weather.py`, `web_search.py`
- Three tools registered: web search (DuckDuckGo), weather (wttr.in), GeoIP (MaxMind)
- `llm/client.py`: added `chat_with_tools()` method with tool-calling loop (up to 5 rounds)
- `modes/jarvis.py`: upgraded to use `chat_with_tools()` with `TOOL_DEFINITIONS`
- Latency improvements applied:
  - STT: `beam_size=5` → `beam_size=1` + `vad_filter=True` (-4 to -6s)
  - LLM: `num_predict=200` → `100`, `num_ctx=2048` → `1024` (-2 to -3s)
  - Removed gemma4:e2b fallback (eliminated 10-30s dead timeout on failure)
  - TTS speed: 1.3 → 1.5 (-0.5s)
- Dependencies: `duckduckgo_search`, `geoip2`, `pyautogui` (Phase A)
- `data/GeoLite2-City.mmdb` placed in `D:\Jarvis\data\`

---

### Phase B2 — BYOLLM + Cloud LLMs ✅ (Complete)

#### What was done
- Replaced Ollama with cloud BYOLLM (Bring Your Own LLM) architecture
- Three providers supported: OpenAI (gpt-4o-mini), Gemini (gemini-2.5-flash-lite), Anthropic (claude-3-5-haiku-latest)
- `llm/client.py`: full rewrite with provider-agnostic message/tool converters and response normalizers
- `config.py`: added `PROVIDER`, `*_API_KEY`, and `*_MODEL` fields
- `stt/transcriber.py`: `compute_type="float16"` on CUDA (was `"int8"` on CPU)
- STT moved to GPU (no VRAM conflict since LLM is now cloud)
- Dependencies: removed `ollama`, added `google-genai`, `openai`, `anthropic`, CUDA runtime packages (`nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`)

#### Provider Architecture
```
LLMClient
    ├── _get_client(provider)   → lazy-loads OpenAI / gemini / Anthropic SDK
    ├── _convert_messages()     → internal format → provider message schema
    ├── _convert_tools()        → OpenAI tool defs → provider tool format
    ├── _normalize_response()   → provider response → unified format
    ├── _chat()                 → dispatches to active provider
    ├── call_raw()              → stateless call (think, generate_content)
    ├── stream_chat()           → streaming chat (best-effort fallback)
    ├── chat()                  → conversational with history
    └── chat_with_tools()       → tool-calling loop (now stores assistant tool_calls in history)
```

---

### Phase B3 — Cloud LLM Optimization + Profile Injection ✅ (Complete)

#### What was done
- Increased `max_tokens` across all LLM calls (was 100/150, now 300/1000/2000) — cloud models handle large output easily
- Made `call_raw()` accept `max_tokens` parameter so each caller can set appropriate budget
- Injected user profile (`profile/profile.md`) into:
  - **LLMClient chat system prompt** — all chat/chat_with_tools conversations have user context
  - **Agent loop think()** — personalized tool selection and routing
  - **generate_content tool** — content generation with user context
- Added history trimming in `chat()` — caps at 20 exchanges to prevent unbounded growth
- Fixed concurrency bug in all 3 modes: audio captured synchronously in `on_release`, passed to pipeline thread instead of having two competing `stop()` calls
- TTS speed normalized: 1.5 → 1.0 (clear speech)
- Web search truncation expanded: 150 → 400 chars per result

---

### Phase C — Memory ✅ (Complete)

#### Architecture: Two ChromaDB Collections

```
memory/
├── __init__.py          # Package marker
├── store.py             # ChromaDB wrapper (add, query, count)
└── session.py           # Session summarizer + save helper

memory_db/               # Auto-created by ChromaDB PersistentClient
```

- **semantic_memory** — User facts and preferences, injected top-5 into think() + top-3 into fallback chat
- **episodic_memory** — Session summaries (on "start fresh" or shutdown), injected top-3 into think() + top-2 into fallback chat

**Embedding:** ChromaDB `DefaultEmbeddingFunction` (all-MiniLM-L6-v2 via ONNX, 384-dim, bundled with chromadb, no external service)

#### Storage Triggers

1. **Explicit ("remember that X")** — Agent detects request → calls `store_memory` tool → saves to semantic collection
2. **"start fresh" / "new session"** — `LLMClient.rotate_session()` summarizes history → stores as episode → resets `self.history`
3. **Auto-extraction** — Background thread after each turn judges if user input contains a durable fact
4. **Shutdown** — `atexit` handler in `main.py` saves current session as episode (best-effort)

#### Memory Injection Points

| Call Point | Where | What's Injected |
|---|---|---|
| Agent think() | System prompt | Top-5 semantic + top-3 episodic (gated by `needs_memory()`) |
| Chat (`llm.chat()`) | `self.history[0]` via `refresh_memories()` | Top-5 semantic + top-3 episodic (gated in fallback) |
| chat_with_tools | `self.history[0]` via `refresh_memories()` | Top-5 semantic + top-3 episodic |
| generate_content | System prompt | Top-3 semantic |

---

### Phase C2 — Auto-Extraction + Persistent Reminders ✅ (Complete)

#### What was done
- **Auto-extraction**: After each turn, a background daemon thread calls the LLM with an extraction prompt to judge if the user's message contains a durable fact. If yes, stores it in the semantic ChromaDB collection with dedup.
- **Persistent reminders**: SQLite-based reminder store with natural language time parsing via `dateparser`. Background scheduler thread checks every 30s and fires due reminders via TTS + desktop notification.
- **Conversation history fix**: `jarvis.py` now stores labeled `"User: ..."` / `"Assistant: ..."` pairs instead of bare user text, so follow-up references like "the three matches" can be resolved from the assistant's last reply.
- **Thread safety**: Added `threading.Lock` to `LLMClient._get_client()` for safe concurrent use from extraction thread.
- **CUDA DLL fix**: `stream_stt.py` switched from PATH-based to `os.add_dll_directory()` for CUDA DLL registration.

---

### Phase D — ReAct Agent Loop ✅ (Complete)

#### What was done
- Replaced planner→executor→summarizer chain with autonomous ReAct agent loop
- New files: `agent/loop.py` (run_agent, think, _extract_json, _stream_summarize), `agent/prompts.py` (SYSTEM_PROMPT, USER_PROMPT_FORMAT, TOOL_DESCRIPTIONS)
- Deleted files: `modes/ultra.py`, `agent/planner.py`, `agent/executor.py`, `agent/summarizer.py`
- Ultra mode abilities absorbed as tools: `read_screen()`, `generate_content()`, `paste_at_cursor()`
- `config.py`: added `MAX_STEPS = 6`, removed `ULTRA_HOTKEY`
- `main.py`: removed UltraMode, two modes (Jarvis + WhisperFlow)
- All 20 existing tools preserved + 3 new tools = 22 total tools
- `execute_tool(name, args, llm=None)` — now accepts optional `llm` parameter for generate_content
- `llm/client.py`: added `stream_chat()` for best-effort fallback streaming
- Agent loop features:
  - JSON parse failure returns safe fallback message (logs raw text)
  - Tool error visible in steps context for agent self-correction
  - MAX_STEPS hard cap prevents infinite loops
  - Delivery modes: speak, paste, both
  - Provider fallback chain: primary → fallback based on API key presence
- Bug fixes applied:
  - Mic resume after "start fresh" in jarvis.py
  - Print + history save in fallback except block in jarvis.py
  - `llm=None` guard in generate_content (registry.py)
  - Syntax error fix in loop.py (trailing `,jj`)

#### ReAct Architecture

```
User Input
    │
    ▼
┌──────────────────┐
│  run_agent()     │  MAX_STEPS=6 hard cap
│  ┌────────────┐  │
│  │  think()   │──│── LLM call (temp=0.1) → JSON decision
│  └─────┬──────┘  │     {"tool": "...", "args": {...}}
│        │         │     or {"done": True, "answer": "...", "delivery": "speak"}
│  ┌─────▼──────┐  │
│  │exec_tool() │  │  tools.registry.execute_tool(name, args, llm)
│  └─────┬──────┘  │
│        │         │
│  ┌─────▼──────┐  │
│  │ observe    │  │  Step result appended → fed back to next think()
│  └────────────┘  │
└──────────────────┘
        │
        ▼
   Streaming TTS → spoken response
```

---

### Phase E — Retrieval Gate + JSONL Tracing ✅ (In Progress)

#### What was done
- **Retrieval Gate** (`memory/retrieval_gate.py`): Cheap fast LLM classifier (temp=0, max_tokens=5) that decides if a turn needs memory before querying ChromaDB. Saves latency when the turn is a simple query that doesn't need user context.
- **Always-On JSONL Tracing** (`ops/tracer.py`): Fire-and-forget tracer that appends one JSON line per event to `.traces/<YYYY-MM-DD>.jsonl`. Events: `turn_start`, `gate`, `think`, `tool`, `final`, `turn_end`, `error`. Zero-setup, never crashes.
- `config.py`: added `RETRIEVAL_GATE = True` and `TRACING = True` feature flags.
- Both gated by feature flags — set to `False` to restore previous behavior.

### Phase F — Packaging (Future)

#### PyInstaller .exe Build
- Only when agent features are stable
- Use `--onedir` mode (not `--onefile`)
- Bundle models or download on first run
- Estimated size: 500MB-1.5GB

---

## Dependencies

```
faster-whisper
sounddevice
numpy
scipy
keyboard
kokoro-onnx
pyautogui                    # Ctrl+A/Ctrl+V paste at cursor (Phase A)
pyperclip                    # Clipboard copy for paste (Phase A)
ddgs                         # Web search tool (Phase B)
geoip2                      # GeoLite2-City.mmdb reader (Phase B)
psutil                       # System info tool
feedparser                   # News RSS tool
winocr                       # Windows native OCR for screen reading tool
google-genai                 # Gemini SDK v2.9.0 (Phase B2)
openai                       # OpenAI SDK (Phase B2)
anthropic                    # Anthropic SDK (Phase B2)
nvidia-cublas-cu12           # CUDA 12.x cuBLAS for faster-whisper (Phase B2)
nvidia-cuda-runtime-cu12     # CUDA 12.x runtime for faster-whisper (Phase B2)
nvidia-cudnn-cu12            # cuDNN 9 for faster-whisper (Phase B2)
chromadb                     # Vector memory store (Phase C)
requests                     # HTTP for fetch_url tool
beautifulsoup4               # HTML parsing for fetch_url tool
dateparser                   # Natural language time parsing for reminders (Phase C2)
plyer                        # Desktop notifications for due reminders (Phase C2)
```

## Setup Steps

```powershell
# 1. Create virtual environment
python -m venv jarvis-env
.\jarvis-env\Scripts\activate

# 2. Install dependencies
#    CUDA 12.x DLLs (cublas, cudnn, cuda_runtime) bundled via pip — ~1.2 GB
pip install -r requirements.txt

# 3. Set your API key in .env (never commit keys)
#    Copy .env.example or create .env with:
#    PROVIDER=gemini
#    GEMINI_API_KEY=your-key-here

# 4. Make sure model files are in D:\Jarvis\
#    - kokoro-v1.0.int8.onnx
#    - voices-v1.0.bin
#    - data/GeoLite2-City.mmdb

# 5. Run terminal as Administrator

# 6. Start Jarvis
python main.py
```

## Known Roadblocks & Mitigations

| Roadblock | Mitigation |
|---|---|
| Pipeline latency (5-8s) | TTS speed=1.15, cloud LLM is fast (~1-2s per call), STT beam_size=1 with VAD filter |
| 4GB VRAM only for TTS | STT on GPU (CUDA float16), LLM is cloud — no GPU needed. Only TTS + Kokoro on CPU |
| Kokoro int8 still slow (~2s) | Use speed=1.15. Pure STT mode has no TTS |
| DirectML incompatible with Kokoro | Stay on CPU for TTS. No GPU acceleration available |
| keyboard needs admin | Run terminal as Administrator |
| Audio cuts out mid-response | `sd.wait()` blocks until playback complete |
| 8GB RAM pressure | Whisper `base` not `small`. Cloud LLM uses no RAM. Monitor with Task Manager |
| Cloud LLM API costs | Use low-cost models: gpt-4o-mini, gemini-2.5-flash-lite, claude-3-5-haiku-latest. Each call ~$0.001-0.003 |
| API key security | Keys stored in .env, loaded via python-dotenv. config.py reads from env vars only — no hardcoded secrets |
| WhisperFlow streaming blocked during hotkey hold | `keyboard` library's `suppress=True` creates `WH_KEYBOARD_LL` hook that intercepts injected input. Tried `SendInput`+`KEYEVENTF_UNICODE` and `PostMessage`+`WM_CHAR` — both fail during hook hold. Final-paste-only is the working approach. |
| CUDA DLLs not found by ctranslate2 | `_add_cuda_dll_dirs()` registers paths via `os.add_dll_directory()` in `transcriber.py`. CUDA 12.x packages installed via pip (~1.2 GB in venv) |
| Gemini free tier quota (20 req/day) | Quota resets ~24h. Use OpenAI or Anthropic for heavy testing, or upgrade to paid tier |
| Gemini requires user-role message | `call_raw()` always sends system + user message pair for Gemini compatibility |
| GeoIP db file missing | Path is `data/GeoLite2-City.mmdb`, ~63MB. Download from MaxMind (free reg) |
| ChromaDB ONNX model download (79MB) | Auto-downloaded on first `MemoryStore()` init to `~/.cache/chroma/`. Required for DefaultEmbeddingFunction. |
| PyInstaller + faster-whisper DLLs | Deferred. Models downloaded on first run, not bundled |
| Retrieval gate adds ~1 LLM call per turn | Saves ChromaDB queries when gate says no — net neutral latency, better accuracy |
| Tracing JSONL files grow unbounded | Each line ~200 bytes. At 100 turns/day, ~20KB/day. Manual cleanup if needed |

## Conventions
- Imports: standard lib first, then third-party, then local
- Error handling: use try/except in LLM client; check empty audio/transcript; tool errors caught and returned as strings
- Print status with emoji indicators throughout pipeline
- No comments in code unless asked
- Type hints on all function signatures
