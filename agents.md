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

## Current Implementation (Phase B2 — BYOLLM + Cloud LLMs)

### Architecture
```
[CTRL+SHIFT+J held]
    -> sounddevice records audio
    -> key released -> stop recording, capture audio synchronously
    -> Thread starts with captured audio (no race on recorder state)
    -> faster-whisper transcribes (CUDA, beam_size=1, vad_filter)
    -> PLANNER (cloud LLM call): classifies intent, outputs JSON plan
       - "chat" intent: short greeting → TTS
       - "tool_request" intent: structured plan with tool steps
    -> EXECUTOR: runs each tool step, collects results
    -> SUMMARIZER (cloud LLM call): formats results into clean spoken response
       - Strips metadata, tool names, internal details
       - 1-2 natural sentences
    -> Kokoro TTS converts response (int8, speed=1.0)
    -> sounddevice plays audio
```

### Three-Stage Agent Pipeline

```
User Input
    │
    ▼
┌──────────────┐
│   PLANNER    │  Cloud LLM call (temp=0.1) — outputs JSON plan
│  (router.py) │  e.g. {"intent": "tool_request", "plan": [...]}
└──────┬───────┘
       │
       ├── "chat" intent → direct reply → TTS
       │
       └── "tool_request" intent
                │
                ▼
          ┌──────────────┐
          │  EXECUTOR    │  Runs tools sequentially
          │ (executor.py)│  Calls tools.registry.execute_tool()
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │  SUMMARIZER  │  Cloud LLM call (temp=0.1) — formats for speech
          │(summarizer.py)│  Strips noise, 1-2 clean sentences
          └──────┬───────┘
                 │
                 ▼
                TTS → spoken response
```

### File Reference

#### `config.py`
- `HOTKEY = "ctrl+shift+j"` — Jarvis agent hotkey
- `WHISPERFLOW_HOTKEY = "ctrl+shift+k"` — Pure STT mode hotkey
- `ULTRA_HOTKEY = "ctrl+shift+u"` — Ultra mode hotkey
- `STT_MODEL = "base"`
- `STT_DEVICE = "cuda"` — GPU accelerated Whisper (no VRAM conflict)
- `SAMPLE_RATE = 16000`
- `CHANNELS = 1`
- `PROVIDER = "gemini"` — LLM provider: "openai" | "gemini" | "anthropic"
- `OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY` — API keys for each provider (loaded from .env via `os.getenv`)
- `OPENAI_MODEL / GEMINI_MODEL / ANTHROPIC_MODEL` — per-provider model selection

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
- `ULTRA_SYSTEM_PROMPT` — content generation prompt for Ultra mode

#### `llm/client.py`
- Class: `LLMClient`
- **BYOLLM architecture** — supports OpenAI, Gemini, Anthropic via unified internal interface
- `__init__()` — initializes session, history with system prompt, client cache. Loads user profile and injects into system prompt.
- `_get_client(provider)` — lazy-loads the right SDK client (OpenAI / gemini / Anthropic)
- `_convert_messages(messages, provider)` — translates internal format to each provider's message schema:
  - **OpenAI**: pass-through with tool_call_id
  - **Gemini**: `parts` array for user/model, `functionCall`/`functionResponse` for tools
  - **Anthropic**: content blocks (`text`, `tool_use`, `tool_result`)
- `_convert_tools(tools, provider)` — translates OpenAI-format tool defs to:
  - **Gemini**: `FunctionDeclaration` wrapped in `function_declarations`
  - **Anthropic**: `input_schema` (renamed from `parameters`)
- `_normalize_response(response, provider)` — converts each provider's response back to unified `{"message": {"content", "tool_calls": [...]}}` format
- `_chat(tools=None)` — internal helper, dispatches to active provider. Uses `max_tokens=300`.
- `call_raw(messages, tools=None, temp=0.3, max_tokens=1000)` — stateless LLM call. `max_tokens` is caller-configurable (planner/summarizer use default 1000, Ultra passes 2000).
- `chat(user_input)` — simple conversational chat. Trims history to last 20 exchanges to prevent unbounded growth.
- `chat_with_tools(user_input, tools)` — tool-calling loop (up to 5 rounds). Now stores assistant `tool_calls` in history for OpenAI/Anthropic protocol compliance.
- `refresh_memories(query)` — queries ChromaDB, injects relevant memories into system prompt
- `rotate_session()` — summarizes current session, stores as episode, resets history
- On model failure: returns error string, pops user message from history

#### `tts/speaker.py`
- Class: `Speaker`
- `__init__()` — loads `Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")`, voice `"af_bella"`
- `speak(text: str)` — creates audio at speed=1.0, `sd.play()`, `sd.wait()`

#### `tools/registry.py`
- `TOOL_DEFINITIONS` — list of OpenAI-compatible function schemas
- `TOOL_MAP` — dict mapping tool name string → callable function
- `execute_tool(name, args) -> str` — dispatches to the registered function, returns result string or error

#### `tools/geoip.py`
- `get_city_info(ip="auto")` — if "auto", resolves current public IP via `api.ipify.org`, then looks up city/region/country from `data/GeoLite2-City.mmdb` via `geoip2`
- Returns string like `"Syracuse, New York, United States"`

#### `tools/weather.py`
- `get_weather(city)` — queries `wttr.in` (no API key needed) with format: condition, temperature, wind, humidity. If `city="auto"`, resolves location via GeoIP first.
- Returns string like `"London: Partly cloudy, +20°C, ↘22km/h, 46% humidity"`

#### `tools/web_search.py`
- `search_web(query, num_results=3)` — uses `ddgs.DDGS`, returns formatted list of title/body/URL
- Returns string with newline-separated entries, each truncated at 400 chars

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

#### `agent/planner.py`
- `create_plan(user_input, llm) -> dict` — intent router + planner
- Uses a dedicated system prompt to classify user intent as `"chat"` or `"tool_request"`
- Outputs a JSON plan: `{"intent": "...", "plan": [...], "message": "..."}`
- For `"chat"`: returns a brief friendly message directly (no tool calls)
- For `"tool_request"`: returns a structured plan array of tool steps
- For `"new_session"`: signals session rotation (start fresh / clear context)
- Strips markdown code blocks from LLM output, parses JSON with fallback
- Temperature=0.1 for deterministic JSON output
- Injects user profile from `profile/profile.md` into prompt for personalization
- Injects top-5 semantic + top-3 episodic memories from ChromaDB into prompt

#### `agent/executor.py`
- `execute_plan(plan) -> list[dict]` — runs each tool step in sequence
- Calls `tools.registry.execute_tool()` for each step
- Returns list of `{"tool": name, "args": args, "result": str}` dicts
- Prints progress: "[tool] → [result]" for each call

#### `agent/summarizer.py`
- `summarize(user_input, tool_results, llm) -> str` — formats tool results for TTS
- Uses a dedicated system prompt to produce clean, natural speech
- Rules: 1-2 sentences, exact numbers from data, no metadata/tool names, natural conversational tone
- Temperature=0.1 to prevent hallucination (especially temperature/numbers)
- Falls back to error message on failure
- Injects user profile from `profile/profile.md` into system message for context
- Injects top-3 semantic + top-2 episodic memories from ChromaDB into system message

#### `modes/jarvis.py`
- Class: `JarvisMode`
- `__init__(recorder, transcriber, llm, speaker)` — stores shared instances
- `on_activate()` — starts recording
- `on_release()` — stops recording, captures audio synchronously, passes to thread:
  1. `agent.planner.create_plan(text, llm)` → plan dict
  2. If intent is "chat": use plan's message directly, or `llm.refresh_memories()` + `llm.chat()`
  3. If intent is "tool_request": `agent.executor.execute_plan(steps)` → results
  4. `agent.summarizer.summarize(text, results, llm)` → clean speech
  5. If intent is "new_session": `llm.rotate_session()` → summarize + store + reset history
  6. Falls back to `llm.chat()` if plan has no steps

#### `modes/ultra.py`
- Class: `UltraMode`
- `__init__(recorder, transcriber, llm)` — stores shared instances
- `on_activate()` — starts recording
- `on_release()` — stops recording, runs the full pipeline:
  1. Screenshot + OCR via `tools.screen_ocr.capture_and_ocr()`
  2. STT via `transcriber.transcribe()`
  3. Profile context loaded via `tools.profile_loader.load_profile()`
  4. Optional company search via `tools.web_search.search_web()`
  5. LLM generation using `ULTRA_SYSTEM_PROMPT` with screen context + profile + user instruction
  6. Copy to clipboard + paste at cursor via `pyperclip` + `pyautogui`
- Press CTRL+SHIFT+U, speak your request, release → generated content appears at cursor
- Works for emails, cover letters, code, summaries, documents — any writing task

#### `tools/screen_ocr.py`
- `capture_and_ocr()` — takes full screenshot via `pyautogui`, OCRs via `winocr` (Windows native)
- Returns extracted text string or `"[No text detected on screen]"` / `"[OCR error: ...]"`

#### `tools/profile_loader.py`
- `load_profile()` — reads `profile/profile.md` from project root
- Returns the file content as a string, or empty string if file doesn't exist
- Used by LLMClient (chat system prompt), Planner, Summarizer, and Ultra mode to inject personal context

#### `memory/store.py`
- `MemoryStore` class — ChromaDB wrapper with two collections (`semantic`, `episodic`)
- `add(collection, content, metadata)` — stores content with timestamp-based ID
- `query(collection, text, n)` — semantic search, returns content strings
- `count(collection)` — returns document count (handles empty collection gracefully)
- Module-level singleton `memory_store` for use by tools and planner

#### `memory/session.py`
- `summarize(llm, history)` — summarizes last 10 exchanges via LLM call
- `save_session(llm, history)` — calls `summarize()` + stores result in episodic collection

#### `modes/whisperflow.py`
- Class: `WhisperFlowMode`
- `__init__(recorder, transcriber)` — stores shared instances
- `on_activate()` — starts recording (no streaming, no terminal output)
- `on_release()` — stops recording, transcribes full audio, copies to clipboard, `ctrl+a` + `ctrl+v` at cursor
- No LLM, no TTS — press CTRL+SHIFT+K, speak, release → text appears
- Streaming was attempted (SendInput KEYEVENTF_UNICODE, PostMessage WM_CHAR) but all methods fail during hotkey hold due to `WH_KEYBOARD_LL` hook interception. Final-paste-only is the current approach.

#### `main.py`
- Global instances: `Recorder`, `Transcriber`, `LLMClient`, `Speaker`
- Three mode instances: `JarvisMode`, `WhisperFlowMode`, `UltraMode` — all with shared `Recorder`, `Transcriber`, `LLMClient`
- Six hotkey registrations (press + release for each of three modes), all with `suppress=True`
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
| **Jarvis** | `CTRL+SHIFT+J` | Hotkey → Record → STT → Planner → Executor → Summarizer → TTS | Full assistant with tools |
| **WhisperFlow** (Type mode) | `CTRL+SHIFT+K` | Hotkey → Record → STT → Paste at cursor | Dictate text anywhere |
| **Ultra** (Generate mode) | `CTRL+SHIFT+U` | Hotkey → Screenshot → Record → STT → OCR → Profile → LLM Gen → Paste | Generate emails, cover letters, code at cursor |

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

#### Agent Architecture (Plan → Execute → Summarize)
```
User speaks → STT → PLANNER (LLM → JSON plan)
                        │
              ┌─────────┴──────────┐
              │                    │
         "chat" intent     "tool_request" intent
              │                    │
              ▼                    ▼
        Direct reply          EXECUTOR
        (friendly msg)     (run tools seq)
                                  │
                                  ▼
                            SUMMARIZER
                            (LLM → clean speech)
                                  │
                                  ▼
                                 TTS
```

#### Pipeline Details

**Jarvis (QA pipeline)**
```
Planning:   LLM call with planner system prompt → {"intent": "...", "plan": [...]}
            - "chat" intent: no tools needed, direct response
            - "tool_request": structured plan with tool names + args

Execution:  for each step in plan → tools.registry.execute_tool(name, args)
            Results collected as list of {tool, args, result}

Summarizer: LLM call with summarizer system prompt
            Input: user question + tool results
            Output: 1-2 clean sentences (no metadata, no tool names)
```

**Ultra (Generation pipeline)**
```
Screen capture: pyautogui.screenshot() → winocr.recognize_pil_sync()
Profile load:   tools.profile_loader.load_profile() → reads profile/profile.md
Search:         If company detected in instruction → search_web(company)
Generation:     LLM call with ULTRA_SYSTEM_PROMPT + screen text + profile + user instruction
Output:         Copy to clipboard → ctrl+v at cursor
```

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
    ├── call_raw()              → stateless call (planner, summarizer, ultra)
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
  - **Planner prompt** — personalized tool selection and routing
  - **Summarizer system message** — context-aware response phrasing
  - Ultra mode already had profile injection (unchanged)
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

- **semantic_memory** — User facts and preferences, injected top-5 into planner + top-3 into summarizer
- **episodic_memory** — Session summaries (on "start fresh" or shutdown), injected top-3 into planner + top-2 into summarizer

**Embedding:** ChromaDB `DefaultEmbeddingFunction` (all-MiniLM-L6-v2 via ONNX, 384-dim, bundled with chromadb, no external service)

#### Storage Triggers

1. **Explicit ("remember that X")** — Planner detects intent → calls `store_memory` tool → saves to semantic collection
2. **"start fresh" / "new session"** — Planner outputs `new_session` intent → `LLMClient.rotate_session()` summarizes history → stores as episode → resets `self.history`
3. **Shutdown** — `atexit` handler in `main.py` saves current session as episode (best-effort)

#### Memory Injection Points

| Call Point | Where | What's Injected |
|---|---|---|
| Planner (`create_plan`) | `{MEMORIES}` in `PLANNER_PROMPT` | Top-5 semantic + top-3 episodic |
| Chat (`llm.chat()`) | `self.history[0]` via `refresh_memories()` | Top-5 semantic + top-3 episodic |
| chat_with_tools | `self.history[0]` via `refresh_memories()` | Top-5 semantic + top-3 episodic |
| Summarizer | System message | Top-3 semantic + top-2 episodic |
| Ultra mode | No injection (generative, not retrieval-based) | — |

#### New Tool: `store_memory`

- **Name:** `store_memory`
- **Params:** `content` (string) — the fact to remember
- **Action:** Adds to `memory_store.semantic` collection with `type: "fact"` metadata
- **Returns:** `"Remembered: {content}"`

#### New Intent: `new_session`

- Planner outputs `{"intent": "new_session", "plan": [], "message": ""}`
- JarvisMode calls `llm.rotate_session()` which:
  1. Summarizes current history via an LLM call
  2. Stores summary in episodic collection
  3. Resets `self.history` to fresh system prompt
- Supported commands: "start fresh", "new session", "clear context", "forget everything"

#### New Methods on LLMClient

- `refresh_memories(query)` — Queries both collections, updates `self.history[0]` with relevant memories
- `rotate_session()` — Summarizes + stores current session as episode, resets history

#### Dependencies

```
chromadb                           # Already installed, added to requirements.txt
```

No Ollama, no sentence-transformers, no external embedding service.

---

### Phase D — Packaging (Future)

### Phase D — Packaging (Future)

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
winocr                       # Windows native OCR for Ultra mode
google-genai                 # Gemini SDK v2.9.0 (Phase B2)
openai                       # OpenAI SDK (Phase B2)
anthropic                    # Anthropic SDK (Phase B2)
nvidia-cublas-cu12           # CUDA 12.x cuBLAS for faster-whisper (Phase B2)
nvidia-cuda-runtime-cu12     # CUDA 12.x runtime for faster-whisper (Phase B2)
nvidia-cudnn-cu12            # cuDNN 9 for faster-whisper (Phase B2)
chromadb                     # Vector memory store (Phase C)
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
| GeoIP db file missing | Path is `data/GeoLite2-City.mmdb`, ~63MB. Download from MaxMind (free reg) |
| ChromaDB ONNX model download (79MB) | Auto-downloaded on first `MemoryStore()` init to `~/.cache/chroma/`. Required for DefaultEmbeddingFunction. |
| PyInstaller + faster-whisper DLLs | Deferred. Models downloaded on first run, not bundled |

## Conventions
- Imports: standard lib first, then third-party, then local
- Error handling: use try/except in LLM client; check empty audio/transcript; tool errors caught and returned as strings
- Print status with emoji indicators throughout pipeline
- No comments in code unless asked
- Type hints on all function signatures
