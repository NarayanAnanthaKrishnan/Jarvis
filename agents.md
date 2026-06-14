# Jarvis — AI Agent Reference

## Project Overview
- **What:** Personal AI assistant — press hotkey, speak, get spoken response
- **Hardware:** GTX 1650 (4GB VRAM), 8GB RAM, Windows
- **Primary LLM:** phi4-mini via Ollama (GPU)
- **Fallback:** None (gemma4:e2b too large for 4GB VRAM, was causing 10-30s dead timeout)
- **STT:** faster-whisper (base, CPU only — never compete for VRAM)
- **TTS:** Kokoro ONNX int8 (CPU, speed=1.15)

## Dev Rules
- Whisper/STT runs on CPU, LLM runs on GPU — never both on GPU
- Terminal must run as Administrator (keyboard lib requirement)
- Python 3.12
- Project lives in `D:\Jarvis\` directly (no subfolder)

---

## Current Implementation (Phase B — Agent + Tool Calling)

### Architecture
```
[CTRL+SHIFT+J held]
    -> sounddevice records audio
    -> key released -> stop recording
    -> faster-whisper transcribes (CPU, beam_size=1, vad_filter)
    -> PLANNER (LLM call): classifies intent, outputs JSON plan
       - "chat" intent: short greeting → TTS
       - "tool_request" intent: structured plan with tool steps
    -> EXECUTOR: runs each tool step, collects results
    -> SUMMARIZER (LLM call): formats results into clean spoken response
       - Strips metadata, tool names, internal details
       - 1-2 natural sentences
    -> Kokoro TTS converts response (int8, speed=1.5)
    -> sounddevice plays audio
```

### Three-Stage Agent Pipeline

```
User Input
    │
    ▼
┌──────────────┐
│   PLANNER    │  LLM call (temp=0.1) — outputs JSON plan
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
          │  SUMMARIZER  │  LLM call (temp=0.1) — formats for speech
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
- `STT_MODEL = "base"`
- `STT_DEVICE = "cpu"`
- `LLM_PRIMARY = "phi4-mini"`
- `SAMPLE_RATE = 16000`
- `CHANNELS = 1`

#### `audio/recorder.py`
- Class: `Recorder`
- `__init__(sample_rate=16000)` — stores sample rate, init empty lists
- `start()` — guards against re-entry via `if self.is_recording: return`, then creates `sd.InputStream`, starts callback-based recording
- `_callback(indata, frames, time, status)` — appends `indata.copy()` if recording
- `stop() -> np.ndarray` — stops stream, closes it, concatenates chunks, flattens
- `get_partial(since: int) -> tuple[np.ndarray, int]` — thread-safe read of audio chunks from an index checkpoint. Returns audio and the new end index. Used by WhisperFlow streaming.

#### `audio/player.py`
- Class: `Player`
- `play(samples, sample_rate)` — `sd.play()`, `sd.wait()`

#### `stt/transcriber.py`
- Class: `Transcriber`
- `__init__(model_size="base", device="cpu")` — loads `WhisperModel` with `compute_type="int8"`
- `transcribe(audio: np.ndarray, sample_rate=16000) -> str` — ignores < 0.5s, transcribes with `beam_size=1, vad_filter=True` for speed

#### `llm/prompts.py`
- `SYSTEM_PROMPT` — strict English, 1-2 sentences, no formatting, no "how can I help"

#### `llm/client.py`
- Class: `LLMClient`
- `__init__()` — initializes session, history with system prompt
- `_chat(model, tools=None)` — internal helper that tries a single model call. Options: `num_predict=100`, `num_ctx=1024`, `temperature=0.3`
- `call_raw(messages, tools=None, temp=0.3)` — stateless LLM call with custom messages (used by planner and summarizer). Does NOT touch `self.history`.
- `chat(user_input)` — simple chat (no tools), for backward compat
- `chat_with_tools(user_input, tools)` — tool-calling loop: sends message with tool defs, executes tool calls via `tools.registry.execute_tool()`, loops up to 5 rounds until LLM gives a direct answer
- On model failure: returns error string, pops user message from history

#### `tts/speaker.py`
- Class: `Speaker`
- `__init__()` — loads `Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")`, voice `"af_bella"`
- `speak(text: str)` — creates audio at speed=1.15, `sd.play()`, `sd.wait()`

#### `tools/registry.py`
- `TOOL_DEFINITIONS` — list of OpenAI-compatible function schemas (Ollama-native format)
- `TOOL_MAP` — dict mapping tool name string → callable function
- `execute_tool(name, args) -> str` — dispatches to the registered function, returns result string or error

#### `tools/geoip.py`
- `get_city_info(ip="auto")` — if "auto", resolves current public IP via `api.ipify.org`, then looks up city/region/country from `data/GeoLite2-City.mmdb` via `geoip2`
- Returns string like `"Syracuse, New York, United States"`

#### `tools/weather.py`
- `get_weather(city)` — queries `wttr.in` (no API key needed) with format: condition, temperature, wind, humidity. If `city="auto"`, resolves location via GeoIP first.
- Returns string like `"London: Partly cloudy, +20°C, ↘22km/h, 46% humidity"`

#### `tools/web_search.py`
- `search_web(query, num_results=5)` — uses `ddgs.DDGS`, returns formatted list of title/body/URL
- Returns string with newline-separated entries

#### `agent/planner.py`
- `create_plan(user_input, llm) -> dict` — intent router + planner
- Uses a dedicated system prompt to classify user intent as `"chat"` or `"tool_request"`
- Outputs a JSON plan: `{"intent": "...", "plan": [...], "message": "..."}`
- For `"chat"`: returns a brief friendly message directly (no tool calls)
- For `"tool_request"`: returns a structured plan array of tool steps
- Strips markdown code blocks from LLM output, parses JSON with fallback
- Temperature=0.1 for deterministic JSON output

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

#### `modes/jarvis.py`
- Class: `JarvisMode`
- `__init__(recorder, transcriber, llm, speaker)` — stores shared instances
- `on_activate()` — starts recording
- `on_release()` — stops recording, transcribes, runs the full agent pipeline:
  1. `agent.planner.create_plan(text, llm)` → plan dict
  2. If intent is "chat": use plan's message directly
  3. If intent is "tool_request": `agent.executor.execute_plan(steps)` → results
  4. `agent.summarizer.summarize(text, results, llm)` → clean speech
  5. Falls back to `llm.chat()` if plan has no steps

#### `modes/whisperflow.py`
- Class: `WhisperFlowMode`
- `__init__(recorder, transcriber)` — stores shared instances
- `on_activate()` — starts recording (no streaming, no terminal output)
- `on_release()` — stops recording, transcribes full audio, copies to clipboard, `ctrl+a` + `ctrl+v` at cursor
- No LLM, no TTS — press CTRL+SHIFT+K, speak, release → text appears
- Streaming was attempted (SendInput KEYEVENTF_UNICODE, PostMessage WM_CHAR) but all methods fail during hotkey hold due to `WH_KEYBOARD_LL` hook interception. Final-paste-only is the current approach.

#### `main.py`
- Global instances: `Recorder`, `Transcriber`, `LLMClient`, `Speaker`
- Two `JarvisMode` and `WhisperFlowMode` instances with shared dependencies
- Four hotkey registrations (press + release for each mode), all with `suppress=True`
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

#### WhisperFlow Mode
| Mode | Hotkey | Pipeline | Purpose |
|---|---|---|---|
| **WhisperFlow** (Type mode) | `CTRL+SHIFT+K` | Hotkey → Record → STT → Paste at cursor | Dictate text anywhere |
| **Jarvis** (Agent mode) | `CTRL+SHIFT+J` | Hotkey → Record → STT → LLM+Tools → TTS | Full assistant |

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

---

### Phase C — Memory (Future)

#### Two-Tier Memory (not yet implemented)
1. **Short-term**: Conversation buffer in `self.history` — already done
2. **Long-term**: ChromaDB with persistent disk storage

#### ChromaDB Integration Pattern
```python
import chromadb
client = chromadb.PersistentClient(path="./memory_db")
collection = client.create_collection(
    name="long_term_memory",
    embedding_function=chromadb.DefaultEmbeddingFunction()
)
# Store: collection.add(ids=[...], documents=[...], metadatas=[...])
# Retrieve: collection.query(query_texts=[...], n_results=5)
```

#### Memory Injection
- Before each LLM call, retrieve top-K relevant memories
- Inject into system prompt as context
- Metadata types: preference, fact, episodic
- Importance scoring + forgetting mechanism for older/low-importance memories

#### Project Structure Addition
```
jarvis/
├── memory/
│   ├── __init__.py
│   ├── store.py          # ChromaDB wrapper
│   └── summarizer.py     # Conversation summarizer
├── jarvis-env/
└── memory_db/            # ChromaDB persistent storage (auto-created)
```

#### Dependencies to Add
```
chromadb
sentence-transformers
```

---

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
ollama
kokoro-onnx
pyautogui                    # Ctrl+A/Ctrl+V paste at cursor (Phase A)
pyperclip                    # Clipboard copy for paste (Phase A)
ddgs                         # Web search tool (Phase B)
geoip2                      # GeoLite2-City.mmdb reader (Phase B)
```

## Setup Steps

```powershell
# 1. Create virtual environment
python -m venv jarvis-env
.\jarvis-env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull LLM models in Ollama
ollama pull phi4-mini

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
| Pipeline latency (5-8s) | TTS speed=1.15, phi4-mini primary, STT beam_size=1 with VAD filter |
| 4GB VRAM tight | Whisper on CPU, LLM on GPU. phi4-mini fits (~2.5GB). Removed gemma4:e2b fallback |
| Kokoro int8 still slow (~2s) | Use speed=1.15. Pure STT mode has no TTS |
| DirectML incompatible with Kokoro | Stay on CPU for TTS. No GPU acceleration available |
| keyboard needs admin | Run terminal as Administrator |
| Audio cuts out mid-response | `sd.wait()` blocks until playback complete |
| 8GB RAM pressure | Whisper `base` not `small`. Monitor with Task Manager |
| Ollama tool calling not supported by some models | phi4-mini supports tools natively. Test qwen2.5:3b as alternative |
| WhisperFlow streaming blocked during hotkey hold | `keyboard` library's `suppress=True` creates `WH_KEYBOARD_LL` hook that intercepts injected input. Tried `SendInput`+`KEYEVENTF_UNICODE` and `PostMessage`+`WM_CHAR` — both fail during hook hold. Final-paste-only is the working approach. |
| GeoIP db file missing | Path is `data/GeoLite2-City.mmdb`, ~63MB. Download from MaxMind (free reg) |
| PyInstaller + faster-whisper DLLs | Deferred. Models downloaded on first run, not bundled |

## Conventions
- Imports: standard lib first, then third-party, then local
- Error handling: use try/except in LLM client; check empty audio/transcript; tool errors caught and returned as strings
- Print status with emoji indicators throughout pipeline
- No comments in code unless asked
- Type hints on all function signatures

## Ollama Model Info (for reference)

| Model | Size | Tool Calling | Notes |
|---|---|---|---|
| phi4-mini:latest | 2.5GB | Yes | Current primary. Works well with 4GB VRAM |
| gemma4:e2b | ~8GB | Yes | Too large for 4GB VRAM. Removed as fallback. Keep pulled in case of hardware upgrade |
| qwen2.5:3b | ~2GB | Yes | Pull and test — likely faster than phi4-mini |
| mistral:latest | 4.17GB | Yes | Pulled but not used. May fit if nothing else loaded |
| nomic-embed-text:latest | 262MB | No | Pulled for future ChromaDB (Phase C) |
