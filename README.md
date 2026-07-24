# Jarvis — Personal AI Assistant

Voice assistant with two modes: full agent (CTRL+SHIFT+J) and dictation (CTRL+SHIFT+K).
Uses cloud LLMs (BYOLLM) — no local GPU needed for inference.

## Project Structure

```
D:\Jarvis\
├── main.py                 # Entry point — registers hotkeys, reminder scheduler
├── config.py               # Hotkeys, STT settings, LLM provider config, feature flags
├── modes/
│   ├── jarvis.py           # Agent mode: orchestrates ReAct loop + async extraction
│   └── whisperflow.py      # Dictation mode: record → STT → paste at cursor
├── agent/
│   ├── __init__.py
│   ├── loop.py             # ReAct agent loop (think → execute → observe)
│   └── prompts.py          # System prompts for ReAct loop and reflection
├── audio/
├── stt/
│   └── stream_stt.py       # RealtimeSTT streaming session recorder
├── llm/
│   ├── client.py           # BYOLLM client — supports OpenAI / Gemini / Anthropic
│   └── prompts.py          # System prompts (chat, ultra)
├── tts/
│   ├── speaker.py          # Kokoro ONNX TTS (int8, speed=1.15)
│   └── stream_tts.py       # Streaming TTS token-by-token
├── tools/
│   ├── __init__.py
│   ├── registry.py         # Tool map + dispatch (22 tools)
│   ├── geoip.py            # GeoLite2-City.mmdb lookup
│   ├── weather.py          # wttr.in weather query (auto-location support)
│   ├── web_search.py       # DuckDuckGo search via ddgs
│   ├── calculator.py       # Safe math expression evaluator
│   ├── datetime_tool.py    # Current date/time
│   ├── app_launcher.py     # Launch Windows apps
│   ├── notes.py            # Take/read/update/delete notes
│   ├── system_info.py      # CPU/RAM/disk usage
│   ├── browser.py          # Open URLs or bookmarks
│   ├── clipboard_tool.py   # Read clipboard contents
│   ├── news.py             # RSS news headlines
│   ├── media.py            # Media playback controls
│   ├── screen_ocr.py       # Screen capture + OCR
│   └── profile_loader.py   # Load user profile
├── memory/
│   ├── __init__.py
│   ├── store.py            # ChromaDB wrapper (semantic + episodic collections)
│   ├── session.py          # Session summarizer + save helper
│   ├── extractor.py        # Async LLM-judge fact extraction
│   ├── reminders.py        # SQLite reminder store
│   └── retrieval_gate.py   # Cheap gate before memory retrieval
├── ops/
│   ├── __init__.py
│   └── tracer.py           # Always-on JSONL tracing
├── profile/
│   └── profile.md          # Your personal context (gitignored)
├── data/
│   └── GeoLite2-City.mmdb  # MaxMind GeoIP database (gitignored)
├── memory_db/              # ChromaDB vector database (auto-created, gitignored)
├── reminders.db            # SQLite reminder database (auto-created, gitignored)
├── .traces/                # Daily JSONL trace files (auto-created, gitignored)
├── requirements.txt
├── agents.md               # Full development plan & reference
└── README.md
```

## Setup

```powershell
# 1. Create & activate virtual environment
python -m venv jarvis-env
.\jarvis-env\Scripts\activate

# 2. Install dependencies (CUDA 12.x DLLs bundled via pip — ~1.2 GB)
pip install -r requirements.txt

# 3. Set your API key in .env (not config.py)
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

## Usage

| Mode | Hotkey | Pipeline | What it does |
|---|---|---|---|
| **Jarvis** (Agent) | Press CTRL+SHIFT+J | ReAct loop → tools → TTS/paste | Full assistant with 22 tools |
| **WhisperFlow** (Dictation) | Press CTRL+SHIFT+K | Record → STT → Paste at cursor | Transcribe speech to text anywhere |

### Jarvis Agent Pipeline
1. **VAD** — voice activity detection auto-submits turn after 1.5s silence
2. **Gate** — cheap LLM call decides if memory retrieval is needed
3. **ReAct Loop** — up to 6 iterations of think (LLM) → execute tool → observe result
4. **Delivery** — speak via TTS, paste at cursor, or both
5. **Extraction** — background async fact extraction after each turn

### Available Tools (22 tools)
- **search_web(query, num_results)** — Search the web via DuckDuckGo
- **fetch_url(url)** — Fetch full page text from a URL
- **get_city_info(ip)** — GeoIP lookup from IP address
- **get_weather(city, days)** — Current weather or forecast
- **get_datetime()** — Current date and time
- **calculate(expression)** — Safe math evaluation
- **open_app(app_name)** — Launch desktop apps
- **take_note(note)** / **read_notes(last_n)** / **update_note(index, content)** / **delete_note(index)** — Notes management
- **store_memory(content)** — Remember a fact or preference
- **get_system_info()** — CPU, RAM, disk usage
- **open_url(url)** — Open URL or named bookmark
- **read_clipboard()** — Read clipboard contents
- **get_news(topic)** — Top headlines (general, tech, science, us)
- **media_control(action)** — Play/pause/next/volume
- **set_reminder(message, when)** — Set a timed reminder with natural language
- **list_reminders()** — Show all pending reminders
- **read_screen()** — OCR text from screen
- **generate_content(instruction, screen_context)** — Draft emails, code, letters
- **paste_at_cursor(text)** — Paste generated content at cursor

## Key Rules
- STT runs on CUDA (GPU), LLM is cloud — no local GPU conflict
- Terminal must run as Administrator
- Python 3.12
- No comments in code
- Type hints on all function signatures
