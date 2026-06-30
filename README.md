# Jarvis — Personal AI Assistant

Voice assistant with three modes: full agent (CTRL+SHIFT+J), dictation (CTRL+SHIFT+K), or content generation (CTRL+SHIFT+U).
Uses cloud LLMs (BYOLLM) — no local GPU needed for inference.

## Project Structure

```
D:\Jarvis\
├── main.py                 # Entry point — registers all three hotkeys
├── config.py               # Hotkeys, STT settings, LLM provider config
├── modes/
│   ├── jarvis.py           # Agent mode: orchestrates Plan→Execute→Summarize
│   ├── whisperflow.py      # Dictation mode: record → STT → paste at cursor
│   └── ultra.py            # Ultra mode: screenshot + OCR + profile → LLM gen → paste
├── main.py                 # Entry point — registers all three hotkeys + reminder scheduler
├── config.py               # Hotkeys, STT settings, LLM provider config, feature flags
├── modes/
│   ├── jarvis.py           # Agent mode: orchestrates Plan→Execute→Summarize + async extraction
│   ├── whisperflow.py      # Dictation mode: record → STT → paste at cursor
│   └── ultra.py            # Ultra mode: screenshot + OCR + profile → LLM gen → paste
├── agent/
│   ├── __init__.py
│   ├── planner.py          # Intent router + step planner (LLM)
│   ├── executor.py         # Tool call executor
│   └── summarizer.py       # Response formatter for TTS (LLM)
├── audio/
│   ├── recorder.py         # sounddevice hold-to-record
│   └── player.py           # sounddevice audio playback
├── stt/
│   ├── transcriber.py      # faster-whisper (CUDA float16 or CPU int8)
│   └── stream_stt.py       # RealtimeSTT streaming session recorder
├── llm/
│   ├── client.py           # BYOLLM client — supports OpenAI / Gemini / Anthropic
│   └── prompts.py          # System prompts
├── tts/
│   ├── speaker.py          # Kokoro ONNX TTS (int8, speed=1.15)
│   └── stream_tts.py       # Streaming TTS token-by-token
├── tools/
│   ├── __init__.py
│   ├── registry.py         # Tool definitions + dispatch (19 tools)
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
│   ├── screen_ocr.py       # Screen capture + OCR (Ultra mode)
│   └── profile_loader.py   # Load user profile
├── memory/
│   ├── __init__.py
│   ├── store.py            # ChromaDB wrapper (semantic + episodic collections)
│   ├── session.py          # Session summarizer + save helper
│   ├── extractor.py        # Async LLM-judge fact extraction (Phase C2)
│   └── reminders.py        # SQLite reminder store (Phase C2)
├── profile/
│   └── profile.md          # Your personal context (gitignored)
├── data/
│   └── GeoLite2-City.mmdb  # MaxMind GeoIP database (gitignored)
├── memory_db/              # ChromaDB vector database (auto-created, gitignored)
├── reminders.db            # SQLite reminder database (auto-created, gitignored)
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

# 3. Set your API key in config.py
#    PROVIDER = "gemini"  # openai | gemini | anthropic
#    GEMINI_API_KEY = "your-key-here"

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
| **Jarvis** (Agent) | Hold CTRL+SHIFT+J | Record → STT → Plan → Execute → Summarize → TTS | Full assistant with tool access |
| **WhisperFlow** (Dictation) | Hold CTRL+SHIFT+K | Record → STT → Paste at cursor | Transcribe speech to text anywhere |
| **Ultra** (Generate) | Hold CTRL+SHIFT+U | Screenshot+OCR → Record → STT → Profile → LLM Gen → Paste | Generate emails, cover letters, code at cursor |

### Jarvis Agent Pipeline
1. **Planner** — Cloud LLM classifies intent: chat or tool request. If tools needed, outputs structured plan.
2. **Executor** — Runs each tool step (web search, weather, GeoIP, etc.) in sequence.
3. **Summarizer** — Cloud LLM formats raw results into 1-2 natural spoken sentences.

### Available Tools (Jarvis mode)
- **search_web(query)** — Searches the web via DuckDuckGo
- **fetch_url(url)** — Fetches full page text content from a URL
- **get_weather(city)** — Current weather for any city (use "auto" for current location)
- **get_city_info(ip)** — GeoIP lookup from IP address
- **get_datetime()** — Current date and time
- **calculate(expression)** — Safe math evaluation
- **open_app(app_name)** — Launch desktop apps (chrome, vscode, notepad, etc.)
- **take_note(note)** / **read_notes(last_n)** / **update_note(index, content)** / **delete_note(index)** — Notes management
- **store_memory(content)** — Remember a fact or preference
- **get_system_info()** — CPU, RAM, disk usage
- **open_url(url)** — Open URL or named bookmark
- **read_clipboard()** — Read clipboard contents
- **get_news(topic)** — Top headlines (general, tech, science, us)
- **media_control(action)** — Play/pause/next/volume
- **set_reminder(message, when)** — Set a timed reminder with natural language
- **list_reminders()** — Show all pending reminders

## Key Rules
- STT runs on CUDA (GPU), LLM is cloud — no local GPU conflict
- Terminal must run as Administrator
- Python 3.12
- No comments in code
- Type hints on all function signatures
