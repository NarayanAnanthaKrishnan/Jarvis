# Jarvis — Personal AI Assistant

Dual-mode voice assistant with agent pipeline: full agent chat (CTRL+SHIFT+J) or instant dictation (CTRL+SHIFT+K).

## Project Structure

```
D:\Jarvis\
├── main.py                 # Entry point — registers both hotkeys
├── config.py               # Hotkeys, model names, samplerate settings
├── modes/
│   ├── jarvis.py           # Agent mode: orchestrates Plan→Execute→Summarize
│   └── whisperflow.py      # Dictation mode: record → STT → paste at cursor
├── agent/
│   ├── __init__.py
│   ├── planner.py          # Intent router + step planner (LLM)
│   ├── executor.py         # Tool call executor
│   └── summarizer.py       # Response formatter for TTS (LLM)
├── audio/
│   ├── recorder.py         # sounddevice hold-to-record
│   └── player.py           # sounddevice audio playback
├── stt/
│   └── transcriber.py      # faster-whisper (base, CPU, int8)
├── llm/
│   ├── client.py           # Ollama client with tool-calling support
│   └── prompts.py          # System prompt
├── tts/
│   └── speaker.py          # Kokoro ONNX TTS (int8, speed=1.5)
├── tools/
│   ├── __init__.py
│   ├── registry.py         # Tool definitions + dispatch
│   ├── geoip.py            # GeoLite2-City.mmdb lookup
│   ├── weather.py          # wttr.in weather query (auto-location support)
│   └── web_search.py       # DuckDuckGo search via ddgs
├── data/
│   └── GeoLite2-City.mmdb  # MaxMind GeoIP database
├── requirements.txt
├── agents.md               # Full development plan & reference
└── README.md
```

## Setup

```powershell
# 1. Create & activate virtual environment
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

## Usage

| Mode | Hotkey | Pipeline | What it does |
|---|---|---|---|
| **Jarvis** (Agent) | Hold CTRL+SHIFT+J | Record → STT → Plan → Execute → Summarize → TTS | Full assistant with tool access |
| **WhisperFlow** (Dictation) | Hold CTRL+SHIFT+K | Record → STT (streaming) → Type at cursor, then correct | Transcribes as you speak, final accurate version replaces streamed text |

### Jarvis Agent Pipeline
1. **Planner** — LLM classifies intent: chat or tool request. If tools needed, outputs structured plan.
2. **Executor** — Runs each tool step (web search, weather, GeoIP) in sequence.
3. **Summarizer** — LLM formats raw results into 1-2 natural spoken sentences.

### Available Tools (Jarvis mode)
- **search_web(query)** — Searches the web via DuckDuckGo
- **get_weather(city)** — Current weather for any city (use "auto" for current location)
- **get_city_info(ip)** — GeoIP lookup from IP address

## Key Rules
- Whisper (STT) runs on CPU, LLM runs on GPU — never compete for VRAM
- Terminal must run as Administrator
- Python 3.12
- No comments in code
- Type hints on all function signatures
