# 🤖 Jarvis — Personal AI Assistant
## Full Build Plan (Phase 1 → Phase 4)

---

## Hardware Context
| Spec | Detail |
|---|---|
| GPU | NVIDIA GTX 1650 (4GB VRAM) |
| RAM | 8GB |
| OS | Windows |
| Primary LLM | Gemma 4 E2B via Ollama (runs on GPU) |
| Fallback LLMs | `phi4-mini`, `llama3.2:3b`, `qwen2.5:3b` via Ollama |

> **VRAM strategy:** Whisper runs on CPU, LLM runs on GPU. Never compete for VRAM.

---

## Tech Stack (All Phases)

| Component | Tool | Phase |
|---|---|---|
| Hotkey detection | `keyboard` | 1 |
| Audio capture | `sounddevice` | 1 |
| Speech-to-text | `faster-whisper` (base, CPU) | 1 |
| LLM inference | `ollama` Python client | 1 |
| Text-to-speech | `Kokoro TTS` | 1 |
| Audio playback | `sounddevice` + `numpy` | 1 |
| Conversation memory | `SQLite` | 2 |
| Long-term memory | `ChromaDB` | 2 |
| Agent orchestration | `LangGraph` | 2 |
| Wake word | `openWakeWord` | 3 |
| Web search | `SearXNG` (self-hosted) or DuckDuckGo | 3 |
| File access | `pathlib` + `os` | 3 |
| RAG over docs | `ChromaDB` + `LangChain` | 3 |
| System control | `pyautogui`, `subprocess` | 4 |
| UI | System tray (`pystray`) | 4 |

---

## Phase 1 — WhisperFlow (Voice Pipeline)
**Goal:** Press hotkey → speak → Gemma responds → hear the answer.
**Timeline:** 1 week

### Architecture
```
[CTRL+SHIFT+J held] 
    → sounddevice records audio
    → key released → stop recording
    → faster-whisper transcribes (CPU)
    → print transcript to terminal
    → send to Ollama (Gemma 4 E2B)
    → stream tokens to terminal
    → Kokoro TTS converts response
    → sounddevice plays audio
```

### Project Structure
```
jarvis/
├── main.py              # Entry point, hotkey listener
├── audio/
│   ├── recorder.py      # sounddevice capture (hold-to-talk)
│   ├── player.py        # sounddevice playback
├── stt/
│   └── transcriber.py   # faster-whisper wrapper
├── llm/
│   ├── client.py        # Ollama client + fallback logic
│   └── prompts.py       # System prompt
├── tts/
│   └── speaker.py       # Kokoro TTS wrapper
├── config.py            # Hotkey, model names, settings
└── requirements.txt
```

### Dependencies
```txt
# requirements.txt
faster-whisper
sounddevice
numpy
scipy
keyboard
ollama
kokoro-onnx        # lightweight Kokoro build for Windows
```

### Implementation

**config.py**
```python
HOTKEY = "ctrl+shift+j"
STT_MODEL = "base"          # faster-whisper model size
STT_DEVICE = "cpu"          # keep GPU free for LLM
LLM_PRIMARY = "gemma4:e2b"
LLM_FALLBACK = "phi4-mini"
SAMPLE_RATE = 16000
CHANNELS = 1
```

**audio/recorder.py**
```python
import sounddevice as sd
import numpy as np

class Recorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = []
        self.is_recording = False

    def start(self):
        self.recording = []
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._callback
        )
        self.stream.start()
        print("🎙  Listening...")

    def _callback(self, indata, frames, time, status):
        if self.is_recording:
            self.recording.append(indata.copy())

    def stop(self) -> np.ndarray:
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        if not self.recording:
            return np.array([])
        return np.concatenate(self.recording, axis=0).flatten()
```

**stt/transcriber.py**
```python
from faster_whisper import WhisperModel
import numpy as np

class Transcriber:
    def __init__(self, model_size="base", device="cpu"):
        print(f"Loading Whisper ({model_size}) on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe(self, audio: np.ndarray, sample_rate=16000) -> str:
        if len(audio) < sample_rate * 0.5:  # ignore < 0.5 sec
            return ""
        segments, _ = self.model.transcribe(audio, beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        return text
```

**llm/client.py**
```python
import ollama
from config import LLM_PRIMARY, LLM_FALLBACK

SYSTEM_PROMPT = """You are Jarvis, a concise personal AI assistant. 
Respond clearly and briefly. You are running locally and privately."""

class LLMClient:
    def __init__(self):
        self.model = LLM_PRIMARY
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        try:
            response = ollama.chat(
                model=self.model,
                messages=self.history,
                stream=False
            )
            reply = response['message']['content']
        except Exception:
            print(f"⚠️  Falling back to {LLM_FALLBACK}")
            self.model = LLM_FALLBACK
            response = ollama.chat(
                model=self.model,
                messages=self.history,
                stream=False
            )
            reply = response['message']['content']

        self.history.append({"role": "assistant", "content": reply})
        return reply
```

**tts/speaker.py**
```python
from kokoro_onnx import Kokoro
import sounddevice as sd
import numpy as np

class Speaker:
    def __init__(self):
        self.kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
        self.voice = "af_bella"   # balanced quality/speed voice

    def speak(self, text: str):
        if not text.strip():
            return
        samples, sample_rate = self.kokoro.create(text, voice=self.voice, speed=1.0)
        sd.play(samples, sample_rate)
        sd.wait()
```

**main.py**
```python
import keyboard
import threading
from audio.recorder import Recorder
from stt.transcriber import Transcriber
from llm.client import LLMClient
from tts.speaker import Speaker
from config import HOTKEY, SAMPLE_RATE

recorder = Recorder(sample_rate=SAMPLE_RATE)
transcriber = Transcriber()
llm = LLMClient()
speaker = Speaker()

def on_activate():
    recorder.start()

def on_release():
    audio = recorder.stop()
    if len(audio) == 0:
        return

    text = transcriber.transcribe(audio)
    if not text:
        print("⚠️  Could not understand audio.")
        return

    print(f"\n🗣  You: {text}")
    reply = llm.chat(text)
    print(f"🤖 Jarvis: {reply}\n")
    speaker.speak(reply)

print("✅ Jarvis is running. Hold CTRL+SHIFT+J to speak.\n")
keyboard.add_hotkey(HOTKEY, on_activate, trigger_on_release=False)
keyboard.on_release_key("j", lambda _: on_release() if keyboard.is_pressed("ctrl+shift") else None)
keyboard.wait()
```

### Setup Steps
```bash
# 1. Create virtual environment
python -m venv jarvis-env
jarvis-env\Scripts\activate

# 2. Install dependencies
pip install faster-whisper sounddevice numpy scipy keyboard ollama kokoro-onnx

# 3. Pull fallback model in Ollama
ollama pull phi4-mini

# 4. Run
python main.py
```

---

## Phase 2 — Memory + Context
**Goal:** Jarvis remembers you across sessions. Knows your name, preferences, past conversations.
**Timeline:** 1 week after Phase 1

### What Gets Added
- `SQLite` — stores every conversation with timestamp
- `ChromaDB` — semantic search over past conversations
- Memory injection — top-k relevant memories injected into system prompt before each query
- Session persistence — conversation history survives restarts

### Key Addition to LLMClient
```python
# Before sending to LLM, retrieve relevant memories
memories = memory_store.search(user_input, top_k=3)
memory_context = "\n".join([f"- {m}" for m in memories])
# Inject into system prompt dynamically
```

---

## Phase 3 — Skills (Tools)
**Goal:** Jarvis can do things, not just answer questions.
**Timeline:** 2 weeks, add one skill at a time

### Skills Roadmap
| Skill | Tool | Trigger example |
|---|---|---|
| Web search | DuckDuckGo API / SearXNG | "What's the weather in Syracuse?" |
| Open app | `subprocess` | "Open VS Code" |
| Read file | `pathlib` | "Summarize my notes.txt" |
| Write file | `pathlib` | "Save this as todo.txt" |
| Run code | `subprocess` sandboxed | "Run my test suite" |
| Calendar | Google Calendar API | "What's on my schedule tomorrow?" |
| RAG over docs | ChromaDB + LangChain | "Search my research papers for RAG" |

### Architecture Upgrade (LangGraph)
Phase 1 is a straight pipeline. Phase 3 upgrades to a LangGraph router:
```
User input → Intent classifier → Route to skill node → Execute → Respond
```
Each skill = one LangGraph node. Easy to add new ones without touching existing code.

---

## Phase 4 — Jarvis (Full Vision)
**Goal:** Proactive, multi-modal, system-aware assistant.
**Timeline:** Ongoing

### What Gets Added
- **Wake word** (`openWakeWord`) — say "Hey Jarvis" instead of hotkey
- **Proactive alerts** — "Interview in 2 hours, here's your prep brief"
- **System tray UI** (`pystray`) — small icon, status indicator
- **Screen awareness** (`pyautogui` screenshot → vision model) — "What's on my screen?"
- **Multi-model routing** — complex tasks → bigger model via OpenCode API
- **Mobile** — FastAPI endpoint → hit from phone

---

## Known Roadblocks

| Roadblock | Mitigation |
|---|---|
| Pipeline latency (5-8s round trip) | Stream TTS as tokens arrive, run STT async |
| 4GB VRAM tight with LLM | Whisper on CPU, LLM on GPU — never both on GPU |
| Gemma hits reasoning limits | Auto-fallback to phi4-mini already in code |
| Kokoro install on Windows | Use `kokoro-onnx` (lighter, Windows-friendly build) |
| keyboard lib needs admin | Run terminal as Administrator |
| Audio cuts out mid-response | Use `sd.wait()` to block until playback complete |
| RAM pressure (8GB tight) | Use Whisper `base` not `small`, monitor with Task Manager |

---

## What You'll Learn
- Real-time audio processing pipelines
- Local model inference optimization (VRAM/CPU split)
- Streaming LLM responses
- LangGraph tool-calling architecture
- Vector memory systems (ChromaDB)
- Async Python for multi-component pipelines
- Windows system-level Python (subprocess, hotkeys, audio)
