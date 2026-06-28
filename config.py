import os

_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        with open(_env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("\"'")
                if key:
                    os.environ.setdefault(key, val)

HOTKEY = "ctrl+shift+j"
WHISPERFLOW_HOTKEY = "ctrl+shift+k"
ULTRA_HOTKEY = "ctrl+shift+u"
STT_MODEL = "base"
STT_DEVICE = "cuda"
STT_PREROLL_SECONDS = 0.5
WHISPERFLOW_TOGGLE = True
STT_STABILIZE_PARTIALS = 2

PROVIDER = os.getenv("PROVIDER", "gemini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "")
