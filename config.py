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
MAX_STEPS = 6
STT_MODEL = "base"
STT_DEVICE = "cpu"
STT_PREROLL_SECONDS = 0.5
STT_STABILIZE_PARTIALS = 2
SESSION_SILENCE_SECONDS = 1.5
STREAMING_ENABLED = True
SESSION_STT_MODEL = "tiny"
TTS_MODEL = os.getenv("TTS_MODEL", "kokoro-v1.0.onnx")
TTS_FALLBACK_MODEL = "kokoro-v1.0.int8.onnx"
TTS_PRELOAD = os.getenv("TTS_PRELOAD", "true").lower() == "true"
TTS_QUEUE_CHUNKS = 2

PROVIDER = os.getenv("PROVIDER", "gemini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

AUTO_EXTRACT = True
REMINDER_CHECK_SECONDS = 30
PARALLEL_WORKERS = 5
REFLECTION_ENABLED = False
RETRIEVAL_GATE = True
TRACING = True

EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_MODEL = os.getenv("EMAIL_MODEL", "gemini-3.8-flash")
EMAIL_THINKING_LEVEL = os.getenv("EMAIL_THINKING_LEVEL", "low").lower()
EMAIL_MODEL_TIMEOUT_SECONDS = 20
EMAIL_MODEL_MAX_TOKENS = 4096
EMAIL_TIMEZONE = os.getenv("EMAIL_TIMEZONE", "America/New_York")
EMAIL_DB_PATH = os.path.join(os.path.dirname(__file__), "email_jobs.db")
GOOGLE_OAUTH_CLIENT_FILE = os.getenv("GOOGLE_OAUTH_CLIENT_FILE", os.path.join(os.path.dirname(__file__), "gmail_client.json"))
EMAIL_POLL_SECONDS = 5
EMAIL_SEND_WINDOW_SECONDS = 60
EMAIL_CONFIRM_SECONDS = 600
JEV_ROUTING_ENABLED = os.getenv("JEV_ROUTING_ENABLED", "false").lower() == "true"
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
JEV_TIMEOUT_SECONDS = 2.0
JEV_MIN_CONFIDENCE = 0.8
