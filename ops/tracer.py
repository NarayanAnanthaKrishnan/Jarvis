import json
import logging
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from config import TRACING


TRACE_DIR = Path(__file__).resolve().parent.parent / ".traces"
_LOCK = threading.Lock()
_HANDLER: RotatingFileHandler | None = None
_FIELDS = frozenset({"turn_id", "step", "steps", "agent_id", "confidence", "source", "fallback_reason", "model", "elapsed_s", "needs_memory", "name", "status", "where", "error_type", "error_code", "draft_id", "job_id", "action_id", "managed_id", "attempt", "output_chars", "finish_reason", "time_present", "time_parse", "time_source", "time_error_code"})


def trace(event: str, **data: Any) -> None:
    global _HANDLER
    if not TRACING:
        return
    try:
        with _LOCK:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            path = TRACE_DIR / "events.jsonl"
            if _HANDLER is None or Path(_HANDLER.baseFilename) != path.resolve():
                if _HANDLER is not None:
                    _HANDLER.close()
                _HANDLER = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event,
                      **{key: value for key, value in data.items() if key in _FIELDS}}
            _HANDLER.emit(logging.LogRecord("jarvis.trace", logging.INFO, "", 0, json.dumps(record, ensure_ascii=False), (), None))
    except Exception:
        pass
