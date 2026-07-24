import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import TRACING

TRACE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / ".traces"


def _ensure_dir():
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def trace(event: str, **data) -> None:
    if not TRACING:
        return
    try:
        _ensure_dir()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = TRACE_DIR / f"{date_str}.jsonl"
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        with open(str(path), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
