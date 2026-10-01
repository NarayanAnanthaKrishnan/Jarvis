import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import config
from email_agent.store import EmailStore
from ops.tracer import trace


class WorkerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file: Any = None

    def __enter__(self) -> "WorkerLock":
        self.file = self.path.open("a+b")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("Email worker is already running") from exc
        return self

    def __exit__(self, *args: Any) -> None:
        self.file.close()


def notify(message: str) -> None:
    print(message, flush=True)
    try:
        from plyer import notification
        notification.notify(title="Jarvis Email", message=message, timeout=10)
    except Exception:
        pass


def run_worker(stop: threading.Event | None = None) -> None:
    from email_agent.gmail import GmailProvider
    from email_agent.service import EmailService
    stop = stop or threading.Event()
    root = Path(config.EMAIL_DB_PATH).parent
    store = EmailStore(config.EMAIL_DB_PATH)
    with WorkerLock(root / "email_worker.lock"):
        provider: Any = None
        last_error = ""
        while not stop.is_set():
            try:
                if not config.EMAIL_ENABLED:
                    raise RuntimeError("email_disabled")
                if provider is None or not provider.connection_current():
                    provider = GmailProvider()
                service = EmailService(store, provider)
                service.maintenance()
                service.deliver_one()
                last_error = ""
                with store.transaction() as conn:
                    rows = conn.execute("SELECT j.id,j.status FROM jobs j LEFT JOIN notifications n ON n.job_id=j.id WHERE j.status IN ('sent','missed','needs_review','failed','delivery_unknown') AND (n.status IS NULL OR n.status != j.status)").fetchall()
                    for row in rows:
                        conn.execute("INSERT INTO notifications(job_id,status) VALUES(?,?) ON CONFLICT(job_id) DO UPDATE SET status=excluded.status", (row["id"], row["status"]))
                for row in rows:
                    notify(f"Email {row['id']}: {row['status'].replace('_', ' ')}. Check Jarvis email status.")
                health = {"pid": os.getpid(), "timestamp": time.time(), "status": "running"}
            except Exception as exc:
                provider = None
                code = getattr(exc, "code", type(exc).__name__)
                if code != last_error:
                    notify(f"Email worker needs attention: {code}. Run python -m email_agent health.")
                    last_error = code
                trace("email_worker", status="unavailable", error_code=code)
                health = {"pid": os.getpid(), "timestamp": time.time(), "status": "unavailable", "error_code": code}
            path = root / "email_worker_status.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(health), encoding="utf-8")
            temporary.replace(path)
            stop.wait(config.EMAIL_POLL_SECONDS)
