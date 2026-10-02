import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from email_agent.models import EmailError


class EmailStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise RuntimeError("Unsupported email database version")
            conn.execute("CREATE TABLE IF NOT EXISTS drafts (id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, remote_id TEXT, payload TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL, operation_key TEXT UNIQUE NOT NULL, updated_at REAL NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, draft_id INTEGER NOT NULL REFERENCES drafts(id), account TEXT NOT NULL, session_id TEXT NOT NULL, action TEXT NOT NULL, payload TEXT NOT NULL, fingerprint TEXT NOT NULL, revision INTEGER NOT NULL, timezone TEXT NOT NULL, due_at REAL NOT NULL, expires_at REAL NOT NULL, status TEXT NOT NULL, presented INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0, lease_until REAL NOT NULL DEFAULT 0, provider_message_id TEXT, last_error TEXT, updated_at REAL NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_due ON jobs(status, due_at, next_attempt)")
            conn.execute("CREATE TABLE IF NOT EXISTS notifications (job_id INTEGER PRIMARY KEY, status TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS calendar_actions (id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, session_id TEXT NOT NULL, action TEXT NOT NULL, managed_id INTEGER, event_id TEXT NOT NULL, payload TEXT NOT NULL, request_key TEXT UNIQUE NOT NULL, expires_at REAL NOT NULL, status TEXT NOT NULL, presented INTEGER NOT NULL DEFAULT 0, error_code TEXT, updated_at REAL NOT NULL)")
            conn.execute("CREATE INDEX IF NOT EXISTS calendar_actions_status ON calendar_actions(status, updated_at)")
            conn.execute("CREATE TABLE IF NOT EXISTS calendar_events (id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, event_id TEXT NOT NULL, payload TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL, updated_at REAL NOT NULL, UNIQUE(account,event_id))")
            conn.execute("PRAGMA user_version=2")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def draft(self, draft_id: int) -> dict:
        if type(draft_id) is not int or draft_id < 1:
            raise ValueError("Draft ID must be a positive integer")
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
        if row is None:
            raise EmailError("That draft ID was not found. Ask me to list your email drafts.")
        return dict(row)

    def job(self, job_id: int) -> dict:
        if type(job_id) is not int or job_id < 1:
            raise ValueError("Email action ID must be a positive integer")
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise EmailError("That email action ID was not found. Ask me to list your email actions.")
        return dict(row)
