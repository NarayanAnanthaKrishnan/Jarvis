import os
import sqlite3
from datetime import datetime

import dateparser

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reminders.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                fire_at TEXT NOT NULL,
                fired INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)


def parse_when(natural: str) -> datetime | None:
    return dateparser.parse(
        natural,
        settings={"PREFER_DATES_FROM": "future"}
    )


def add_reminder(message: str, fire_at: datetime) -> str:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO reminders (message, fire_at, created_at) VALUES (?, ?, ?)",
            (message, fire_at.isoformat(), datetime.now().isoformat())
        )
    return f"Reminder set for {fire_at.strftime('%I:%M %p on %A, %B %d')}: {message}"


def get_due_reminders() -> list[dict]:
    now = datetime.now().isoformat()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, message, fire_at FROM reminders WHERE fire_at <= ? AND fired = 0",
            (now,)
        ).fetchall()
    return [dict(r) for r in rows]


def mark_fired(reminder_id: int) -> None:
    with _get_conn() as conn:
        conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))


def list_pending() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, message, fire_at FROM reminders WHERE fired = 0 ORDER BY fire_at"
        ).fetchall()
    return [dict(r) for r in rows]


def format_pending() -> str:
    pending = list_pending()
    if not pending:
        return "No pending reminders."
    lines = [f"{r['id']}. {r['message']} — due {r['fire_at']}" for r in pending]
    return "\n".join(lines)


init_db()
