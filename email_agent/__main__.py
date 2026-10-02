import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import config


def worker_task(install: bool) -> None:
    if os.name != "nt":
        raise RuntimeError("Windows Task Scheduler is required")
    root = Path(__file__).resolve().parent.parent
    executable = Path(sys.executable).with_name("pythonw.exe")
    if not executable.exists():
        raise RuntimeError("pythonw.exe is missing from this Python environment")
    if install:
        script = """$ErrorActionPreference = 'Stop'
$action = New-ScheduledTaskAction -Execute $env:JARVIS_WORKER_PYTHON -Argument '-m email_agent worker' -WorkingDirectory $env:JARVIS_WORKER_ROOT
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'JarvisEmailWorker' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName 'JarvisEmailWorker'
"""
    else:
        script = """$ErrorActionPreference = 'Stop'
Stop-ScheduledTask -TaskName 'JarvisEmailWorker' -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName 'JarvisEmailWorker' -Confirm:$false
"""
    env = dict(os.environ, JARVIS_WORKER_PYTHON=str(executable), JARVIS_WORKER_ROOT=str(root))
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", script], env=env, check=True, creationflags=subprocess.CREATE_NO_WINDOW)


def main() -> int:
    parser = argparse.ArgumentParser(description="Jarvis Google Workspace email and Calendar connection")
    parser.add_argument("command", choices=["connect", "disconnect", "health", "status", "worker", "install-worker", "remove-worker", "resolve-delivery"])
    parser.add_argument("--client-file")
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--outcome", choices=["sent", "not-sent"])
    args = parser.parse_args()
    try:
        if args.command == "connect":
            from email_agent.gmail import connect
            print(f"Connected Google Workspace account: {connect(args.client_file)}")
            from email_agent.store import EmailStore
            with EmailStore(config.EMAIL_DB_PATH).transaction() as conn:
                conn.execute("UPDATE jobs SET status='needs_review',last_error='reconnected' WHERE status IN ('awaiting_confirmation','scheduled','checking')")
            print("Set EMAIL_ENABLED=true in .env. Install the worker to deliver scheduled Gmail messages.")
        elif args.command == "disconnect":
            from email_agent.gmail import disconnect
            from email_agent.store import EmailStore
            with EmailStore(config.EMAIL_DB_PATH).transaction() as conn:
                conn.execute("UPDATE jobs SET status='needs_review',last_error='disconnected' WHERE status IN ('awaiting_confirmation','scheduled','checking')")
                conn.execute("UPDATE calendar_actions SET status='needs_review',error_code='disconnected',updated_at=? WHERE status IN ('awaiting_confirmation','processing')", (time.time(),))
            disconnect()
            print("Google Workspace disconnected. Pending email and meeting actions require review.")
        elif args.command in ("install-worker", "remove-worker"):
            if args.command == "install-worker" and not config.EMAIL_ENABLED:
                raise RuntimeError("Set EMAIL_ENABLED=true before installing the worker")
            worker_task(args.command == "install-worker")
            print("Email worker installed and started." if args.command == "install-worker" else "Email worker removed.")
        elif args.command == "worker":
            from email_agent.worker import run_worker
            from ops.logging_setup import configure_logging
            configure_logging()
            run_worker()
        elif args.command == "status":
            from email_agent.store import EmailStore
            with EmailStore(config.EMAIL_DB_PATH).transaction() as conn:
                rows = conn.execute("SELECT id,draft_id,status,action,due_at,timezone,last_error,provider_message_id FROM jobs ORDER BY id DESC LIMIT 50").fetchall()
                calendar_actions = conn.execute("SELECT id,managed_id,status,action,error_code,updated_at FROM calendar_actions ORDER BY id DESC LIMIT 50").fetchall()
                meetings = conn.execute("SELECT id,status,updated_at FROM calendar_events ORDER BY id DESC LIMIT 50").fetchall()
            print(json.dumps({"email_jobs": [dict(row) for row in rows],
                              "calendar_actions": [dict(row) for row in calendar_actions],
                              "managed_meetings": [dict(row) for row in meetings]}, indent=2))
        elif args.command == "resolve-delivery":
            if args.job_id is None or args.outcome is None:
                raise ValueError("Inspect Gmail Sent first, then specify --job-id and --outcome sent|not-sent")
            from email_agent.runtime import get_service
            print(get_service().resolve_delivery(args.job_id, args.outcome == "sent").message)
        else:
            from email_agent.gmail import GmailProvider
            from calendar_agent.google_calendar import GoogleCalendarProvider
            print(f"Email enabled: {config.EMAIL_ENABLED}")
            gmail = GmailProvider()
            GoogleCalendarProvider(gmail.credentials, gmail.account).verify()
            print(f"Connected Google Workspace account: {gmail.account}")
            path = Path(config.EMAIL_DB_PATH).parent / "email_worker_status.json"
            print(f"Worker heartbeat: {path.read_text(encoding='utf-8') if path.exists() else 'not started'}")
        return 0
    except Exception as exc:
        print(f"Email operation failed: {getattr(exc, 'code', str(exc) if isinstance(exc, (RuntimeError, ValueError)) else type(exc).__name__)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
