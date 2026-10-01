import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from agent.context import TurnContext
from agent.loop import run_agent
from email_agent.models import EmailPayload
from email_agent.runtime import execute_email
from email_agent.service import EmailService
from email_agent.store import EmailStore
from llm.client import LLMClient


class SyntheticMailbox:
    account = "owner@example.com"

    def __init__(self) -> None:
        self.drafts: dict[str, EmailPayload] = {}

    def create_draft(self, payload: EmailPayload) -> str:
        key = str(len(self.drafts) + 1)
        self.drafts[key] = payload
        return key

    def update_draft(self, draft_id: str, payload: EmailPayload) -> None:
        self.drafts[draft_id] = payload

    def get_draft(self, draft_id: str) -> EmailPayload:
        return self.drafts[draft_id]

    def send_draft(self, draft_id: str, payload: EmailPayload) -> str:
        raise AssertionError("This diagnostic must never send email")


def check_gmail_draft(root: Path) -> None:
    from unittest.mock import Mock
    from email_agent.gmail import GmailProvider

    provider = GmailProvider()
    remote_ids: list[str] = []
    create = provider.create_draft

    def capture(payload: EmailPayload) -> str:
        remote_id = create(payload)
        remote_ids.append(remote_id)
        return remote_id

    def reject_send(*args: object, **kwargs: object) -> str:
        raise AssertionError("Gmail diagnostic may only create, update, read and remove its own temporary draft")

    with tempfile.TemporaryDirectory(prefix="gmail-draft-", dir=root) as directory, \
         patch.object(provider, "create_draft", capture), patch.object(provider, "send_draft", reject_send), \
         patch("ops.tracer.TRACING", False):
        service = EmailService(EmailStore(Path(directory) / "check.db"), provider)
        context = TurnContext(agent_id="email", email_service=service)
        writer = Mock()
        writer.call_raw.return_value = {"message": {"content": json.dumps({"subject": "Jarvis temporary draft verification", "body": "Synthetic verification draft. This message will not be sent."})}}
        try:
            result = execute_email("email_draft", {"instruction": "Synthetic draft verification"}, context, writer)
            assert result.status == "draft" and context.email_workflow.awaiting == "recipient"
            before = provider.get_draft(remote_ids[0])
            assert not before.to and not before.cc and not before.bcc
            result = execute_email("email_recipients", {"draft_id": result.draft_id, "to": [provider.account]}, context, writer)
            after = provider.get_draft(remote_ids[0])
            assert result.status == "draft" and before.subject == after.subject and before.body == after.body
            assert after.to == [provider.account]
            writer.call_raw.assert_called_once()
            print("Gmail draft check passed: recipient-free creation, recipient update and unchanged content verified.", flush=True)
        finally:
            for remote_id in remote_ids:
                provider._execute(provider.api.users().drafts().delete(userId="me", id=remote_id), write=True)
            print(f"Removed {len(remote_ids)} temporary verification draft(s). No email sent.", flush=True)


def check_scenario(name: str, utterances: list[str], root: Path, llm: LLMClient) -> dict:
    mailbox = SyntheticMailbox()
    service = EmailService(EmailStore(root / (name + ".db")), mailbox)
    context = TurnContext(email_service=service)
    history: list[str] = []
    records = []
    for utterance in utterances:
        context = TurnContext(session_id=context.session_id, email_service=service,
                              active_draft_id=context.active_draft_id, email_workflow=context.email_workflow)
        started = time.monotonic()
        result = run_agent(utterance, history, llm, context)
        history.extend(["User: " + utterance, "Assistant: " + result["output"]])
        record = {"scenario": name, "turn": len(records) + 1, "elapsed_s": round(time.monotonic() - started, 2),
                  "status": result["email_result"].status if result["email_result"] else "clarification",
                  "awaiting": context.email_workflow.awaiting, "drafts": len(mailbox.drafts),
                  "confirmation": result["confirmation_id"] is not None}
        print(json.dumps(record), flush=True)
        records.append(record)
        if result["email_result"] and result["email_result"].status == "error":
            raise AssertionError(f"{name}: email operation failed")
        if result.get("stream") is not None or context.agent_id != "email":
            raise AssertionError(f"{name}: email follow-up routed incorrectly")
    assert len(mailbox.drafts) == 1, f"{name}: expected one reusable draft"
    with service.store.transaction() as conn:
        jobs = conn.execute("SELECT status FROM jobs").fetchall()
    assert all(job["status"] == "awaiting_confirmation" for job in jobs)
    if name == "progressive_schedule":
        assert [row["awaiting"] for row in records] == ["recipient", "time", None]
        assert records[-1]["confirmation"] and mailbox.drafts["1"].to == ["alex@example.com"]
    elif name == "purpose_then_recipient":
        assert records[0]["drafts"] == 0 and records[0]["awaiting"] == "purpose"
        assert mailbox.drafts["1"].to == ["sam@example.com"] and not jobs
    elif name == "incomplete_address":
        assert not mailbox.drafts["1"].to and records[-1]["awaiting"] == "recipient" and not jobs
    elif name == "draft_and_revise":
        assert mailbox.drafts["1"].to == ["robin@example.com"] and not jobs
    return {"scenario": name, "passed": True, "turns": records}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Exercise real email model decisions using synthetic prompts and an in-memory mailbox. No Gmail access or sending.")
    parser.add_argument("--live-model", action="store_true", help="Allow Gemini calls for synthetic conversations; API usage is billed normally.")
    parser.add_argument("--gmail-draft", action="store_true", help="Create, verify and delete one temporary Gmail draft, adding only the connected account as recipient. Never send.")
    args = parser.parse_args()
    if not args.live_model and not args.gmail_draft:
        parser.error("Choose --live-model or --gmail-draft to enable the corresponding live check")
    root = Path(__file__).resolve().parent.parent / ".test_runs"
    root.mkdir(exist_ok=True)
    if args.gmail_draft:
        check_gmail_draft(root)
    if not args.live_model:
        return
    scenarios = {
        "progressive_schedule": ["Schedule an email asking Alex about a software developer opening. I don't have the address or time yet.", "alex@example.com", "Tomorrow at 9am"],
        "purpose_then_recipient": ["Draft an email", "Ask Sam if the project review is ready", "sam@example.com"],
        "incomplete_address": ["Draft an application email for a software developer opening. The recipient's address was garbled; I'll spell it later."],
        "draft_and_revise": ["Draft an email to alex@example.com asking about a project review. Just save a draft.", "Make it shorter and warmer", "Change the recipient to robin@example.com"],
    }
    with patch("llm.client.load_profile", return_value=""), patch("agent.loop.load_profile", return_value=""), \
         patch("config.JEV_ROUTING_ENABLED", False), patch("config.RETRIEVAL_GATE", True), \
         patch("agent.loop.needs_memory", return_value=False), patch("ops.tracer.TRACING", False), \
         tempfile.TemporaryDirectory(prefix="email-live-", dir=root) as directory:
        llm = LLMClient()
        results = [check_scenario(name, utterances, Path(directory), llm) for name, utterances in scenarios.items()]
    report = root / "email_harness_live.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Passed {len(results)} synthetic conversations; report: {report}")


if __name__ == "__main__":
    main()
