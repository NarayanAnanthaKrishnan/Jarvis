import json
import threading
import uuid
from functools import partial
from pathlib import Path
from typing import Any

import config
from email_agent.models import EmailPayload, ProviderError


SCOPES = ["https://www.googleapis.com/auth/gmail.compose", "https://www.googleapis.com/auth/calendar.events.owned"]
KEYRING_SERVICE = "Jarvis.Gmail"
KEYRING_USER = "primary"
CONNECTION_FILE = Path(config.EMAIL_DB_PATH).parent / ".gmail_connection.json"


def connection_generation() -> str:
    try:
        return CONNECTION_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _changed_connection() -> None:
    temporary = CONNECTION_FILE.with_suffix(".tmp")
    temporary.write_text(uuid.uuid4().hex, encoding="utf-8")
    temporary.replace(CONNECTION_FILE)


def _keyring() -> Any:
    import keyring
    from keyring.backends.Windows import WinVaultKeyring
    backend = keyring.get_keyring()
    if not isinstance(backend, WinVaultKeyring):
        raise ProviderError("windows_credential_manager_required")
    return backend


def save_credentials(credentials: Any) -> None:
    _keyring().set_password(KEYRING_SERVICE, KEYRING_USER, credentials.to_json())


def load_credentials() -> Any:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    raw = _keyring().get_password(KEYRING_SERVICE, KEYRING_USER)
    if not raw:
        raise ProviderError("gmail_not_connected")
    try:
        info = json.loads(raw)
        if not set(SCOPES).issubset(set(info.get("scopes") or [])):
            raise ProviderError("google_reconnect_required")
        credentials = Credentials.from_authorized_user_info(info, SCOPES)
        if not credentials.valid:
            credentials.refresh(partial(Request(), timeout=15))
            save_credentials(credentials)
        return credentials
    except Exception as exc:
        raise ProviderError("gmail_reconnect_required") from exc


class GmailProvider:
    def __init__(self, credentials: Any = None) -> None:
        import httplib2
        from google_auth_httplib2 import AuthorizedHttp
        from googleapiclient.discovery import build
        self.credentials = credentials or load_credentials()
        self.generation = connection_generation()
        self._lock = threading.RLock()
        http = AuthorizedHttp(self.credentials, http=httplib2.Http(timeout=15))
        self.api = build("gmail", "v1", http=http, cache_discovery=False, static_discovery=True)
        self.account = self._execute(self.api.users().getProfile(userId="me"))["emailAddress"]

    def connection_current(self) -> bool:
        return self.generation == connection_generation()

    def _execute(self, request: Any, write: bool = False) -> dict:
        from google.auth.exceptions import RefreshError
        from googleapiclient.errors import HttpError
        with self._lock:
            if not self.connection_current():
                raise ProviderError("gmail_connection_changed")
            try:
                result = request.execute(num_retries=0)
            except RefreshError as exc:
                raise ProviderError("gmail_reconnect_required") from exc
            except HttpError as exc:
                status = exc.resp.status
                if status in (401, 403):
                    raise ProviderError("gmail_authorization_or_quota_rejected") from exc
                if status == 404:
                    raise ProviderError("draft_missing") from exc
                if status == 429:
                    raise ProviderError("gmail_rate_limited", retryable=True) from exc
                raise ProviderError("gmail_server_error" if status >= 500 else "gmail_request_rejected", uncertain=write and status >= 500, retryable=not write and status >= 500) from exc
            except Exception as exc:
                raise ProviderError("gmail_connection_failed", uncertain=write, retryable=not write) from exc
            return result

    def create_draft(self, payload: EmailPayload) -> str:
        return self._execute(self.api.users().drafts().create(userId="me", body={"message": {"raw": payload.raw()}}), write=True)["id"]

    def update_draft(self, draft_id: str, payload: EmailPayload) -> None:
        self._execute(self.api.users().drafts().update(userId="me", id=draft_id, body={"message": {"raw": payload.raw()}}), write=True)

    def get_draft(self, draft_id: str) -> EmailPayload:
        value = self._execute(self.api.users().drafts().get(userId="me", id=draft_id, format="raw"))
        return EmailPayload.from_raw(value["message"]["raw"])

    def send_draft(self, draft_id: str, payload: EmailPayload) -> str:
        return self._execute(self.api.users().drafts().send(userId="me", body={"id": draft_id, "message": {"raw": payload.raw()}}), write=True)["id"]


def connect(client_file: str | None = None) -> str:
    from google_auth_oauthlib.flow import InstalledAppFlow
    path = Path(client_file or config.GOOGLE_OAUTH_CLIENT_FILE)
    if not path.is_file():
        raise ProviderError("google_desktop_oauth_client_file_missing")
    flow = InstalledAppFlow.from_client_secrets_file(str(path), SCOPES, autogenerate_code_verifier=True)
    credentials = flow.run_local_server(host="127.0.0.1", port=0, access_type="offline", prompt="consent", timeout_seconds=180)
    account = GmailProvider(credentials).account
    from calendar_agent.google_calendar import GoogleCalendarProvider
    GoogleCalendarProvider(credentials, account).verify()
    if not credentials.refresh_token:
        raise ProviderError("offline_access_not_granted")
    save_credentials(credentials)
    _changed_connection()
    return account


def disconnect() -> None:
    import requests
    backend = _keyring()
    raw = backend.get_password(KEYRING_SERVICE, KEYRING_USER)
    if not raw:
        return
    token = json.loads(raw).get("refresh_token")
    if token:
        try:
            response = requests.post("https://oauth2.googleapis.com/revoke", data={"token": token}, timeout=15)
            if response.status_code not in (200, 400):
                raise ProviderError("token_revocation_failed")
        except requests.RequestException as exc:
            raise ProviderError("token_revocation_failed") from exc
    backend.delete_password(KEYRING_SERVICE, KEYRING_USER)
    _changed_connection()
