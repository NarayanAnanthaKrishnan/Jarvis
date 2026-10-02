import threading
from typing import Any

from email_agent.models import ProviderError


class GoogleCalendarProvider:
    def __init__(self, credentials: Any = None, account: str | None = None) -> None:
        import httplib2
        from google_auth_httplib2 import AuthorizedHttp
        from googleapiclient.discovery import build
        from email_agent.gmail import connection_generation, load_credentials

        self.credentials = credentials or load_credentials()
        self.account = account or ""
        self.generation = connection_generation()
        self._lock = threading.RLock()
        http = AuthorizedHttp(self.credentials, http=httplib2.Http(timeout=15))
        self.api = build("calendar", "v3", http=http, cache_discovery=False, static_discovery=True)

    def connection_current(self) -> bool:
        from email_agent.gmail import connection_generation
        return self.generation == connection_generation()

    def _execute(self, request: Any, write: bool = False) -> dict:
        from google.auth.exceptions import RefreshError
        from googleapiclient.errors import HttpError

        with self._lock:
            if not self.connection_current():
                raise ProviderError("google_connection_changed")
            try:
                return request.execute(num_retries=0)
            except RefreshError as exc:
                raise ProviderError("google_reconnect_required") from exc
            except HttpError as exc:
                status = exc.resp.status
                if status in (401, 403):
                    raise ProviderError("calendar_authorization_or_quota_rejected") from exc
                if status == 404:
                    raise ProviderError("calendar_event_missing") from exc
                if status == 429:
                    raise ProviderError("calendar_rate_limited", retryable=not write) from exc
                raise ProviderError("calendar_server_error" if status >= 500 else "calendar_request_rejected",
                                    uncertain=write and status >= 500, retryable=not write and status >= 500) from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError("calendar_connection_failed", uncertain=write, retryable=not write) from exc

    def verify(self) -> None:
        self._execute(self.api.events().list(calendarId="primary", maxResults=1))

    def insert(self, event_id: str, body: dict) -> dict:
        payload = dict(body)
        payload["id"] = event_id
        return self._execute(self.api.events().insert(calendarId="primary", body=payload, sendUpdates="all"), write=True)

    def get(self, event_id: str) -> dict | None:
        try:
            return self._execute(self.api.events().get(calendarId="primary", eventId=event_id))
        except ProviderError as exc:
            if exc.code == "calendar_event_missing":
                return None
            raise

    def update(self, event_id: str, body: dict) -> dict:
        return self._execute(self.api.events().patch(calendarId="primary", eventId=event_id, body=body,
                                                      sendUpdates="all"), write=True)

    def delete(self, event_id: str) -> None:
        self._execute(self.api.events().delete(calendarId="primary", eventId=event_id,
                                                sendUpdates="all"), write=True)

    def list_events(self, start: str, end: str, limit: int = 20) -> list[dict]:
        result = self._execute(self.api.events().list(calendarId="primary", timeMin=start, timeMax=end,
                                                       maxResults=limit, singleEvents=True,
                                                       orderBy="startTime"))
        return result.get("items", [])
