# Gmail and Jev setup

Run these commands from `D:\Jarvis` after activating `jarvis-env`. Use the same Windows account for Gmail connection, Jarvis and the scheduled worker.

## 1. Create the Google OAuth client

1. Create or select a Google Cloud project and enable the Gmail API.
2. Configure Google Auth Platform branding and audience. For a personal Gmail account, use External and add your Gmail address as a test user while the app is in Testing.
3. Create an OAuth client of type **Desktop app** and download its JSON file to `D:\Jarvis\gmail_client.json`.

These steps follow Google's [Gmail Python quickstart](https://developers.google.com/workspace/gmail/api/quickstart/python) and [consent configuration guide](https://developers.google.com/workspace/guides/configure-oauth-consent). Jarvis requests `https://www.googleapis.com/auth/gmail.compose`, which permits draft management and sending. Google classifies this as a restricted scope. See [Gmail scope definitions](https://developers.google.com/workspace/gmail/api/auth/scopes) for distribution and verification requirements.

External apps in Testing receive refresh tokens that expire after seven days for these permissions. Reconnect when required; do not assume a testing connection will last indefinitely. [Google refresh token expiration](https://developers.google.com/identity/protocols/oauth2#expiration).

## 2. Connect and enable Gmail

Add or update these entries in `.env`:

```dotenv
EMAIL_ENABLED=false
EMAIL_TIMEZONE=America/New_York
GOOGLE_OAUTH_CLIENT_FILE=D:\Jarvis\gmail_client.json
```

```powershell
python -m email_agent connect
```

Select the intended account in the browser and approve the requested Gmail access. Connection uses a temporary loopback callback with PKCE. Access and refresh credentials are saved under `Jarvis.Gmail` in Windows Credential Manager. No `token.json` is created. The Gmail address printed on completion is the sending account.

After connection, set `EMAIL_ENABLED=true` in `.env`. Restart Jarvis to reload configuration.

## 3. Install the scheduling worker

```powershell
python -m email_agent install-worker
python -m email_agent health
```

Installation registers and starts `JarvisEmailWorker` in Windows Task Scheduler. It uses this virtual environment's `pythonw.exe`, runs hidden as the current logged-in user, and starts at that user's next logon. It loads no voice or language models. Only one worker can hold the local worker lock.

Keep the PC awake, logged in and online. The worker polls every five seconds; normal delivery should begin within approximately ten seconds when Gmail is responsive. This is local scheduling: queued messages remain Gmail drafts until the worker sends them. They do not appear in Gmail's native Scheduled folder.

The heartbeat shown by `health` should say `running` and have a recent timestamp. `health` also verifies access to the connected account. It does not send a message. For foreground troubleshooting, stop the scheduled task first, then run:

```powershell
Stop-ScheduledTask -TaskName JarvisEmailWorker
python -m email_agent worker
```

After stopping the foreground worker with CTRL+C, restart the scheduled task:

```powershell
Start-ScheduledTask -TaskName JarvisEmailWorker
```

Changes to `.env` require restarting the worker. A job more than 60 seconds overdue is marked `missed` on the next successful worker check. Wake-up, network recovery or a restart will not send it late. Request a new time and confirm a new preview.

## 4. Draft, review and confirm

Start Jarvis and toggle a session with CTRL+SHIFT+J. You can start with an incomplete request as long as the purpose is understandable. Jarvis saves the draft first, then asks one focused question for missing recipients or timing. This version cannot look up contacts or guess ambiguous addresses.

For example: “Schedule an email asking Alex about a software developer opening” saves a draft and asks for the address. Reply “alex@example.com”, then “tomorrow at 9 AM” when asked for a time. These replies update the same draft and produce a delivery preview. Recipient-only changes do not rewrite the message. An initial “draft an email” with no purpose prompts for what it should say. Relative delivery times are anchored when drafting starts, so a later recipient reply does not move the deadline forward.

Pending action and clarification state belong to the voice session and clear on session reset or account change. Drafts remain in Gmail and can be reopened by their local draft ID in a new session.

| Say | Result |
|---|---|
| “Draft an email to alex@example.com about tomorrow's meeting” | Saves a plain text Gmail draft and prints it |
| “Make it shorter” | Revises the active draft, preserving omitted recipients |
| “Send draft one” | Prints a send preview and an action number |
| “Schedule draft one for tomorrow at 9 AM” | Prints the message, local date/time, timezone and action number |
| “Confirm email two” | Approves displayed action 2 in the current session |
| “List my emails” | Prints managed draft IDs and delivery states |
| “Cancel email action two” | Cancels a pending action and retains its draft |
| “Reschedule draft one for Friday at 2 PM” | Replaces the old schedule with a new preview requiring confirmation |

Read the full From, To, Cc, Bcc, subject, body and timing before confirming. A draft request never schedules or sends by itself. A general “yes” is insufficient. Approval must name the displayed email action in a separate utterance within ten minutes. Ending or resetting the session expires unconfirmed previews. Already confirmed schedules remain queued.

Editing a draft pauses its pending delivery. Request a new preview after an edit. The worker checks the live Gmail draft and its revision against the approved snapshot before sending. If Gmail content changes, delivery requires review. Use Jarvis to edit managed drafts; Gmail edits can introduce HTML or MIME structures outside this version's plain text support.

Use AM/PM or an unambiguous 24-hour time, plus an IANA timezone when different from the default, such as `Europe/London`. Voice forms such as “later today 6 PM,” “six PM,” and “in half an hour” are normalized to a time before the scheduling model runs. If a request offers multiple times, Jarvis asks you to choose one. Ambiguous numeric dates, timezone abbreviations, past times and repeated/nonexistent daylight-saving clock times require clarification. An ISO timestamp with an explicit offset can disambiguate a daylight-saving transition.

When a spoken recipient is normalized, Jarvis reads the candidate back using “at” and “dot” and waits for confirmation before adding it. Say “no” and repeat or spell the full address to correct it. Canonical addresses already written with `@` are accepted directly.

## Delivery states and recovery

```powershell
python -m email_agent status
```

This command reads local status without contacting Gmail. State is persisted in `email_jobs.db`, including message snapshots, revisions, action IDs and any provider message ID.

| State | Meaning and next step |
|---|---|
| `awaiting_confirmation` | Preview exists; confirm its action number in the same session |
| `scheduled` | Approved and queued; keep the worker running |
| `checking` / `sending` | Claimed for delivery; cancellation cannot be guaranteed |
| `sent` | Gmail accepted the message, or the user manually resolved an unknown delivery as sent; this is not proof of recipient receipt |
| `cancelled` / `expired` | No pending delivery for this action; the draft is retained |
| `missed` | More than 60 seconds overdue; request a new schedule |
| `needs_review` / `failed` | Connection, account or draft needs attention; check details before requesting a new preview |
| `delivery_unknown` | Gmail may have accepted a request whose response was lost; no automatic send retry |

For `delivery_unknown`, inspect Gmail Sent manually. Only after checking the outcome, record it with one of:

```powershell
python -m email_agent resolve-delivery --job-id 2 --outcome sent
python -m email_agent resolve-delivery --job-id 2 --outcome not-sent
```

Choose the command matching what you found. Neither command sends email. `not-sent` requires that the original draft still exists; request a fresh preview to try again. The voice agent cannot resolve unknown delivery itself.

A draft in `creating`, `editing` or `write_unknown` after interruption is held for manual inspection. Check Gmail for the result before making a new draft. There is no automatic draft import or reconciliation in version one.

To reconnect an expired account, run `connect` again. Connection changes pause outstanding unclaimed jobs for review. To disconnect and remove the worker:

```powershell
python -m email_agent disconnect
python -m email_agent remove-worker
```

Disconnect revokes the OAuth credential and clears Windows Credential Manager. If revocation fails, the command reports failure and pending jobs remain paused; reconnect or retry once the connection is restored.

## Optional Jev routing

Jev uses TypeSafe's `Choice` for agent selection and `Noul` for memory retrieval in one request. These are typed classification primitives supported by the [TypeSafe API](https://docs.typesafe.ai/introduction). Email planning and composition use the dedicated Gemini email model after routing or handoff.

Add a TypeSafe API key to `.env`, then enable routing after evaluating it with your typical requests:

```dotenv
TYPESAFE_API_KEY=your-key-here
JEV_MODEL=jev-latest
JEV_ROUTING_ENABLED=true
```

Restart Jarvis. Routing calls have a two-second timeout and no retries. Direct Jev email routing requires confidence of at least 0.8. Mixed or unclear requests and low confidence use the general agent; it can gather context and hand off once. When Jev is disabled, missing its key or unavailable, clear email requests and pending address/time replies use a local route to the email specialist. Routine email turns skip the separate memory classifier; requests involving remembered background can still retrieve memory.

Check `.traces/events.jsonl` route events using representative requests: general questions, direct email requests, draft follow-ups, research followed by email, and ambiguous instructions. Compare observed routes, fallback counts and latency before relying on direct routing. Automated tests verify the SDK wire format and fallback behavior; live routing accuracy requires your TypeSafe account and representative inputs. Set `JEV_ROUTING_ENABLED=false` to disable it.

## Data and diagnostics

- Email requires `GEMINI_API_KEY`. The default is `EMAIL_MODEL=gemini-3.8-flash`, with `EMAIL_THINKING_LEVEL=low`; medium and high are configurable. Structured decisions and drafts are validated locally before execution. Calls have a 20-second timeout and 4,096-token output ceiling, with no automatic model fallback or transport retry. This profile is independent of the general assistant's model and its API usage is billed by Gemini.
- Invalid tool arguments receive one correction attempt within the shared step budget. Unexpected failures log the operation stage, exception type, argument names/types and traceback locations in `.logs/jarvis.log`, without argument values or raw provider responses.
- Invalid or truncated model decisions receive one repair attempt before tools execute. Decision traces include the turn, agent, attempt number, output length and provider finish reason, excluding response contents. Failed interpretation never claims that an email action succeeded.
- OAuth credentials stay in Windows Credential Manager. Keep `.env` and the Desktop OAuth client JSON out of version control.
- Email content and approved recipient snapshots are stored as plain text in the local SQLite database and printed for review. Protect the Windows account and disk accordingly. The database and related files are ignored by Git.
- Draft writing sends the instruction, existing draft when editing, and relevant memory context to the configured LLM. Agent planning receives recent conversation. If enabled, Jev receives the utterance, the last eight history entries and the active draft ID. Account passwords and OAuth tokens are never included in model requests.
- Email turns are excluded from automatic memory extraction and session summaries. Routing traces record identifiers, status, confidence and latency rather than subjects, bodies or recipients.
- New application logs and traces rotate at 2 MB with three backups each. Existing historical logs, including `err.txt`, are left intact. Avoid redirecting console output to an unbounded file because the console includes speech transcripts and full previews.
- `python -m pip check` checks installed dependency consistency. `python -m pytest -q` runs offline automated checks. Live Gmail delivery, hotkeys, real microphone interaction and Windows logon scheduling require a configured interactive installation to verify.
