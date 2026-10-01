import json
import logging
import traceback

import pyautogui
import pyperclip

from agent.context import TurnContext
from email_agent.models import EmailError, EmailResult
from email_agent.contracts import safe_key
from llm.client import ModelRequestError
from llm.prompts import ULTRA_SYSTEM_PROMPT
from tools.web_search import search_web, fetch_url
from tools.geoip import get_city_info
from tools.weather import get_weather
from tools.datetime_tool import get_datetime
from tools.calculator import calculate
from tools.app_launcher import open_app
from tools.notes import take_note, read_notes, update_note, delete_note
from tools.system_info import get_system_info
from tools.browser import open_url
from tools.clipboard_tool import read_clipboard
from tools.news import get_news
from tools.media import media_control
from tools.screen_ocr import capture_and_ocr
from tools.profile_loader import load_profile


def store_memory(content: str) -> str:
    from memory.store import memory_store
    memory_store.add("semantic", content, metadata={"type": "fact"})
    return f"Remembered: {content}"


def set_reminder(message: str, when: str) -> str:
    from memory.reminders import add_reminder, parse_when
    fire_at = parse_when(when)
    if fire_at is None:
        return f"Could not parse time: '{when}'. Try something like '3pm tomorrow' or 'in 20 minutes'."
    return add_reminder(message, fire_at)


def list_reminders() -> str:
    from memory.reminders import format_pending
    return format_pending()


def read_screen() -> str:
    return capture_and_ocr()


def generate_content(instruction: str, screen_context: str = "", llm=None) -> str:
    if llm is None:
        return "[Generation failed: LLM not available]"
    profile = load_profile()
    from memory.store import memory_store
    memories = memory_store.query("semantic", instruction, n=3)
    parts = []
    if profile:
        parts.append(f"About the user:\n{profile}\n")
    if memories:
        parts.append(f"Relevant context:\n" + "\n".join(f"- {m}" for m in memories))
    if screen_context:
        parts.append(f"Screen content:\n{screen_context}\n")
    parts.append(f"User instruction: {instruction}")
    messages = [
        {"role": "system", "content": ULTRA_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)}
    ]
    response = llm.call_raw(messages, temp=0.3, max_tokens=2000)
    if response is None:
        return "[Generation failed]"
    return response["message"]["content"]


def paste_at_cursor(text: str) -> str:
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    return f"Pasted {len(text)} characters at cursor."


TOOL_MAP = {
    "search_web": search_web,
    "fetch_url": fetch_url,
    "get_city_info": get_city_info,
    "get_weather": get_weather,
    "get_datetime": get_datetime,
    "calculate": calculate,
    "open_app": open_app,
    "take_note": take_note,
    "read_notes": read_notes,
    "get_system_info": get_system_info,
    "open_url": open_url,
    "read_clipboard": read_clipboard,
    "get_news": get_news,
    "media_control": media_control,
    "store_memory": store_memory,
    "update_note": update_note,
    "delete_note": delete_note,
    "set_reminder": set_reminder,
    "list_reminders": list_reminders,
    "read_screen": read_screen,
    "generate_content": generate_content,
    "paste_at_cursor": paste_at_cursor,
}


READ_ONLY_TOOLS = frozenset({"search_web", "fetch_url", "get_city_info", "get_weather", "get_datetime", "calculate", "read_notes", "get_system_info", "read_clipboard", "get_news", "list_reminders", "read_screen", "email_get", "email_list"})
EMAIL_TOOLS = frozenset({"email_draft", "email_get", "email_recipients", "email_prepare", "email_list", "email_cancel"})
GENERAL_TOOLS = frozenset(TOOL_MAP) | {"handoff_email"}


def execute_tool(name: str, args: dict, llm=None, context: TurnContext | None = None, parallel: bool = False) -> str | EmailResult:
    if not isinstance(args, dict) or not isinstance(name, str):
        return "Error: Tool name and arguments have invalid types"
    allowed = EMAIL_TOOLS | {"get_datetime"} if context and context.agent_id == "email" else GENERAL_TOOLS
    if name not in allowed:
        return "Error: Tool is not available to this agent"
    if parallel and name not in READ_ONLY_TOOLS:
        return "Error: Only read-only tools can run in parallel"
    if context:
        context.check_active()
    if name in EMAIL_TOOLS:
        if context is None:
            return "Error: Email requires an active session"
        context.email_touched = True
        try:
            from email_agent.runtime import execute_email
            key = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
            if name not in READ_ONLY_TOOLS and key in context.mutation_results:
                return context.mutation_results[key]
            result = execute_email(name, args, context, llm)
            if name not in READ_ONLY_TOOLS and result.status != "invalid_arguments":
                context.mutation_results[key] = result
            return result
        except InterruptedError:
            raise
        except EmailError as exc:
            return EmailResult("error", str(exc))
        except ModelRequestError as exc:
            return EmailResult("error", str(exc))
        except Exception as exc:
            logging.getLogger("jarvis.email").error("Email failure: turn=%s operation=%s stage=%s type=%s argument_types=%s frames=%s",
                context.turn_id, name, context.email_stage, type(exc).__name__,
                {safe_key(key): type(value).__name__ for key, value in args.items()},
                [(frame.filename, frame.lineno, frame.name) for frame in traceback.extract_tb(exc.__traceback__)])
            return EmailResult("error", f"Email operation could not complete ({type(exc).__name__}). Check email status before retrying.")
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'"
    try:
        if name == "generate_content":
            return fn(
                instruction=args.get("instruction", ""),
                screen_context=args.get("screen_context", ""),
                llm=llm
            )
        key = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if context and name not in READ_ONLY_TOOLS and key in context.mutation_results:
            return context.mutation_results[key]
        result = fn(**args)
        if context and name not in READ_ONLY_TOOLS:
            context.mutation_results[key] = result
        return result
    except Exception as e:
        return f"Error executing {name}: {e}"
