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
from memory.store import memory_store
from memory.reminders import add_reminder, parse_when, format_pending
from llm.prompts import ULTRA_SYSTEM_PROMPT
from tools.profile_loader import load_profile
import pyperclip
import pyautogui


def store_memory(content: str) -> str:
    memory_store.add("semantic", content, metadata={"type": "fact"})
    return f"Remembered: {content}"


def set_reminder(message: str, when: str) -> str:
    fire_at = parse_when(when)
    if fire_at is None:
        return f"Could not parse time: '{when}'. Try something like '3pm tomorrow' or 'in 20 minutes'."
    return add_reminder(message, fire_at)


def list_reminders() -> str:
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


def execute_tool(name: str, args: dict, llm=None) -> str:
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
        return fn(**args)
    except Exception as e:
        return f"Error executing {name}: {e}"
