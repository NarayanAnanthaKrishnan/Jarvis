from tools.web_search import search_web
from tools.geoip import get_city_info
from tools.weather import get_weather
from tools.datetime_tool import get_datetime
from tools.calculator import calculate
from tools.app_launcher import open_app
from tools.notes import take_note, read_notes
from tools.system_info import get_system_info
from tools.browser import open_url
from tools.clipboard_tool import read_clipboard
from tools.news import get_news
from tools.media import media_control


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_info",
            "description": "Get city, region, and country for an IP address",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {
                        "type": "string",
                        "description": "IP address to look up (default: auto-detect current IP)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Safely evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '(2 + 3) * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Launch a desktop application on Windows",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the app to open (chrome, vscode, notepad, explorer, spotify, terminal, cmd, calculator, brave)"
                    }
                },
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_note",
            "description": "Save a note or reminder to the user's notes file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "The note content to save"}
                },
                "required": ["note"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_notes",
            "description": "Read the user's saved notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "description": "Number of recent notes to return. Default 5."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get current CPU usage, RAM usage, and disk space on this Windows machine.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL or named bookmark in the browser. Use for GitHub, Gmail, YouTube, or any website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL or bookmark name like 'github', 'youtube', 'gmail'"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_clipboard",
            "description": "Read the current contents of the user's clipboard.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Fetch top news headlines. Topics: general, tech, science, us.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "News topic: general, tech, science, or us. Default general."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": "Control media playback. Actions: play, pause, next, previous, volume up, volume down, mute.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Media action: play, pause, next, previous, volume up, volume down, mute"}
                },
                "required": ["action"]
            }
        }
    },
]

TOOL_MAP = {
    "search_web": search_web,
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
}


def execute_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'"
    try:
        return fn(**args)
    except Exception as e:
        return f"Error executing {name}: {e}"
