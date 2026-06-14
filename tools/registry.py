from tools.web_search import search_web
from tools.geoip import get_city_info
from tools.weather import get_weather


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
    }
]

TOOL_MAP = {
    "search_web": search_web,
    "get_city_info": get_city_info,
    "get_weather": get_weather
}


def execute_tool(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'"
    try:
        return fn(**args)
    except Exception as e:
        return f"Error executing {name}: {e}"
