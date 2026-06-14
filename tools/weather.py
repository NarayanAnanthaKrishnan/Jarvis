import requests
from urllib.parse import quote


def get_weather(city: str) -> str:
    if city == "auto":
        from tools.geoip import get_city_info
        location = get_city_info("auto")
        city_name = location.split(",")[0].strip()
        url = f"https://wttr.in/{quote(city_name)}?format=%l:+%C,+%t,+%w,+%h+humidity&m"
    else:
        url = f"https://wttr.in/{quote(city)}?format=%l:+%C,+%t,+%w,+%h+humidity&m"
    resp = requests.get(url, timeout=10)
    return resp.text.strip()
