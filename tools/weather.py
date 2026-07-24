import re
import requests
from urllib.parse import quote


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def get_weather(city: str, days: int = 0) -> str:
    if city == "auto":
        from tools.geoip import get_city_info
        location = get_city_info("auto")
        city_name = location.split(",")[0].strip()
    else:
        city_name = city

    encoded = quote(city_name)
    if days > 0:
        days = min(days, 3)
        view_params = "&".join(str(d) for d in range(days + 1))
        url = f"https://wttr.in/{encoded}?m&{view_params}"
        resp = requests.get(url, timeout=10)
        raw = resp.text
        stripped = _strip_ansi(raw)
        lines = [l.strip() for l in stripped.splitlines() if l.strip()]
        weather_lines = [l for l in lines if any(c in l for c in "°%:CloudySunnyRainOvercastClearPartlyFoggyWindHumidity")]
        if weather_lines:
            return "\n".join(weather_lines[:15])
        return "\n".join(lines[:20])
    else:
        url = f"https://wttr.in/{encoded}?format=%l:+%C,+%t,+%w,+%h+humidity&m"
        resp = requests.get(url, timeout=10)
        return resp.text.strip()
