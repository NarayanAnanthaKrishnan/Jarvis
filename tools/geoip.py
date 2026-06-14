import geoip2.database
import requests
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "GeoLite2-City.mmdb"


def get_city_info(ip: str = "auto") -> str:
    if ip == "auto":
        ip = requests.get("https://api.ipify.org", timeout=5).text.strip()

    reader = geoip2.database.Reader(str(DB_PATH))
    try:
        response = reader.city(ip)
        city = response.city.name or "Unknown"
        region = response.subdivisions.most_specific.name or "Unknown"
        country = response.country.name or "Unknown"
        return f"{city}, {region}, {country}"
    finally:
        reader.close()
