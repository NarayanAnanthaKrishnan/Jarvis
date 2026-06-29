import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def search_web(query: str, num_results: int = 8) -> str:
    with DDGS(timeout=12) as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    if not results:
        return "No results found."
    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"- {title}: {body} ({href})")
    return "\n".join(lines)


def fetch_url(url: str) -> str:
    try:
        headers = {"User-Agent": _USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = "\n".join(lines)
        if len(content) > 4000:
            content = content[:4000] + "\n...[truncated]"
        return content if content else "Page contained no readable text."
    except Exception as e:
        return f"Error fetching URL: {e}"
