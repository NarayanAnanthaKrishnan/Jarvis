from ddgs import DDGS


def search_web(query: str, num_results: int = 5) -> str:
    with DDGS(timeout=15) as ddgs:
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
