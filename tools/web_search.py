from ddgs import DDGS


def search_web(query: str, num_results: int = 3) -> str:
    with DDGS(timeout=8) as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    if not results:
        return "No results found."
    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        line = f"- {title}: {body} ({href})"
        if len(line) > 150:
            line = line[:147] + "..."
        lines.append(line)
    return "\n".join(lines)
