import webbrowser


BOOKMARK_REGISTRY = {
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "youtube": "https://youtube.com",
    "localhost": "http://localhost:3000",
    "jarvis": "http://localhost:8000",
    "linkedin": "https://linkedin.com",
    "claude": "https://claude.ai",
}


def open_url(url: str) -> str:
    try:
        key = url.lower().strip()
        if key in BOOKMARK_REGISTRY:
            url = BOOKMARK_REGISTRY[key]
        elif not url.startswith(("http://", "https://")):
            url = "https://" + url
        webbrowser.open(url)
        return f"Opened {url} in browser."
    except Exception as e:
        return f"Failed to open URL: {e}"
