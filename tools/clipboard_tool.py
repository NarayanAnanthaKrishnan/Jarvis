import pyperclip


def read_clipboard() -> str:
    try:
        content = pyperclip.paste()
        if not content or not content.strip():
            return "Clipboard is empty."
        if len(content) > 500:
            content = content[:500] + "... (truncated)"
        return content
    except Exception as e:
        return f"Could not read clipboard: {e}"
