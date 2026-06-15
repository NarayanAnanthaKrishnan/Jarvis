import os
import subprocess


APP_REGISTRY = {
    "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "vscode": "C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe",
    "vs code": "C:\\Users\\%USERNAME%\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe",
    "notepad": "notepad.exe",
    "explorer": "explorer.exe",
    "spotify": "C:\\Users\\%USERNAME%\\AppData\\Roaming\\Spotify\\Spotify.exe",
    "terminal": "wt.exe",
    "cmd": "cmd.exe",
    "calculator": "calc.exe",
    "brave": "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
}


def open_app(app_name: str) -> str:
    name = app_name.lower().strip()
    try:
        path = APP_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(APP_REGISTRY))
        return f"App '{app_name}' not found. Available: {available}"
    try:
        resolved = path.replace("%USERNAME%", os.environ.get("USERNAME", ""))
        subprocess.Popen(resolved)
        return f"Opened {app_name}"
    except Exception as e:
        return f"Failed to open {app_name}: {e}"
