import keyboard


ACTION_MAP = {
    "play": "play/pause",
    "pause": "play/pause",
    "play/pause": "play/pause",
    "toggle": "play/pause",
    "next": "next track",
    "skip": "next track",
    "previous": "previous track",
    "back": "previous track",
    "prev": "previous track",
    "volume up": "volume up",
    "volume down": "volume down",
    "mute": "volume mute",
}


def media_control(action: str) -> str:
    key = action.lower().strip()
    mapped = ACTION_MAP.get(key)
    if mapped is None:
        return f"Unknown media action: {action}"
    try:
        if key in ("volume up", "volume down"):
            keyboard.send(mapped, do_press=True, do_release=True)
            keyboard.send(mapped, do_press=True, do_release=True)
        else:
            keyboard.send(mapped, do_press=True, do_release=True)
        return "Done."
    except Exception as e:
        return f"Media control failed: {e}"
