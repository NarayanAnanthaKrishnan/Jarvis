from pathlib import Path


PROFILE_PATH = Path(__file__).parent.parent / "profile" / "profile.md"


def load_profile() -> str:
    if not PROFILE_PATH.exists():
        return ""
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content
