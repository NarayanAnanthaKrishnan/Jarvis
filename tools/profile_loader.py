from pathlib import Path


PROFILE_PATH = Path(__file__).parent.parent / "profile" / "profile.md"
_cache: str | None = None


def load_profile() -> str:
    global _cache
    if _cache is not None:
        return _cache
    if not PROFILE_PATH.exists():
        _cache = ""
        return _cache
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        _cache = f.read().strip()
    return _cache
