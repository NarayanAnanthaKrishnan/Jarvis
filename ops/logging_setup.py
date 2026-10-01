import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging() -> None:
    path = Path(__file__).resolve().parent.parent / ".logs"
    path.mkdir(exist_ok=True)
    logger = logging.getLogger()
    handler = next((h for h in logger.handlers if getattr(h, "_jarvis", False)), None)
    if handler is None:
        handler = RotatingFileHandler(path / "jarvis.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        handler._jarvis = True
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logging.getLogger("jarvis.audio").setLevel(logging.INFO)
    stt_logger = logging.getLogger("realtimestt")
    if not any(getattr(h, "_jarvis", False) for h in stt_logger.handlers):
        stt_handler = RotatingFileHandler(path / "speech.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        stt_handler._jarvis = True
        stt_handler.setLevel(logging.WARNING)
        stt_handler.setFormatter(handler.formatter)
        stt_logger.addHandler(stt_handler)
