import logging
import os
from pathlib import Path


def setup_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name or __name__)

    if logger.handlers:
        return logger

    debug_mode = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(level)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(logs_dir / "app.log")
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


default_logger = setup_logger("study_app")
