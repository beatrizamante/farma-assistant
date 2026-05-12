import logging
import sys
from pathlib import Path


def setup_logging() -> logging.Logger:
    """Setup logging configuration for the entire application"""

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    app_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    debug_handler = logging.FileHandler(logs_dir / "debug.log", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, app_handler, debug_handler],
        force=True,
    )

    for uvicorn_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv = logging.getLogger(uvicorn_logger_name)
        uv.handlers = [console_handler, app_handler, debug_handler]
        uv.propagate = False

    app_logger = logging.getLogger("farmaassistant")
    app_logger.info("Logging configuration completed")

    return app_logger
