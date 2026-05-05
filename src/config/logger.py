import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup logging configuration for the entire application"""

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / "app.log", encoding='utf-8'),
            logging.FileHandler(logs_dir / "debug.log", encoding='utf-8')
        ],
        force=True
    )

    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.stream.reconfigure(encoding='utf-8')

    app_logger = logging.getLogger("farmaassistant")
    app_logger.info("Logging configuration completed")

    return app_logger
