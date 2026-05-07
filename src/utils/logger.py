"""Project logging configuration."""

import sys
from typing import Any

from loguru import logger

from config.config import LOG_PATH

_LOGGER_CONFIGURED = False


def setup_logger(name: str) -> Any:
    """Configure Loguru for structured audit logging.

    Args:
        name: Logical module name to bind to emitted records.

    Returns:
        A configured Loguru logger bound with the module name.
    """
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        logger.remove()
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | <cyan>{extra[module]}</cyan> | "
                "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            ),
            level="INFO",
        )
        logger.add(
            LOG_PATH,
            rotation="10 MB",
            retention="10 days",
            level="DEBUG",
            serialize=True,
        )
        _LOGGER_CONFIGURED = True

    return logger.bind(module=name)


sys_logger = setup_logger("healthcare_system")
