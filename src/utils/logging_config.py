"""
logging_config.py

Centralized logging configuration for AWS-CapacityForecaster project.
Provides consistent logging setup across all modules with both console and file output.

Usage:
    from src.utils.logging_config import setup_logger
    logger = setup_logger(__name__, 'data_generation')
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file_prefix: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with both console and file handlers.

    Args:
        name: Logger name (typically __name__)
        log_file_prefix: Prefix for log file (e.g., 'data_generation' -> 'data_generation_20240123_143052.log')
        level: Logging level (default INFO)
        log_dir: Directory for log files (default: project_root/logs)

    Returns:
        Configured logger instance
    """
    # Determine project root and log directory
    if log_dir is None:
        # Try to find project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # src/utils/logging_config.py -> project root
        log_dir = project_root / 'logs'
    else:
        log_dir = Path(log_dir)

    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{log_file_prefix}_{timestamp}.log'

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler - concise logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log initial message
    logger.info(f"Logger initialized. Log file: {log_file}")

    return logger


def get_log_file_path(logger: logging.Logger) -> Optional[Path]:
    """
    Get the log file path from a logger.

    Args:
        logger: Logger instance

    Returns:
        Path to log file or None if no file handler
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


class LogSection:
    """
    Context manager for logging section headers.

    Usage:
        with LogSection(logger, "Data Generation"):
            # do work
    """

    def __init__(self, logger: logging.Logger, section_name: str, width: int = 70):
        self.logger = logger
        self.section_name = section_name
        self.width = width
        self.start_time = None

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info("=" * self.width)
        self.logger.info(f" {self.section_name}")
        self.logger.info("=" * self.width)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"[OK] {self.section_name} completed in {duration:.2f}s")
        else:
            self.logger.error(f"[FAILED] {self.section_name} failed after {duration:.2f}s: {exc_val}")
        self.logger.info("-" * self.width)
        return False  # Don't suppress exceptions
