"""
utils.py -- Shared utility functions used across the project.

Provides:
  - setup_logger : Create a file + console logger
  - timestamp    : YYYYMMDD_HHMMSS string
  - ensure_dir   : Create directories if they don't exist
"""

import os
import logging
from datetime import datetime


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to both a file and the console.

    Parameters
    ----------
    name     : str            - Logger name (e.g. 'autoencoder')
    log_file : str            - Absolute path to the log file
    level    : logging level  - Default INFO

    Returns
    -------
    logging.Logger
    """
    ensure_dir(os.path.dirname(log_file))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def timestamp() -> str:
    """Return current timestamp as YYYYMMDD_HHMMSS string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    """Create directory (and parents) if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)
