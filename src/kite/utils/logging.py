"""Logging utilities."""

from __future__ import annotations

import logging


_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
