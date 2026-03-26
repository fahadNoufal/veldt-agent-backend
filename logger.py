"""
logger.py — Centralized logging configuration for Maison Elara AI.

Usage
-----
    from logger import get_logger, get_thinking_logger
    log = get_logger(__name__)
    think_log = get_thinking_logger()

    log.info("Something happened")
    think_log.debug("Model reasoning: ...")

Outputs
-------
- Console      : INFO and above, colored by level
- logs/agent.log   : DEBUG and above (all app logs, rotates at 5 MB)
- logs/thinking.log: Gemini chain-of-thought/thinking blocks only (rotates at 10 MB)
"""

import logging
import logging.handlers
import os
import sys

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

LOG_DIR        = os.getenv("LOG_DIR",        "logs")
LOG_FILE       = os.getenv("LOG_FILE",       os.path.join(LOG_DIR, "agent.log"))
THINKING_FILE  = os.getenv("THINKING_FILE",  os.path.join(LOG_DIR, "thinking.log"))
LOG_LEVEL      = os.getenv("LOG_LEVEL",      "DEBUG")   # file level
CON_LEVEL      = os.getenv("CON_LEVEL",      "INFO")    # console level

# ─────────────────────────────────────────────────────────────
# ANSI COLOR FORMATTER (console only)
# ─────────────────────────────────────────────────────────────

_COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """Adds ANSI color codes around the level name for console output."""

    BASE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        color = _COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return logging.Formatter(self.BASE_FMT, datefmt=self.DATE_FMT).format(record)


# ─────────────────────────────────────────────────────────────
# ROOT LOGGER SETUP  (called once at import time)
# ─────────────────────────────────────────────────────────────

def _configure_root() -> None:
    """Configure the root logger with file + console handlers."""

    root = logging.getLogger()

    # Avoid duplicate handlers if module is reloaded
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── File handler ──────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # ── Console handler ───────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, CON_LEVEL.upper(), logging.INFO))
    console_handler.setFormatter(ColorFormatter())

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Silence noisy third-party loggers
    for lib in ("httpx", "httpcore", "hpack", "urllib3", "faiss"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def _configure_thinking_logger() -> logging.Logger:
    """
    Dedicated logger for Gemini chain-of-thought / thinking blocks.
    Writes ONLY to logs/thinking.log — never to console or agent.log.
    """
    tlog = logging.getLogger("thinking")

    if tlog.handlers:
        return tlog   # already configured

    tlog.setLevel(logging.DEBUG)
    tlog.propagate = False   # don't bubble up to root (keeps thinking out of agent.log)

    os.makedirs(LOG_DIR, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        THINKING_FILE,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s\n%(message)s\n" + "─" * 80,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    tlog.addHandler(handler)
    return tlog


_configure_root()
_thinking_logger = _configure_thinking_logger()


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a child logger for the given module name."""
    return logging.getLogger(name)


def get_thinking_logger() -> logging.Logger:
    """Return the dedicated thinking/reasoning logger (writes to logs/thinking.log)."""
    return _thinking_logger
