"""Structured JSON logging. No print() anywhere in src/ — everything goes
through here, either directly (structlog) or via the stdlib logger that
third-party libraries (SQLAlchemy, Alembic) already use.
"""

from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path
from typing import Any, cast

import structlog
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LOGGING_CONFIG = _PROJECT_ROOT / "config" / "logging.yaml"

_RESERVED_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JSONFormatter(logging.Formatter):
    """Renders stdlib LogRecords as single-line JSON, so every log line —
    ours and third-party libraries' — shares one format regardless of
    which logging API produced it."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_RECORD_ATTRS}
        if extras:
            payload["context"] = extras
        return json.dumps(payload, default=str)


def configure_logging(config_path: Path | str = _DEFAULT_LOGGING_CONFIG) -> None:
    """Load config/logging.yaml and wire structlog through stdlib logging.

    Call once at process start (CLI entrypoints, API startup, pipeline
    scripts). structlog funnels into stdlib logging rather than rendering
    independently, so the JSON formatter above governs everything
    uniformly instead of maintaining two separate logging configs.
    """
    with Path(config_path).open("r", encoding="utf-8") as fh:
        dict_config = yaml.safe_load(fh)
    logging.config.dictConfig(dict_config)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.render_to_log_kwargs,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    # structlog.get_logger()'s stub return type is Any (it's a proxy
    # object resolved lazily); cast reflects the actual runtime type once
    # configure_logging() has set wrapper_class=BoundLogger.
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
