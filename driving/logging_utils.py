"""Structured logging with a per-run correlation id.

Library code gets a logger; the CLI configures handlers once. Set
``EVENT_CBF_JSON_LOGS=1`` (or pass ``json_logs=True``) for machine-readable
logs that can be shipped to an aggregator.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from driving.io_utils import utc_now

_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("event_cbf_run_id", default=None)

_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}


def set_run_id(run_id: Optional[str]) -> None:
    _run_id.set(run_id)


def get_run_id() -> Optional[str]:
    return _run_id.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        run_id = _run_id.get()
        if run_id:
            payload["run_id"] = run_id
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, allow_nan=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        run_id = _run_id.get()
        prefix = f"[{run_id}] " if run_id else ""
        base = f"{record.levelname:<7} {prefix}{record.name}: {record.getMessage()}"
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        if extras:
            rendered = " ".join(f"{k}={v!r}" for k, v in sorted(extras.items()))
            base = f"{base} | {rendered}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(
    *,
    level: str = "INFO",
    json_logs: bool = False,
    log_file: Optional[Path | str] = None,
    force: bool = False,
) -> None:
    root = logging.getLogger()
    if getattr(root, "_event_cbf_configured", False) and not force:
        root.setLevel(level.upper())
        return
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())
    root.addHandler(handler)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter() if json_logs else ConsoleFormatter())
        root.addHandler(file_handler)
    root.setLevel(level.upper())
    root._event_cbf_configured = True  # type: ignore[attr-defined]
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"event": event, **fields})


def json_logs_requested() -> bool:
    return os.environ.get("EVENT_CBF_JSON_LOGS", "").lower() in {"1", "true", "yes", "on"}
