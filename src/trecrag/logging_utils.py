"""Central logging setup so scripts and library code share one format."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console logging, and optionally tee everything to `log_file`.

    The file handler uses full timestamps (the console format keeps just the
    time) because a multi-hour run's log is usually read days later.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers[0].setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
    # These libraries log one line per HTTP request; at ~1000 hits/topic that
    # buries the pipeline's own progress output.
    for noisy in ("httpx", "urllib3", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
