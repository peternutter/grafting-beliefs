from __future__ import annotations

import logging
import sys
from pathlib import Path

from why_gen.paths import LOGS_DIR


def setup_logging(
    name: str,
    *,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    reset: bool = True,
) -> logging.Logger:
    if reset:
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        if not path.is_absolute():
            path = LOGS_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(name)
