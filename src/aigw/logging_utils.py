"""Logging setup shared by all components. Logs always go to stderr."""

import logging
import os
import sys


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=(level or os.environ.get("AIGW_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
