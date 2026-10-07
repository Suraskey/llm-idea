"""Project-wide logger factory. Level pulled from Settings.LOG_LEVEL.

Phase 2.0 will likely replace basicConfig with structured JSON logging for the
LLM call audit trail (SCOPE §22.2 reproducibility). Keep get_logger(name)
signatures stable so later refactors don't cascade.
"""

from __future__ import annotations

import logging

from ascension.common.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Call with ``__name__`` at module scope so log lines are tagged with the
    importing module's dotted path (e.g., ``ascension.common.db``).
    """
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name)
