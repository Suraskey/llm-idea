"""Runtime settings for the released code.

The development repository used a larger pydantic-settings class covering
model rosters, budgets, and provider credentials. The released packages read
only the three fields below, so this module replaces that class with plain
environment lookups. Nothing here contacts a network service.
"""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
        # Where benchmark_run_dir() / simulator artifacts are written.
        self.RUN_ARTIFACTS_DIR: Path = Path(os.environ.get("RUN_ARTIFACTS_DIR", "runs"))
        # Only the optional database scripts (scripts/score_alien_run.py,
        # scripts/dump_run_artifacts.py, scripts/export_prompt_appendix.py) use this.
        self.DATABASE_URL: str = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/ascension")

    def model_for(self, role: str) -> str:
        """Model id recorded for a role in a run manifest (env MODEL_<ROLE>)."""
        return os.environ.get(f"MODEL_{role.upper()}", "unspecified")


settings = Settings()
