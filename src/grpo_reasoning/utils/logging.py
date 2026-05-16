# src/grpo_reasoning/utils/logging.py
from __future__ import annotations
import logging
from typing import Any
from contextlib import contextmanager

logger = logging.getLogger("grpo_reasoning")

class MetricLogger:
    """Thin wrapper over wandb so the rest of the codebase doesn't import it directly."""

    def __init__(self, project: str, run_name: str, config: dict[str, Any], enabled: bool = True):
        self.enabled = enabled
        if enabled:
            import wandb
            self._wandb = wandb
            self.run = wandb.init(project=project, name=run_name, config=config)
        else:
            self._wandb = None
            self.run = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled:
            self._wandb.log(metrics, step=step)
        else:
            logger.info(f"step={step} {metrics}")

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )