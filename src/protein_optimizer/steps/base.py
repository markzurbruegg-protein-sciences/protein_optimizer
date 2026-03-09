"""Abstract base class for all pipeline steps.

Every step inherits from BaseStep and implements the `run` method.
Steps declare their name, tier, dependencies, and CLI metadata.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from protein_optimizer.models import StepResult

logger = logging.getLogger(__name__)


# Global step registry — populated by BaseStep.__init_subclass__
_STEP_REGISTRY: dict[str, type[BaseStep]] = {}


class BaseStep(ABC):
    """Base class for all pipeline steps.

    Subclasses must define:
        name: str           — unique step identifier (used as CLI subcommand)
        tier: int           — tier number (1-5)
        title: str          — human-readable title
        description: str    — one-line description (used in CLI --help)
        requires: list[str] — names of steps that must run before this one

    And implement:
        run(input: StepResult, config: dict) -> StepResult
    """

    name: str = ""
    tier: int = 0
    title: str = ""
    description: str = ""
    requires: list[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses."""
        super().__init_subclass__(**kwargs)
        if cls.name:  # skip abstract intermediaries
            _STEP_REGISTRY[cls.name] = cls

    @abstractmethod
    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        """Execute this step.

        Args:
            step_input: StepResult from a previous step (or initial FASTA load).
            config: Step-specific configuration dict from the pipeline YAML.

        Returns:
            A new StepResult containing the original candidates plus any
            new/modified candidates produced by this step.
        """
        ...

    def validate_input(self, step_input: StepResult) -> None:
        """Optional input validation. Override to add checks."""
        if not step_input.candidates:
            raise ValueError(f"Step '{self.name}' received empty input (no candidates).")

    def execute(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        """Validate, run, and log. Called by the pipeline orchestrator."""
        logger.info(f"[Tier {self.tier}] Starting step: {self.title}")
        self.validate_input(step_input)
        result = self.run(step_input, config)
        logger.info(
            f"[Tier {self.tier}] Completed step: {self.title} "
            f"({len(result.candidates)} candidates)"
        )
        return result


def get_step(name: str) -> BaseStep:
    """Instantiate a step by name."""
    if name not in _STEP_REGISTRY:
        available = ", ".join(sorted(_STEP_REGISTRY.keys()))
        raise KeyError(f"Unknown step '{name}'. Available steps: {available}")
    return _STEP_REGISTRY[name]()


def list_steps() -> dict[str, type[BaseStep]]:
    """Return the full step registry."""
    return dict(_STEP_REGISTRY)
