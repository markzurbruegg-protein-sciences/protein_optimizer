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
        import time as _time

        n_in = len(step_input.candidates)
        logger.info(f"[Tier {self.tier}] Starting step: {self.title}  (input: {n_in} candidates)")
        t0 = _time.time()
        self.validate_input(step_input)
        result = self.run(step_input, config)
        elapsed = _time.time() - t0

        n_out = len(result.candidates)
        n_new = n_out - n_in
        # Count new mutations across all new candidates
        n_mut = sum(len(c.mutations) for c in result.candidates if c.parent_id is not None)

        logger.info(
            f"[Tier {self.tier}] Completed step: {self.title}  "
            f"({n_out} candidates, +{n_new} new, {n_mut} mutations)  "
            f"[{elapsed:.1f}s]"
        )

        if result.warnings:
            for w in result.warnings:
                logger.warning(f"  ⚠ {self.name}: {w}")

        # Log score summary for new candidates
        scored = [c for c in result.candidates if c.scores]
        if scored:
            score_keys = set()
            for c in scored:
                score_keys.update(c.scores.keys())
            for key in sorted(score_keys):
                vals = [c.scores[key] for c in scored if key in c.scores]
                if vals:
                    logger.info(
                        f"  scores/{key}: min={min(vals):.4f}  "
                        f"max={max(vals):.4f}  mean={sum(vals)/len(vals):.4f}  "
                        f"(n={len(vals)})"
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
