"""Pipeline orchestrator — runs steps in sequence with dependency resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from protein_optimizer.config import load_config
from protein_optimizer.io_utils import load_input
from protein_optimizer.models import StepResult
from protein_optimizer.steps.base import get_step, list_steps

logger = logging.getLogger(__name__)


def _get_step_config(config: dict[str, Any], step_name: str) -> dict[str, Any]:
    """Resolve step-specific config from either nested 'steps' key or top-level."""
    # Try config['steps'][step_name] first (DEFAULT_CONFIG layout)
    steps_section = config.get("steps", {})
    if isinstance(steps_section, dict) and step_name in steps_section:
        return dict(steps_section[step_name])
    # Fallback: top-level key (YAML layout)
    if step_name in config and isinstance(config[step_name], dict):
        return dict(config[step_name])
    return {}


class Pipeline:
    """Orchestrates the execution of pipeline steps.

    Respects step dependencies, passes StepResults between steps,
    and saves intermediate results to disk.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.results: dict[str, StepResult] = {}

    @classmethod
    def from_config_file(cls, path: str | Path) -> Pipeline:
        config = load_config(path)
        return cls(config)

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> StepResult:
        """Execute the full pipeline.

        Args:
            input_path: Path to FASTA or StepResult JSON.
            output_dir: Directory for intermediate and final outputs.

        Returns:
            The StepResult from the last step.
        """
        output_dir = Path(output_dir or self.config["global"].get("output_dir", "./results"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load initial input
        step_input = load_input(input_path)
        self.results["input"] = step_input

        # Get step names from config
        step_names = self.config["pipeline"]["steps"]
        logger.info(f"Pipeline will execute {len(step_names)} steps: {step_names}")

        # Ensure all step modules are imported (triggers registration)
        _import_all_steps()

        current = step_input
        for step_name in step_names:
            step = get_step(step_name)

            # Check dependencies
            for dep in step.requires:
                if dep not in self.results:
                    logger.warning(
                        f"Step '{step_name}' requires '{dep}' which hasn't run. "
                        f"Attempting to run '{dep}' first."
                    )
                    dep_step = get_step(dep)
                    dep_config = _get_step_config(self.config, dep)
                    dep_config["_global"] = self.config.get("global", {})
                    dep_config["_prior_results"] = self.results
                    dep_result = dep_step.execute(current, dep_config)
                    self.results[dep] = dep_result
                    dep_result.save(output_dir / f"{dep}.json")

            # Run step
            step_config = _get_step_config(self.config, step_name)
            step_config["_global"] = self.config.get("global", {})
            step_config["_prior_results"] = self.results

            current = step.execute(current, step_config)
            self.results[step_name] = current

            # Save intermediate result
            current.save(output_dir / f"{step_name}.json")
            logger.info(f"Saved {step_name} result to {output_dir / f'{step_name}.json'}")

        # Aggregate scores and generate report
        self._finalize(current, output_dir)

        logger.info(f"Pipeline complete. {len(current.candidates)} final candidates.")
        return current

    def _finalize(self, result: StepResult, output_dir: Path) -> None:
        """Run score aggregation, filtering, and report generation."""
        try:
            from protein_optimizer.scoring.aggregator import aggregate_scores
            weights = self.config.get("scoring", {}).get("weights")
            aggregate_scores(result, weights=weights)
            logger.info("Score aggregation complete.")
        except Exception as e:
            logger.warning(f"Score aggregation skipped: {e}")

        try:
            from protein_optimizer.scoring.filters import filter_candidates
            filter_config = self.config.get("scoring", {}).get("filters", {})
            passed, failed = filter_candidates(result, filter_config)
            logger.info(f"Filtering: {len(passed)} passed, {len(failed)} failed.")
        except Exception as e:
            logger.warning(f"Filtering skipped: {e}")

        try:
            from protein_optimizer.reporting.html_report import generate_html_report
            report_path = output_dir / "report.html"
            generate_html_report(
                results=self.results,
                output_path=report_path,
                config=self.config,
            )
        except Exception as e:
            logger.warning(f"Report generation skipped: {e}")


def _import_all_steps() -> None:
    """Import all step modules to trigger BaseStep.__init_subclass__ registration."""
    import importlib
    step_modules = [
        "protein_optimizer.steps.cysteine_scan",
        "protein_optimizer.steps.motif_scan",
        "protein_optimizer.steps.sequence_complexity",
        "protein_optimizer.steps.find_homologs",
        "protein_optimizer.steps.consensus_design",
        "protein_optimizer.steps.pssm_analysis",
        "protein_optimizer.steps.predict_structure",
        "protein_optimizer.steps.stability_ddg",
        "protein_optimizer.steps.disulfide_design",
        "protein_optimizer.steps.cavity_fill",
        "protein_optimizer.steps.surface_patch",
        "protein_optimizer.steps.e1_score",
        "protein_optimizer.steps.esm1v_score",
        "protein_optimizer.steps.proteinmpnn_design",
        "protein_optimizer.steps.esmif1_score",
        "protein_optimizer.steps.combine_variants",
        "protein_optimizer.steps.rfdiffusion_diversify",
        "protein_optimizer.steps.design_validate",
        "protein_optimizer.steps.motif_scaffold",
    ]
    for module_name in step_modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            logger.debug(f"Could not import {module_name}: {e}")
