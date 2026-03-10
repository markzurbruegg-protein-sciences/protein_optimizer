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
    """Resolve step-specific config from either top-level YAML key or nested 'steps'.

    Priority: top-level key (user YAML) > config['steps'][step_name] (defaults).
    When both exist, the top-level values override the defaults.
    """
    result: dict[str, Any] = {}

    # Start with defaults from config['steps'][step_name] if present
    steps_section = config.get("steps", {})
    if isinstance(steps_section, dict) and step_name in steps_section:
        result = dict(steps_section[step_name])

    # Override with top-level key (user YAML layout)
    if step_name in config and isinstance(config[step_name], dict):
        result.update(config[step_name])

    return result


class Pipeline:
    """Orchestrates the execution of pipeline steps.

    Respects step dependencies, passes StepResults between steps,
    and saves intermediate results to disk.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.results: dict[str, StepResult] = {}

    def _setup_file_logging(self, output_dir: Path) -> None:
        """Add a file handler so every log message is written to pipeline.log."""
        log_path = output_dir / "pipeline.log"
        root = logging.getLogger()
        # Avoid duplicate file handlers on re-runs
        for h in root.handlers[:]:
            if isinstance(h, logging.FileHandler) and "pipeline.log" in str(getattr(h, 'baseFilename', '')):
                root.removeHandler(h)
        fh = logging.FileHandler(str(log_path), mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(fh)
        logger.info(f"Logging to {log_path}")

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

        # Set up file logging into the output directory
        self._setup_file_logging(output_dir)

        # Load initial input
        step_input = load_input(input_path)
        self.results["input"] = step_input

        # Get step names from config
        step_names = self.config["pipeline"]["steps"]
        logger.info("=" * 72)
        logger.info(f"PIPELINE START — {len(step_names)} steps on {Path(input_path).name}")
        logger.info(f"  Input: {len(step_input.candidates)} candidate(s)")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Steps: {', '.join(step_names)}")
        logger.info("=" * 72)

        # Ensure all step modules are imported (triggers registration)
        _import_all_steps()

        import time as _time
        t_pipeline = _time.time()

        current = step_input
        for step_idx, step_name in enumerate(step_names, 1):
            step = get_step(step_name)
            logger.info("─" * 72)
            logger.info(
                f"Step {step_idx}/{len(step_names)}: {step.title} "
                f"({step.name})  [Tier {step.tier}]"
            )
            logger.info("─" * 72)

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
        self._finalize(current, output_dir, input_path=Path(input_path))

        total_time = _time.time() - t_pipeline
        logger.info("=" * 72)
        logger.info(
            f"PIPELINE COMPLETE — {len(current.candidates)} final candidates  "
            f"[{total_time:.1f}s total]"
        )
        n_with_scores = sum(1 for c in current.candidates if c.scores)
        n_with_muts = sum(1 for c in current.candidates if c.mutations)
        logger.info(f"  Candidates with scores: {n_with_scores}")
        logger.info(f"  Candidates with mutations: {n_with_muts}")
        if hasattr(current, 'candidates') and current.candidates:
            agg = [c.scores.get('aggregate_score', 0) for c in current.candidates if 'aggregate_score' in c.scores]
            if agg:
                agg_sorted = sorted(agg, reverse=True)
                logger.info(f"  Top-5 aggregate scores: {[round(s, 4) for s in agg_sorted[:5]]}")
        logger.info(f"  Results: {output_dir}")
        logger.info("=" * 72)
        return current

    def _finalize(
        self,
        result: StepResult,
        output_dir: Path,
        *,
        input_path: Path | None = None,
    ) -> None:
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

        # Determine report output directory: same folder as the input FASTA
        report_dir = output_dir
        if input_path:
            fasta_dir = Path(input_path).resolve().parent
            if fasta_dir.is_dir():
                report_dir = fasta_dir

        # ── v2 report (3D viewer + metrics) ──
        try:
            from protein_optimizer.reporting.report_v2 import generate_report_v2

            # Find PDB file from structure prediction
            pdb_path = None
            struct_result = self.results.get("predict_structure")
            if struct_result:
                for c in struct_result.candidates:
                    sp = c.metadata.get("structure_path", "")
                    if sp and Path(sp).exists():
                        pdb_path = sp
                        break
            # Fallback: look in structures/
            if not pdb_path:
                struct_dir = output_dir / "structures"
                for ext in ("*.pdb", "*.PDB"):
                    pdbs = list(struct_dir.glob(ext))
                    if pdbs:
                        pdb_path = str(pdbs[0])
                        break

            parent_name = ""
            for c in result.candidates:
                if c.parent_id is None:
                    parent_name = c.name
                    break
            report_title = f"{parent_name} — Protein Optimization Report"

            report_path = report_dir / "report.html"
            generate_report_v2(
                results=self.results,
                output_path=report_path,
                pdb_path=pdb_path,
                title=report_title,
                config=self.config,
            )
            logger.info(f"Report v2 saved to {report_path}")

            # Also save a copy in the results directory if different
            if report_dir != output_dir:
                generate_report_v2(
                    results=self.results,
                    output_path=output_dir / "report.html",
                    pdb_path=pdb_path,
                    title=report_title,
                    config=self.config,
                )
        except Exception as e:
            logger.warning(f"Report v2 generation failed: {e}", exc_info=True)
            # Fallback to v1
            try:
                from protein_optimizer.reporting.html_report import generate_html_report
                generate_html_report(
                    results=self.results,
                    output_path=output_dir / "report.html",
                    config=self.config,
                )
            except Exception as e2:
                logger.warning(f"Report generation skipped: {e2}")


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
