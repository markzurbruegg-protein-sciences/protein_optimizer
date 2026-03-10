"""Tier 3 — Stability ΔΔG Predictions via ThermoMPNN.

Uses ThermoMPNN (Kuhlman Lab) to predict ΔΔG of single-point mutations.
ThermoMPNN is a transfer-learning model built on ProteinMPNN's structure
encoder, fine-tuned on the Megascale thermostability dataset.

Negative ΔΔG = stabilizing mutation.

Dispatches computation to a conda environment (default: 'protopt') that
has the required dependencies (torch, omegaconf, pytorch-lightning).
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Path to the helper script (relative to package root)
_HELPER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "thermompnn_helper.py"


class StabilityDDGStep(BaseStep):
    name = "stability_ddg"
    tier = 3
    title = "Stability ΔΔG"
    description = "Predict mutation stability effects with ThermoMPNN."
    requires = ["predict_structure"]

    def validate_input(self, step_input: StepResult) -> None:
        super().validate_input(step_input)
        has_structure = any(
            c.structure_path or c.metadata.get("structure_path")
            for c in step_input.candidates
        )
        if not has_structure:
            raise ValueError(
                "stability_ddg requires structure data. "
                "Run predict_structure first or provide a PDB."
            )

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        ddg_threshold = config.get("ddg_threshold", -1.0)
        saturation = config.get("saturation_mutagenesis", False)
        target_positions = config.get("positions", [])
        conda_env = config.get("conda_env", "protopt")
        thermompnn_dir = config.get(
            "thermompnn_dir", str(Path.home() / "ThermoMPNN")
        )
        model_weights = config.get(
            "model_weights",
            str(Path(thermompnn_dir) / "models" / "thermoMPNN_default.pt"),
        )
        chain = config.get("chain", "A")
        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            if parent.parent_id is not None:
                continue

            candidates.append(parent)

            pdb_path = parent.structure_path
            if not pdb_path or not Path(pdb_path).exists():
                pdb_path = parent.metadata.get("structure_path", "")
            if not pdb_path or not Path(pdb_path).exists():
                warnings.append(
                    f"{parent.name}: No structure available, skipping ΔΔG."
                )
                continue

            positions = _get_target_positions(
                parent, target_positions, protected, saturation
            )

            if not positions:
                warnings.append(
                    f"{parent.name}: No positions to scan for ΔΔG."
                )
                continue

            logger.info(
                f"{parent.name}: Scanning {len(positions)} positions "
                f"with ThermoMPNN"
            )

            ddg_results = _run_thermompnn(
                pdb_path=pdb_path,
                positions=positions,
                conda_env=conda_env,
                thermompnn_dir=thermompnn_dir,
                model_weights=model_weights,
                chain=chain,
            )

            if not ddg_results:
                warnings.append(
                    f"{parent.name}: ThermoMPNN returned no results. "
                    f"Check that the '{conda_env}' conda env has torch, "
                    f"omegaconf, pytorch-lightning installed."
                )
                continue

            stabilizing = [
                (pos, wt, mut_aa, ddg)
                for pos, wt, mut_aa, ddg in ddg_results
                if ddg < ddg_threshold
            ]

            logger.info(
                f"{parent.name}: {len(stabilizing)} stabilizing mutations "
                f"(ΔΔG < {ddg_threshold}) out of {len(ddg_results)} scored"
            )

            for pos, wt, mut_aa, ddg in stabilizing:
                mut = Mutation(
                    position=pos, wt=wt, mut=mut_aa,
                    source_step=self.name, score=ddg,
                    metadata={"ddg": ddg, "method": "thermompnn"},
                )
                try:
                    variant = parent.apply_mutation(mut)
                    variant.scores["ddg"] = ddg
                    candidates.append(variant)
                except ValueError as e:
                    logger.debug(f"Skipping {wt}{pos}{mut_aa}: {e}")

        return StepResult(
            step_name=self.name, candidates=candidates,
            config_used=config, warnings=warnings,
        )


def _get_target_positions(
    parent: ProteinCandidate,
    specified_positions: list[int],
    protected: set[int],
    saturation: bool,
) -> list[int]:
    """Determine which 1-based positions to scan."""
    if specified_positions:
        return [p for p in specified_positions if p not in protected]
    if saturation:
        return [
            i + 1 for i in range(len(parent.sequence))
            if (i + 1) not in protected
        ]
    # Default: scan all positions (full SSM) — ThermoMPNN is fast enough
    return [
        i + 1 for i in range(len(parent.sequence))
        if (i + 1) not in protected
    ]


def _run_thermompnn(
    pdb_path: str,
    positions: list[int],
    conda_env: str,
    thermompnn_dir: str,
    model_weights: str,
    chain: str,
) -> list[tuple[int, str, str, float]]:
    """Dispatch ThermoMPNN ΔΔG prediction to a conda env via subprocess."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "thermompnn_input.json"
            out_path = Path(tmpdir) / "thermompnn_output.json"

            helper_input = {
                "pdb_path": str(pdb_path),
                "positions": positions,
                "chain": chain,
                "thermompnn_dir": thermompnn_dir,
                "model_weights": model_weights,
            }

            with open(in_path, "w") as f:
                json.dump(helper_input, f)

            helper = str(_HELPER_SCRIPT)
            if not Path(helper).exists():
                logger.error(f"ThermoMPNN helper script not found: {helper}")
                return []

            cmd = [
                "conda", "run", "--no-capture-output", "-n", conda_env,
                "python", helper,
                str(in_path), str(out_path),
            ]

            logger.info(
                f"Dispatching ThermoMPNN to conda env '{conda_env}' "
                f"({len(positions)} positions)..."
            )

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200,
            )

            if result.returncode != 0:
                logger.error(
                    f"ThermoMPNN helper failed:\n{result.stderr[:2000]}"
                )
                return []

            if not out_path.exists():
                logger.error("ThermoMPNN helper produced no output file")
                return []

            with open(out_path) as f:
                output = json.load(f)

            predictions = output.get("predictions", [])
            return [
                (p["position"], p["wt"], p["mut"], p["ddg"])
                for p in predictions
            ]

    except subprocess.TimeoutExpired:
        logger.error("ThermoMPNN scoring timed out after 7200s")
        return []
    except Exception as e:
        logger.error(f"ThermoMPNN subprocess dispatch failed: {e}")
        return []
