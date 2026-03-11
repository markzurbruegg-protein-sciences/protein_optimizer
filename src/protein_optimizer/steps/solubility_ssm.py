"""Tier 3 — Targeted Solubility SSM via ProtSolM.

Performs in silico site-saturation mutagenesis at surface-exposed
hydrophobic and aggregation-prone region (APR) positions using ProtSolM
(Tan et al., IEEE BIBM 2024).  The scan identifies mutations that
improve predicted solubility.

Only positions most relevant to solubility are scanned (surface-exposed
hydrophobics + APR residues), keeping compute manageable (~2-5 min on
A100 for a 300-residue protein).

Dispatches to a conda environment with ESM2, torch, torch_geometric,
and ProtSolM dependencies.
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

_HELPER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "protsolm_helper.py"


class SolubilitySsmStep(BaseStep):
    name = "solubility_ssm"
    tier = 3
    title = "Solubility SSM (ProtSolM)"
    description = "Targeted saturation mutagenesis for solubility improvement."
    requires = ["predict_structure", "protein_characterization"]

    def validate_input(self, step_input: StepResult) -> None:
        super().validate_input(step_input)
        has_structure = any(
            c.structure_path or c.metadata.get("structure_path")
            for c in step_input.candidates
        )
        if not has_structure:
            raise ValueError(
                "solubility_ssm requires structure data. "
                "Run predict_structure first or provide a PDB."
            )

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        conda_env = config.get("conda_env", "protopt")
        min_delta = config.get("min_delta_solubility", 0.01)
        batch_size = config.get("batch_size", 32)

        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        prior_results: dict[str, StepResult] = config.get("_prior_results", {})

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
                    f"{parent.name}: No structure available, skipping solubility SSM."
                )
                continue

            # Collect target positions from characterization + surface_patch
            target_positions = _collect_target_positions(
                parent, prior_results, protected
            )

            if not target_positions:
                warnings.append(
                    f"{parent.name}: No target positions for solubility SSM."
                )
                continue

            logger.info(
                f"{parent.name}: Scanning {len(target_positions)} positions "
                f"× 19 AAs with ProtSolM ({len(target_positions) * 19} variants)"
            )

            # Generate mutant sequences
            wt_seq = parent.sequence
            mutant_entries = []  # (position, wt_aa, mut_aa, sequence)
            for pos in target_positions:
                idx = pos - 1  # 0-based
                wt_aa = wt_seq[idx]
                for mut_aa in AMINO_ACIDS:
                    if mut_aa == wt_aa:
                        continue
                    seq_list = list(wt_seq)
                    seq_list[idx] = mut_aa
                    mutant_entries.append(
                        (pos, wt_aa, mut_aa, "".join(seq_list))
                    )

            # Score WT first, then all mutants
            all_sequences = [wt_seq] + [e[3] for e in mutant_entries]

            predictions = _run_protsolm_batch(
                pdb_path=str(pdb_path),
                sequences=all_sequences,
                conda_env=conda_env,
            )

            if not predictions:
                warnings.append(
                    f"{parent.name}: ProtSolM batch scoring failed."
                )
                continue

            wt_prob = predictions[0]["probability"]
            logger.info(f"{parent.name}: WT P(soluble) = {wt_prob:.3f}")

            # Find beneficial mutations
            beneficial = []
            for i, (pos, wt_aa, mut_aa, _seq) in enumerate(mutant_entries):
                pred = predictions[i + 1]  # +1 because WT is first
                delta_sol = pred["probability"] - wt_prob
                if delta_sol >= min_delta:
                    beneficial.append({
                        "position": pos,
                        "wt": wt_aa,
                        "mut": mut_aa,
                        "delta_solubility": delta_sol,
                        "probability": pred["probability"],
                    })

            beneficial.sort(key=lambda x: x["delta_solubility"], reverse=True)

            logger.info(
                f"{parent.name}: {len(beneficial)} solubility-improving mutations "
                f"(ΔSol ≥ {min_delta}) out of {len(mutant_entries)} scanned"
            )

            # Store summary in parent metadata
            parent.metadata["solubility_ssm_summary"] = {
                "wt_probability": wt_prob,
                "n_beneficial": len(beneficial),
                "n_total_scanned": len(mutant_entries),
                "n_positions": len(target_positions),
                "top_5_mutations": beneficial[:5],
            }

            # Create candidate variants for beneficial mutations
            for entry in beneficial:
                mut = Mutation(
                    position=entry["position"],
                    wt=entry["wt"],
                    mut=entry["mut"],
                    source_step=self.name,
                    score=entry["delta_solubility"],
                    metadata={
                        "delta_solubility": entry["delta_solubility"],
                        "probability_soluble": entry["probability"],
                        "method": "protsolm_ssm",
                    },
                )
                try:
                    variant = parent.apply_mutation(mut)
                    variant.scores["delta_solubility"] = entry["delta_solubility"]
                    candidates.append(variant)
                except ValueError as e:
                    logger.debug(f"Skipping {entry['wt']}{entry['position']}{entry['mut']}: {e}")

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _collect_target_positions(
    parent: ProteinCandidate,
    prior_results: dict[str, StepResult],
    protected: set[int],
) -> list[int]:
    """Collect positions relevant to solubility: surface hydrophobics + APRs."""
    target = set()
    seq = parent.sequence
    char = parent.metadata.get("characterization", {})

    HYDROPHOBIC = set("ILMFVW")

    # 1. APR positions from characterization aggregation analysis
    tango_aprs = char.get("tango_aprs", [])
    for apr in tango_aprs:
        for pos in range(apr.get("start", 0), apr.get("end", 0) + 1):
            if 1 <= pos <= len(seq) and pos not in protected:
                target.add(pos)

    # 2. Nucleation core positions (highest aggregation propensity)
    tango_cores = char.get("tango_nucleation_cores", [])
    for core in tango_cores:
        for pos in range(core.get("start", 0), core.get("end", 0) + 1):
            if 1 <= pos <= len(seq) and pos not in protected:
                target.add(pos)

    # 3. Surface-exposed hydrophobic residues from surface_patch step
    if "surface_patch" in prior_results:
        sp_result = prior_results["surface_patch"]
        for c in sp_result.candidates:
            if c.parent_id is not None:
                for m in c.mutations:
                    if m.source_step == "surface_patch":
                        target.add(m.position)

    # 4. If we don't have enough targets from structural data,
    #    add sequence-based hydrophobic positions
    if len(target) < 10:
        for i, aa in enumerate(seq):
            if aa in HYDROPHOBIC and (i + 1) not in protected:
                target.add(i + 1)

    # Remove protected residues
    target -= protected

    return sorted(target)


def _run_protsolm_batch(
    pdb_path: str,
    sequences: list[str],
    conda_env: str,
) -> list[dict] | None:
    """Dispatch batch ProtSolM prediction to conda env."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "protsolm_batch_input.json"
            out_path = Path(tmpdir) / "protsolm_batch_output.json"

            batch_input = {
                "pdb_path": pdb_path,
                "sequences": sequences,
            }

            with open(in_path, "w") as f:
                json.dump(batch_input, f)

            helper = str(_HELPER_SCRIPT)
            if not Path(helper).exists():
                logger.error(f"ProtSolM helper not found: {helper}")
                return None

            cmd = [
                "conda", "run", "--no-capture-output", "-n", conda_env,
                "python", helper,
                str(in_path), str(out_path), "--batch",
            ]

            logger.info(
                f"Dispatching ProtSolM batch to '{conda_env}' "
                f"({len(sequences)} sequences)..."
            )

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200,
            )

            if result.returncode != 0:
                logger.error(
                    f"ProtSolM batch helper failed:\n{result.stderr[:2000]}"
                )
                return None

            if not out_path.exists():
                logger.error("ProtSolM batch helper produced no output")
                return None

            with open(out_path) as f:
                output = json.load(f)

            return output.get("predictions", [])

    except subprocess.TimeoutExpired:
        logger.error("ProtSolM batch scoring timed out (7200s)")
        return None
    except Exception as e:
        logger.error(f"ProtSolM batch dispatch failed: {e}")
        return None
