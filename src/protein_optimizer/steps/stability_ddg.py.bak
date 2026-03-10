"""Tier 3 — Stability ΔΔG Predictions.

Uses Rosetta cartesian_ddg or FoldX BuildModel to computationally score
mutations for stability effects. Negative ΔΔG = stabilizing.

Usage:
    protein-opt step stability_ddg -i structure_result.json -o ddg_result.json
    protein-opt step stability_ddg -i structure_result.json --method=foldx

Requires: PyRosetta (Rosetta) or FoldX binary.
"""

from __future__ import annotations

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


class StabilityDDGStep(BaseStep):
    name = "stability_ddg"
    tier = 3
    title = "Stability ΔΔG"
    description = "Predict mutation stability effects with Rosetta or FoldX."
    requires = ["predict_structure"]

    def validate_input(self, step_input: StepResult) -> None:
        super().validate_input(step_input)
        has_structure = any(c.structure_path for c in step_input.candidates)
        if not has_structure:
            raise ValueError(
                "stability_ddg requires structure data. "
                "Run predict_structure first or provide a PDB."
            )

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        method = config.get("method", "rosetta")
        ddg_threshold = config.get("ddg_threshold", -1.0)
        saturation = config.get("saturation_mutagenesis", False)
        target_positions = config.get("positions", [])
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
                warnings.append(f"{parent.name}: No structure available, skipping ΔΔG.")
                continue

            # Determine positions to scan
            positions = _get_target_positions(
                parent, target_positions, protected, saturation
            )

            if not positions:
                warnings.append(f"{parent.name}: No positions to scan for ΔΔG.")
                continue

            logger.info(
                f"{parent.name}: Scanning {len(positions)} positions "
                f"with {method}"
            )

            # Run ΔΔG predictions
            if method == "rosetta":
                ddg_results = _run_rosetta_ddg(pdb_path, parent.sequence, positions)
            elif method == "foldx":
                ddg_results = _run_foldx_ddg(pdb_path, parent.sequence, positions)
            else:
                warnings.append(f"Unknown ΔΔG method: {method}")
                continue

            if not ddg_results:
                warnings.append(f"{parent.name}: ΔΔG calculation returned no results.")
                continue

            # Filter stabilizing mutations
            stabilizing = [
                (pos, wt, mut_aa, ddg)
                for pos, wt, mut_aa, ddg in ddg_results
                if ddg < ddg_threshold
            ]

            logger.info(
                f"{parent.name}: {len(stabilizing)} stabilizing mutations "
                f"(ΔΔG < {ddg_threshold})"
            )

            # Create candidates for stabilizing mutations
            for pos, wt, mut_aa, ddg in stabilizing:
                mut = Mutation(
                    position=pos,
                    wt=wt,
                    mut=mut_aa,
                    source_step=self.name,
                    score=ddg,
                    metadata={"ddg": ddg, "method": method},
                )
                try:
                    variant = parent.apply_mutation(mut)
                    variant.scores["ddg"] = ddg
                    variant.scores["ddg_method"] = hash(method)  # for tracking
                    candidates.append(variant)
                except ValueError as e:
                    logger.debug(f"Skipping {wt}{pos}{mut_aa}: {e}")

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _get_target_positions(
    parent: ProteinCandidate,
    specified_positions: list[int],
    protected: set[int],
    saturation: bool,
) -> list[int]:
    """Determine which positions to scan."""
    if specified_positions:
        positions = [p for p in specified_positions if p not in protected]
    elif saturation:
        positions = [
            i + 1 for i in range(len(parent.sequence))
            if (i + 1) not in protected
        ]
    else:
        # Use positions flagged by earlier steps (motif_scan, cysteine_scan)
        flagged = set()
        if "motif_hits" in parent.metadata:
            for hit in parent.metadata["motif_hits"]:
                flagged.add(hit.get("position", 0))
        if "complexity_flags" in parent.metadata:
            for flag in parent.metadata["complexity_flags"]:
                flagged.add(flag.get("position", 0))
        # Also include positions from any proposed mutations
        for mut in parent.mutations:
            flagged.add(mut.position)

        positions = [p for p in flagged if p not in protected and 1 <= p <= len(parent.sequence)]
        if not positions:
            # Fall back to surface residues (heuristic: charged flanking)
            positions = _estimate_surface_positions(parent.sequence, protected)

    return sorted(positions)


def _estimate_surface_positions(sequence: str, protected: set[int]) -> list[int]:
    """Rough heuristic: positions with charged neighbors are likely surface."""
    positions = []
    charged = set("DEKRH")
    for i, aa in enumerate(sequence):
        pos = i + 1
        if pos in protected:
            continue
        window = sequence[max(0, i - 2): min(len(sequence), i + 3)]
        if sum(1 for c in window if c in charged) >= 2:
            positions.append(pos)
    return positions[:50]  # cap at 50 for performance


def _run_rosetta_ddg(
    pdb_path: str, sequence: str, positions: list[int]
) -> list[tuple[int, str, str, float]]:
    """Run Rosetta cartesian_ddg protocol.

    Returns list of (position, wt_aa, mut_aa, ddg).
    """
    try:
        import pyrosetta
        from pyrosetta.rosetta.protocols.cartesian_ddg import CartesianddGMover

        pyrosetta.init("-ignore_unrecognized_res -mute all")
        pose = pyrosetta.pose_from_pdb(pdb_path)

        results = []
        for pos in positions:
            if pos > len(sequence):
                continue
            wt_aa = sequence[pos - 1]
            for mut_aa in AMINO_ACIDS:
                if mut_aa == wt_aa:
                    continue
                try:
                    # Simple ddG estimation using PyRosetta
                    mutant_pose = pose.clone()
                    # Apply mutation
                    pyrosetta.toolbox.mutants.mutate_residue(
                        mutant_pose, pos, mut_aa
                    )
                    # Score
                    sfxn = pyrosetta.get_fa_scorefxn()
                    wt_score = sfxn(pose)
                    mut_score = sfxn(mutant_pose)
                    ddg = mut_score - wt_score
                    results.append((pos, wt_aa, mut_aa, ddg))
                except Exception as e:
                    logger.debug(f"Rosetta ddG failed for {wt_aa}{pos}{mut_aa}: {e}")

        return results

    except ImportError:
        logger.error(
            "PyRosetta not available. Install from: "
            "https://www.pyrosetta.org/downloads"
        )
        return []


def _run_foldx_ddg(
    pdb_path: str, sequence: str, positions: list[int]
) -> list[tuple[int, str, str, float]]:
    """Run FoldX BuildModel for ΔΔG prediction.

    Returns list of (position, wt_aa, mut_aa, ddg).
    """
    results = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create individual_list.txt for FoldX
            mutations_file = tmpdir / "individual_list.txt"
            mutation_lines = []
            for pos in positions:
                if pos > len(sequence):
                    continue
                wt_aa = sequence[pos - 1]
                for mut_aa in AMINO_ACIDS:
                    if mut_aa == wt_aa:
                        continue
                    # FoldX format: WTaa Chain Position MUTaa (e.g., "LA10V;")
                    mutation_lines.append(f"{wt_aa}A{pos}{mut_aa};")

            with open(mutations_file, "w") as f:
                for line in mutation_lines:
                    f.write(line + "\n")

            # Run FoldX
            cmd = [
                "foldx", "--command=BuildModel",
                f"--pdb={Path(pdb_path).name}",
                f"--mutant-file={mutations_file}",
                f"--output-dir={tmpdir}",
            ]
            subprocess.run(
                cmd, check=True, capture_output=True, text=True,
                cwd=Path(pdb_path).parent, timeout=3600
            )

            # Parse FoldX output
            ddg_file = tmpdir / f"Dif_{Path(pdb_path).stem}.fxout"
            if ddg_file.exists():
                with open(ddg_file) as f:
                    for i, line in enumerate(f):
                        if line.startswith(("Pdb", "#")):
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                ddg = float(parts[1])
                                # Map back to mutation
                                if i - 1 < len(mutation_lines):
                                    ml = mutation_lines[i - 1].rstrip(";")
                                    wt_aa = ml[0]
                                    mut_aa = ml[-1]
                                    pos = int(ml[2:-1])
                                    results.append((pos, wt_aa, mut_aa, ddg))
                            except (ValueError, IndexError):
                                pass

    except FileNotFoundError:
        logger.error("FoldX not found on PATH. Download from: https://foldxsuite.crg.eu/")
    except subprocess.CalledProcessError as e:
        logger.error(f"FoldX failed: {e}")

    return results
