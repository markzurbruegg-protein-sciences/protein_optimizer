"""Tier 4 — ProteinMPNN / SolubleMPNN Design.

Runs ProteinMPNN (or SolubleMPNN) on predicted structures to redesign
surface or selected positions for improved solubility, stability, or
cysteine removal while keeping the backbone fixed.

Usage:
    protein-opt step proteinmpnn_design -i structure_result.json -o mpnn_result.json

Requirements:
    git clone https://github.com/dauparas/ProteinMPNN
    Set PROTEINMPNN_DIR environment variable or config proteinmpnn_dir.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class ProteinMPNNDesignStep(BaseStep):
    name = "proteinmpnn_design"
    tier = 4
    title = "ProteinMPNN Design"
    description = "Structure-based sequence design with SolubleMPNN."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        mpnn_dir = config.get(
            "proteinmpnn_dir",
            os.environ.get("PROTEINMPNN_DIR", ""),
        )
        use_soluble = config.get("use_soluble_model", True)
        omit_aas = config.get("omit_aas", "C")  # default: remove cysteines
        sampling_temp = config.get("sampling_temp", 0.1)
        num_sequences = config.get("num_sequences", 8)
        redesign_mode = config.get("redesign_mode", "all")  # all | surface | flagged

        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        if not mpnn_dir or not Path(mpnn_dir).exists():
            warnings.append(
                f"ProteinMPNN directory not found: '{mpnn_dir}'. "
                "Set PROTEINMPNN_DIR or config proteinmpnn_dir."
            )
            return StepResult(
                step_name=self.name,
                candidates=list(step_input.candidates),
                config_used=config,
                warnings=warnings,
            )

        for parent in step_input.candidates:
            if parent.parent_id is not None:
                continue

            candidates.append(parent)

            pdb_path = parent.structure_path
            if not pdb_path or not Path(pdb_path).exists():
                warnings.append(f"{parent.name}: No structure for MPNN design.")
                continue

            # Build fixed positions JSON
            fixed_positions = _get_fixed_positions(
                parent, protected, redesign_mode
            )

            # Run ProteinMPNN
            designed_seqs = _run_proteinmpnn(
                mpnn_dir=mpnn_dir,
                pdb_path=pdb_path,
                use_soluble=use_soluble,
                omit_aas=omit_aas,
                sampling_temp=sampling_temp,
                num_sequences=num_sequences,
                fixed_positions=fixed_positions,
            )

            if not designed_seqs:
                warnings.append(f"{parent.name}: ProteinMPNN produced no sequences.")
                continue

            logger.info(
                f"{parent.name}: ProteinMPNN generated {len(designed_seqs)} designs"
            )

            for i, (seq, score, recovery) in enumerate(designed_seqs):
                # Find mutations relative to parent
                mutations = _find_mutations(parent.sequence, seq, self.name)

                variant = ProteinCandidate(
                    sequence=seq,
                    name=f"{parent.name}_mpnn_{i+1}",
                    parent_id=parent.candidate_id,
                    mutations=mutations,
                )
                variant.scores["mpnn_score"] = score
                variant.scores["mpnn_recovery"] = recovery
                variant.metadata["design_method"] = (
                    "SolubleMPNN" if use_soluble else "ProteinMPNN"
                )
                candidates.append(variant)

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _get_fixed_positions(
    parent: ProteinCandidate,
    protected: set[int],
    redesign_mode: str,
) -> list[int]:
    """Determine which positions should be fixed (not redesigned)."""
    seq_len = len(parent.sequence)
    all_positions = set(range(1, seq_len + 1))

    if redesign_mode == "all":
        # Fix only protected positions
        return sorted(protected)

    elif redesign_mode == "flagged":
        # Redesign only positions flagged by earlier analysis steps
        flagged = set()
        for mut in parent.mutations:
            flagged.add(mut.position)

        # Pull flagged positions from metadata
        for key in ["cys_positions", "deamidation_sites", "oxidation_sites",
                     "complexity_flags", "motif_hits"]:
            if key in parent.metadata:
                for item in parent.metadata[key]:
                    if isinstance(item, dict) and "position" in item:
                        flagged.add(item["position"])
                    elif isinstance(item, int):
                        flagged.add(item)

        # Fix everything except flagged (but respect protected)
        fixed = (all_positions - flagged) | protected
        return sorted(fixed)

    elif redesign_mode == "surface":
        # Will redesign surface residues only; need structure
        # For now, fix core residues (those with high contact number)
        return sorted(protected)

    return sorted(protected)


def _run_proteinmpnn(
    mpnn_dir: str,
    pdb_path: str,
    use_soluble: bool,
    omit_aas: str,
    sampling_temp: float,
    num_sequences: int,
    fixed_positions: list[int],
) -> list[tuple[str, float, float]]:
    """Run ProteinMPNN subprocess.

    Returns list of (sequence, score, recovery_rate).
    """
    mpnn_path = Path(mpnn_dir)
    script = mpnn_path / "protein_mpnn_run.py"
    if not script.exists():
        logger.error(f"protein_mpnn_run.py not found in {mpnn_dir}")
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Create PDB list
        pdb_list = Path(tmpdir) / "pdb_list.txt"
        pdb_list.write_text(pdb_path + "\n")

        # Create fixed positions JSONL if needed
        jsonl_path = None
        if fixed_positions:
            jsonl_path = Path(tmpdir) / "fixed_positions.jsonl"
            pdb_name = Path(pdb_path).stem
            # ProteinMPNN format: {"pdb_name": {"A": [1,2,3,...]}}
            entry = {pdb_name: {"A": fixed_positions}}
            jsonl_path.write_text(json.dumps(entry) + "\n")

        # Build command
        cmd = [
            "python", str(script),
            "--pdb_path", pdb_path,
            "--out_folder", str(output_dir),
            "--num_seq_per_target", str(num_sequences),
            "--sampling_temp", str(sampling_temp),
            "--seed", "42",
            "--batch_size", "1",
        ]

        if use_soluble:
            cmd.append("--use_soluble_model")

        if omit_aas:
            cmd.extend(["--omit_AAs", omit_aas])

        if jsonl_path:
            cmd.extend(["--fixed_positions_jsonl", str(jsonl_path)])

        try:
            logger.info(f"Running ProteinMPNN: {' '.join(cmd[:6])}...")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=str(mpnn_path),
            )
            if result.returncode != 0:
                logger.error(f"ProteinMPNN failed:\n{result.stderr[:500]}")
                return []
        except subprocess.TimeoutExpired:
            logger.error("ProteinMPNN timed out after 600s")
            return []
        except FileNotFoundError:
            logger.error("Python not found for ProteinMPNN subprocess")
            return []

        # Parse output FASTA
        return _parse_mpnn_output(output_dir, pdb_path)


def _parse_mpnn_output(
    output_dir: Path, pdb_path: str
) -> list[tuple[str, float, float]]:
    """Parse ProteinMPNN output FASTA files.

    Returns list of (sequence, global_score, recovery).
    """
    results = []
    fasta_dir = output_dir / "seqs"
    if not fasta_dir.exists():
        # Try alternative output structure
        fasta_dir = output_dir

    pdb_name = Path(pdb_path).stem

    for fasta_file in sorted(fasta_dir.glob("*.fa")):
        with open(fasta_file) as f:
            lines = f.readlines()

        seq = None
        score = 0.0
        recovery = 0.0

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(">"):
                # Parse header: >T=0.1, sample=1, score=1.234, ...
                parts = line[1:].split(",")
                for part in parts:
                    part = part.strip()
                    if part.startswith("score="):
                        try:
                            score = float(part.split("=")[1])
                        except ValueError:
                            pass
                    elif part.startswith("seq_recovery="):
                        try:
                            recovery = float(part.split("=")[1])
                        except ValueError:
                            pass
            else:
                if line and not line.startswith("#"):
                    seq = line

            if seq and i > 0:  # Skip the first entry (input sequence)
                results.append((seq, score, recovery))
                seq = None
                score = 0.0
                recovery = 0.0

    return results


def _find_mutations(
    parent_seq: str, designed_seq: str, source_step: str
) -> list[Mutation]:
    """Find all mutations between parent and designed sequence."""
    mutations = []
    for i in range(min(len(parent_seq), len(designed_seq))):
        if parent_seq[i] != designed_seq[i]:
            mutations.append(Mutation(
                position=i + 1,
                wt=parent_seq[i],
                mut=designed_seq[i],
                source_step=source_step,
            ))
    return mutations
