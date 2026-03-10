"""Tier 5 — Motif Scaffolding.

Uses RFdiffusion2 motif scaffolding to graft key functional motifs
(active sites, binding regions) into de novo backbones.

Dispatches to a conda environment (default: 'rfd3') where RFdiffusion2
is installed.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

_DEFAULT_RFDIFF_DIR = os.path.expanduser(
    "~/library-design/rfdiffusion2-lib/RFdiffusion2"
)


class MotifScaffoldStep(BaseStep):
    name = "motif_scaffold"
    tier = 5
    title = "Motif Scaffolding"
    description = "De novo scaffold around key functional motifs via RFdiffusion."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        rfdiff_dir = config.get(
            "rfdiffusion_dir",
            os.environ.get("RFDIFFUSION_DIR", _DEFAULT_RFDIFF_DIR),
        )
        motif_residues = config.get("motif_residues", [])
        scaffold_length = config.get("scaffold_length", [80, 120])
        num_designs = config.get("num_designs", 5)
        conda_env = config.get("conda_env", "rfd3")

        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        if not rfdiff_dir or not Path(rfdiff_dir).exists():
            warnings.append(
                f"RFdiffusion directory not found: '{rfdiff_dir}'. "
                "Set RFDIFFUSION_DIR or config rfdiffusion_dir."
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
                pdb_path = parent.metadata.get("structure_path", "")
            if not pdb_path or not Path(pdb_path).exists():
                prior = config.get("_prior_results", {})
                ps = prior.get("predict_structure")
                if ps and hasattr(ps, "candidates"):
                    for pc in ps.candidates:
                        sp = getattr(pc, "structure_path", "") or pc.metadata.get("structure_path", "")
                        if sp and Path(sp).exists():
                            pdb_path = sp
                            break
            if not pdb_path or not Path(pdb_path).exists():
                warnings.append(f"{parent.name}: No structure for motif scaffolding.")
                continue

            if not motif_residues:
                motif_residues = _auto_detect_motif(parent, protected)

            if not motif_residues:
                warnings.append(
                    f"{parent.name}: No motif residues specified. "
                    "Set motif_residues in config or use protected_residues."
                )
                continue

            contig_str = _build_motif_contig(
                motif_residues, parent.sequence, scaffold_length
            )

            output_pdbs = _run_motif_scaffolding(
                rfdiff_dir=rfdiff_dir,
                pdb_path=str(pdb_path),
                contig_str=contig_str,
                num_designs=num_designs,
                conda_env=conda_env,
                protein_name=parent.name,
            )

            if not output_pdbs:
                warnings.append(f"{parent.name}: Motif scaffolding produced no outputs.")
                continue

            logger.info(
                f"{parent.name}: Generated {len(output_pdbs)} motif scaffolds"
            )

            for i, out_pdb in enumerate(output_pdbs):
                variant = ProteinCandidate(
                    sequence=parent.sequence,
                    name=f"{parent.name}_scaffold_{i + 1}",
                    parent_id=parent.candidate_id,
                    structure_path=str(out_pdb),
                )
                variant.metadata["scaffold_output"] = str(out_pdb)
                variant.metadata["motif_residues"] = motif_residues
                variant.metadata["needs_sequence_design"] = True
                candidates.append(variant)

        return StepResult(
            step_name=self.name, candidates=candidates,
            config_used=config, warnings=warnings,
        )


def _auto_detect_motif(
    parent: ProteinCandidate, protected: set[int]
) -> list[int]:
    motif = set(protected)
    if "catalytic_residues" in parent.metadata:
        motif.update(parent.metadata["catalytic_residues"])
    if "binding_residues" in parent.metadata:
        motif.update(parent.metadata["binding_residues"])
    return sorted(motif)


def _build_motif_contig(
    motif_residues: list[int],
    sequence: str,
    scaffold_length: list[int],
) -> str:
    if not motif_residues:
        return f"[{scaffold_length[0]}-{scaffold_length[1]}]"

    sorted_motif = sorted(motif_residues)

    # Group consecutive residues into segments
    segments = []
    current_start = sorted_motif[0]
    current_end = sorted_motif[0]

    for r in sorted_motif[1:]:
        if r == current_end + 1:
            current_end = r
        else:
            segments.append((current_start, current_end))
            current_start = r
            current_end = r
    segments.append((current_start, current_end))

    min_gap = 5
    max_gap = 25

    parts = []
    parts.append(f"{min_gap}-{max_gap}")
    for i, (start, end) in enumerate(segments):
        parts.append(f"A{start}-{end}")
        if i < len(segments) - 1:
            parts.append(f"{min_gap}-{max_gap}")
    parts.append(f"{min_gap}-{max_gap}")

    return "[" + "/".join(parts) + "]"


def _run_motif_scaffolding(
    rfdiff_dir: str,
    pdb_path: str,
    contig_str: str,
    num_designs: int,
    conda_env: str,
    protein_name: str,
) -> list[Path]:
    rfdiff_path = Path(rfdiff_dir)
    script = rfdiff_path / "rf_diffusion" / "run_inference.py"
    if not script.exists():
        logger.error(f"run_inference.py not found at {script}")
        return []

    output_prefix = Path(pdb_path).parent / f"{protein_name}_scaffold"

    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", str(script),
        f"inference.input_pdb={pdb_path}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
        f"contigmap.contigs={contig_str}",
        "diffuser.T=200",
    ]

    try:
        logger.info("Running RFdiffusion motif scaffolding...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            logger.error(f"RFdiffusion failed:\n{result.stderr[:500]}")
            return []
    except subprocess.TimeoutExpired:
        logger.error("RFdiffusion timed out after 60 minutes")
        return []
    except FileNotFoundError:
        logger.error("conda not found — ensure RFdiffusion environment is set up")
        return []

    output_pdbs = sorted(
        Path(pdb_path).parent.glob(f"{protein_name}_scaffold_*.pdb")
    )
    return output_pdbs
