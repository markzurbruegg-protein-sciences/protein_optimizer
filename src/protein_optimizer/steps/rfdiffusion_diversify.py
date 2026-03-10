"""Tier 5 — RFdiffusion Partial Diffusion / Diversification.

Uses RFdiffusion2 to generate backbone-diversified variants of the input
protein via partial diffusion.  The input structure is noised for a small
number of timesteps (partial_T) then denoised, producing similar but
structurally distinct backbones.

Dispatches to a conda environment (default: 'rfd3') where RFdiffusion2
is installed.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

_DEFAULT_RFDIFF_DIR = os.path.expanduser(
    "~/library-design/rfdiffusion2-lib/RFdiffusion2"
)


class RFdiffusionDiversifyStep(BaseStep):
    name = "rfdiffusion_diversify"
    tier = 4
    title = "RFdiffusion Diversify"
    description = "Generate structurally diverse variants via partial diffusion."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        rfdiff_dir = config.get(
            "rfdiffusion_dir",
            os.environ.get("RFDIFFUSION_DIR", _DEFAULT_RFDIFF_DIR),
        )
        partial_T = config.get("partial_T", 15)
        total_T = config.get("T", 50)
        num_designs = config.get("num_designs", 5)
        conda_env = config.get("conda_env", "rfd3")

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
                warnings.append(f"{parent.name}: No structure for RFdiffusion.")
                continue

            output_pdbs = _run_rfdiffusion(
                rfdiff_dir=rfdiff_dir,
                pdb_path=str(pdb_path),
                partial_T=partial_T,
                total_T=total_T,
                num_designs=num_designs,
                conda_env=conda_env,
                protein_name=parent.name,
            )

            if not output_pdbs:
                warnings.append(f"{parent.name}: RFdiffusion produced no outputs.")
                continue

            logger.info(
                f"{parent.name}: RFdiffusion generated {len(output_pdbs)} backbones"
            )

            for i, out_pdb in enumerate(output_pdbs):
                variant = ProteinCandidate(
                    sequence=parent.sequence,
                    name=f"{parent.name}_rfdiff_{i + 1}",
                    parent_id=parent.candidate_id,
                    structure_path=str(out_pdb),
                )
                variant.metadata["rfdiffusion_output"] = str(out_pdb)
                variant.metadata["partial_T"] = partial_T
                variant.metadata["needs_sequence_design"] = True
                candidates.append(variant)

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _run_rfdiffusion(
    rfdiff_dir: str,
    pdb_path: str,
    partial_T: int,
    total_T: int,
    num_designs: int,
    conda_env: str,
    protein_name: str,
) -> list[Path]:
    rfdiff_path = Path(rfdiff_dir)
    script = rfdiff_path / "rf_diffusion" / "run_inference.py"
    if not script.exists():
        logger.error(f"run_inference.py not found at {script}")
        return []

    output_prefix = Path(pdb_path).parent / f"{protein_name}_rfdiff"
    ckpt_path = rfdiff_path / "rf_diffusion" / "model_weights" / "RFD_140.pt"
    if not ckpt_path.exists():
        ckpt_path = rfdiff_path / "rf_diffusion" / "model_weights" / "RFD_173.pt"
    if not ckpt_path.exists():
        logger.error(f"No model weights found in {rfdiff_path / 'rf_diffusion' / 'model_weights'}")
        return []

    seq_len = _get_pdb_length(pdb_path)
    contig = f"A1-{seq_len}" if seq_len else "A1-999"

    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", str(script),
        f"inference.input_pdb={pdb_path}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={num_designs}",
        f"inference.ckpt_path={ckpt_path}",
        f"diffuser.partial_T={partial_T}",
        f"diffuser.T={total_T}",
        f"contigmap.contigs=[{contig}]",
    ]

    # RFdiffusion2 expects its repo root on PYTHONPATH for internal imports
    env = os.environ.copy()
    python_path = str(rfdiff_path)
    if "PYTHONPATH" in env:
        python_path = python_path + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = python_path

    try:
        logger.info(f"Running RFdiffusion (partial_T={partial_T}, T={total_T})...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env,
        )
        if result.returncode != 0:
            logger.error(f"RFdiffusion failed:\n{result.stderr[:500]}")
            return []
    except subprocess.TimeoutExpired:
        logger.error("RFdiffusion timed out after 30 minutes")
        return []
    except FileNotFoundError:
        logger.error("conda not found — ensure RFdiffusion environment is set up")
        return []

    output_pdbs = sorted(
        Path(pdb_path).parent.glob(f"{protein_name}_rfdiff_*.pdb")
    )
    # RFdiffusion2 appends suffixes like '-atomized-bb-False' to output names
    if not output_pdbs:
        output_pdbs = sorted(
            Path(pdb_path).parent.glob(f"{protein_name}_rfdiff*-*.pdb")
        )
    return output_pdbs


def _get_pdb_length(pdb_path: str) -> int | None:
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("p", pdb_path)
        chain = list(structure[0].get_chains())[0]
        return len([r for r in chain.get_residues() if r.get_id()[0] == " "])
    except Exception:
        return None
