"""Tier 3 — Structure Prediction.

Predicts 3D structure using Boltz-2 (default) or ESMFold (lightweight fallback).
The output PDB is required for downstream structure-based steps.

Usage:
    protein-opt step predict_structure -i my_enzyme.fasta -o structure_result.json
    protein-opt step predict_structure -i my_enzyme.fasta --method=esmfold

Requires:
    - Boltz-2: pip install boltz[cuda]
    - ESMFold: pip install fair-esm[esmfold]
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from protein_optimizer.io_utils import write_fasta
from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class PredictStructureStep(BaseStep):
    name = "predict_structure"
    tier = 3
    title = "Structure Prediction"
    description = "Predict 3D structure with Boltz-2 (or ESMFold fallback)."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        method = config.get("method", "boltz2")
        use_msa_server = config.get("use_msa_server", True)
        output_dir = Path(config.get("_global", {}).get("output_dir", "./results"))
        structures_dir = output_dir / "structures"
        structures_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            # Skip if structure already exists
            if parent.structure_path and Path(parent.structure_path).exists():
                logger.info(f"{parent.name}: Structure already exists, skipping prediction.")
                candidates.append(parent)
                continue

            pdb_path = structures_dir / f"{parent.name or parent.candidate_id}.pdb"

            if method == "boltz2":
                success = _predict_boltz2(parent, pdb_path, use_msa_server, structures_dir)
            elif method == "esmfold":
                success = _predict_esmfold(parent, pdb_path)
            else:
                warnings.append(f"Unknown method '{method}'. Use 'boltz2' or 'esmfold'.")
                candidates.append(parent)
                continue

            if success and pdb_path.exists():
                parent.structure_path = str(pdb_path)
                plddt = _parse_plddt(pdb_path)
                if plddt is not None:
                    parent.scores["plddt"] = plddt
                    logger.info(f"{parent.name}: pLDDT = {plddt:.1f}")
            else:
                warnings.append(f"{parent.name}: Structure prediction failed with {method}.")

            candidates.append(parent)

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _predict_boltz2(
    candidate: ProteinCandidate,
    output_pdb: Path,
    use_msa_server: bool,
    work_dir: Path,
) -> bool:
    """Predict structure using Boltz-2.

    Boltz expects YAML input describing the biomolecule.
    """
    try:
        # Create Boltz-2 YAML input
        yaml_dir = work_dir / "boltz_inputs"
        yaml_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = yaml_dir / f"{candidate.name or candidate.candidate_id}.yaml"

        yaml_content = (
            f"version: 1\n"
            f"sequences:\n"
            f"  - protein:\n"
            f"      id: A\n"
            f"      sequence: {candidate.sequence}\n"
        )
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        # Run Boltz predict
        cmd = ["boltz", "predict", str(yaml_path)]
        if use_msa_server:
            cmd.append("--use_msa_server")
        cmd.extend(["--out_dir", str(work_dir / "boltz_output")])

        logger.info(f"Running Boltz-2 for {candidate.name}...")
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=1800
        )

        # Find output PDB
        boltz_out = work_dir / "boltz_output"
        pdb_files = list(boltz_out.rglob("*.pdb"))
        if pdb_files:
            import shutil
            shutil.copy(pdb_files[0], output_pdb)
            return True
        else:
            # Try CIF format
            cif_files = list(boltz_out.rglob("*.cif"))
            if cif_files:
                import shutil
                shutil.copy(cif_files[0], output_pdb.with_suffix(".cif"))
                # Also save as reference
                output_pdb_cif = output_pdb.with_suffix(".cif")
                return True

        logger.warning("Boltz-2 ran but no output PDB found.")
        return False

    except FileNotFoundError:
        logger.error(
            "Boltz-2 not found. Install: pip install boltz[cuda]"
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Boltz-2 failed: {e.stderr[:500]}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Boltz-2 timed out after 1800s")
        return False


def _predict_esmfold(candidate: ProteinCandidate, output_pdb: Path) -> bool:
    """Predict structure using ESMFold (single-sequence, fast)."""
    try:
        import torch
        import esm

        logger.info(f"Running ESMFold for {candidate.name}...")
        model = esm.pretrained.esmfold_v1()
        model = model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        with torch.no_grad():
            pdb_string = model.infer_pdb(candidate.sequence)

        with open(output_pdb, "w") as f:
            f.write(pdb_string)

        return True

    except ImportError:
        logger.error(
            "ESMFold not available. Install: pip install 'fair-esm[esmfold]'"
        )
        return False
    except Exception as e:
        logger.error(f"ESMFold failed: {e}")
        return False


def _parse_plddt(pdb_path: Path) -> float | None:
    """Parse average pLDDT from B-factor column of a PDB file."""
    try:
        b_factors = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    # B-factor is columns 61-66
                    try:
                        bfactor = float(line[60:66].strip())
                        b_factors.append(bfactor)
                    except (ValueError, IndexError):
                        pass
        if b_factors:
            return sum(b_factors) / len(b_factors)
    except Exception:
        pass
    return None
