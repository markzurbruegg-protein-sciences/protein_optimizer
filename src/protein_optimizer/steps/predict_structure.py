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
        conda_env = config.get("conda_env", "boltz2-env")
        output_dir = Path(config.get("_global", {}).get("output_dir", "./results"))
        structures_dir = output_dir / "structures"
        structures_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        # Only predict structure for parent/WT candidates (not every variant)
        parents_needing_structure = []
        for parent in step_input.candidates:
            candidates.append(parent)
            if parent.parent_id is not None:
                continue  # variants inherit parent structure
            if parent.structure_path and Path(parent.structure_path).exists():
                logger.info(f"{parent.name}: Structure already exists, skipping.")
                continue
            parents_needing_structure.append(parent)

        for parent in parents_needing_structure:
            pdb_path = structures_dir / f"{parent.name or parent.candidate_id}.pdb"

            if method == "boltz2":
                success = _predict_boltz2(
                    parent, pdb_path, use_msa_server, structures_dir, conda_env
                )
            elif method == "esmfold":
                success = _predict_esmfold(parent, pdb_path)
            else:
                warnings.append(f"Unknown method '{method}'. Use 'boltz2' or 'esmfold'.")
                continue

            if success and pdb_path.exists():
                parent.structure_path = str(pdb_path)
                plddt = _parse_plddt(pdb_path)
                if plddt is not None:
                    parent.scores["plddt"] = plddt
                    logger.info(f"{parent.name}: pLDDT = {plddt:.1f}")
            else:
                # Also try CIF
                cif_path = pdb_path.with_suffix(".cif")
                if cif_path.exists():
                    parent.structure_path = str(cif_path)
                    logger.info(f"{parent.name}: Structure saved as CIF")
                else:
                    warnings.append(
                        f"{parent.name}: Structure prediction failed with {method}."
                    )

        # Propagate structure path to variants that share the same parent
        parent_structures = {
            c.candidate_id: c.structure_path
            for c in candidates
            if c.parent_id is None and c.structure_path
        }
        for c in candidates:
            if c.parent_id and not c.structure_path:
                if c.parent_id in parent_structures:
                    c.structure_path = parent_structures[c.parent_id]

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
    conda_env: str = "boltz2-env",
) -> bool:
    """Predict structure using Boltz-2 via conda env dispatch.

    Boltz expects YAML input describing the biomolecule.
    """
    try:
        import shutil

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

        boltz_out = work_dir / "boltz_output"

        # Build command — dispatch to conda env
        boltz_cmd = ["boltz", "predict", str(yaml_path)]
        if use_msa_server:
            boltz_cmd.append("--use_msa_server")
        boltz_cmd.extend(["--out_dir", str(boltz_out)])

        cmd = [
            "conda", "run", "--no-capture-output", "-n", conda_env,
        ] + boltz_cmd

        logger.info(
            f"Running Boltz-2 for {candidate.name} in conda env '{conda_env}'..."
        )
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )

        if result.returncode != 0:
            logger.error(f"Boltz-2 failed (exit {result.returncode}):\n{result.stderr[:500]}")
            return False

        # Find output PDB or CIF
        pdb_files = list(boltz_out.rglob("*.pdb"))
        if pdb_files:
            shutil.copy(pdb_files[0], output_pdb)
            return True

        cif_files = list(boltz_out.rglob("*.cif"))
        if cif_files:
            cif_dest = output_pdb.with_suffix(".cif")
            shutil.copy(cif_files[0], cif_dest)
            # Convert CIF→PDB with BioPython if possible
            if _cif_to_pdb(cif_dest, output_pdb):
                return True
            # Keep CIF as-is — downstream steps can handle it
            logger.info(f"Structure saved as CIF: {cif_dest}")
            return True

        logger.warning("Boltz-2 ran but no output structure found.")
        return False

    except FileNotFoundError:
        logger.error(
            "conda or Boltz-2 not found. Ensure boltz2-env is set up."
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Boltz-2 failed: {e.stderr[:500]}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Boltz-2 timed out after 1800s")
        return False


def _cif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
    """Convert CIF to PDB format using BioPython."""
    try:
        from Bio.PDB import MMCIFParser, PDBIO
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", str(cif_path))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path))
        return True
    except Exception as e:
        logger.debug(f"CIF→PDB conversion failed: {e}")
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
