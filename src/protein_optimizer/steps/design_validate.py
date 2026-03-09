"""Tier 5 — Design-Validate Loop.

Takes RFdiffusion-generated backbones (or any backbone PDBs), runs
ProteinMPNN to design sequences, then validates each design with
Boltz-2 structure prediction and computes self-consistency (scTM/RMSD).

This creates the full generative design loop:
    RFdiffusion backbone → ProteinMPNN sequence → Boltz-2 validation

Usage:
    protein-opt step design_validate -i rfdiff_result.json -o validated.json

Requirements:
    ProteinMPNN, Boltz-2
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class DesignValidateStep(BaseStep):
    name = "design_validate"
    tier = 5
    title = "Design-Validate Loop"
    description = "ProteinMPNN → Boltz-2 self-consistency validation loop."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        min_plddt = config.get("min_plddt", 70.0)
        max_rmsd = config.get("max_rmsd", 2.0)
        mpnn_config = config.get("mpnn", {})
        boltz_config = config.get("boltz", {})

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        # Find candidates that need design (from RFdiffusion or with structures)
        design_targets = []
        for c in step_input.candidates:
            if c.metadata.get("needs_sequence_design"):
                design_targets.append(c)
            elif c.parent_id is None:
                candidates.append(c)
            else:
                candidates.append(c)

        if not design_targets:
            # Validate existing designs instead
            design_targets = [
                c for c in step_input.candidates
                if c.structure_path and Path(c.structure_path).exists()
                and c.parent_id is not None
            ]

        for target in design_targets:
            pdb_path = target.structure_path
            if not pdb_path or not Path(pdb_path).exists():
                warnings.append(f"{target.name}: No structure for design-validate.")
                continue

            # Step 1: Design sequences with ProteinMPNN
            from protein_optimizer.steps.proteinmpnn_design import (
                _run_proteinmpnn, _find_mutations,
            )

            designed_seqs = _run_proteinmpnn(
                mpnn_dir=mpnn_config.get(
                    "proteinmpnn_dir",
                    config.get("proteinmpnn_dir", ""),
                ),
                pdb_path=pdb_path,
                use_soluble=mpnn_config.get("use_soluble_model", True),
                omit_aas=mpnn_config.get("omit_aas", "C"),
                sampling_temp=mpnn_config.get("sampling_temp", 0.1),
                num_sequences=mpnn_config.get("num_sequences", 4),
                fixed_positions=[],
            )

            if not designed_seqs:
                warnings.append(f"{target.name}: ProteinMPNN produced no sequences.")
                continue

            # Step 2: Validate each with Boltz-2
            for i, (seq, mpnn_score, recovery) in enumerate(designed_seqs):
                variant_name = f"{target.name}_dv_{i+1}"

                # Predict structure with Boltz-2
                val_pdb = _predict_boltz2(seq, variant_name, boltz_config)

                if not val_pdb:
                    logger.warning(f"{variant_name}: Boltz-2 validation failed")
                    continue

                # Step 3: Compute self-consistency metrics
                plddt = _get_mean_plddt(val_pdb)
                rmsd = _compute_ca_rmsd(pdb_path, val_pdb)

                mutations = _find_mutations(target.sequence, seq, self.name)

                variant = ProteinCandidate(
                    sequence=seq,
                    name=variant_name,
                    parent_id=target.candidate_id,
                    structure_path=str(val_pdb),
                    mutations=mutations,
                )
                variant.scores["mpnn_score"] = mpnn_score
                variant.scores["mpnn_recovery"] = recovery
                variant.scores["val_plddt"] = plddt or 0.0
                variant.scores["val_rmsd"] = rmsd or 99.0
                variant.metadata["design_backbone"] = pdb_path
                variant.metadata["validation_structure"] = str(val_pdb)

                # Pass/fail based on thresholds
                passed = True
                if plddt is not None and plddt < min_plddt:
                    passed = False
                if rmsd is not None and rmsd > max_rmsd:
                    passed = False

                variant.metadata["validation_passed"] = passed

                if passed:
                    logger.info(
                        f"{variant_name}: PASS (pLDDT={plddt:.1f}, "
                        f"RMSD={rmsd:.2f}Å)"
                    )
                else:
                    logger.info(
                        f"{variant_name}: FAIL (pLDDT={plddt or 0:.1f}, "
                        f"RMSD={rmsd or 99:.2f}Å)"
                    )

                candidates.append(variant)

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _predict_boltz2(
    sequence: str, name: str, boltz_config: dict
) -> Path | None:
    """Run Boltz-2 on a single sequence for validation."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write YAML input
            yaml_content = (
                f"version: 1\n"
                f"sequences:\n"
                f"  - protein:\n"
                f"      id: A\n"
                f"      sequence: {sequence}\n"
            )
            yaml_path = tmpdir / f"{name}.yaml"
            yaml_path.write_text(yaml_content)

            output_dir = tmpdir / "output"
            output_dir.mkdir()

            cmd = [
                "boltz", "predict",
                str(yaml_path),
                "--out_dir", str(output_dir),
            ]

            if boltz_config.get("use_msa_server", False):
                cmd.append("--use_msa_server")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )

            if result.returncode != 0:
                logger.warning(f"Boltz-2 failed for {name}: {result.stderr[:200]}")
                return None

            # Find output PDB
            pdbs = list(output_dir.rglob("*.pdb"))
            if pdbs:
                # Copy to persistent location
                persistent_dir = Path(boltz_config.get("output_dir", ".")) / "val_structures"
                persistent_dir.mkdir(parents=True, exist_ok=True)
                out_path = persistent_dir / f"{name}_val.pdb"

                import shutil
                shutil.copy2(pdbs[0], out_path)
                return out_path

            return None

    except Exception as e:
        logger.error(f"Boltz-2 validation failed for {name}: {e}")
        return None


def _get_mean_plddt(pdb_path: Path) -> float | None:
    """Extract mean pLDDT from B-factor column."""
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("val", str(pdb_path))
        bfactors = [
            a.get_bfactor()
            for a in structure.get_atoms()
            if a.name == "CA"
        ]
        if bfactors:
            return sum(bfactors) / len(bfactors)
        return None
    except Exception:
        return None


def _compute_ca_rmsd(ref_pdb: str, query_pdb: str | Path) -> float | None:
    """Compute backbone Cα RMSD between two structures."""
    try:
        from Bio.PDB import PDBParser, Superimposer
        import numpy as np

        parser = PDBParser(QUIET=True)
        ref = parser.get_structure("ref", ref_pdb)
        query = parser.get_structure("query", str(query_pdb))

        ref_atoms = [
            a for a in ref.get_atoms() if a.name == "CA"
        ]
        query_atoms = [
            a for a in query.get_atoms() if a.name == "CA"
        ]

        # Align on overlapping length
        n = min(len(ref_atoms), len(query_atoms))
        if n < 10:
            return None

        ref_atoms = ref_atoms[:n]
        query_atoms = query_atoms[:n]

        sup = Superimposer()
        sup.set_atoms(ref_atoms, query_atoms)

        return sup.rms

    except Exception as e:
        logger.debug(f"RMSD computation failed: {e}")
        return None
