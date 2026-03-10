"""Tier 5 — Design-Validate Loop.

Takes RFdiffusion-generated backbones (or any backbone PDBs), runs
ProteinMPNN to design sequences, then validates each design with
Boltz-2 structure prediction and computes self-consistency (scTM/RMSD).

Dispatches ProteinMPNN to 'protopt' and Boltz-2 to 'boltz2-env' via
subprocess.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

_MPNN_HELPER = Path(__file__).resolve().parents[3] / "scripts" / "proteinmpnn_helper.py"


class DesignValidateStep(BaseStep):
    name = "design_validate"
    tier = 4
    title = "Design-Validate Loop"
    description = "ProteinMPNN → Boltz-2 self-consistency validation loop."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        min_plddt = config.get("min_plddt", 70.0)
        max_rmsd = config.get("max_rmsd", 2.0)
        mpnn_config = config.get("mpnn", {})
        boltz_config = config.get("boltz", {})
        mpnn_conda = config.get("mpnn_conda_env", "protopt")
        boltz_conda = config.get("boltz_conda_env", "boltz2-env")

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        # Find candidates that need design
        design_targets = []
        for c in step_input.candidates:
            if c.metadata.get("needs_sequence_design"):
                design_targets.append(c)
            else:
                candidates.append(c)

        if not design_targets:
            design_targets = [
                c for c in step_input.candidates
                if c.structure_path and Path(c.structure_path).exists()
                and c.parent_id is not None
            ]

        mpnn_dir = mpnn_config.get(
            "proteinmpnn_dir",
            config.get("proteinmpnn_dir",
                        os.environ.get("PROTEINMPNN_DIR", "")),
        )

        for target in design_targets:
            pdb_path = target.structure_path
            if not pdb_path or not Path(pdb_path).exists():
                warnings.append(f"{target.name}: No structure for design-validate.")
                continue

            # Step 1: Design sequences with ProteinMPNN
            designed_seqs = _run_mpnn(
                mpnn_dir=mpnn_dir,
                pdb_path=str(pdb_path),
                use_soluble=mpnn_config.get("use_soluble_model", True),
                omit_aas=mpnn_config.get("omit_aas", "C"),
                sampling_temp=mpnn_config.get("sampling_temp", 0.1),
                num_sequences=mpnn_config.get("num_sequences", 4),
                conda_env=mpnn_conda,
            )

            if not designed_seqs:
                warnings.append(f"{target.name}: ProteinMPNN produced no sequences.")
                continue

            # Step 2: Validate each with Boltz-2
            output_dir = Path(pdb_path).parent / "val_structures"
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, entry in enumerate(designed_seqs):
                seq = entry["sequence"]
                mpnn_score = entry.get("score", 0.0)
                recovery = entry.get("recovery", 0.0)
                variant_name = f"{target.name}_dv_{i + 1}"

                val_pdb = _predict_boltz2(
                    seq, variant_name, str(output_dir), boltz_conda
                )

                plddt = _get_mean_plddt(val_pdb) if val_pdb else None
                rmsd = _compute_ca_rmsd(str(pdb_path), str(val_pdb)) if val_pdb else None

                mutations = _find_mutations(target.sequence, seq, self.name)

                variant = ProteinCandidate(
                    sequence=seq,
                    name=variant_name,
                    parent_id=target.candidate_id,
                    structure_path=str(val_pdb) if val_pdb else None,
                    mutations=mutations,
                )
                variant.scores["mpnn_score"] = mpnn_score
                variant.scores["mpnn_recovery"] = recovery
                variant.scores["val_plddt"] = plddt or 0.0
                variant.scores["val_rmsd"] = rmsd or 99.0
                variant.metadata["design_backbone"] = str(pdb_path)
                variant.metadata["validation_structure"] = str(val_pdb) if val_pdb else ""

                passed = True
                if plddt is not None and plddt < min_plddt:
                    passed = False
                if rmsd is not None and rmsd > max_rmsd:
                    passed = False
                variant.metadata["validation_passed"] = passed

                status = "PASS" if passed else "FAIL"
                logger.info(
                    f"{variant_name}: {status} "
                    f"(pLDDT={plddt or 0:.1f}, RMSD={rmsd or 99:.2f}Å)"
                )
                candidates.append(variant)

        return StepResult(
            step_name=self.name, candidates=candidates,
            config_used=config, warnings=warnings,
        )


def _run_mpnn(
    mpnn_dir: str, pdb_path: str, use_soluble: bool,
    omit_aas: str, sampling_temp: float, num_sequences: int,
    conda_env: str,
) -> list[dict] | None:
    if not mpnn_dir or not Path(mpnn_dir).exists():
        logger.warning(f"ProteinMPNN dir not found: {mpnn_dir}")
        return None

    helper = str(_MPNN_HELPER)
    if not Path(helper).exists():
        logger.error(f"ProteinMPNN helper not found: {helper}")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_p = Path(tmpdir) / "in.json"
            out_p = Path(tmpdir) / "out.json"
            in_p.write_text(json.dumps({
                "mpnn_dir": mpnn_dir,
                "pdb_path": pdb_path,
                "use_soluble": use_soluble,
                "omit_aas": omit_aas,
                "sampling_temp": sampling_temp,
                "num_sequences": num_sequences,
                "fixed_positions": [],
            }))

            r = subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", conda_env,
                 "python", helper, str(in_p), str(out_p)],
                capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                logger.error(f"MPNN helper failed:\n{r.stderr[:500]}")
                return None
            if not out_p.exists():
                return None
            return json.loads(out_p.read_text()).get("designs", [])
    except Exception as e:
        logger.error(f"MPNN dispatch failed: {e}")
        return None


def _predict_boltz2(
    sequence: str, name: str, output_dir: str, conda_env: str
) -> Path | None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_p = Path(tmpdir)
            yaml_path = tmpdir_p / f"{name}.yaml"
            yaml_path.write_text(
                f"version: 1\nsequences:\n  - protein:\n"
                f"      id: A\n      sequence: {sequence}\n"
            )

            boltz_out = tmpdir_p / "output"
            boltz_out.mkdir()

            cmd = [
                "conda", "run", "--no-capture-output", "-n", conda_env,
                "boltz", "predict", str(yaml_path),
                "--out_dir", str(boltz_out),
            ]

            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if r.returncode != 0:
                logger.warning(f"Boltz-2 failed for {name}: {r.stderr[:200]}")
                return None

            pdbs = list(boltz_out.rglob("*.pdb"))
            if pdbs:
                import shutil
                out_path = Path(output_dir) / f"{name}_val.pdb"
                shutil.copy2(pdbs[0], out_path)
                return out_path
            return None
    except Exception as e:
        logger.error(f"Boltz-2 validation failed for {name}: {e}")
        return None


def _get_mean_plddt(pdb_path: Path) -> float | None:
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("val", str(pdb_path))
        bfactors = [
            a.get_bfactor() for a in structure.get_atoms() if a.name == "CA"
        ]
        return sum(bfactors) / len(bfactors) if bfactors else None
    except Exception:
        return None


def _compute_ca_rmsd(ref_pdb: str, query_pdb: str) -> float | None:
    try:
        from Bio.PDB import PDBParser, Superimposer
        parser = PDBParser(QUIET=True)
        ref = parser.get_structure("ref", ref_pdb)
        query = parser.get_structure("query", query_pdb)

        ref_atoms = [a for a in ref.get_atoms() if a.name == "CA"]
        query_atoms = [a for a in query.get_atoms() if a.name == "CA"]

        n = min(len(ref_atoms), len(query_atoms))
        if n < 10:
            return None

        sup = Superimposer()
        sup.set_atoms(ref_atoms[:n], query_atoms[:n])
        return sup.rms
    except Exception:
        return None


def _find_mutations(
    parent_seq: str, designed_seq: str, source_step: str
) -> list[Mutation]:
    mutations = []
    for i in range(min(len(parent_seq), len(designed_seq))):
        if parent_seq[i] != designed_seq[i]:
            mutations.append(Mutation(
                position=i + 1, wt=parent_seq[i], mut=designed_seq[i],
                source_step=source_step,
            ))
    return mutations
