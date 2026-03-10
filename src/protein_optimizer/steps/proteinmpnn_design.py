"""Tier 4 — ProteinMPNN / SolubleMPNN Design.

Runs ProteinMPNN (or SolubleMPNN) on predicted structures to redesign
surface or selected positions for improved solubility, stability, or
cysteine removal while keeping the backbone fixed.

Dispatches to a conda environment (default: 'protopt') with torch CUDA.
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

_HELPER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "proteinmpnn_helper.py"


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
        omit_aas = config.get("omit_aas", "C")
        sampling_temp = config.get("sampling_temp", 0.1)
        num_sequences = config.get("num_sequences", 8)
        redesign_mode = config.get("redesign_mode", "all")
        conda_env = config.get("conda_env", "protopt")

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
                pdb_path = parent.metadata.get("structure_path", "")
            if not pdb_path or not Path(pdb_path).exists():
                # Try from prior results
                prior = config.get("_prior_results", {})
                ps = prior.get("predict_structure")
                if ps and hasattr(ps, "candidates"):
                    for pc in ps.candidates:
                        sp = getattr(pc, "structure_path", "") or pc.metadata.get("structure_path", "")
                        if sp and Path(sp).exists():
                            pdb_path = sp
                            break
            if not pdb_path or not Path(pdb_path).exists():
                warnings.append(f"{parent.name}: No structure for MPNN design.")
                continue

            # Build fixed positions
            fixed_positions = _get_fixed_positions(parent, protected, redesign_mode)

            helper_input = {
                "mpnn_dir": str(mpnn_dir),
                "pdb_path": str(pdb_path),
                "use_soluble": use_soluble,
                "omit_aas": omit_aas,
                "sampling_temp": sampling_temp,
                "num_sequences": num_sequences,
                "fixed_positions": fixed_positions,
            }

            designed_seqs = _run_subprocess(helper_input, conda_env)
            if not designed_seqs:
                warnings.append(f"{parent.name}: ProteinMPNN produced no sequences.")
                continue

            logger.info(f"{parent.name}: ProteinMPNN generated {len(designed_seqs)} designs")

            for i, entry in enumerate(designed_seqs):
                seq = entry["sequence"]
                score = entry.get("score", 0.0)
                recovery = entry.get("recovery", 0.0)

                mutations = _find_mutations(parent.sequence, seq, self.name)

                variant = ProteinCandidate(
                    sequence=seq,
                    name=f"{parent.name}_mpnn_{i + 1}",
                    parent_id=parent.candidate_id,
                    mutations=mutations,
                )
                variant.scores["mpnn_score"] = score
                variant.scores["mpnn_recovery"] = recovery
                variant.metadata["design_method"] = (
                    "SolubleMPNN" if use_soluble else "ProteinMPNN"
                )
                candidates.append(variant)

        # ── RFdiffusion backbone designs ──────────────────────────────────────
        # ProteinMPNN is used to design sequences on top of new RFdiffusion
        # backbones.  Those variants carry needs_sequence_design=True and store
        # the diffused backbone PDB path in metadata["rfdiffusion_output"].
        prior = config.get("_prior_results", {})
        rfdiff_result = prior.get("rfdiffusion_diversify")
        rfdiff_candidates: list[ProteinCandidate] = []
        if rfdiff_result is not None and hasattr(rfdiff_result, "candidates"):
            rfdiff_candidates = rfdiff_result.candidates
        # Also scan step_input itself in case rfdiffusion ran just before this step
        for c in step_input.candidates:
            if c.metadata.get("needs_sequence_design") and c not in rfdiff_candidates:
                rfdiff_candidates.append(c)

        # Find WT parent sequence for mutation comparison
        wt_seq: str = ""
        for c in step_input.candidates:
            if c.parent_id is None:
                wt_seq = c.sequence
                break

        for backbone in rfdiff_candidates:
            if not backbone.metadata.get("needs_sequence_design"):
                continue

            pdb_path = backbone.metadata.get("rfdiffusion_output", "")
            if not pdb_path or not Path(pdb_path).exists():
                warnings.append(f"{backbone.name}: RFdiffusion PDB not found ({pdb_path}).")
                continue

            # No fixed positions for backbone redesigns – allow full redesign
            # (caller can still protect residues via protected_residues config)
            fixed_positions = sorted(protected)

            helper_input = {
                "mpnn_dir": str(mpnn_dir),
                "pdb_path": str(pdb_path),
                "use_soluble": use_soluble,
                "omit_aas": omit_aas,
                "sampling_temp": sampling_temp,
                "num_sequences": num_sequences,
                "fixed_positions": fixed_positions,
            }

            designed_seqs = _run_subprocess(helper_input, conda_env)
            if not designed_seqs:
                warnings.append(f"{backbone.name}: ProteinMPNN produced no sequences for RFdiffusion backbone.")
                continue

            logger.info(
                f"{backbone.name}: ProteinMPNN designed {len(designed_seqs)} "
                "sequences on RFdiffusion backbone"
            )

            ref_seq = wt_seq or backbone.sequence
            # Parent for MPNN designs is the WT (backbone's parent), so that
            # combine_variants can discover them as direct children of WT.
            effective_parent_id = backbone.parent_id or backbone.candidate_id
            for i, entry in enumerate(designed_seqs):
                seq = entry["sequence"]
                score = entry.get("score", 0.0)
                recovery = entry.get("recovery", 0.0)

                mutations = _find_mutations(ref_seq, seq, self.name)

                variant = ProteinCandidate(
                    sequence=seq,
                    name=f"{backbone.name}_mpnn_{i + 1}",
                    parent_id=effective_parent_id,
                    mutations=mutations,
                )
                variant.scores["mpnn_score"] = score
                variant.scores["mpnn_recovery"] = recovery
                variant.metadata["design_method"] = (
                    "SolubleMPNN" if use_soluble else "ProteinMPNN"
                )
                variant.metadata["backbone_source"] = "rfdiffusion"
                variant.metadata["backbone_name"] = backbone.name
                variant.metadata["backbone_pdb"] = str(pdb_path)
                candidates.append(variant)

        return StepResult(
            step_name=self.name, candidates=candidates,
            config_used=config, warnings=warnings,
        )


def _get_fixed_positions(
    parent: ProteinCandidate, protected: set[int], redesign_mode: str
) -> list[int]:
    seq_len = len(parent.sequence)
    all_positions = set(range(1, seq_len + 1))

    if redesign_mode == "all":
        return sorted(protected)
    elif redesign_mode == "flagged":
        flagged = set()
        for mut in parent.mutations:
            flagged.add(mut.position)
        for key in ["cys_positions", "deamidation_sites", "oxidation_sites",
                     "complexity_flags", "motif_hits"]:
            if key in parent.metadata:
                for item in parent.metadata[key]:
                    if isinstance(item, dict) and "position" in item:
                        flagged.add(item["position"])
                    elif isinstance(item, int):
                        flagged.add(item)
        fixed = (all_positions - flagged) | protected
        return sorted(fixed)
    elif redesign_mode == "surface":
        return sorted(protected)

    return sorted(protected)


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


def _run_subprocess(helper_input: dict, conda_env: str) -> list[dict] | None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_p = Path(tmpdir) / "in.json"
            out_p = Path(tmpdir) / "out.json"
            in_p.write_text(json.dumps(helper_input))

            helper = str(_HELPER_SCRIPT)
            if not Path(helper).exists():
                logger.error(f"ProteinMPNN helper not found: {helper}")
                return None

            logger.info(f"Dispatching ProteinMPNN to '{conda_env}'...")

            r = subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", conda_env,
                 "python", helper, str(in_p), str(out_p)],
                capture_output=True, text=True, timeout=1200,
            )
            if r.returncode != 0:
                logger.error(f"ProteinMPNN helper failed:\n{r.stderr[:1000]}")
                return None
            if not out_p.exists():
                logger.error("ProteinMPNN helper produced no output")
                return None
            data = json.loads(out_p.read_text())
            return data.get("designs", [])
    except subprocess.TimeoutExpired:
        logger.error("ProteinMPNN timed out (1200s)")
        return None
    except Exception as e:
        logger.error(f"ProteinMPNN dispatch failed: {e}")
        return None
