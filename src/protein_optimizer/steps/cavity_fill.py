"""Tier 3 — Cavity Filling.

Identifies internal cavities in the protein structure and proposes
small→large hydrophobic mutations (Ala→Val, Val→Ile/Leu, Gly→Ala)
to improve core packing and thermostability.

Usage:
    protein-opt step cavity_fill -i structure_result.json -o cavity_result.json
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

# Conservative size-increasing hydrophobic substitutions
CAVITY_FILL_SUBSTITUTIONS: dict[str, list[str]] = {
    "G": ["A"],           # Gly → Ala
    "A": ["V", "L"],      # Ala → Val, Leu
    "V": ["I", "L"],      # Val → Ile, Leu
    "S": ["T", "V"],      # Ser → Thr, Val (fill + maintain some polarity)
    "T": ["V", "I"],      # Thr → Val, Ile
}


class CavityFillStep(BaseStep):
    name = "cavity_fill"
    tier = 3
    title = "Cavity Filling"
    description = "Fill internal cavities with larger hydrophobics for better packing."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        min_volume = config.get("min_volume", 20.0)
        max_mutations = config.get("max_mutations", 10)
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
                warnings.append(f"{parent.name}: No structure for cavity analysis.")
                continue

            # Find buried positions with packing defects
            buried_positions = _find_underpacked_buried(pdb_path, parent.sequence, protected)

            if not buried_positions:
                warnings.append(f"{parent.name}: No cavity-lining residues found.")
                continue

            # Propose mutations
            proposed = []
            for pos, wt_aa, burial_score in buried_positions:
                if wt_aa not in CAVITY_FILL_SUBSTITUTIONS:
                    continue
                for mut_aa in CAVITY_FILL_SUBSTITUTIONS[wt_aa]:
                    proposed.append(Mutation(
                        position=pos,
                        wt=wt_aa,
                        mut=mut_aa,
                        source_step=self.name,
                        score=burial_score,
                        metadata={"burial_score": burial_score},
                    ))

            # Sort by burial score (higher = more buried = better candidate)
            proposed.sort(key=lambda m: m.score or 0, reverse=True)
            proposed = proposed[:max_mutations]

            logger.info(f"{parent.name}: {len(proposed)} cavity-fill mutations proposed")

            for mut in proposed:
                try:
                    variant = parent.apply_mutation(mut)
                    variant.scores["cavity_burial"] = mut.score or 0.0
                    candidates.append(variant)
                except ValueError as e:
                    logger.debug(f"Skipping {mut.label}: {e}")

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _find_underpacked_buried(
    pdb_path: str,
    sequence: str,
    protected: set[int],
) -> list[tuple[int, str, float]]:
    """Find buried residues with suboptimal packing.

    Uses a simple neighbor-count / contact-number approach as a proxy for
    burial. Residues that are buried (high contact number) but have small
    side chains (Gly, Ala, Val, Ser, Thr) are candidates for cavity filling.

    Returns list of (position, aa, burial_score).
    """
    try:
        from Bio.PDB import PDBParser, NeighborSearch
    except ImportError:
        logger.error("BioPython required for cavity analysis.")
        return []

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
    except Exception as e:
        logger.error(f"Failed to parse PDB: {e}")
        return []

    model = structure[0]
    chain = list(model.get_chains())[0]
    residues = list(chain.get_residues())

    # Get all heavy atoms for neighbor search
    all_atoms = [a for a in chain.get_atoms() if a.element != "H"]
    if not all_atoms:
        return []

    ns = NeighborSearch(all_atoms)

    buried_candidates = []
    small_aas = set(CAVITY_FILL_SUBSTITUTIONS.keys())

    for i, res in enumerate(residues):
        pos = i + 1
        if pos in protected:
            continue
        if res.get_id()[0] != " ":
            continue

        aa = sequence[pos - 1] if pos <= len(sequence) else ""
        if aa not in small_aas:
            continue

        # Compute burial: count Cα atoms within 10Å of this residue's Cα
        if "CA" not in res:
            continue

        ca_coord = res["CA"].get_vector().get_array()
        neighbors = ns.search(ca_coord, 10.0, level="R")
        contact_count = len(neighbors) - 1  # exclude self

        # Consider "buried" if high contact count (>15 neighbors within 10Å)
        if contact_count > 15:
            burial_score = contact_count / 30.0  # normalize ~0.5–1.0
            buried_candidates.append((pos, aa, burial_score))

    return buried_candidates
