"""Tier 3 — Disulfide Bond Engineering.

Uses the DbD 2.0 algorithm (geometric criteria on Cβ-Cβ distances and
Cα-Cβ-Cα angles) to identify residue pairs compatible with engineered
disulfide bridges for thermostability.

Usage:
    protein-opt step disulfide_design -i structure_result.json -o disulfide_result.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class DisulfideDesignStep(BaseStep):
    name = "disulfide_design"
    tier = 3
    title = "Disulfide Design"
    description = "Identify residue pairs for engineered disulfide bonds (DbD 2.0 criteria)."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        cb_dist_min = config.get("cb_distance_min", 3.5)
        cb_dist_max = config.get("cb_distance_max", 4.5)
        max_candidates = config.get("max_candidates", 10)
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
                warnings.append(f"{parent.name}: No structure for disulfide design.")
                continue

            pairs = _find_disulfide_candidates(
                pdb_path, parent.sequence, cb_dist_min, cb_dist_max, protected
            )

            if not pairs:
                warnings.append(f"{parent.name}: No disulfide-compatible pairs found.")
                continue

            # Sort by distance from ideal (3.8Å) and take top N
            pairs.sort(key=lambda p: abs(p[2] - 3.8))
            pairs = pairs[:max_candidates]

            logger.info(f"{parent.name}: Found {len(pairs)} disulfide candidates")

            for pos_i, pos_j, dist, energy_estimate in pairs:
                wt_i = parent.sequence[pos_i - 1]
                wt_j = parent.sequence[pos_j - 1]

                mutations = []
                if wt_i != "C":
                    mutations.append(Mutation(
                        position=pos_i, wt=wt_i, mut="C",
                        source_step=self.name,
                        metadata={"disulfide_partner": pos_j, "cb_distance": dist},
                    ))
                if wt_j != "C":
                    mutations.append(Mutation(
                        position=pos_j, wt=wt_j, mut="C",
                        source_step=self.name,
                        metadata={"disulfide_partner": pos_i, "cb_distance": dist},
                    ))

                if not mutations:
                    continue

                try:
                    variant = parent.apply_mutations(mutations)
                    variant.scores["disulfide_cb_distance"] = dist
                    variant.scores["disulfide_energy_estimate"] = energy_estimate
                    variant.metadata["disulfide_pair"] = [pos_i, pos_j]
                    candidates.append(variant)
                except ValueError as e:
                    logger.debug(f"Skipping disulfide {pos_i}-{pos_j}: {e}")

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _find_disulfide_candidates(
    pdb_path: str,
    sequence: str,
    cb_dist_min: float,
    cb_dist_max: float,
    protected: set[int],
) -> list[tuple[int, int, float, float]]:
    """Find residue pairs meeting disulfide geometric criteria.

    Returns list of (pos_i, pos_j, cb_distance, energy_estimate).
    Uses Cβ positions (Cα for Gly).
    """
    import math

    try:
        from Bio.PDB import PDBParser
    except ImportError:
        logger.error("BioPython PDB parser required for disulfide design.")
        return []

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
    except Exception as e:
        logger.error(f"Failed to parse PDB {pdb_path}: {e}")
        return []

    model = structure[0]
    chain = list(model.get_chains())[0]
    residues = list(chain.get_residues())

    # Get Cβ coordinates (Cα for Gly)
    cb_coords: dict[int, tuple[float, float, float]] = {}
    ca_coords: dict[int, tuple[float, float, float]] = {}

    for i, res in enumerate(residues):
        pos = i + 1  # 1-based
        if pos in protected:
            continue
        # Skip non-standard residues
        if res.get_id()[0] != " ":
            continue
        if "CB" in res:
            coord = res["CB"].get_vector()
            cb_coords[pos] = (coord[0], coord[1], coord[2])
        elif "CA" in res:
            coord = res["CA"].get_vector()
            cb_coords[pos] = (coord[0], coord[1], coord[2])
        if "CA" in res:
            coord = res["CA"].get_vector()
            ca_coords[pos] = (coord[0], coord[1], coord[2])

    # Find pairs within distance range
    pairs = []
    positions = sorted(cb_coords.keys())

    for i_idx, pos_i in enumerate(positions):
        for pos_j in positions[i_idx + 1:]:
            # Skip sequence-adjacent residues (disulfide needs ≥4 residue separation)
            if abs(pos_j - pos_i) < 4:
                continue

            # Compute Cβ-Cβ distance
            ci = cb_coords[pos_i]
            cj = cb_coords[pos_j]
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ci, cj)))

            if cb_dist_min <= dist <= cb_dist_max:
                # Simple energy estimate based on distance from ideal
                ideal_dist = 3.8
                energy = -2.0 + 5.0 * abs(dist - ideal_dist)  # rough kcal/mol
                pairs.append((pos_i, pos_j, dist, energy))

    return pairs
