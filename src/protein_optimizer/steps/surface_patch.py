"""Tier 3 — Surface Hydrophobic Patch Analysis.

Identifies solvent-exposed hydrophobic patches that can cause aggregation
and proposes surface hydrophobic→polar/charged substitutions to improve
solubility.

Usage:
    protein-opt step surface_patch -i structure_result.json -o surface_result.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

# Hydrophobic residues that should not be on the surface
SURFACE_HYDROPHOBICS = set("ILMFVW")

# Conservative mutations to reduce surface hydrophobicity
SURFACE_SUBSTITUTIONS: dict[str, list[str]] = {
    "I": ["T", "K"],       # Ile → Thr (branched, similar shape) or Lys
    "L": ["K", "Q"],       # Leu → Lys (similar size) or Gln
    "M": ["K", "Q"],       # Met → Lys, Gln
    "F": ["Y", "R"],       # Phe → Tyr (add OH) or Arg
    "V": ["T", "D"],       # Val → Thr or Asp
    "W": ["R", "Y"],       # Trp → Arg (large) or Tyr
}

# Kyte–Doolittle hydrophobicity scale
KD_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


class SurfacePatchStep(BaseStep):
    name = "surface_patch"
    tier = 3
    title = "Surface Patch Analysis"
    description = "Reduce solvent-exposed hydrophobic patches to lower aggregation."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        sasa_threshold = config.get("sasa_threshold", 0.25)  # fraction of max SASA
        patch_min_size = config.get("patch_min_size", 3)      # min residues in a patch
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
                warnings.append(f"{parent.name}: No structure for surface analysis.")
                continue

            patches = _find_hydrophobic_patches(
                pdb_path, parent.sequence, protected, sasa_threshold, patch_min_size
            )

            if not patches:
                logger.info(f"{parent.name}: No concerning hydrophobic patches found.")
                continue

            # Generate mutations from patches
            proposed = []
            for patch in patches:
                for pos, aa, sap_score in patch:
                    if aa not in SURFACE_SUBSTITUTIONS:
                        continue
                    for mut_aa in SURFACE_SUBSTITUTIONS[aa]:
                        proposed.append(Mutation(
                            position=pos,
                            wt=aa,
                            mut=mut_aa,
                            source_step=self.name,
                            score=sap_score,
                            metadata={
                                "sap_score": sap_score,
                                "patch_size": len(patch),
                            },
                        ))

            # Prioritize largest patches, highest SAP scores
            proposed.sort(key=lambda m: (
                m.metadata.get("patch_size", 0),
                m.score or 0,
            ), reverse=True)
            proposed = proposed[:max_mutations]

            logger.info(
                f"{parent.name}: {len(patches)} hydrophobic patches, "
                f"{len(proposed)} mutations proposed"
            )

            for mut in proposed:
                try:
                    variant = parent.apply_mutation(mut)
                    variant.scores["surface_sap"] = mut.score or 0.0
                    candidates.append(variant)
                except ValueError as e:
                    logger.debug(f"Skipping {mut.label}: {e}")

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _find_hydrophobic_patches(
    pdb_path: str,
    sequence: str,
    protected: set[int],
    sasa_threshold: float,
    patch_min_size: int,
) -> list[list[tuple[int, str, float]]]:
    """Find clusters of solvent-exposed hydrophobic residues.

    Uses DSSP/Shrake-Rupley SASA estimation + spatial proximity to identify
    contiguous hydrophobic patches on the protein surface.

    Returns list of patches, each a list of (position, aa, sap_score).
    """
    try:
        from Bio.PDB import PDBParser, ShrakeRupley, NeighborSearch
    except ImportError:
        logger.error("BioPython required for surface patch analysis.")
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

    # Compute SASA
    sr = ShrakeRupley()
    try:
        sr.compute(model, level="R")
    except Exception as e:
        logger.warning(f"SASA computation failed: {e}, using B-factor fallback")
        return _fallback_surface_scan(residues, sequence, protected, patch_min_size)

    # Max SASA reference values (Gly-X-Gly tripeptides, Å²)
    MAX_SASA = {
        "A": 129, "R": 274, "N": 195, "D": 193, "C": 167,
        "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
        "L": 201, "K": 236, "M": 224, "F": 240, "P": 159,
        "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
    }

    # Find surface-exposed hydrophobic residues
    surface_hydrophobics: list[tuple[int, str, float]] = []
    residue_ca_coords = {}

    for i, res in enumerate(residues):
        pos = i + 1
        if pos in protected or res.get_id()[0] != " ":
            continue
        if pos > len(sequence):
            continue

        aa = sequence[pos - 1]
        if aa not in SURFACE_HYDROPHOBICS:
            continue

        sasa = res.sasa if hasattr(res, "sasa") else 0.0
        max_sasa = MAX_SASA.get(aa, 200)
        rel_sasa = sasa / max_sasa

        if rel_sasa < sasa_threshold:
            continue  # not exposed enough

        # SAP-like score: relative SASA × hydrophobicity
        sap_score = rel_sasa * KD_HYDROPHOBICITY.get(aa, 0.0)

        if "CA" in res:
            residue_ca_coords[len(surface_hydrophobics)] = res["CA"].get_vector().get_array()

        surface_hydrophobics.append((pos, aa, sap_score))

    if len(surface_hydrophobics) < patch_min_size:
        return []

    # Cluster spatially close surface hydrophobics into patches
    patches = _cluster_residues(surface_hydrophobics, residue_ca_coords, distance=8.0)

    # Filter by minimum patch size
    patches = [p for p in patches if len(p) >= patch_min_size]

    return patches


def _cluster_residues(
    residue_list: list[tuple[int, str, float]],
    ca_coords: dict[int, Any],
    distance: float = 8.0,
) -> list[list[tuple[int, str, float]]]:
    """Simple single-linkage clustering of residues by Cα distance."""
    import numpy as np

    n = len(residue_list)
    if n == 0:
        return []

    # Build adjacency by distance
    clusters: list[set[int]] = [{i} for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if i not in ca_coords or j not in ca_coords:
                # Fallback: cluster by sequence proximity
                if abs(residue_list[i][0] - residue_list[j][0]) <= 5:
                    _merge_clusters(clusters, i, j)
                continue
            dist = np.linalg.norm(
                np.array(ca_coords[i]) - np.array(ca_coords[j])
            )
            if dist <= distance:
                _merge_clusters(clusters, i, j)

    # Deduplicate clusters
    unique_clusters: list[set[int]] = []
    seen: set[int] = set()
    for c in clusters:
        key = min(c)
        if key not in seen:
            unique_clusters.append(c)
            seen.update(c)

    result = []
    for cluster in unique_clusters:
        patch = [residue_list[idx] for idx in sorted(cluster)]
        result.append(patch)

    return result


def _merge_clusters(clusters: list[set[int]], i: int, j: int) -> None:
    """Merge the clusters containing i and j."""
    ci = cj = None
    for c in clusters:
        if i in c:
            ci = c
        if j in c:
            cj = c
    if ci is not None and cj is not None and ci is not cj:
        ci.update(cj)
        clusters.remove(cj)


def _fallback_surface_scan(
    residues, sequence: str, protected: set[int], patch_min_size: int
) -> list[list[tuple[int, str, float]]]:
    """Sequence-based fallback when SASA can't be computed."""
    window = 9
    patches = []
    current_patch = []

    for i in range(len(sequence)):
        pos = i + 1
        if pos in protected:
            if len(current_patch) >= patch_min_size:
                patches.append(current_patch)
            current_patch = []
            continue

        aa = sequence[i]
        if aa in SURFACE_HYDROPHOBICS:
            score = KD_HYDROPHOBICITY.get(aa, 0.0)
            current_patch.append((pos, aa, score))
        else:
            if len(current_patch) >= patch_min_size:
                patches.append(current_patch)
            current_patch = []

    if len(current_patch) >= patch_min_size:
        patches.append(current_patch)

    return patches
