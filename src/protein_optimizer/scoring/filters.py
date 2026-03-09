"""Candidate filtering by quality thresholds and rules.

Applies hard filters (e.g., max mutation count, minimum pLDDT,
no new cysteines) and soft warnings to narrow down the variant
library before final ranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configurable filter thresholds."""

    max_mutations: int = 15
    min_plddt: float = 60.0
    max_ddg: float = 5.0             # kcal/mol — reject highly destabilising
    min_e1_fitness: float | None = None
    min_esm1v_delta: float | None = None
    min_recovery: float = 0.0         # MPNN sequence recovery
    forbid_new_cysteines: bool = True  # for E. coli
    forbid_prolines_in_helix: bool = True
    max_hydrophobic_surface: float | None = None
    require_scores: list[str] = field(default_factory=list)


def filter_candidates(
    result: StepResult,
    config: dict[str, Any] | FilterConfig | None = None,
) -> tuple[list[ProteinCandidate], list[ProteinCandidate]]:
    """Filter candidates, returning (passed, failed).

    Args:
        result: StepResult with scored candidates.
        config: Filter configuration dict or FilterConfig.

    Returns:
        Tuple of (passed_candidates, failed_candidates).
    """
    if config is None:
        fc = FilterConfig()
    elif isinstance(config, dict):
        fc = FilterConfig(**{k: v for k, v in config.items() if hasattr(FilterConfig, k)})
    else:
        fc = config

    passed = []
    failed = []

    for candidate in result.candidates:
        # Parents always pass
        if candidate.parent_id is None:
            passed.append(candidate)
            continue

        reasons = _check_filters(candidate, fc)

        if reasons:
            candidate.metadata["filter_fail_reasons"] = reasons
            failed.append(candidate)
            logger.debug(f"FAIL {candidate.name}: {', '.join(reasons)}")
        else:
            passed.append(candidate)

    logger.info(
        f"Filter: {len(passed)} passed, {len(failed)} failed "
        f"out of {len(result.candidates)} total"
    )

    return passed, failed


def _check_filters(
    candidate: ProteinCandidate, fc: FilterConfig
) -> list[str]:
    """Check all filters, returning list of failure reasons."""
    reasons = []

    # Mutation count
    if len(candidate.mutations) > fc.max_mutations:
        reasons.append(f"too many mutations ({len(candidate.mutations)} > {fc.max_mutations})")

    # pLDDT
    plddt = candidate.scores.get("val_plddt") or candidate.scores.get("plddt")
    if plddt is not None and plddt < fc.min_plddt:
        reasons.append(f"low pLDDT ({plddt:.1f} < {fc.min_plddt})")

    # ΔΔG
    ddg = candidate.scores.get("ddg")
    if ddg is not None and ddg > fc.max_ddg:
        reasons.append(f"high ΔΔG ({ddg:.2f} > {fc.max_ddg} kcal/mol)")

    # E1 fitness
    if fc.min_e1_fitness is not None:
        e1 = candidate.scores.get("e1_fitness")
        if e1 is not None and e1 < fc.min_e1_fitness:
            reasons.append(f"low E1 fitness ({e1:.4f} < {fc.min_e1_fitness})")

    # ESM-1v delta
    if fc.min_esm1v_delta is not None:
        delta = candidate.scores.get("esm1v_delta")
        if delta is not None and delta < fc.min_esm1v_delta:
            reasons.append(f"negative ESM-1v Δ ({delta:.4f})")

    # MPNN recovery
    recovery = candidate.scores.get("mpnn_recovery")
    if recovery is not None and recovery < fc.min_recovery:
        reasons.append(f"low MPNN recovery ({recovery:.2f})")

    # No new cysteines check
    if fc.forbid_new_cysteines:
        for mut in candidate.mutations:
            if mut.mut == "C":
                reasons.append(f"introduces Cys at {mut.position}")
                break

    # Proline in helix (sequence-based heuristic)
    if fc.forbid_prolines_in_helix:
        for mut in candidate.mutations:
            if mut.mut == "P":
                # Simple check: if the position was in a predicted helix
                helix_info = candidate.metadata.get("secondary_structure", "")
                if helix_info and len(helix_info) > mut.position - 1:
                    if helix_info[mut.position - 1] == "H":
                        reasons.append(f"Pro in helix at {mut.position}")

    # Required scores
    for score_key in fc.require_scores:
        if score_key not in candidate.scores:
            reasons.append(f"missing required score: {score_key}")

    return reasons


def apply_diversity_filter(
    candidates: list[ProteinCandidate],
    min_sequence_distance: int = 2,
    max_candidates: int = 50,
) -> list[ProteinCandidate]:
    """Ensure diversity in final candidate set.

    Removes candidates that are too similar (differ by fewer than
    min_sequence_distance mutations from any already-selected candidate).
    """
    if not candidates:
        return []

    selected = [candidates[0]]  # best candidate always included

    for candidate in candidates[1:]:
        if len(selected) >= max_candidates:
            break

        is_diverse = True
        for sel in selected:
            dist = _hamming_distance(candidate.sequence, sel.sequence)
            if dist < min_sequence_distance:
                is_diverse = False
                break

        if is_diverse:
            selected.append(candidate)

    logger.info(
        f"Diversity filter: {len(selected)} selected from {len(candidates)}"
    )

    return selected


def _hamming_distance(seq1: str, seq2: str) -> int:
    """Count positions where sequences differ."""
    return sum(a != b for a, b in zip(seq1, seq2))
