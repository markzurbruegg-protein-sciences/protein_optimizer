"""Score aggregation across multiple scoring steps.

Normalizes and weights scores from different sources (E1, ESM-1v,
ESM-IF1, ΔΔG, MPNN, etc.) into a single composite fitness score
for ranking variant candidates.
"""

from __future__ import annotations

import logging
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult

logger = logging.getLogger(__name__)

# Default weights for each scoring method
DEFAULT_WEIGHTS: dict[str, float] = {
    "e1_fitness": 0.25,
    "esm1v_delta": 0.20,
    "esmif1_delta": 0.15,
    "mpnn_score": 0.10,
    "consensus_score": 0.10,
    "pssm_delta": 0.10,
    "ddg": 0.05,
    "cavity_burial": 0.025,
    "surface_sap": 0.025,
    "delta_solubility": 0.10,
    "evidence_score": 0.05,
}


def aggregate_scores(
    result: StepResult,
    weights: dict[str, float] | None = None,
    normalize: bool = True,
) -> StepResult:
    """Add composite_score to each candidate by weighted combination.

    Args:
        result: StepResult with scored candidates.
        weights: Custom score weights (overrides defaults).
        normalize: Whether to z-score normalize each score before weighting.

    Returns:
        Same StepResult with composite_score added to each candidate.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    candidates = result.candidates
    if not candidates:
        return result

    # Collect all available score keys
    all_keys: set[str] = set()
    for c in candidates:
        all_keys.update(c.scores.keys())

    scored_keys = [k for k in all_keys if k in w]

    if not scored_keys:
        logger.warning("No weightable scores found — skipping aggregation.")
        return result

    # Compute statistics for normalization
    if normalize:
        stats = _compute_stats(candidates, scored_keys)
    else:
        stats = {}

    # Compute composite score
    for candidate in candidates:
        composite = 0.0
        total_weight = 0.0

        for key in scored_keys:
            if key not in candidate.scores:
                continue

            raw_value = candidate.scores[key]
            weight = w.get(key, 0.0)

            if normalize and key in stats:
                mean, std = stats[key]
                if std > 1e-8:
                    value = (raw_value - mean) / std
                else:
                    value = 0.0
            else:
                value = raw_value

            # ΔΔG is "lower is better" → invert
            if key == "ddg":
                value = -value
            # SAP score is "higher is worse" → invert
            if key == "surface_sap":
                value = -value

            composite += weight * value
            total_weight += weight

        if total_weight > 0:
            candidate.scores["composite_score"] = composite / total_weight
        else:
            candidate.scores["composite_score"] = 0.0

    return result


def _compute_stats(
    candidates: list[ProteinCandidate],
    keys: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute mean and std for each score key."""
    stats = {}

    for key in keys:
        values = [c.scores[key] for c in candidates if key in c.scores]
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        stats[key] = (mean, std)

    return stats


def rank_candidates(
    result: StepResult,
    score_key: str = "composite_score",
    ascending: bool = False,
) -> list[ProteinCandidate]:
    """Rank candidates by a score, returning sorted list.

    Args:
        result: StepResult with scored candidates.
        score_key: Which score to rank by.
        ascending: If True, lower is better (e.g., ΔΔG).
    """
    scored = [c for c in result.candidates if score_key in c.scores]
    unscored = [c for c in result.candidates if score_key not in c.scores]

    scored.sort(
        key=lambda c: c.scores.get(score_key, float("-inf")),
        reverse=not ascending,
    )

    return scored + unscored
