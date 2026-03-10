"""Tier 4 — Combine Variants.

Collects the best individual mutations from all prior steps and creates
combinatorial variant libraries.  Uses a greedy additive approach:
takes the top-N scoring single mutations and builds all pairwise and
triple combinations (optionally higher-order).

Usage:
    protein-opt step combine_variants -i scored_result.json -o combined.json
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class CombineVariantsStep(BaseStep):
    name = "combine_variants"
    tier = 4
    title = "Combine Top Variants"
    description = "Build combinatorial libraries from top-ranked individual mutations."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        max_single = config.get("max_single_mutations", 15)
        max_order = config.get("max_combination_order", 3)
        max_library_size = config.get("max_library_size", 200)
        ranking_score = config.get("ranking_score", None)  # auto-detect
        min_score = config.get("min_score_threshold", 0.0)
        exclude_steps = set(config.get("exclude_steps", []))

        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        # --- Collect mutations from ALL prior steps, not just the last ---
        prior_results: dict[str, StepResult] = config.get("_prior_results", {})

        # Identify parent candidates from the current input
        parents = [c for c in step_input.candidates if c.parent_id is None]
        variants = [c for c in step_input.candidates if c.parent_id is not None]

        # Also collect all variants from prior step results
        all_prior_variants: list[ProteinCandidate] = list(variants)
        seen_ids: set[str] = {v.candidate_id for v in variants}
        for step_name_key, prior_result in prior_results.items():
            if step_name_key == self.name:
                continue
            if not isinstance(prior_result, StepResult):
                continue
            for c in prior_result.candidates:
                if c.parent_id is not None and c.candidate_id not in seen_ids:
                    all_prior_variants.append(c)
                    seen_ids.add(c.candidate_id)

        logger.info(
            f"Collected {len(all_prior_variants)} total variants from "
            f"{len(prior_results)} prior steps + current input"
        )

        for parent in parents:
            candidates.append(parent)

            # Gather all single-point mutations for this parent
            parent_variants = [
                v for v in all_prior_variants if v.parent_id == parent.candidate_id
            ]

            # Collect unique mutations with their best scores
            mutation_scores: dict[str, tuple[Mutation, float]] = {}

            for variant in parent_variants:
                if not variant.mutations:
                    # Pass through generative designs (e.g. RFdiffusion, ProteinMPNN)
                    # that carry no explicit mutation list — treat them as multi-mutants.
                    candidates.append(variant)
                    continue

                source = variant.mutations[0].source_step if variant.mutations else ""
                if source in exclude_steps:
                    continue

                # Only consider single-point variants for combination
                if len(variant.mutations) != 1:
                    candidates.append(variant)  # keep multi-mutants as-is
                    continue

                candidates.append(variant)

                mut = variant.mutations[0]
                if mut.position in protected:
                    continue

                # Score for ranking
                score = _get_best_score(variant, ranking_score)
                key = mut.label

                if key not in mutation_scores or score > mutation_scores[key][1]:
                    mutation_scores[key] = (mut, score)

            # Rank mutations
            ranked_mutations = sorted(
                mutation_scores.values(),
                key=lambda x: x[1],
                reverse=True,
            )

            # Filter by minimum score
            ranked_mutations = [
                (m, s) for m, s in ranked_mutations if s >= min_score
            ]

            top_mutations = [m for m, s in ranked_mutations[:max_single]]

            if len(top_mutations) < 2:
                logger.info(
                    f"{parent.name}: Only {len(top_mutations)} mutations — "
                    "skipping combinations."
                )
                continue

            logger.info(
                f"{parent.name}: Combining top {len(top_mutations)} mutations "
                f"(order 2–{max_order})"
            )

            # Generate combinations
            combo_count = 0
            for order in range(2, max_order + 1):
                if combo_count >= max_library_size:
                    break

                for combo in itertools.combinations(top_mutations, order):
                    if combo_count >= max_library_size:
                        break

                    # Check for position conflicts
                    positions = [m.position for m in combo]
                    if len(positions) != len(set(positions)):
                        continue  # conflicting mutations at same position

                    # Apply all mutations
                    try:
                        variant = parent.apply_mutations(list(combo))
                        # Aggregate scores from individual mutations
                        individual_scores = []
                        for m in combo:
                            key = m.label
                            if key in mutation_scores:
                                individual_scores.append(mutation_scores[key][1])

                        variant.scores["combo_sum"] = sum(individual_scores)
                        variant.scores["combo_mean"] = (
                            sum(individual_scores) / len(individual_scores)
                            if individual_scores else 0.0
                        )
                        variant.scores["n_mutations"] = len(combo)
                        variant.metadata["combination_order"] = order
                        variant.metadata["component_mutations"] = [
                            m.label for m in combo
                        ]

                        candidates.append(variant)
                        combo_count += 1
                    except ValueError as e:
                        logger.debug(f"Skipping combination: {e}")

            logger.info(
                f"{parent.name}: Generated {combo_count} combinatorial variants"
            )

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _get_best_score(
    variant: ProteinCandidate, preferred_score: str | None
) -> float:
    """Get the best available score for ranking.

    Priority tuned for pre-PLM ranking since PLMs score at the end.
    Structural/evolutionary scores first, then PLMs if available.
    
    Note: ddg is inverted (more negative = better = higher rank).
    """
    if preferred_score and preferred_score in variant.scores:
        val = variant.scores[preferred_score]
        # Invert scores where lower is better
        if preferred_score in ("ddg", "surface_sap"):
            return -val
        return val

    # Priority tuned for pre-PLM ranking (PLMs score at end):
    # structural scores first, then evolutionary, then PLM if available
    priority = [
        "ddg", "pssm_log_odds", "consensus_conservation",
        "mpnn_score", "cavity_burial", "disulfide_energy_estimate",
        "surface_sap", "motif_risk_fixed", "cys_risk",
        "e1_fitness", "esm1v_delta", "esmif1_delta",
    ]

    # Scores where lower = better (invert for ranking)
    _INVERT = {"ddg", "surface_sap"}

    for key in priority:
        if key in variant.scores:
            val = variant.scores[key]
            return -val if key in _INVERT else val

    # Return any score
    if variant.scores:
        return max(variant.scores.values())

    return 0.0
