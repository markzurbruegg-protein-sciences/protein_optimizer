"""Tier 1 — Cysteine Scanner & Replacement.

Identifies cysteine residues and proposes Cys→Ser or Cys→Ala mutations.
In E. coli's reducing cytoplasm, unpaired cysteines cause intermolecular
disulfide bonds and aggregation.

Usage:
    protein-opt step cysteine_scan -i my_enzyme.fasta -o cys_results.json
    protein-opt step cysteine_scan -i my_enzyme.fasta --protected-residues 34,78
"""

from __future__ import annotations

from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep


class CysteineScanStep(BaseStep):
    name = "cysteine_scan"
    tier = 1
    title = "Cysteine Scanner"
    description = "Identify cysteines and propose Cys→Ser/Ala replacements for E. coli expression."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        replacement = config.get("replacement", "S")
        generate_remove_all = config.get("generate_remove_all", True)
        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            # Always include the parent (wild type)
            candidates.append(parent)

            cys_positions = []
            for i, aa in enumerate(parent.sequence):
                pos = i + 1  # 1-based
                if aa == "C":
                    cys_positions.append(pos)

            if not cys_positions:
                warnings.append(f"{parent.name}: No cysteines found. No changes needed.")
                continue

            # Score each cysteine by context risk
            mutable_positions = [p for p in cys_positions if p not in protected]
            protected_positions = [p for p in cys_positions if p in protected]

            if protected_positions:
                warnings.append(
                    f"{parent.name}: Protected cysteines at positions "
                    f"{protected_positions} — skipped."
                )

            # Generate individual Cys→replacement candidates
            for pos in mutable_positions:
                mut = Mutation(
                    position=pos,
                    wt="C",
                    mut=replacement,
                    source_step=self.name,
                    score=_cysteine_risk_score(parent.sequence, pos),
                    metadata={"context": _get_context(parent.sequence, pos)},
                )
                variant = parent.apply_mutation(mut)
                variant.scores["cys_risk"] = mut.score or 0.0
                candidates.append(variant)

            # Generate remove-all-cysteines variant
            if generate_remove_all and len(mutable_positions) > 1:
                mutations = [
                    Mutation(
                        position=pos,
                        wt="C",
                        mut=replacement,
                        source_step=self.name,
                        score=_cysteine_risk_score(parent.sequence, pos),
                    )
                    for pos in mutable_positions
                ]
                all_removed = parent.apply_mutations(mutations)
                all_removed.name = f"{parent.name}_no_cys"
                all_removed.scores["cys_risk"] = sum(
                    m.score for m in mutations if m.score
                )
                candidates.append(all_removed)

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
            metadata={"replacement_aa": replacement},
        )


def _cysteine_risk_score(sequence: str, pos: int) -> float:
    """Heuristic risk score for an unpaired cysteine (0-1 scale).

    Higher = more problematic. Factors:
    - Flanking charged residues (surface-exposed → higher risk)
    - Proximity to other cysteines (potential unwanted disulfide)
    - N/C-terminal position (more exposed)
    """
    idx = pos - 1
    seq_len = len(sequence)
    risk = 0.5  # baseline

    # Terminal proximity increases exposure risk
    terminal_distance = min(idx, seq_len - 1 - idx)
    if terminal_distance < 5:
        risk += 0.2

    # Flanking charged residues suggest surface exposure
    window = sequence[max(0, idx - 3): min(seq_len, idx + 4)]
    charged = sum(1 for aa in window if aa in "DEKRH")
    if charged >= 2:
        risk += 0.15

    # Nearby cysteines: could form unwanted disulfide
    nearby_cys = sum(
        1 for i, aa in enumerate(sequence)
        if aa == "C" and i != idx and abs(i - idx) < 20
    )
    if nearby_cys > 0:
        risk += 0.15

    return min(risk, 1.0)


def _get_context(sequence: str, pos: int, window: int = 5) -> str:
    """Return sequence context around a position."""
    idx = pos - 1
    start = max(0, idx - window)
    end = min(len(sequence), idx + window + 1)
    context = sequence[start:end]
    # Mark the target position
    local_idx = idx - start
    return context[:local_idx] + f"[{context[local_idx]}]" + context[local_idx + 1:]
