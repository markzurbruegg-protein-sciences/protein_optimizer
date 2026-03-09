"""Tier 2 — Consensus Sequence Design.

Computes a consensus sequence from the MSA and proposes back-to-consensus
mutations. Each consensus mutation typically adds +1–2°C thermostability.

Usage:
    protein-opt step consensus_design -i homologs_result.json -o consensus_result.json

Requires: MSA output from find_homologs step (or user-provided MSA FASTA).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from Bio import AlignIO

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class ConsensusDesignStep(BaseStep):
    name = "consensus_design"
    tier = 2
    title = "Consensus Design"
    description = "Propose back-to-consensus mutations from MSA for thermostability."
    requires = ["find_homologs"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        min_conservation = config.get("min_conservation", 0.5)
        max_mutations = config.get("max_mutations", 20)
        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            if parent.parent_id is not None:
                # Skip non-WT candidates (only process original sequences)
                continue

            candidates.append(parent)

            # Get MSA path
            msa_path = parent.metadata.get("msa_fasta")
            if not msa_path or not Path(msa_path).exists():
                warnings.append(
                    f"{parent.name}: No MSA found. Run find_homologs first "
                    f"or provide --msa-path."
                )
                continue

            # Parse MSA
            try:
                alignment = AlignIO.read(msa_path, "fasta")
            except Exception as e:
                warnings.append(f"{parent.name}: Failed to parse MSA: {e}")
                continue

            # Compute per-position consensus
            consensus_mutations = _compute_consensus_mutations(
                parent.sequence,
                alignment,
                min_conservation=min_conservation,
                protected=protected,
            )

            if not consensus_mutations:
                warnings.append(f"{parent.name}: No consensus mutations found above threshold.")
                continue

            logger.info(
                f"{parent.name}: Found {len(consensus_mutations)} consensus mutations"
            )

            # Sort by conservation score (highest first) and take top N
            consensus_mutations.sort(key=lambda m: m.score or 0, reverse=True)
            consensus_mutations = consensus_mutations[:max_mutations]

            # Generate individual mutant candidates
            for mut in consensus_mutations:
                try:
                    variant = parent.apply_mutation(mut)
                    variant.scores["consensus_conservation"] = mut.score or 0.0
                    candidates.append(variant)
                except ValueError as e:
                    logger.debug(f"Skipping mutation {mut.label}: {e}")

            # Generate a "consensus-optimized" candidate with top mutations
            top_n = min(5, len(consensus_mutations))
            if top_n > 1:
                try:
                    combo = parent.apply_mutations(consensus_mutations[:top_n])
                    combo.name = f"{parent.name}_consensus_top{top_n}"
                    combo.scores["consensus_conservation"] = sum(
                        m.score for m in consensus_mutations[:top_n] if m.score
                    )
                    candidates.append(combo)
                except ValueError as e:
                    logger.debug(f"Could not apply combined consensus mutations: {e}")

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _compute_consensus_mutations(
    wt_sequence: str,
    alignment,
    min_conservation: float,
    protected: set[int],
) -> list[Mutation]:
    """Compare wild-type to MSA consensus and find stabilizing mutations."""
    mutations = []
    num_seqs = len(alignment)

    # Find the WT sequence in the alignment
    wt_aligned = None
    for record in alignment:
        ungapped = str(record.seq).replace("-", "")
        if ungapped == wt_sequence:
            wt_aligned = str(record.seq)
            break

    if wt_aligned is None:
        # WT should be first sequence in MSA
        wt_aligned = str(alignment[0].seq)

    # Map alignment columns to WT residue positions
    wt_pos = 0  # 0-based position in ungapped WT
    for col_idx in range(alignment.get_alignment_length()):
        wt_aa = wt_aligned[col_idx] if col_idx < len(wt_aligned) else "-"

        if wt_aa == "-":
            continue  # insertion in homologs, not in WT

        wt_pos += 1  # now 1-based

        if wt_pos in protected:
            continue

        # Count amino acids at this column
        col_counts: Counter[str] = Counter()
        for record in alignment:
            aa = str(record.seq)[col_idx] if col_idx < len(record.seq) else "-"
            if aa != "-":
                col_counts[aa] += 1

        total = sum(col_counts.values())
        if total < 3:
            continue  # too few sequences

        # Find consensus amino acid
        consensus_aa, consensus_count = col_counts.most_common(1)[0]
        conservation = consensus_count / total

        # Only propose mutation if consensus differs from WT and is well-conserved
        if consensus_aa != wt_aa and conservation >= min_conservation:
            wt_freq = col_counts.get(wt_aa, 0) / total
            mutations.append(Mutation(
                position=wt_pos,
                wt=wt_aa,
                mut=consensus_aa,
                source_step="consensus_design",
                score=conservation,
                metadata={
                    "conservation": conservation,
                    "wt_frequency": wt_freq,
                    "column_depth": total,
                },
            ))

    return mutations
