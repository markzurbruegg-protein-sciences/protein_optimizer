"""Tier 2 — PSSM Analysis.

Builds a Position-Specific Scoring Matrix from the MSA and scores
every possible single-point mutation by log-odds ratio. Identifies
stabilizing mutations where the wild-type residue is suboptimal.

Usage:
    protein-opt step pssm_analysis -i homologs_result.json -o pssm_result.json
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

from Bio import AlignIO

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
BG_FREQ = {aa: 1.0 / 20 for aa in AMINO_ACIDS}  # uniform background


class PSSMAnalysisStep(BaseStep):
    name = "pssm_analysis"
    tier = 2
    title = "PSSM Analysis"
    description = "Score mutations by PSSM log-odds from MSA for stability engineering."
    requires = ["find_homologs"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        min_log_odds = config.get("min_log_odds", 2.0)
        max_mutations = config.get("max_mutations", 30)
        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            if parent.parent_id is not None:
                continue

            candidates.append(parent)

            msa_path = parent.metadata.get("msa_fasta")
            if not msa_path or not Path(msa_path).exists():
                warnings.append(f"{parent.name}: No MSA found. Run find_homologs first.")
                continue

            try:
                alignment = AlignIO.read(msa_path, "fasta")
            except Exception as e:
                warnings.append(f"{parent.name}: Failed to parse MSA: {e}")
                continue

            # Build PSSM
            pssm = _build_pssm(alignment, parent.sequence)
            if not pssm:
                warnings.append(f"{parent.name}: Could not build PSSM.")
                continue

            # Store PSSM in metadata
            parent.metadata["pssm"] = pssm

            # Find beneficial mutations (mutant log-odds > WT + threshold)
            beneficial = _find_beneficial_mutations(
                parent.sequence, pssm, min_log_odds, protected
            )

            if not beneficial:
                warnings.append(f"{parent.name}: No beneficial PSSM mutations found.")
                continue

            beneficial.sort(key=lambda m: m.score or 0, reverse=True)
            beneficial = beneficial[:max_mutations]

            logger.info(f"{parent.name}: Found {len(beneficial)} PSSM-beneficial mutations")

            for mut in beneficial:
                try:
                    variant = parent.apply_mutation(mut)
                    variant.scores["pssm_log_odds"] = mut.score or 0.0
                    candidates.append(variant)
                except ValueError as e:
                    logger.debug(f"Skipping mutation {mut.label}: {e}")

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _build_pssm(
    alignment,
    wt_sequence: str,
    pseudocount: float = 0.5,
) -> list[dict[str, float]]:
    """Build PSSM (log-odds matrix) from MSA.

    Returns list of dicts, one per WT position, mapping AA → log-odds score.
    """
    num_seqs = len(alignment)
    aln_length = alignment.get_alignment_length()

    # Find WT in alignment
    wt_aligned = str(alignment[0].seq)

    pssm: list[dict[str, float]] = []
    wt_pos = 0

    for col_idx in range(aln_length):
        wt_aa = wt_aligned[col_idx]
        if wt_aa == "-":
            continue

        wt_pos += 1

        # Count amino acids at this column
        counts: dict[str, float] = {aa: pseudocount for aa in AMINO_ACIDS}
        total_seqs = 0
        for record in alignment:
            aa = str(record.seq)[col_idx] if col_idx < len(record.seq) else "-"
            if aa in counts:
                counts[aa] += 1
                total_seqs += 1

        total = sum(counts.values())

        # Compute log-odds
        log_odds: dict[str, float] = {}
        for aa in AMINO_ACIDS:
            freq = counts[aa] / total
            bg = BG_FREQ[aa]
            log_odds[aa] = math.log2(freq / bg) if freq > 0 else -10.0

        pssm.append(log_odds)

    return pssm


def _find_beneficial_mutations(
    wt_sequence: str,
    pssm: list[dict[str, float]],
    min_log_odds: float,
    protected: set[int],
) -> list[Mutation]:
    """Find mutations where the mutant has a significantly higher PSSM score."""
    mutations = []

    for i, (wt_aa, scores) in enumerate(zip(wt_sequence, pssm)):
        pos = i + 1  # 1-based
        if pos in protected:
            continue

        wt_score = scores.get(wt_aa, 0.0)

        for aa in AMINO_ACIDS:
            if aa == wt_aa:
                continue
            mut_score = scores.get(aa, -10.0)
            delta = mut_score - wt_score

            if delta >= min_log_odds:
                mutations.append(Mutation(
                    position=pos,
                    wt=wt_aa,
                    mut=aa,
                    source_step="pssm_analysis",
                    score=delta,
                    metadata={
                        "wt_pssm_score": wt_score,
                        "mut_pssm_score": mut_score,
                        "delta_log_odds": delta,
                    },
                ))

    return mutations
