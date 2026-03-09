"""Tier 1 — Problematic Motif Scanner.

Flags and proposes fixes for sequence motifs known to cause problems
in protein production:
  - Deamidation sites (Asn-Gly, Asn-Ser, Asn-His, Asp-Gly)
  - Oxidation-prone surface Met residues
  - Protease-susceptible dibasic sites (Arg-Arg, Lys-Arg, etc.)
  - Aggregation-prone hydrophobic runs

Usage:
    protein-opt step motif_scan -i my_enzyme.fasta -o motif_results.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep


@dataclass
class MotifHit:
    """A detected problematic motif."""
    category: str
    pattern: str
    position: int  # 1-based start position
    residues: str  # matched substring
    risk: float  # 0–1
    fix_position: int  # 1-based position to mutate
    fix_wt: str
    fix_mut: str
    rationale: str


# ── Motif definitions ─────────────────────────────────────────────────────

DEAMIDATION_PATTERNS: list[tuple[str, float, str]] = [
    # (regex, risk_score, rationale)
    ("NG", 1.0, "Asn-Gly: highest deamidation rate (~10x baseline)"),
    ("NS", 0.7, "Asn-Ser: high deamidation rate"),
    ("NH", 0.5, "Asn-His: moderate deamidation rate"),
    ("NA", 0.3, "Asn-Ala: moderate deamidation rate"),
    ("NT", 0.3, "Asn-Thr: moderate deamidation rate"),
    ("DG", 0.6, "Asp-Gly: isomerization-prone (succinimide intermediate)"),
]

OXIDATION_RESIDUES = "M"  # methionine
OXIDATION_REPLACEMENTS = {"M": "L"}  # Met → Leu (conservative)

DIBASIC_PATTERNS: list[tuple[str, float, str]] = [
    ("RR", 0.8, "Arg-Arg: protease-susceptible dibasic site"),
    ("KR", 0.7, "Lys-Arg: furin/kexin protease recognition"),
    ("RK", 0.6, "Arg-Lys: protease-susceptible dibasic site"),
    ("KK", 0.4, "Lys-Lys: moderate protease susceptibility"),
]


class MotifScanStep(BaseStep):
    name = "motif_scan"
    tier = 1
    title = "Motif Scanner"
    description = "Flag deamidation, oxidation, proteolysis, and aggregation-prone motifs."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        check_deamidation = config.get("check_deamidation", True)
        check_oxidation = config.get("check_oxidation", True)
        check_proteolysis = config.get("check_proteolysis", True)
        check_aggregation = config.get("check_aggregation", True)
        hydrophobic_run_len = config.get("hydrophobic_run_length", 5)
        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            candidates.append(parent)
            seq = parent.sequence
            all_hits: list[MotifHit] = []

            if check_deamidation:
                all_hits.extend(_scan_deamidation(seq))
            if check_oxidation:
                all_hits.extend(_scan_oxidation(seq))
            if check_proteolysis:
                all_hits.extend(_scan_dibasic(seq))
            if check_aggregation:
                all_hits.extend(_scan_hydrophobic_runs(seq, hydrophobic_run_len))

            if not all_hits:
                warnings.append(f"{parent.name}: No problematic motifs found.")
                continue

            # Store all hits in parent metadata
            parent.metadata["motif_hits"] = [
                {
                    "category": h.category,
                    "pattern": h.pattern,
                    "position": h.position,
                    "residues": h.residues,
                    "risk": h.risk,
                    "rationale": h.rationale,
                }
                for h in all_hits
            ]
            parent.scores["motif_risk_total"] = sum(h.risk for h in all_hits)
            parent.scores["motif_count"] = len(all_hits)

            # Generate fix candidates for each hit (skip protected)
            for hit in all_hits:
                if hit.fix_position in protected:
                    continue
                if hit.fix_wt == hit.fix_mut:
                    continue
                mut = Mutation(
                    position=hit.fix_position,
                    wt=hit.fix_wt,
                    mut=hit.fix_mut,
                    source_step=self.name,
                    score=hit.risk,
                    metadata={
                        "category": hit.category,
                        "rationale": hit.rationale,
                    },
                )
                try:
                    variant = parent.apply_mutation(mut)
                    variant.scores["motif_risk_fixed"] = hit.risk
                    candidates.append(variant)
                except ValueError:
                    # Mutation can't be applied (e.g., already mutated position)
                    pass

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _scan_deamidation(seq: str) -> list[MotifHit]:
    hits = []
    for pattern, risk, rationale in DEAMIDATION_PATTERNS:
        for m in re.finditer(pattern, seq):
            pos = m.start() + 1  # 1-based
            # Fix the Asn (first residue of the pair) → Gln (conservative)
            hits.append(MotifHit(
                category="deamidation",
                pattern=pattern,
                position=pos,
                residues=m.group(),
                risk=risk,
                fix_position=pos,
                fix_wt=seq[m.start()],
                fix_mut="Q" if seq[m.start()] == "N" else "E",  # Asn→Gln or Asp→Glu
                rationale=rationale,
            ))
    return hits


def _scan_oxidation(seq: str) -> list[MotifHit]:
    hits = []
    for i, aa in enumerate(seq):
        if aa in OXIDATION_RESIDUES:
            # Estimate surface exposure from flanking charges
            window = seq[max(0, i - 3): min(len(seq), i + 4)]
            charged = sum(1 for c in window if c in "DEKRH")
            # Higher risk if likely surface-exposed
            risk = 0.3 + 0.15 * min(charged, 3)
            hits.append(MotifHit(
                category="oxidation",
                pattern=f"Met{i + 1}",
                position=i + 1,
                residues=aa,
                risk=risk,
                fix_position=i + 1,
                fix_wt="M",
                fix_mut=OXIDATION_REPLACEMENTS["M"],
                rationale=f"Surface-exposed Met: oxidation risk (flanking charged: {charged})",
            ))
    return hits


def _scan_dibasic(seq: str) -> list[MotifHit]:
    hits = []
    for pattern, risk, rationale in DIBASIC_PATTERNS:
        for m in re.finditer(pattern, seq):
            pos = m.start() + 1
            # Fix: mutate the second residue to maintain charge with Gln
            fix_pos = m.start() + 2  # 1-based, second residue
            hits.append(MotifHit(
                category="proteolysis",
                pattern=pattern,
                position=pos,
                residues=m.group(),
                risk=risk,
                fix_position=fix_pos,
                fix_wt=seq[m.start() + 1],
                fix_mut="Q",  # Arg/Lys → Gln (maintains polar character, removes basicity)
                rationale=rationale,
            ))
    return hits


def _scan_hydrophobic_runs(seq: str, min_length: int = 5) -> list[MotifHit]:
    """Detect runs of ≥min_length consecutive hydrophobic residues."""
    hits = []
    hydrophobic = set("ILMFVW")
    pattern = re.compile(f"[ILMFVW]{{{min_length},}}")
    for m in pattern.finditer(seq):
        pos = m.start() + 1
        length = len(m.group())
        risk = min(0.3 + 0.1 * (length - min_length), 1.0)
        # Fix: replace the middle residue with Lys (charge break)
        mid = m.start() + length // 2
        hits.append(MotifHit(
            category="aggregation",
            pattern=f"hydrophobic_run_{length}",
            position=pos,
            residues=m.group(),
            risk=risk,
            fix_position=mid + 1,
            fix_wt=seq[mid],
            fix_mut="K",  # insert a charged residue to break the run
            rationale=f"Hydrophobic run of {length} residues: aggregation risk",
        ))
    return hits
