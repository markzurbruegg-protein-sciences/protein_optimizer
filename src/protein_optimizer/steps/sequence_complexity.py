"""Tier 1 — Sequence Complexity Checks.

Flags problematic sequence features for E. coli expression:
  - Homopolymeric amino acid runs (≥5 identical residues)
  - Proline-rich stretches (ribosome stalling)
  - Low-complexity / repetitive regions
  - Extreme charge clusters

Usage:
    protein-opt step sequence_complexity -i my_enzyme.fasta -o complexity_results.json
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep


@dataclass
class ComplexityFlag:
    """A detected sequence complexity issue."""
    category: str
    position: int  # 1-based start
    length: int
    residues: str
    risk: float
    description: str


class SequenceComplexityStep(BaseStep):
    name = "sequence_complexity"
    tier = 1
    title = "Sequence Complexity"
    description = "Flag homopolymeric runs, proline stretches, and low-complexity regions."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        homo_threshold = config.get("homopolymer_threshold", 5)
        pro_threshold = config.get("proline_run_threshold", 3)

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            seq = parent.sequence
            flags: list[ComplexityFlag] = []

            # Homopolymeric runs
            flags.extend(_scan_homopolymers(seq, homo_threshold))

            # Proline-rich regions
            flags.extend(_scan_proline_runs(seq, pro_threshold))

            # Charge clusters
            flags.extend(_scan_charge_clusters(seq))

            # Low complexity (Shannon entropy)
            flags.extend(_scan_low_complexity_windows(seq))

            # Annotate parent with flags
            parent.metadata["complexity_flags"] = [
                {
                    "category": f.category,
                    "position": f.position,
                    "length": f.length,
                    "residues": f.residues,
                    "risk": f.risk,
                    "description": f.description,
                }
                for f in flags
            ]
            parent.scores["complexity_issues"] = len(flags)
            parent.scores["complexity_risk_total"] = sum(f.risk for f in flags)

            if not flags:
                warnings.append(f"{parent.name}: No complexity issues detected.")

            candidates.append(parent)

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
            metadata={"total_flags": sum(
                c.scores.get("complexity_issues", 0) for c in candidates
            )},
        )


def _scan_homopolymers(seq: str, threshold: int) -> list[ComplexityFlag]:
    """Find runs of ≥threshold identical amino acids."""
    flags = []
    pattern = re.compile(r"(.)\1{" + str(threshold - 1) + r",}")
    for m in pattern.finditer(seq):
        length = len(m.group())
        flags.append(ComplexityFlag(
            category="homopolymer",
            position=m.start() + 1,
            length=length,
            residues=m.group(),
            risk=min(0.3 + 0.1 * (length - threshold), 1.0),
            description=(
                f"Homopolymeric run of {length}x {m.group()[0]}: "
                f"may cause translational frameshifting or recombination in E. coli"
            ),
        ))
    return flags


def _scan_proline_runs(seq: str, threshold: int) -> list[ComplexityFlag]:
    """Detect consecutive proline residues (ribosome stalling)."""
    flags = []
    pattern = re.compile(f"P{{{threshold},}}")
    for m in pattern.finditer(seq):
        length = len(m.group())
        flags.append(ComplexityFlag(
            category="proline_run",
            position=m.start() + 1,
            length=length,
            residues=m.group(),
            risk=min(0.4 + 0.2 * (length - threshold), 1.0),
            description=(
                f"Consecutive proline run ({length}x Pro): "
                f"causes ribosome stalling in E. coli (requires EF-P)"
            ),
        ))

    # Also check for Pro-Pro-X-Pro type motifs (known stalling triggers)
    for m in re.finditer(r"PP.P", seq):
        flags.append(ComplexityFlag(
            category="proline_stall",
            position=m.start() + 1,
            length=4,
            residues=m.group(),
            risk=0.4,
            description="Pro-Pro-X-Pro motif: ribosome stalling risk",
        ))
    return flags


def _scan_charge_clusters(seq: str, window: int = 10, threshold: int = 7) -> list[ComplexityFlag]:
    """Detect windows with extreme charge density."""
    flags = []
    charged = set("DEKRH")
    for i in range(len(seq) - window + 1):
        w = seq[i: i + window]
        count = sum(1 for aa in w if aa in charged)
        if count >= threshold:
            flags.append(ComplexityFlag(
                category="charge_cluster",
                position=i + 1,
                length=window,
                residues=w,
                risk=0.3 + 0.1 * (count - threshold),
                description=(
                    f"High charge density ({count}/{window} charged): "
                    f"may cause solubility issues or non-specific interactions"
                ),
            ))
    return flags


def _scan_low_complexity_windows(
    seq: str, window: int = 20, entropy_threshold: float = 1.5
) -> list[ComplexityFlag]:
    """Detect windows with low Shannon entropy (repetitive composition)."""
    import math
    flags = []
    if len(seq) < window:
        return flags

    for i in range(len(seq) - window + 1):
        w = seq[i: i + window]
        # Shannon entropy
        freq: dict[str, int] = {}
        for aa in w:
            freq[aa] = freq.get(aa, 0) + 1
        entropy = -sum(
            (c / window) * math.log2(c / window) for c in freq.values()
        )
        if entropy < entropy_threshold:
            flags.append(ComplexityFlag(
                category="low_complexity",
                position=i + 1,
                length=window,
                residues=w,
                risk=0.3,
                description=(
                    f"Low complexity region (entropy={entropy:.2f}): "
                    f"may indicate disordered / aggregation-prone segment"
                ),
            ))
    # Deduplicate overlapping low-complexity hits (keep highest risk per region)
    return _deduplicate_flags(flags)


def _deduplicate_flags(flags: list[ComplexityFlag], gap: int = 10) -> list[ComplexityFlag]:
    """Merge overlapping flags of the same category."""
    if not flags:
        return flags
    flags.sort(key=lambda f: f.position)
    merged = [flags[0]]
    for f in flags[1:]:
        prev = merged[-1]
        if f.category == prev.category and f.position <= prev.position + prev.length + gap:
            # Extend previous flag
            end = max(prev.position + prev.length, f.position + f.length)
            prev.length = end - prev.position
            prev.risk = max(prev.risk, f.risk)
        else:
            merged.append(f)
    return merged
