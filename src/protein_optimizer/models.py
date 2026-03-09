"""Core data models for the protein optimization pipeline.

Defines ProteinCandidate, Mutation, and StepResult — the universal
interchange format between all pipeline steps.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Mutation:
    """A single amino acid substitution."""

    position: int  # 1-based residue index
    wt: str  # wild-type amino acid (1-letter)
    mut: str  # mutant amino acid (1-letter)
    source_step: str = ""  # which pipeline step proposed this
    score: float | None = None  # step-specific score for this mutation
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-readable label like 'C45S'."""
        return f"{self.wt}{self.position}{self.mut}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Mutation:
        return cls(**d)


@dataclass
class ProteinCandidate:
    """A protein sequence variant with associated scores and provenance.

    This is the central data object passed between pipeline steps.
    """

    sequence: str
    name: str = ""
    candidate_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None  # ID of the candidate this was derived from
    structure_path: str | None = None  # path to PDB file, if available
    mutations: list[Mutation] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mutation_labels(self) -> list[str]:
        return [m.label for m in self.mutations]

    @property
    def num_mutations(self) -> int:
        return len(self.mutations)

    def apply_mutation(self, mutation: Mutation) -> ProteinCandidate:
        """Return a new ProteinCandidate with the mutation applied."""
        seq_list = list(self.sequence)
        idx = mutation.position - 1  # convert to 0-based
        if idx < 0 or idx >= len(seq_list):
            raise ValueError(
                f"Mutation position {mutation.position} out of range "
                f"for sequence of length {len(seq_list)}"
            )
        if seq_list[idx] != mutation.wt:
            raise ValueError(
                f"Expected {mutation.wt} at position {mutation.position}, "
                f"found {seq_list[idx]}"
            )
        seq_list[idx] = mutation.mut
        new_mutations = list(self.mutations) + [mutation]
        return ProteinCandidate(
            sequence="".join(seq_list),
            name=f"{self.name}_{mutation.label}" if self.name else mutation.label,
            parent_id=self.candidate_id,
            mutations=new_mutations,
            scores={},  # scores are invalidated when sequence changes
            metadata={"derived_from": self.name},
        )

    def apply_mutations(self, mutations: list[Mutation]) -> ProteinCandidate:
        """Return a new ProteinCandidate with multiple mutations applied."""
        seq_list = list(self.sequence)
        for m in mutations:
            idx = m.position - 1
            if idx < 0 or idx >= len(seq_list):
                raise ValueError(
                    f"Mutation position {m.position} out of range "
                    f"for sequence of length {len(seq_list)}"
                )
            if seq_list[idx] != m.wt:
                raise ValueError(
                    f"Expected {m.wt} at position {m.position}, found {seq_list[idx]}"
                )
            seq_list[idx] = m.mut

        label = "+".join(m.label for m in mutations)
        return ProteinCandidate(
            sequence="".join(seq_list),
            name=f"{self.name}_{label}" if self.name else label,
            parent_id=self.candidate_id,
            mutations=list(self.mutations) + list(mutations),
            scores={},
            metadata={"derived_from": self.name},
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ProteinCandidate:
        mutations = [Mutation.from_dict(m) for m in d.pop("mutations", [])]
        return cls(mutations=mutations, **d)


@dataclass
class StepResult:
    """Output of a pipeline step — the universal interchange format.

    Every step reads a StepResult (or raw FASTA) and writes a StepResult.
    This is serialized to/from JSON for inter-step communication.
    """

    step_name: str
    candidates: list[ProteinCandidate]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_used: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def wild_type(self) -> ProteinCandidate | None:
        """Return the first candidate (conventionally the wild type)."""
        return self.candidates[0] if self.candidates else None

    def get_candidate(self, candidate_id: str) -> ProteinCandidate | None:
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        return None

    def top_candidates(self, score_key: str, n: int = 10, ascending: bool = True) -> list[ProteinCandidate]:
        """Return top N candidates sorted by a score key."""
        scored = [c for c in self.candidates if score_key in c.scores]
        scored.sort(key=lambda c: c.scores[score_key], reverse=not ascending)
        return scored[:n]

    def to_dict(self) -> dict:
        # Strip non-serializable internal keys from config_used
        config_clean = {
            k: v for k, v in self.config_used.items()
            if k not in ("_prior_results",) and _is_json_serializable(v)
        }
        return {
            "step_name": self.step_name,
            "timestamp": self.timestamp,
            "config_used": config_clean,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, d: dict) -> StepResult:
        candidates = [ProteinCandidate.from_dict(c) for c in d.pop("candidates", [])]
        return cls(candidates=candidates, **d)

    def save(self, path: str | Path) -> None:
        """Serialize to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> StepResult:
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _is_json_serializable(value: Any) -> bool:
    """Check whether a value can be JSON-serialized."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False
