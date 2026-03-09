"""I/O utilities for reading FASTA, PDB references, and StepResult JSON."""

from __future__ import annotations

import json
from pathlib import Path

from Bio import SeqIO

from protein_optimizer.models import ProteinCandidate, StepResult


def read_fasta(path: str | Path) -> list[ProteinCandidate]:
    """Read a FASTA file and return a list of ProteinCandidates.

    The first sequence is treated as the wild-type / primary candidate.
    """
    candidates = []
    for record in SeqIO.parse(str(path), "fasta"):
        candidates.append(
            ProteinCandidate(
                sequence=str(record.seq),
                name=record.id,
            )
        )
    if not candidates:
        raise ValueError(f"No sequences found in {path}")
    return candidates


def fasta_to_step_result(path: str | Path) -> StepResult:
    """Load a FASTA file as a StepResult (convenience for step entry points)."""
    candidates = read_fasta(path)
    return StepResult(
        step_name="input",
        candidates=candidates,
        metadata={"source_file": str(path)},
    )


def load_input(path: str | Path) -> StepResult:
    """Smart loader: auto-detect FASTA vs StepResult JSON.

    This is the primary entry point used by CLI subcommands.
    """
    path = Path(path)
    if path.suffix in (".json",):
        return StepResult.load(path)
    elif path.suffix in (".fasta", ".fa", ".faa", ".fas"):
        return fasta_to_step_result(path)
    else:
        # Try JSON first, fall back to FASTA
        try:
            return StepResult.load(path)
        except (json.JSONDecodeError, KeyError):
            return fasta_to_step_result(path)


def write_fasta(candidates: list[ProteinCandidate], path: str | Path) -> None:
    """Write candidates to a FASTA file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in candidates:
            header = c.name or c.candidate_id
            if c.mutations:
                header += f" mutations={','.join(c.mutation_labels)}"
            f.write(f">{header}\n{c.sequence}\n")


def parse_protected_residues(value: str | list[int] | None) -> set[int]:
    """Parse protected residues from CLI arg or config.

    Accepts:
        - comma-separated string: "10,25,30"
        - list of ints: [10, 25, 30]
        - None → empty set
    """
    if value is None:
        return set()
    if isinstance(value, (list, set)):
        return {int(x) for x in value}
    return {int(x.strip()) for x in str(value).split(",") if x.strip()}
