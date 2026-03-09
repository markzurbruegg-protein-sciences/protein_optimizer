"""Tier 2 — Homolog Search.

Runs MMseqs2 against UniRef90 (or a user-provided database) to find
homologous sequences. Outputs a FASTA of homologs + MSA for downstream
consensus design and retrieval-augmented E1 scoring.

Usage:
    protein-opt step find_homologs -i my_enzyme.fasta -o homologs_result.json

Requires:
    - mmseqs2 binary on PATH
    - MAFFT binary on PATH (for MSA generation)
    - A sequence database (UniRef90 recommended)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from Bio import SeqIO

from protein_optimizer.io_utils import write_fasta
from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class FindHomologsStep(BaseStep):
    name = "find_homologs"
    tier = 2
    title = "Homolog Search"
    description = "Find homologous sequences via MMseqs2 for consensus design and E1 retrieval."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        database = config.get("database", "")
        max_hits = config.get("max_hits", 500)
        min_identity = config.get("min_identity", 0.3)
        evalue = config.get("evalue", 1e-5)

        output_dir = Path(config.get("_global", {}).get("output_dir", "./results"))

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            candidates.append(parent)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                query_fasta = tmpdir / "query.fasta"
                write_fasta([parent], query_fasta)

                # --- Run MMseqs2 ---
                homolog_sequences = _run_mmseqs2(
                    query_fasta=query_fasta,
                    database=database,
                    tmpdir=tmpdir,
                    max_hits=max_hits,
                    min_identity=min_identity,
                    evalue=evalue,
                )

                if not homolog_sequences:
                    warnings.append(
                        f"{parent.name}: No homologs found. "
                        f"Check database path or relax search parameters."
                    )
                    continue

                logger.info(f"Found {len(homolog_sequences)} homologs for {parent.name}")

                # Save homologs FASTA
                homologs_fasta = output_dir / f"{parent.name}_homologs.fasta"
                homologs_fasta.parent.mkdir(parents=True, exist_ok=True)
                with open(homologs_fasta, "w") as f:
                    for name, seq in homolog_sequences:
                        f.write(f">{name}\n{seq}\n")

                # --- Build MSA with MAFFT ---
                msa_fasta = output_dir / f"{parent.name}_msa.fasta"
                combined_fasta = tmpdir / "combined.fasta"
                with open(combined_fasta, "w") as f:
                    f.write(f">{parent.name}\n{parent.sequence}\n")
                    for name, seq in homolog_sequences:
                        f.write(f">{name}\n{seq}\n")

                _run_mafft(combined_fasta, msa_fasta)

                # Store paths in parent metadata
                parent.metadata["homologs_fasta"] = str(homologs_fasta)
                parent.metadata["msa_fasta"] = str(msa_fasta)
                parent.metadata["num_homologs"] = len(homolog_sequences)

                # Store homolog sequences for retrieval-augmented E1
                parent.metadata["homolog_sequences"] = [
                    seq for _, seq in homolog_sequences[:50]
                ]

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _run_mmseqs2(
    query_fasta: Path,
    database: str,
    tmpdir: Path,
    max_hits: int,
    min_identity: float,
    evalue: float,
) -> list[tuple[str, str]]:
    """Run MMseqs2 easy-search and return (name, sequence) pairs."""
    if not database:
        logger.warning(
            "No MMseqs2 database specified. Set 'database' in config "
            "(e.g., path to UniRef90 MMseqs2 DB). Skipping search."
        )
        return []

    result_tsv = tmpdir / "result.tsv"
    try:
        cmd = [
            "mmseqs", "easy-search",
            str(query_fasta),
            database,
            str(result_tsv),
            str(tmpdir / "tmp"),
            "--min-seq-id", str(min_identity),
            "-e", str(evalue),
            "--max-seqs", str(max_hits),
            "--format-output", "target,tseq",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        logger.error(
            "mmseqs2 not found on PATH. Install: "
            "conda install -c bioconda mmseqs2"
        )
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"MMseqs2 failed: {e.stderr}")
        return []
    except subprocess.TimeoutExpired:
        logger.error("MMseqs2 timed out after 600s")
        return []

    # Parse results
    sequences = []
    if result_tsv.exists():
        with open(result_tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    sequences.append((parts[0], parts[1]))
    return sequences


def _run_mafft(input_fasta: Path, output_fasta: Path) -> bool:
    """Run MAFFT multiple sequence alignment."""
    try:
        with open(output_fasta, "w") as out:
            subprocess.run(
                ["mafft", "--auto", "--quiet", str(input_fasta)],
                stdout=out,
                check=True,
                timeout=300,
            )
        return True
    except FileNotFoundError:
        logger.error("MAFFT not found on PATH. Install: conda install -c bioconda mafft")
        # Fall back: just copy input as-is
        import shutil
        shutil.copy(input_fasta, output_fasta)
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"MAFFT failed: {e}")
        return False
