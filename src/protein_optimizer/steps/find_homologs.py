"""Tier 2 — Homolog Search.

Finds homologous sequences for downstream consensus design and
retrieval-augmented E1 scoring.

Supports two search backends:
  - **colabfold** (default): Uses the free ColabFold MMseqs2 API server.
    No local database needed, returns results in ~30 seconds.
  - **local**: Runs MMseqs2 locally against a user-provided database
    (UniRef50/90). Requires mmseqs2 on PATH + a local DB.

In both cases, MAFFT builds a multiple sequence alignment from the hits.

Usage:
    protein-opt step find_homologs -i my_enzyme.fasta -o homologs_result.json
    protein-opt step find_homologs -i input.json search_method=local database=/path/to/uniref50

Requires:
    - MAFFT binary on PATH (for MSA generation)
    - For local mode: mmseqs2 binary + sequence database
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

from Bio import SeqIO

from protein_optimizer.io_utils import write_fasta
from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

COLABFOLD_API = "https://api.colabfold.com"


class FindHomologsStep(BaseStep):
    name = "find_homologs"
    tier = 2
    title = "Homolog Search"
    description = "Find homologous sequences via ColabFold API or local MMseqs2."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        search_method = config.get("search_method", "colabfold")
        database = config.get("database", "")
        max_hits = config.get("max_hits", 500)
        min_identity = config.get("min_identity", 0.3)
        evalue = config.get("evalue", 1e-5)

        output_dir = Path(config.get("_global", {}).get("output_dir", "./results"))

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        # Only search once using the first parent (WT) sequence—
        # all Tier 1 variants share the same backbone
        wt_parents = [c for c in step_input.candidates if c.parent_id is None]
        query_parent = wt_parents[0] if wt_parents else step_input.candidates[0]

        logger.info(
            f"Searching for homologs of {query_parent.name} "
            f"using method={search_method}"
        )

        # Search for homologs
        if search_method == "colabfold":
            homolog_sequences = _run_colabfold_search(
                query_parent.sequence,
                max_hits=max_hits,
            )
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                query_fasta = tmpdir_path / "query.fasta"
                write_fasta([query_parent], query_fasta)
                homolog_sequences = _run_mmseqs2(
                    query_fasta=query_fasta,
                    database=database,
                    tmpdir=tmpdir_path,
                    max_hits=max_hits,
                    min_identity=min_identity,
                    evalue=evalue,
                )

        if not homolog_sequences:
            warnings.append(
                f"{query_parent.name}: No homologs found. "
                f"Check search parameters or try a different method."
            )
            for parent in step_input.candidates:
                candidates.append(parent)
            return StepResult(
                step_name=self.name,
                candidates=candidates,
                config_used=config,
                warnings=warnings,
            )

        logger.info(
            f"Found {len(homolog_sequences)} homologs for {query_parent.name}"
        )

        # Save homologs FASTA
        homologs_fasta = output_dir / f"{query_parent.name}_homologs.fasta"
        homologs_fasta.parent.mkdir(parents=True, exist_ok=True)
        with open(homologs_fasta, "w") as f:
            for hname, hseq in homolog_sequences:
                f.write(f">{hname}\n{hseq}\n")

        # Build MSA with MAFFT (query + homologs)
        msa_fasta = output_dir / f"{query_parent.name}_msa.fasta"
        with tempfile.TemporaryDirectory() as tmpdir:
            combined_fasta = Path(tmpdir) / "combined.fasta"
            with open(combined_fasta, "w") as f:
                f.write(f">{query_parent.name}\n{query_parent.sequence}\n")
                for hname, hseq in homolog_sequences:
                    f.write(f">{hname}\n{hseq}\n")
            _run_mafft(combined_fasta, msa_fasta)

        # Attach homolog/MSA metadata to ALL candidates
        homolog_seqs_for_e1 = [seq for _, seq in homolog_sequences[:50]]
        for parent in step_input.candidates:
            parent.metadata["homologs_fasta"] = str(homologs_fasta)
            parent.metadata["msa_fasta"] = str(msa_fasta)
            parent.metadata["num_homologs"] = len(homolog_sequences)
            parent.metadata["homolog_sequences"] = homolog_seqs_for_e1
            parent.metadata["search_method"] = search_method
            candidates.append(parent)

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
            metadata={
                "n_homologs": len(homolog_sequences),
                "search_method": search_method,
                "database": database if search_method == "local" else "colabfold_api",
            },
        )


def _run_colabfold_search(
    sequence: str,
    max_hits: int = 500,
    databases: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Search for homologs using the ColabFold MMseqs2 API server.

    The server runs MMseqs2 against pre-indexed UniRef30 + environmental
    databases and returns an A3M-formatted MSA. We parse the aligned
    sequences, strip gaps, and return (name, sequence) pairs.
    """
    if databases is None:
        databases = ["uniref30_2302_db", "colabfold_envdb_202108_db"]

    # Submit the search job
    logger.info("Submitting sequence to ColabFold API server...")
    try:
        form_data = urllib.parse.urlencode({
            "q": f">query\n{sequence}\n",
            "mode": "all",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{COLABFOLD_API}/ticket/msa",
            data=form_data,
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            ticket_data = json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.error(f"ColabFold API submission failed: {e}")
        return []

    ticket_id = ticket_data.get("id")
    if not ticket_id:
        logger.error(f"ColabFold API returned no ticket ID: {ticket_data}")
        return []

    logger.info(f"ColabFold ticket: {ticket_id}. Polling for results...")

    # Poll for completion
    max_wait = 300  # 5 minutes
    poll_interval = 5
    elapsed = 0
    status = "PENDING"

    while elapsed < max_wait:
        try:
            req = urllib.request.Request(f"{COLABFOLD_API}/ticket/{ticket_id}")
            with urllib.request.urlopen(req, timeout=15) as resp:
                status_data = json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            time.sleep(poll_interval)
            elapsed += poll_interval
            continue

        status = status_data.get("status", "UNKNOWN")
        if status == "COMPLETE":
            break
        elif status in ("ERROR", "UNKNOWN"):
            logger.error(f"ColabFold search failed with status: {status}")
            return []

        logger.info(f"ColabFold search status: {status} ({elapsed}s elapsed)")
        time.sleep(poll_interval)
        elapsed += poll_interval

    if status != "COMPLETE":
        logger.error(f"ColabFold search timed out after {max_wait}s")
        return []

    # Download the A3M result
    try:
        req = urllib.request.Request(
            f"{COLABFOLD_API}/result/download/{ticket_id}"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result_data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.error(f"ColabFold result download failed: {e}")
        return []

    # The result is a tar.gz containing .a3m files
    import tarfile
    sequences = []
    try:
        with tarfile.open(fileobj=io.BytesIO(result_data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".a3m"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    a3m_content = f.read().decode("utf-8")
                    sequences.extend(_parse_a3m(a3m_content))
    except (tarfile.TarError, Exception) as e:
        logger.error(f"Failed to parse ColabFold result: {e}")
        # Try as plain A3M text
        try:
            a3m_text = result_data.decode("utf-8")
            sequences = _parse_a3m(a3m_text)
        except Exception:
            return []

    # Remove the query sequence (first entry) and deduplicate
    if sequences and sequences[0][0] in ("query", "101"):
        sequences = sequences[1:]

    # Deduplicate by sequence
    seen = set()
    unique = []
    for name, seq in sequences:
        if seq not in seen and len(seq) > 0:
            seen.add(seq)
            unique.append((name, seq))
    unique = unique[:max_hits]

    logger.info(f"ColabFold returned {len(unique)} unique homologs")
    return unique


def _parse_a3m(a3m_text: str) -> list[tuple[str, str]]:
    """Parse A3M format: like FASTA but with lowercase insertions.

    Returns (name, sequence) pairs with insertions and gaps removed
    to get the raw ungapped sequences.
    """
    sequences = []
    current_name = ""
    current_seq: list[str] = []

    for line in a3m_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_name and current_seq:
                # Strip lowercase insertions and gap characters
                raw_seq = "".join(
                    c for c in "".join(current_seq)
                    if c.isupper() or c == "-"
                )
                # Remove gaps to get raw sequence
                ungapped = raw_seq.replace("-", "")
                if ungapped:
                    sequences.append((current_name, ungapped))
            current_name = line[1:].split()[0]  # take first word
            current_seq = []
        elif line.startswith("#"):
            continue
        else:
            current_seq.append(line)

    # Last entry
    if current_name and current_seq:
        raw_seq = "".join(
            c for c in "".join(current_seq)
            if c.isupper() or c == "-"
        )
        ungapped = raw_seq.replace("-", "")
        if ungapped:
            sequences.append((current_name, ungapped))

    return sequences


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
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
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
