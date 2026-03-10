#!/usr/bin/env python
"""ProteinMPNN design helper — runs inside the 'protopt' conda env.

Called by the proteinmpnn_design pipeline step via subprocess.
Reads input JSON, runs ProteinMPNN, writes output JSON with designed sequences.

Usage:
    conda run -n protopt python scripts/proteinmpnn_helper.py input.json output.json
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_mpnn(
    mpnn_dir: str,
    pdb_path: str,
    use_soluble: bool,
    omit_aas: str,
    sampling_temp: float,
    num_sequences: int,
    fixed_positions: list[int],
) -> list[dict]:
    """Run ProteinMPNN and return designed sequences."""
    mpnn_path = Path(mpnn_dir)
    script = mpnn_path / "protein_mpnn_run.py"
    if not script.exists():
        logger.error(f"protein_mpnn_run.py not found in {mpnn_dir}")
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Create fixed positions JSONL if needed
        jsonl_path = None
        if fixed_positions:
            jsonl_path = Path(tmpdir) / "fixed_positions.jsonl"
            pdb_name = Path(pdb_path).stem
            entry = {pdb_name: {"A": fixed_positions}}
            jsonl_path.write_text(json.dumps(entry) + "\n")

        # Build command
        cmd = [
            sys.executable, str(script),
            "--pdb_path", pdb_path,
            "--out_folder", str(output_dir),
            "--num_seq_per_target", str(num_sequences),
            "--sampling_temp", str(sampling_temp),
            "--seed", "42",
            "--batch_size", "1",
        ]

        if use_soluble:
            cmd.append("--use_soluble_model")

        if omit_aas:
            cmd.extend(["--omit_AAs", omit_aas])

        if jsonl_path:
            cmd.extend(["--fixed_positions_jsonl", str(jsonl_path)])

        logger.info(f"Running ProteinMPNN: {' '.join(cmd[:6])}...")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                cwd=str(mpnn_path),
            )
            if result.returncode != 0:
                logger.error(f"ProteinMPNN failed:\n{result.stderr[:500]}")
                return []
        except subprocess.TimeoutExpired:
            logger.error("ProteinMPNN timed out after 600s")
            return []

        # Parse output FASTA
        return parse_mpnn_output(output_dir, pdb_path)


def parse_mpnn_output(
    output_dir: Path, pdb_path: str
) -> list[dict]:
    """Parse ProteinMPNN output FASTA files."""
    results = []
    fasta_dir = output_dir / "seqs"
    if not fasta_dir.exists():
        fasta_dir = output_dir

    for fasta_file in sorted(fasta_dir.glob("*.fa")):
        with open(fasta_file) as f:
            lines = f.readlines()

        first_entry = True
        seq = None
        score = 0.0
        recovery = 0.0

        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                # Save previous entry (skip input sequence)
                if seq is not None and not first_entry:
                    results.append({
                        "sequence": seq,
                        "score": score,
                        "recovery": recovery,
                    })
                elif seq is not None:
                    first_entry = False

                seq = None
                score = 0.0
                recovery = 0.0

                # Parse header: >T=0.1, sample=1, score=1.234, ...
                parts = line[1:].split(",")
                for part in parts:
                    part = part.strip()
                    if part.startswith("score="):
                        try:
                            score = float(part.split("=")[1])
                        except ValueError:
                            pass
                    elif part.startswith("seq_recovery="):
                        try:
                            recovery = float(part.split("=")[1])
                        except ValueError:
                            pass

                if first_entry:
                    first_entry = False
                    seq = ""  # placeholder
            else:
                if line and not line.startswith("#"):
                    seq = line

        # Don't forget the last entry
        if seq is not None and score != 0.0:
            results.append({
                "sequence": seq,
                "score": score,
                "recovery": recovery,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Input JSON with MPNN config")
    parser.add_argument("output_json", help="Output JSON with designed sequences")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    designs = run_mpnn(
        mpnn_dir=data["mpnn_dir"],
        pdb_path=data["pdb_path"],
        use_soluble=data.get("use_soluble", True),
        omit_aas=data.get("omit_aas", "C"),
        sampling_temp=data.get("sampling_temp", 0.1),
        num_sequences=data.get("num_sequences", 8),
        fixed_positions=data.get("fixed_positions", []),
    )

    output = {"designs": designs}
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Done. {len(designs)} designs written to {args.output_json}")


if __name__ == "__main__":
    main()
