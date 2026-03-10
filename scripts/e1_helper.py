#!/usr/bin/env python
"""Profluent E1 scoring helper — runs inside the 'e1' conda env (Python 3.12).

Uses the native E1 scorer API (wildtype-marginal scoring) to score protein
variants against the parent sequence.  Also supports masked-marginal saturation
scanning to discover beneficial single-point mutations.

Usage:
    conda run -n e1 python scripts/e1_helper.py input.json output.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def load_model(model_name: str) -> "E1ForMaskedLM":
    """Load E1 model from HuggingFace."""
    from E1.modeling import E1ForMaskedLM

    logger.info(f"Loading E1 model: {model_name}")
    model = E1ForMaskedLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("  -> GPU")
    else:
        logger.info("  -> CPU")
    return model


def score_sequences_native(
    model: "E1ForMaskedLM",
    parent_sequence: str,
    sequences: list[dict],
    homolog_seqs: list[str] | None = None,
) -> list[dict]:
    """Score sequences using E1's native wildtype-marginal scorer."""
    from E1.scorer import E1Scorer, EncoderScoreMethod

    scorer = E1Scorer(
        model=model,
        method=EncoderScoreMethod.WILDTYPE_MARGINAL,
        max_batch_tokens=32768,
    )

    # Build context dict from homologs (retrieval-augmented mode)
    context_seqs: dict[str, str] | None = None
    if homolog_seqs:
        context_seqs = {f"h{i}": h for i, h in enumerate(homolog_seqs)}

    # Collect unique variant sequences (exclude parent -- score is 0 by defn)
    seq_ids: list[str] = []
    seq_list: list[str] = []
    skipped = 0
    parent_len = len(parent_sequence)
    for entry in sequences:
        seq = entry["sequence"]
        if seq == parent_sequence:
            continue
        # E1 scorer requires all seqs same length as parent, uppercase, no X
        if len(seq) != parent_len:
            skipped += 1
            continue
        if not seq.isalpha() or not seq.isupper() or "X" in seq:
            skipped += 1
            continue
        seq_ids.append(entry["id"])
        seq_list.append(seq)
    if skipped:
        logger.warning(f"Skipped {skipped} sequences (length mismatch or invalid chars)")

    results: list[dict] = []

    # Parent always gets score 0 (wildtype marginal diff from itself)
    for entry in sequences:
        if entry["sequence"] == parent_sequence:
            results.append({"id": entry["id"], "e1_fitness": 0.0})

    if seq_list:
        logger.info(f"Scoring {len(seq_list)} variant(s) with E1 wildtype-marginal...")
        raw_scores = scorer.score(
            parent_sequence=parent_sequence,
            sequences=seq_list,
            sequence_ids=seq_ids,
            context_seqs=context_seqs,
            context_reduction="mean",
        )
        for rs in raw_scores:
            results.append({"id": rs["id"], "e1_fitness": float(rs["score"])})

    return results


def masked_marginal_scan(
    model: "E1ForMaskedLM",
    parent_sequence: str,
    protected: list[int],
    homolog_seqs: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Scan all positions via masked-marginal scoring to find beneficial mutations."""
    from E1.scorer import E1Scorer, EncoderScoreMethod
    from E1.tokenizer import get_tokenizer

    scorer = E1Scorer(
        model=model,
        method=EncoderScoreMethod.MASKED_MARGINAL,
        max_batch_tokens=32768,
    )

    tokenizer = get_tokenizer()
    vocab = tokenizer.get_vocab()
    protected_set = set(protected)

    context_seqs: dict[str, str] | None = None
    if homolog_seqs:
        context_seqs = {f"h{i}": h for i, h in enumerate(homolog_seqs)}

    # Generate all single-point mutants for non-protected positions
    mutant_seqs: list[str] = []
    mutant_ids: list[str] = []
    mutant_info: list[dict] = []

    for i, wt_aa in enumerate(parent_sequence):
        pos = i + 1  # 1-indexed
        if pos in protected_set:
            continue
        for mut_aa in AA_ALPHABET:
            if mut_aa == wt_aa:
                continue
            mut_seq = parent_sequence[:i] + mut_aa + parent_sequence[i + 1 :]
            mid = f"p{pos}_{wt_aa}{mut_aa}"
            mutant_seqs.append(mut_seq)
            mutant_ids.append(mid)
            mutant_info.append({"position": pos, "wt": wt_aa, "mut": mut_aa})

    if not mutant_seqs:
        return []

    logger.info(f"Masked-marginal scan: {len(mutant_seqs)} single-point mutants...")
    raw_scores = scorer.score(
        parent_sequence=parent_sequence,
        sequences=mutant_seqs,
        sequence_ids=mutant_ids,
        context_seqs=context_seqs,
        context_reduction="mean",
    )

    # Aggregate by position -- find best mutation at each position
    pos_best: dict[int, dict] = {}
    for rs, info in zip(raw_scores, mutant_info):
        gain = float(rs["score"])
        pos = info["position"]
        if gain > 0:
            if pos not in pos_best or gain > pos_best[pos]["best_gain"]:
                pos_best[pos] = {
                    "position": pos,
                    "wt": info["wt"],
                    "best_mut": info["mut"],
                    "best_gain": gain,
                    "mutations": {},
                }
            pos_best[pos]["mutations"][info["mut"]] = round(gain, 4)

    position_scores = sorted(pos_best.values(), key=lambda x: x["best_gain"], reverse=True)
    return position_scores[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    model_name = data.get("model_name", "Profluent-Bio/E1-600m")
    sequences = data["sequences"]  # [{id, sequence}, ...]
    homolog_seqs = data.get("homolog_seqs", [])
    saturation_cfg = data.get("saturation", None)

    model = load_model(model_name)

    # Determine parent sequence
    parent_seq = None
    for entry in sequences:
        if entry.get("is_parent", False):
            parent_seq = entry["sequence"]
            break
    # Fallback: first sequence is parent
    if parent_seq is None and sequences:
        parent_seq = sequences[0]["sequence"]

    # --- Score all variants against parent ---
    results = score_sequences_native(model, parent_seq, sequences, homolog_seqs or None)

    for r in results:
        logger.info(f"  {r['id']}: e1_fitness={r['e1_fitness']:.4f}")

    # --- Optional saturation scan ---
    sat_mutations: list[dict] = []
    if saturation_cfg and parent_seq:
        protected = saturation_cfg.get("protected_positions", [])
        top_k = saturation_cfg.get("top_k_positions", 5)
        scan = masked_marginal_scan(
            model, parent_seq, protected,
            homolog_seqs=homolog_seqs or None,
            top_k=top_k,
        )
        for entry in scan:
            sat_mutations.append({
                "position": entry["position"],
                "wt": entry["wt"],
                "mut": entry["best_mut"],
                "gain": entry["best_gain"],
            })
        logger.info(f"Saturation scan found {len(sat_mutations)} beneficial mutations")

    output = {"scores": results}
    if sat_mutations:
        output["saturation_mutations"] = sat_mutations

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote {len(results)} scores to {args.output_json}")


if __name__ == "__main__":
    main()
