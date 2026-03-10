#!/usr/bin/env python
"""ESM-IF1 inverse-folding scoring helper — runs inside the 'protopt' conda env.

Called by the esmif1_score pipeline step via subprocess.
Reads input JSON with sequences + PDB path, scores each, writes output JSON.

GPU-accelerated: coords are extracted once, then all sequences are scored
on GPU using the ESM-IF1 model.

Usage:
    conda run -n protopt python scripts/esmif1_helper.py input.json output.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    """Load ESM-IF1 model and move to GPU if available."""
    import esm
    import torch

    logger.info("Loading ESM-IF1 (esm_if1_gvp4_t16_142M_UR50)...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
        logger.info(f"  → GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.info("  → CPU")

    return model, alphabet, device


def extract_coords(pdb_path: str, chain_id: str):
    """Extract backbone coordinates from a PDB file (done once)."""
    from esm.inverse_folding.util import load_structure, extract_coords_from_structure

    logger.info(f"Extracting coords from {pdb_path} (chain {chain_id})...")
    structure = load_structure(pdb_path, chain_id)
    coords, native_seq = extract_coords_from_structure(structure)
    logger.info(f"  → {len(native_seq)} residues")
    return coords, native_seq


def score_sequences(model, alphabet, device, coords, sequences: list[tuple[str, str]]) -> dict[str, float | None]:
    """Score a list of (seq_id, sequence) pairs against the backbone coords.

    Re-implements ESM's score_sequence/get_sequence_loss with explicit device
    handling — moves tensors to GPU before the forward pass and back to CPU
    for numpy conversion afterward.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from esm.inverse_folding.util import CoordBatchConverter

    batch_converter = CoordBatchConverter(alphabet)
    results: dict[str, float | None] = {}
    n = len(sequences)

    for i, (seq_id, sequence) in enumerate(sequences):
        try:
            batch = [(coords, None, sequence)]
            coords_t, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)

            prev_output_tokens = tokens[:, :-1].to(device)
            target = tokens[:, 1:].to(device)
            target_padding_mask = (target == alphabet.padding_idx)

            with torch.no_grad():
                logits, _ = model.forward(coords_t, padding_mask, confidence, prev_output_tokens)

            loss = F.cross_entropy(logits, target, reduction="none")
            loss_np = loss[0].cpu().detach().numpy()
            target_padding_np = target_padding_mask[0].cpu().numpy()

            ll_fullseq = -np.sum(loss_np * ~target_padding_np) / np.sum(~target_padding_np)
            results[seq_id] = float(ll_fullseq)

            if (i + 1) % 50 == 0 or i == 0 or i == n - 1:
                logger.info(f"  [{i+1}/{n}] {seq_id}: {ll_fullseq:.4f}")

        except Exception as e:
            logger.warning(f"  [{i+1}/{n}] {seq_id} failed: {e}")
            results[seq_id] = None

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    pdb_path = data["pdb_path"]
    chain_id = data.get("chain_id", "A")
    entries = data["sequences"]  # [{id, sequence}, ...]

    t0 = time.time()

    model, alphabet, device = load_model()
    coords, native_seq = extract_coords(pdb_path, chain_id)

    # Deduplicate — score each unique sequence only once
    seq_to_rep: dict[str, str] = {}  # sequence → representative id
    for entry in entries:
        if entry["sequence"] not in seq_to_rep:
            seq_to_rep[entry["sequence"]] = entry["id"]

    unique_seqs = [(rep_id, seq) for seq, rep_id in seq_to_rep.items()]
    n_dup = len(entries) - len(unique_seqs)
    logger.info(
        f"Scoring {len(unique_seqs)} unique sequences "
        f"({len(entries)} total, {n_dup} duplicates skipped)"
    )

    scores = score_sequences(model, alphabet, device, coords, unique_seqs)

    # Map back to all entries (re-use score for duplicates)
    output_results = []
    for entry in entries:
        rep_id = seq_to_rep[entry["sequence"]]
        output_results.append({
            "id": entry["id"],
            "esmif1_score": scores.get(rep_id),
        })

    elapsed = time.time() - t0
    n_ok = sum(1 for r in output_results if r["esmif1_score"] is not None)
    logger.info(f"Done in {elapsed:.1f}s — {n_ok}/{len(output_results)} scored")

    with open(args.output_json, "w") as f:
        json.dump({"scores": output_results}, f, indent=2)

    logger.info(f"Scores written to {args.output_json}")


if __name__ == "__main__":
    main()
