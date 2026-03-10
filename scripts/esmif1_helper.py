#!/usr/bin/env python
"""ESM-IF1 inverse-folding scoring helper — runs inside the 'protopt' conda env.

Called by the esmif1_score pipeline step via subprocess.
Reads input JSON with sequences + PDB path, scores each, writes output JSON.

Usage:
    conda run -n protopt python scripts/esmif1_helper.py input.json output.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    """Load ESM-IF1 model."""
    import esm
    import torch

    logger.info("Loading ESM-IF1 (esm_if1_gvp4_t16_142M_UR50D)...")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50D()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("  → GPU")
    else:
        logger.info("  → CPU")
    return model, alphabet


def score_sequence(model, alphabet, sequence: str, pdb_path: str, chain_id: str) -> float | None:
    """Score a sequence against a structure using ESM-IF1."""
    import torch
    import esm
    from esm.inverse_folding.util import load_structure, extract_coords_from_structure

    try:
        structure = load_structure(pdb_path, chain_id)
        coords, _ = extract_coords_from_structure(structure)
    except Exception as e:
        logger.warning(f"Could not extract coords from {pdb_path}: {e}")
        return None

    device = next(model.parameters()).device

    try:
        batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
        batch = [(coords, None, sequence)]
        coords_batch, confidence, strs, tokens, padding_mask = batch_converter(batch)

        coords_batch = coords_batch.to(device)
        tokens = tokens.to(device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device)

        with torch.no_grad():
            logits = model(coords_batch, padding_mask, tokens)
            log_probs = torch.log_softmax(logits, dim=-1)

            total_ll = 0.0
            n_scored = 0

            for i, aa in enumerate(sequence):
                tok_idx = i + 1  # BOS offset
                if tok_idx >= log_probs.shape[1]:
                    break
                aa_idx = alphabet.get_idx(aa)
                if aa_idx is not None:
                    total_ll += log_probs[0, tok_idx, aa_idx].item()
                    n_scored += 1

            return total_ll / max(n_scored, 1)

    except Exception as e:
        logger.warning(f"ESM-IF1 scoring failed: {e}")
        try:
            ll, _ = esm.inverse_folding.util.score_sequence(
                model, alphabet, coords, sequence
            )
            return ll
        except Exception as e2:
            logger.error(f"ESM-IF1 fallback also failed: {e2}")
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    pdb_path = data["pdb_path"]
    chain_id = data.get("chain_id", "A")
    sequences = data["sequences"]  # [{id, sequence}, ...]

    model, alphabet = load_model()

    results = []
    for entry in sequences:
        score = score_sequence(model, alphabet, entry["sequence"], pdb_path, chain_id)
        results.append({
            "id": entry["id"],
            "esmif1_score": score,
        })
        logger.info(f"  {entry['id']}: {score:.4f}" if score else f"  {entry['id']}: FAILED")

    with open(args.output_json, "w") as f:
        json.dump({"scores": results}, f, indent=2)

    logger.info(f"Wrote {len(results)} scores to {args.output_json}")


if __name__ == "__main__":
    main()
