#!/usr/bin/env python
"""ESM-1v scoring helper — runs inside the 'plm' conda env.

Called by the esm1v_score pipeline step via subprocess.
Reads input JSON, computes masked-marginal PLL scores, writes output JSON.

Usage:
    conda run -n plm python scripts/esm1v_helper.py input.json output.json [--n-models 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_models(n_models: int):
    """Load ESM-1v ensemble."""
    import esm
    import torch

    models = []
    for i in range(1, n_models + 1):
        name = f"esm1v_t33_650M_UR90S_{i}"
        logger.info(f"Loading {name}...")
        model, alphabet = esm.pretrained.load_model_and_alphabet(name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"  → GPU")
        batch_converter = alphabet.get_batch_converter()
        models.append((model, alphabet, batch_converter))
    logger.info(f"Loaded {len(models)}/{n_models} ESM-1v models")
    return models


def compute_pll(models, sequence: str) -> float:
    """Masked-marginal pseudo-log-likelihood averaged across model ensemble."""
    import torch

    plls = []
    seq = sequence[:1022]  # ESM-1v max length

    for model, alphabet, batch_converter in models:
        device = next(model.parameters()).device
        data = [("protein", seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        mask_idx = alphabet.mask_idx
        seq_len = len(seq)

        log_prob_sum = 0.0
        with torch.no_grad():
            for i in range(seq_len):
                tok_idx = i + 1  # BOS offset
                if tok_idx >= tokens.shape[1] - 1:
                    break
                masked = tokens.clone()
                true_token = masked[0, tok_idx].item()
                masked[0, tok_idx] = mask_idx
                result = model(masked)
                logits = result["logits"][0, tok_idx]
                log_probs = torch.log_softmax(logits, dim=-1)
                log_prob_sum += log_probs[true_token].item()

        plls.append(log_prob_sum / max(seq_len, 1))

    return sum(plls) / len(plls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Input JSON with sequences to score")
    parser.add_argument("output_json", help="Output JSON with scores")
    parser.add_argument("--n-models", type=int, default=5)
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    sequences = data["sequences"]  # list of {id, sequence}
    logger.info(f"Scoring {len(sequences)} sequences with ESM-1v ({args.n_models} models)")

    models = load_models(args.n_models)

    results = []
    for i, entry in enumerate(sequences):
        seq_id = entry["id"]
        sequence = entry["sequence"]
        logger.info(f"  [{i+1}/{len(sequences)}] {seq_id} ({len(sequence)} aa)")
        pll = compute_pll(models, sequence)
        results.append({"id": seq_id, "esm1v_pll": round(pll, 6)})

    output = {"scores": results}
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Done. Scores written to {args.output_json}")


if __name__ == "__main__":
    main()
