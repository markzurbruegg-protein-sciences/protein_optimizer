#!/usr/bin/env python
"""ESM-1v scoring helper — runs inside the 'plm' conda env.

Called by the esm1v_score pipeline step via subprocess.

Two scoring modes:
  1. **Variant scoring** (default, fast): Compute WT masked-marginal log-probs
     once, then score each variant by summing log-prob differences at mutated
     positions.  Only L × N_models forward passes needed for the WT, regardless
     of how many variants are being scored.
  2. **Full PLL** (--full-pll): Compute full masked-marginal PLL for every
     sequence independently.  Needed when there's no shared WT or for
     absolute fitness comparisons.  Very slow for many sequences.

Usage:
    conda run -n plm python scripts/esm1v_helper.py input.json output.json [--n-models 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import logging
import time

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


def compute_wt_marginals(models, wt_sequence: str):
    """Compute masked-marginal log-probabilities for every position in the WT.

    Returns a numpy array of shape (L, vocab_size) averaged over the ensemble,
    where L = len(wt_sequence[:1022]).
    """
    import torch
    import numpy as np

    seq = wt_sequence[:1022]
    seq_len = len(seq)
    all_logprobs = []

    for model, alphabet, batch_converter in models:
        device = next(model.parameters()).device
        data = [("protein", seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        mask_idx = alphabet.mask_idx
        vocab_size = len(alphabet)

        logprobs_model = np.zeros((seq_len, vocab_size), dtype=np.float32)

        with torch.no_grad():
            for i in range(seq_len):
                tok_idx = i + 1  # BOS offset
                if tok_idx >= tokens.shape[1] - 1:
                    break
                masked = tokens.clone()
                masked[0, tok_idx] = mask_idx
                result = model(masked)
                logits = result["logits"][0, tok_idx]
                lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
                logprobs_model[i] = lp

        all_logprobs.append(logprobs_model)
        logger.info(f"  WT marginals computed for model ({seq_len} positions)")

    # Average log-probabilities across ensemble
    return np.mean(all_logprobs, axis=0), models[0][1]  # (L, V), alphabet


def score_variants_from_marginals(wt_marginals, alphabet, wt_sequence, variants):
    """Score variants using pre-computed WT masked-marginal log-probs.

    Each variant is a dict with 'id' and 'mutations' (list of {pos, wt_aa, mut_aa}).
    The score is the sum of log P(mut_aa | masked WT) - log P(wt_aa | masked WT)
    at each mutated position.
    """
    seq = wt_sequence[:1022]
    results = []

    for v in variants:
        var_id = v["id"]
        mutations = v.get("mutations", [])
        if not mutations:
            results.append({"id": var_id, "esm1v_score": 0.0, "esm1v_pll": 0.0})
            continue

        total_delta = 0.0
        valid = True
        for mut in mutations:
            pos = mut["pos"]  # 0-indexed
            wt_aa = mut["wt_aa"]
            mut_aa = mut["mut_aa"]

            if pos >= len(seq):
                logger.warning(f"  {var_id}: position {pos} out of range ({len(seq)}), skipping")
                valid = False
                break

            wt_tok = alphabet.get_idx(wt_aa)
            mut_tok = alphabet.get_idx(mut_aa)

            if wt_tok is None or mut_tok is None:
                logger.warning(f"  {var_id}: unknown AA '{wt_aa}'→'{mut_aa}', skipping")
                valid = False
                break

            log_p_mut = wt_marginals[pos, mut_tok]
            log_p_wt = wt_marginals[pos, wt_tok]
            total_delta += float(log_p_mut - log_p_wt)

        if valid:
            # Also compute a pseudo-PLL for the variant (WT PLL + delta)
            results.append({
                "id": var_id,
                "esm1v_score": round(total_delta, 6),
            })
        else:
            results.append({"id": var_id, "esm1v_score": None})

    return results


def compute_wt_pll(wt_marginals, alphabet, wt_sequence):
    """Compute WT PLL from pre-computed marginals."""
    seq = wt_sequence[:1022]
    pll_sum = 0.0
    for i, aa in enumerate(seq):
        tok = alphabet.get_idx(aa)
        if tok is not None and i < wt_marginals.shape[0]:
            pll_sum += float(wt_marginals[i, tok])
    return pll_sum / max(len(seq), 1)


def compute_full_pll(models, sequence: str) -> float:
    """Full masked-marginal PLL for a single sequence (slow)."""
    import torch

    plls = []
    seq = sequence[:1022]

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
                tok_idx = i + 1
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
    parser.add_argument("input_json", help="Input JSON with sequences/variants to score")
    parser.add_argument("output_json", help="Output JSON with scores")
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--full-pll", action="store_true",
                        help="Score each sequence with full PLL (slow)")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    models = load_models(args.n_models)

    t0 = time.time()

    if args.full_pll or "variants" not in data:
        # Legacy mode: full PLL per sequence
        sequences = data["sequences"]
        logger.info(f"Full-PLL scoring {len(sequences)} sequences ({args.n_models} models)")

        results = []
        for i, entry in enumerate(sequences):
            seq_id = entry["id"]
            sequence = entry["sequence"]
            logger.info(f"  [{i+1}/{len(sequences)}] {seq_id} ({len(sequence)} aa)")
            pll = compute_full_pll(models, sequence)
            results.append({"id": seq_id, "esm1v_pll": round(pll, 6)})

    else:
        # Fast variant scoring mode
        wt_sequence = data["wt_sequence"]
        variants = data["variants"]
        logger.info(
            f"Variant-mode scoring: {len(variants)} variants against "
            f"WT ({len(wt_sequence)} aa) with {args.n_models} models"
        )

        logger.info("Computing WT masked-marginal log-probs...")
        wt_marginals, alphabet = compute_wt_marginals(models, wt_sequence)

        wt_pll = compute_wt_pll(wt_marginals, alphabet, wt_sequence)
        logger.info(f"WT PLL: {wt_pll:.4f}")

        logger.info(f"Scoring {len(variants)} variants...")
        results = score_variants_from_marginals(
            wt_marginals, alphabet, wt_sequence, variants
        )

        # Add WT PLL to all results for reference
        for r in results:
            r["wt_pll"] = round(wt_pll, 6)

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s. Scored {len(results)} entries.")

    with open(args.output_json, "w") as f:
        json.dump({"scores": results}, f, indent=2)

    logger.info(f"Scores written to {args.output_json}")


if __name__ == "__main__":
    main()
