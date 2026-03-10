#!/usr/bin/env python
"""ThermoMPNN ΔΔG helper — runs inside the 'protopt' conda env.

Called by the stability_ddg pipeline step via subprocess.
Reads input JSON (PDB path, positions), runs ThermoMPNN site-saturation
mutagenesis, writes output JSON with ΔΔG predictions.

ThermoMPNN predicts ΔΔG of single-point mutations using a ProteinMPNN
backbone with a transfer-learned stability prediction head trained on the
Megascale dataset. Negative ΔΔG = stabilizing.

Usage:
    conda run -n protopt python scripts/thermompnn_helper.py input.json output.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"  # 20 canonical, no X


def load_model(thermompnn_dir: str, model_weights: str):
    """Load ThermoMPNN model from checkpoint."""
    import torch
    from omegaconf import OmegaConf

    # Add ThermoMPNN paths
    sys.path.insert(0, thermompnn_dir)
    sys.path.insert(0, os.path.join(thermompnn_dir, "analysis"))

    from train_thermompnn import TransferModelPL

    # ThermoMPNN config
    config = OmegaConf.create({
        "platform": {"thermompnn_dir": thermompnn_dir},
        "training": {
            "num_workers": 1,
            "learn_rate": 0.001,
            "epochs": 100,
            "lr_schedule": True,
        },
        "model": {
            "hidden_dims": [64, 32],
            "subtract_mut": True,
            "num_final_layers": 2,
            "freeze_weights": True,
            "load_pretrained": True,
            "lightattn": True,
            "lr_schedule": True,
        },
    })

    logger.info(f"Loading ThermoMPNN from {model_weights}")
    model = TransferModelPL.load_from_checkpoint(
        model_weights, cfg=config
    ).model

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("  → GPU")
    else:
        logger.info("  → CPU")

    return model


def run_ssm(model, pdb_path: str, positions: list[int] | None, chain: str = "A"):
    """Run site-saturation mutagenesis on specified positions.

    Args:
        model: Loaded ThermoMPNN model
        pdb_path: Path to PDB file
        positions: 1-based positions to scan (None = all positions)
        chain: Chain ID to use

    Returns:
        list of (position_1based, wt_aa, mut_aa, ddg_float)
    """
    import torch

    thermompnn_dir = sys.path[0] if "ThermoMPNN" in sys.path[0] else sys.path[1]
    sys.path.insert(0, thermompnn_dir)
    sys.path.insert(0, os.path.join(thermompnn_dir, "analysis"))

    from protein_mpnn_utils import alt_parse_PDB
    from datasets import Mutation as TMutation

    # Parse PDB
    logger.info(f"Parsing PDB: {pdb_path}")
    pdb_dict_list = alt_parse_PDB(pdb_path, input_chain_list=[chain])

    if not pdb_dict_list:
        logger.error("Failed to parse PDB — no chains found")
        return []

    pdb = pdb_dict_list[0]
    seq = pdb.get("seq", "")
    logger.info(f"Sequence length: {len(seq)}")

    if not seq:
        logger.error("No sequence extracted from PDB")
        return []

    # Determine which positions to scan (convert to 0-based for ThermoMPNN)
    if positions:
        scan_positions_0based = [p - 1 for p in positions if 0 <= p - 1 < len(seq)]
    else:
        scan_positions_0based = list(range(len(seq)))

    logger.info(f"Scanning {len(scan_positions_0based)} positions × 19 mutations = "
                f"{len(scan_positions_0based) * 19} predictions")

    # Build mutation list
    mutation_list = []
    for seq_pos in scan_positions_0based:
        wt_aa = seq[seq_pos]
        if wt_aa == "-" or wt_aa not in STANDARD_AAS:
            continue
        for mut_aa in STANDARD_AAS:
            if mut_aa == wt_aa:
                continue
            mutation_list.append(
                TMutation(
                    position=seq_pos,
                    wildtype=wt_aa,
                    mutation=mut_aa,
                    ddG=None,
                    pdb=pdb["name"],
                )
            )

    if not mutation_list:
        logger.warning("No valid mutations generated")
        return []

    # Run predictions in batches to manage memory
    device = next(model.parameters()).device
    results = []
    batch_size = 1000  # mutations per batch (they share the same structure)

    with torch.no_grad():
        for batch_start in range(0, len(mutation_list), batch_size):
            batch_muts = mutation_list[batch_start : batch_start + batch_size]
            logger.info(
                f"  Predicting batch {batch_start // batch_size + 1} "
                f"({len(batch_muts)} mutations)..."
            )

            preds, _ = model([pdb], batch_muts)

            for mut, pred in zip(batch_muts, preds):
                if pred is not None and "ddG" in pred:
                    ddg_val = pred["ddG"].cpu().item()
                    # ThermoMPNN uses 0-based positions; convert to 1-based
                    results.append(
                        (mut.position + 1, mut.wildtype, mut.mutation, ddg_val)
                    )

    logger.info(f"Predicted {len(results)} ΔΔG values")
    return results


def main():
    parser = argparse.ArgumentParser(description="ThermoMPNN ΔΔG helper")
    parser.add_argument("input_json", help="Input JSON with PDB path and positions")
    parser.add_argument("output_json", help="Output JSON with ΔΔG predictions")
    args = parser.parse_args()

    with open(args.input_json) as f:
        data = json.load(f)

    pdb_path = data["pdb_path"]
    positions = data.get("positions")  # None = full SSM
    chain = data.get("chain", "A")
    thermompnn_dir = data["thermompnn_dir"]
    model_weights = data["model_weights"]

    model = load_model(thermompnn_dir, model_weights)
    results = run_ssm(model, pdb_path, positions, chain)

    output = {
        "predictions": [
            {
                "position": pos,
                "wt": wt,
                "mut": mut_aa,
                "ddg": round(ddg, 4),
            }
            for pos, wt, mut_aa, ddg in results
        ]
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Done. {len(results)} predictions written to {args.output_json}")


if __name__ == "__main__":
    main()
