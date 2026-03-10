"""Tier 4 — ESM-IF1 Inverse Folding Scoring.

Scores designed sequences against their predicted structures using
ESM-IF1 (inverse folding).  Higher scores mean the sequence is more
compatible with the 3D backbone — a key consistency check after
ProteinMPNN design or manual mutations.

Usage:
    protein-opt step esmif1_score -i variants_result.json -o esmif1_scored.json

Requirements:
    pip install fair-esm torch biotite
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class ESMIF1ScoreStep(BaseStep):
    name = "esmif1_score"
    tier = 4
    title = "ESM-IF1 Inverse Folding"
    description = "Score sequence-structure compatibility with ESM-IF1."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        chain_id = config.get("chain_id", "A")

        model_data = _load_esmif1()
        if model_data is None:
            return StepResult(
                step_name=self.name,
                candidates=list(step_input.candidates),
                config_used=config,
                warnings=["ESM-IF1 could not be loaded — pip install fair-esm"],
            )

        model, alphabet = model_data
        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        # Find the structure for each candidate (use parent's if variant)
        parent_structures: dict[str, str] = {}
        for c in step_input.candidates:
            if c.structure_path and Path(c.structure_path).exists():
                parent_structures[c.candidate_id] = c.structure_path

        for candidate in step_input.candidates:
            candidates.append(candidate)

            # Find structure to score against
            pdb_path = None
            if candidate.structure_path and Path(candidate.structure_path).exists():
                pdb_path = candidate.structure_path
            elif candidate.parent_id and candidate.parent_id in parent_structures:
                pdb_path = parent_structures[candidate.parent_id]

            if not pdb_path:
                continue

            score = _score_sequence_structure(
                model, alphabet, candidate.sequence, pdb_path, chain_id
            )
            if score is not None:
                candidate.scores["esmif1_score"] = score
                logger.debug(f"{candidate.name}: ESM-IF1 = {score:.4f}")

        # Compute delta relative to parent
        parent_scores = {}
        for c in candidates:
            if c.parent_id is None and "esmif1_score" in c.scores:
                parent_scores[c.candidate_id] = c.scores["esmif1_score"]

        for c in candidates:
            if c.parent_id and c.parent_id in parent_scores:
                if "esmif1_score" in c.scores:
                    c.scores["esmif1_delta"] = (
                        c.scores["esmif1_score"] - parent_scores[c.parent_id]
                    )

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _load_esmif1():
    """Load ESM-IF1 model."""
    try:
        import esm
        import torch

        logger.info("Loading ESM-IF1 model...")
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50D()
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        logger.info("ESM-IF1 loaded successfully")
        return model, alphabet
    except ImportError:
        logger.error("ESM-IF1 requires: pip install fair-esm torch")
        return None
    except Exception as e:
        logger.error(f"Failed to load ESM-IF1: {e}")
        return None


def _score_sequence_structure(
    model, alphabet, sequence: str, pdb_path: str, chain_id: str
) -> float | None:
    """Score sequence against structure using ESM-IF1.

    Returns average log-likelihood (higher = more compatible).
    """
    import torch

    try:
        import esm
        from esm.inverse_folding.util import load_structure, extract_coords_from_structure
    except ImportError:
        logger.error("ESM inverse folding utilities not available")
        return None

    try:
        structure = load_structure(pdb_path, chain_id)
        coords, native_seq = extract_coords_from_structure(structure)
    except Exception as e:
        logger.warning(f"Could not extract coords from {pdb_path}: {e}")
        return None

    device = next(model.parameters()).device

    # Use the GVP-GNN encoder to get backbone features
    try:
        coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)

        # Encode structure
        batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)
        batch = [(coords, None, sequence)]
        coords_batch, confidence, strs, tokens, padding_mask = batch_converter(batch)

        coords_batch = coords_batch.to(device)
        tokens = tokens.to(device)
        padding_mask = padding_mask.to(device) if padding_mask is not None else None

        with torch.no_grad():
            logits = model(coords_batch, padding_mask, tokens)
            # logits shape: (batch, seq_len, vocab_size)

            # Compute log-probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Score each position
            total_ll = 0.0
            n_scored = 0

            for i, aa in enumerate(sequence):
                tok_idx = i + 1  # account for BOS
                if tok_idx >= log_probs.shape[1]:
                    break
                aa_idx = alphabet.get_idx(aa)
                if aa_idx is not None:
                    total_ll += log_probs[0, tok_idx, aa_idx].item()
                    n_scored += 1

            return total_ll / max(n_scored, 1)

    except Exception as e:
        logger.warning(f"ESM-IF1 scoring failed: {e}")
        # Fallback: use the simpler API if available
        try:
            ll, _ = esm.inverse_folding.util.score_sequence(
                model, alphabet, coords, sequence
            )
            return ll
        except Exception as e2:
            logger.error(f"ESM-IF1 fallback also failed: {e2}")
            return None
