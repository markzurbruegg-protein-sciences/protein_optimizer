"""Tier 4 — ESM-1v Variant Effect Prediction.

Scores variant candidates with the ESM-1v 5-model ensemble for
zero-shot prediction of mutation effects.  Uses masked-marginal
scoring which is well-validated on ProteinGym benchmarks.

Usage:
    protein-opt step esm1v_score -i previous_result.json -o esm1v_scored.json

Requirements:
    pip install fair-esm torch
"""

from __future__ import annotations

import logging
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

ESM1V_MODELS = [
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
]

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class ESM1vScoreStep(BaseStep):
    name = "esm1v_score"
    tier = 4
    title = "ESM-1v Ensemble Scoring"
    description = "5-model ensemble variant effect prediction with ESM-1v."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        n_models = config.get("n_models", 5)
        score_existing = config.get("score_existing", True)
        run_saturation = config.get("site_saturation", False)
        top_k_positions = config.get("top_k_positions", 5)
        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        models = _load_esm1v_ensemble(min(n_models, 5))
        if not models:
            return StepResult(
                step_name=self.name,
                candidates=list(step_input.candidates),
                config_used=config,
                warnings=["ESM-1v models could not be loaded — pip install fair-esm"],
            )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        # Group: find parents, score all
        parents = [c for c in step_input.candidates if c.parent_id is None]
        variants = [c for c in step_input.candidates if c.parent_id is not None]

        for parent in parents:
            candidates.append(parent)
            if len(parent.sequence) > 1022:
                warnings.append(
                    f"{parent.name}: Sequence length {len(parent.sequence)} "
                    f"exceeds ESM-1v max (1022). Truncating."
                )
                seq = parent.sequence[:1022]
            else:
                seq = parent.sequence

            # Score parent (pseudo-log-likelihood)
            parent_pll = _ensemble_pll(models, seq)
            parent.scores["esm1v_pll"] = parent_pll

            # Score mutations in existing variant candidates
            if score_existing:
                for variant in variants:
                    if variant.parent_id != parent.candidate_id:
                        continue
                    vseq = variant.sequence[:1022]
                    variant_pll = _ensemble_pll(models, vseq)
                    variant.scores["esm1v_pll"] = variant_pll
                    variant.scores["esm1v_delta"] = variant_pll - parent_pll
                    candidates.append(variant)

            # Saturation scan for new mutation proposals
            if run_saturation:
                marginals = _ensemble_masked_marginals(models, seq, protected)
                top_positions = sorted(
                    marginals.items(),
                    key=lambda x: x[1]["max_gain"],
                    reverse=True,
                )[:top_k_positions]

                for pos, info in top_positions:
                    if info["max_gain"] <= 0:
                        continue
                    wt_aa = seq[pos - 1]
                    for mut_aa, gain in info["mutations"].items():
                        if gain <= 0:
                            continue
                        mut = Mutation(
                            position=pos,
                            wt=wt_aa,
                            mut=mut_aa,
                            source_step=self.name,
                            score=gain,
                            metadata={"esm1v_gain": gain},
                        )
                        try:
                            variant = parent.apply_mutation(mut)
                            variant.scores["esm1v_pll"] = parent_pll + gain
                            variant.scores["esm1v_delta"] = gain
                            candidates.append(variant)
                        except ValueError:
                            pass

        # Add remaining variants that weren't scored (different parent)
        scored_ids = {c.candidate_id for c in candidates}
        for v in variants:
            if v.candidate_id not in scored_ids:
                candidates.append(v)

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _load_esm1v_ensemble(n_models: int) -> list:
    """Load ESM-1v models (up to 5)."""
    try:
        import esm
        import torch
    except ImportError:
        logger.error("ESM-1v requires: pip install fair-esm torch")
        return []

    loaded = []
    for i in range(1, n_models + 1):
        model_name = f"esm1v_t33_650M_UR90S_{i}"
        try:
            logger.info(f"Loading {model_name}...")
            model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            batch_converter = alphabet.get_batch_converter()
            loaded.append((model, alphabet, batch_converter))
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")

    logger.info(f"Loaded {len(loaded)}/{n_models} ESM-1v models")
    return loaded


def _ensemble_pll(models: list, sequence: str) -> float:
    """Compute ensemble pseudo-log-likelihood (average across models)."""
    import torch

    if not models:
        return 0.0

    plls = []
    for model, alphabet, batch_converter in models:
        pll = _single_model_pll(model, alphabet, batch_converter, sequence)
        plls.append(pll)

    return sum(plls) / len(plls)


def _single_model_pll(model, alphabet, batch_converter, sequence: str) -> float:
    """Masked-marginal pseudo-log-likelihood for one ESM-1v model."""
    import torch

    device = next(model.parameters()).device
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    mask_idx = alphabet.mask_idx
    seq_len = len(sequence)

    log_prob_sum = 0.0

    with torch.no_grad():
        for i in range(seq_len):
            tok_idx = i + 1  # ESM adds BOS token at position 0
            if tok_idx >= tokens.shape[1] - 1:
                break

            masked = tokens.clone()
            true_token = masked[0, tok_idx].item()
            masked[0, tok_idx] = mask_idx

            result = model(masked)
            logits = result["logits"][0, tok_idx]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_prob_sum += log_probs[true_token].item()

    return log_prob_sum / max(seq_len, 1)


def _ensemble_masked_marginals(
    models: list, sequence: str, protected: set[int]
) -> dict[int, dict]:
    """Compute ensemble-averaged masked-marginal scores for all positions."""
    import torch

    if not models:
        return {}

    position_scores: dict[int, dict[str, float]] = {}

    for model, alphabet, batch_converter in models:
        device = next(model.parameters()).device
        data = [("protein", sequence)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        mask_idx = alphabet.mask_idx

        # Build AA token mapping
        aa_tok = {}
        for aa in AA_ALPHABET:
            idx = alphabet.get_idx(aa)
            if idx is not None:
                aa_tok[aa] = idx

        with torch.no_grad():
            for i in range(len(sequence)):
                pos = i + 1
                if pos in protected:
                    continue

                tok_idx = i + 1
                if tok_idx >= tokens.shape[1] - 1:
                    break

                wt_aa = sequence[i]
                masked = tokens.clone()
                masked[0, tok_idx] = mask_idx

                result = model(masked)
                logits = result["logits"][0, tok_idx]
                log_probs = torch.log_softmax(logits, dim=-1)

                wt_logp = log_probs[aa_tok.get(wt_aa, 0)].item()

                if pos not in position_scores:
                    position_scores[pos] = {}

                for aa in AA_ALPHABET:
                    if aa == wt_aa:
                        continue
                    tid = aa_tok.get(aa)
                    if tid is None:
                        continue
                    gain = log_probs[tid].item() - wt_logp
                    key = f"{aa}_gain"
                    if key not in position_scores[pos]:
                        position_scores[pos][key] = 0.0
                    position_scores[pos][key] += gain

    # Average across models and restructure
    n_models = len(models)
    result = {}
    for pos, scores in position_scores.items():
        mutations = {}
        wt_aa = sequence[pos - 1]
        for aa in AA_ALPHABET:
            if aa == wt_aa:
                continue
            key = f"{aa}_gain"
            if key in scores:
                avg_gain = scores[key] / n_models
                if avg_gain > 0:
                    mutations[aa] = round(avg_gain, 4)

        if mutations:
            result[pos] = {
                "max_gain": max(mutations.values()),
                "mutations": mutations,
            }

    return result
