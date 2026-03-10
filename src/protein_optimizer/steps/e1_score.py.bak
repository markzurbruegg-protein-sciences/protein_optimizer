"""Tier 4 — Profluent E1 Fitness Scoring.

Uses the Profluent E1 protein language model (600M) for zero-shot
fitness prediction via masked-marginal scoring.  Supports retrieval-
augmented mode: if homologs were found by ``find_homologs``, the top
homolog sequences are prepended (comma-separated, then ``?`` mask)
to bias predictions toward the evolutionary family.

Usage:
    protein-opt step e1_score -i previous_result.json -o e1_scored.json

Requirements:
    pip install git+https://github.com/Profluent-AI/E1.git
    Model weights: Profluent-Bio/E1-600m (auto-downloaded from HuggingFace)
"""

from __future__ import annotations

import logging
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)


class E1ScoreStep(BaseStep):
    name = "e1_score"
    tier = 4
    title = "Profluent E1 Scoring"
    description = "Zero-shot fitness prediction with Profluent E1 language model."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        model_name = config.get("model", "Profluent-Bio/E1-600m")
        use_retrieval = config.get("retrieval_augmented", True)
        max_homologs = config.get("max_homologs_context", 8)
        batch_size = config.get("batch_size", 4)
        score_existing = config.get("score_existing", True)

        # Also propose new mutations via saturation at impactful positions
        run_saturation = config.get("site_saturation", False)
        top_k_positions = config.get("top_k_positions", 5)

        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        # Try to get homologs from prior results for retrieval-augmented mode
        prior_results = config.get("_prior_results", {})
        homolog_seqs: list[str] = []
        if use_retrieval and "find_homologs" in prior_results:
            fh_result = prior_results["find_homologs"]
            for c in fh_result.get("candidates", []):
                meta = c.get("metadata", {})
                if "homolog_sequences" in meta:
                    homolog_seqs = meta["homolog_sequences"][:max_homologs]
                    break

        model, tokenizer = _load_e1(model_name)
        if model is None:
            return StepResult(
                step_name=self.name,
                candidates=list(step_input.candidates),
                config_used=config,
                warnings=["E1 model could not be loaded — install E1 package."],
            )

        candidates: list[ProteinCandidate] = []
        warnings: list[str] = []

        for parent in step_input.candidates:
            candidates.append(parent)

            if not score_existing and parent.parent_id is not None:
                # Score existing variants
                score = _score_sequence(
                    model, tokenizer, parent.sequence, homolog_seqs
                )
                parent.scores["e1_fitness"] = score
                continue

            # Score each candidate
            score = _score_sequence(
                model, tokenizer, parent.sequence, homolog_seqs
            )
            parent.scores["e1_fitness"] = score

            if parent.parent_id is not None:
                continue  # only do saturation from parents

            if run_saturation:
                # Find most impactful positions via masked-marginal
                position_scores = _masked_marginal_scan(
                    model, tokenizer, parent.sequence, homolog_seqs, protected
                )
                top_positions = sorted(
                    position_scores.items(),
                    key=lambda x: x[1]["max_gain"],
                    reverse=True,
                )[:top_k_positions]

                for pos, info in top_positions:
                    if info["max_gain"] <= 0:
                        continue
                    wt_aa = parent.sequence[pos - 1]
                    for mut_aa, gain in info["mutations"].items():
                        if gain <= 0 or mut_aa == wt_aa:
                            continue
                        mut = Mutation(
                            position=pos,
                            wt=wt_aa,
                            mut=mut_aa,
                            source_step=self.name,
                            score=gain,
                            metadata={"e1_gain": gain},
                        )
                        try:
                            variant = parent.apply_mutation(mut)
                            variant.scores["e1_fitness"] = score + gain
                            candidates.append(variant)
                        except ValueError:
                            pass

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _load_e1(model_name: str):
    """Load Profluent E1 model and tokenizer."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        logger.info(f"Loading E1 model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("E1 model loaded on GPU")
        else:
            logger.info("E1 model loaded on CPU")

        return model, tokenizer
    except ImportError:
        logger.error(
            "E1 requires: pip install git+https://github.com/Profluent-AI/E1.git"
        )
        return None, None
    except Exception as e:
        logger.error(f"Failed to load E1 model: {e}")
        return None, None


def _build_retrieval_input(sequence: str, homolog_seqs: list[str]) -> str:
    """Build retrieval-augmented input for E1.

    Format: homolog1,homolog2,...,homologN?target_sequence
    The ``?`` token separates the retrieval context from the query.
    """
    if homolog_seqs:
        context = ",".join(homolog_seqs)
        return f"{context}?{sequence}"
    return sequence


def _score_sequence(model, tokenizer, sequence: str, homolog_seqs: list[str]) -> float:
    """Compute pseudo-log-likelihood of a sequence under E1.

    Uses masked-marginal scoring: for each position, mask it and compute
    the log-probability of the true amino acid.
    """
    import torch
    import math

    device = next(model.parameters()).device
    input_seq = _build_retrieval_input(sequence, homolog_seqs)

    tokens = tokenizer(input_seq, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)

    # Find the token positions corresponding to the target sequence
    # (after the ? token if retrieval-augmented)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # E1 uses ? as mask — find its token id
        mask_token_id = tokenizer.convert_tokens_to_ids("?")

    seq_len = len(sequence)
    total_len = input_ids.shape[1]
    # Target sequence tokens are the last seq_len tokens before EOS
    # This is an approximation — tokenizer may add special tokens
    start_idx = max(1, total_len - seq_len - 1)  # skip BOS, before EOS

    log_prob_sum = 0.0
    n_scored = 0

    with torch.no_grad():
        for i in range(seq_len):
            tok_idx = start_idx + i
            if tok_idx >= total_len - 1:
                break

            masked = input_ids.clone()
            true_token = masked[0, tok_idx].item()
            masked[0, tok_idx] = mask_token_id

            outputs = model(masked)
            logits = outputs.logits[0, tok_idx]
            log_probs = torch.log_softmax(logits, dim=-1)
            log_prob_sum += log_probs[true_token].item()
            n_scored += 1

    return log_prob_sum / max(n_scored, 1)


def _masked_marginal_scan(
    model, tokenizer, sequence: str, homolog_seqs: list[str], protected: set[int],
) -> dict[int, dict]:
    """Scan all non-protected positions for beneficial mutations.

    Returns {position: {"max_gain": float, "mutations": {aa: gain}}}.
    """
    import torch
    import string

    device = next(model.parameters()).device
    input_seq = _build_retrieval_input(sequence, homolog_seqs)

    tokens = tokenizer(input_seq, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("?")

    seq_len = len(sequence)
    total_len = input_ids.shape[1]
    start_idx = max(1, total_len - seq_len - 1)

    AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    aa_token_ids = {
        aa: tokenizer.convert_tokens_to_ids(aa) for aa in AA_ALPHABET
    }

    position_scores = {}

    with torch.no_grad():
        for i in range(seq_len):
            pos = i + 1
            if pos in protected:
                continue

            tok_idx = start_idx + i
            if tok_idx >= total_len - 1:
                break

            wt_aa = sequence[i]
            masked = input_ids.clone()
            masked[0, tok_idx] = mask_token_id

            outputs = model(masked)
            logits = outputs.logits[0, tok_idx]
            log_probs = torch.log_softmax(logits, dim=-1)

            wt_logp = log_probs[aa_token_ids.get(wt_aa, 0)].item()

            mutations = {}
            for aa in AA_ALPHABET:
                if aa == wt_aa:
                    continue
                tid = aa_token_ids.get(aa)
                if tid is None:
                    continue
                gain = log_probs[tid].item() - wt_logp
                if gain > 0:
                    mutations[aa] = round(gain, 4)

            if mutations:
                max_gain = max(mutations.values())
                position_scores[pos] = {
                    "max_gain": max_gain,
                    "mutations": mutations,
                }

    return position_scores
