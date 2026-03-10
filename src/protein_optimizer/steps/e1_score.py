"""Tier 5 — Profluent E1 Fitness Scoring.

Uses the Profluent E1 protein language model (600M) for zero-shot
fitness prediction via wildtype-marginal scoring.

Dispatches computation to a conda environment (default: 'e1') that
has E1 (Python 3.12), torch (CUDA), and tokenizers installed.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from protein_optimizer.io_utils import parse_protected_residues
from protein_optimizer.models import Mutation, ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

_HELPER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "e1_helper.py"


class E1ScoreStep(BaseStep):
    name = "e1_score"
    tier = 5
    title = "Profluent E1 Scoring"
    description = "Zero-shot fitness prediction with Profluent E1 language model."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        model_name = config.get("model", "Profluent-Bio/E1-600m")
        use_retrieval = config.get("retrieval_augmented", True)
        max_homologs = config.get("max_homologs_context", 8)
        conda_env = config.get("conda_env", "e1")

        run_saturation = config.get("site_saturation", False)
        top_k_positions = config.get("top_k_positions", 5)

        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        # Get homolog sequences for retrieval-augmented mode
        prior_results = config.get("_prior_results", {})
        homolog_seqs: list[str] = []
        if use_retrieval and "find_homologs" in prior_results:
            fh_result = prior_results["find_homologs"]
            for c in getattr(fh_result, "candidates", []):
                meta = getattr(c, "metadata", {})
                if "homolog_sequences" in meta:
                    homolog_seqs = meta["homolog_sequences"][:max_homologs]
                    break

        candidates: list[ProteinCandidate] = list(step_input.candidates)
        warnings: list[str] = []

        # Prepare sequences for scoring
        seq_map: dict[str, list[ProteinCandidate]] = {}
        parent_seq: str | None = None
        for c in candidates:
            seq_map.setdefault(c.sequence, []).append(c)
            if c.parent_id is None:
                parent_seq = c.sequence

        helper_input: dict[str, Any] = {
            "model_name": model_name,
            "homolog_seqs": homolog_seqs,
            "sequences": [
                {"id": f"seq_{i}", "sequence": seq, "is_parent": (seq == parent_seq)}
                for i, seq in enumerate(seq_map.keys())
            ],
        }

        # Add saturation scan config if requested
        if run_saturation and parent_seq:
            helper_input["saturation"] = {
                "parent_sequence": parent_seq,
                "top_k_positions": top_k_positions,
                "protected_positions": sorted(protected),
            }

        result = _run_subprocess(helper_input, conda_env)
        if result is None:
            warnings.append("E1 scoring failed — check e1 conda env.")
            return StepResult(
                step_name=self.name, candidates=candidates,
                config_used=config, warnings=warnings,
            )

        # Map scores back
        scores = result.get("scores", [])
        seq_keys = list(seq_map.keys())
        for entry in scores:
            idx = int(entry["id"].split("_")[1])
            val = entry.get("e1_fitness")
            if val is None:
                continue
            for c in seq_map[seq_keys[idx]]:
                c.scores["e1_fitness"] = val

        # Add saturation-derived mutations
        sat_mutations = result.get("saturation_mutations", [])
        if sat_mutations and parent_seq:
            parent_cands = [c for c in candidates if c.parent_id is None]
            parent = parent_cands[0] if parent_cands else None
            if parent:
                parent_score = parent.scores.get("e1_fitness", 0.0)
                for sm in sat_mutations:
                    pos = sm["position"]
                    wt_aa = sm["wt"]
                    mut_aa = sm["mut"]
                    gain = sm["gain"]
                    if gain <= 0:
                        continue
                    mut = Mutation(
                        position=pos, wt=wt_aa, mut=mut_aa,
                        source_step=self.name, score=gain,
                        metadata={"e1_gain": gain},
                    )
                    try:
                        variant = parent.apply_mutation(mut)
                        variant.scores["e1_fitness"] = parent_score + gain
                        candidates.append(variant)
                    except ValueError:
                        pass

        n_scored = sum(1 for c in candidates if "e1_fitness" in c.scores)
        logger.info(f"E1 scored {n_scored}/{len(candidates)} candidates")

        return StepResult(
            step_name=self.name, candidates=candidates,
            config_used=config, warnings=warnings,
        )


def _run_subprocess(helper_input: dict, conda_env: str) -> dict | None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_p = Path(tmpdir) / "in.json"
            out_p = Path(tmpdir) / "out.json"
            in_p.write_text(json.dumps(helper_input))

            helper = str(_HELPER_SCRIPT)
            if not Path(helper).exists():
                logger.error(f"E1 helper not found: {helper}")
                return None

            n = len(helper_input["sequences"])
            logger.info(f"Dispatching E1 to '{conda_env}' ({n} seqs)...")

            r = subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", conda_env,
                 "python", helper, str(in_p), str(out_p)],
                capture_output=True, text=True, timeout=3600,
            )
            if r.returncode != 0:
                logger.error(f"E1 helper failed:\n{r.stderr[:1000]}")
                return None
            if not out_p.exists():
                logger.error("E1 helper produced no output")
                return None
            return json.loads(out_p.read_text())
    except subprocess.TimeoutExpired:
        logger.error("E1 timed out (3600s)")
        return None
    except Exception as e:
        logger.error(f"E1 dispatch failed: {e}")
        return None
