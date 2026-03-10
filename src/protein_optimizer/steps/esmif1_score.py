"""Tier 4 — ESM-IF1 Inverse Folding Scoring.

Scores designed sequences against their predicted structures using
ESM-IF1 (inverse folding).  Higher scores mean the sequence is more
compatible with the 3D backbone.

Dispatches computation to a conda environment (default: 'protopt') that
has fair-esm, torch (CUDA), and biotite < 1.0.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.steps.base import BaseStep

logger = logging.getLogger(__name__)

_HELPER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "esmif1_helper.py"


class ESMIF1ScoreStep(BaseStep):
    name = "esmif1_score"
    tier = 5
    title = "ESM-IF1 Inverse Folding"
    description = "Score sequence-structure compatibility with ESM-IF1."
    requires = ["predict_structure"]

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        chain_id = config.get("chain_id", "A")
        conda_env = config.get("conda_env", "protopt")

        candidates: list[ProteinCandidate] = list(step_input.candidates)
        warnings: list[str] = []

        # Find PDB path
        pdb_path = _find_pdb(candidates, config)
        if not pdb_path:
            warnings.append("No PDB structure found for ESM-IF1 scoring.")
            return StepResult(
                step_name=self.name, candidates=candidates,
                config_used=config, warnings=warnings,
            )

        # Deduplicate sequences
        seq_map: dict[str, list[ProteinCandidate]] = {}
        for c in candidates:
            seq_map.setdefault(c.sequence, []).append(c)

        helper_input = {
            "pdb_path": str(pdb_path),
            "chain_id": chain_id,
            "sequences": [
                {"id": f"seq_{i}", "sequence": seq}
                for i, seq in enumerate(seq_map.keys())
            ],
        }

        scores = _run_subprocess(helper_input, conda_env)
        if scores is None:
            warnings.append("ESM-IF1 scoring failed — check protopt env.")
            return StepResult(
                step_name=self.name, candidates=candidates,
                config_used=config, warnings=warnings,
            )

        # Map scores back
        seq_keys = list(seq_map.keys())
        for entry in scores:
            idx = int(entry["id"].split("_")[1])
            val = entry.get("esmif1_score")
            if val is None:
                continue
            for c in seq_map[seq_keys[idx]]:
                c.scores["esmif1_score"] = val

        # Compute delta
        parent_scores = {
            c.candidate_id: c.scores["esmif1_score"]
            for c in candidates
            if c.parent_id is None and "esmif1_score" in c.scores
        }
        for c in candidates:
            if c.parent_id and c.parent_id in parent_scores and "esmif1_score" in c.scores:
                c.scores["esmif1_delta"] = c.scores["esmif1_score"] - parent_scores[c.parent_id]

        n_scored = sum(1 for c in candidates if "esmif1_score" in c.scores)
        logger.info(f"ESM-IF1 scored {n_scored}/{len(candidates)} candidates")

        return StepResult(
            step_name=self.name, candidates=candidates,
            config_used=config, warnings=warnings,
        )


def _find_pdb(candidates, config):
    for c in candidates:
        sp = c.structure_path or c.metadata.get("structure_path", "")
        if sp and Path(sp).exists() and c.parent_id is None:
            return sp
    prior = config.get("_prior_results", {})
    sr = prior.get("predict_structure")
    if sr and hasattr(sr, "candidates"):
        for c in sr.candidates:
            sp = getattr(c, "structure_path", "") or c.metadata.get("structure_path", "")
            if sp and Path(sp).exists():
                return sp
    return None


def _run_subprocess(helper_input, conda_env):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_p = Path(tmpdir) / "in.json"
            out_p = Path(tmpdir) / "out.json"
            in_p.write_text(json.dumps(helper_input))

            helper = str(_HELPER_SCRIPT)
            if not Path(helper).exists():
                logger.error(f"ESM-IF1 helper not found: {helper}")
                return None

            n = len(helper_input["sequences"])
            logger.info(f"Dispatching ESM-IF1 to '{conda_env}' ({n} seqs)...")

            r = subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", conda_env,
                 "python", helper, str(in_p), str(out_p)],
                capture_output=True, text=True, timeout=3600,
            )
            if r.returncode != 0:
                logger.error(f"ESM-IF1 helper failed:\n{r.stderr[:1000]}")
                return None
            if not out_p.exists():
                logger.error("ESM-IF1 helper produced no output")
                return None
            return json.loads(out_p.read_text()).get("scores", [])
    except subprocess.TimeoutExpired:
        logger.error("ESM-IF1 timed out (3600s)")
        return None
    except Exception as e:
        logger.error(f"ESM-IF1 dispatch failed: {e}")
        return None
