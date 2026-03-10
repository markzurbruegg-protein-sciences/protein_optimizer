"""Tier 4 — ESM-1v Variant Effect Prediction.

Scores variant candidates with the ESM-1v 5-model ensemble for
zero-shot prediction of mutation effects.  Uses masked-marginal
scoring which is well-validated on ProteinGym benchmarks.

Dispatches computation to a conda environment (default: 'plm') that
has fair-esm and a CUDA-enabled PyTorch installed.

Usage:
    protein-opt step esm1v_score -i previous_result.json -o esm1v_scored.json

Requirements:
    conda env with: fair-esm torch (CUDA)
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

# Path to the helper script (relative to package root)
_HELPER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "esm1v_helper.py"


class ESM1vScoreStep(BaseStep):
    name = "esm1v_score"
    tier = 5
    title = "ESM-1v Ensemble Scoring"
    description = "5-model ensemble variant effect prediction with ESM-1v."
    requires = []

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        n_models = config.get("n_models", 5)
        conda_env = config.get("conda_env", "plm")

        candidates: list[ProteinCandidate] = list(step_input.candidates)
        warnings: list[str] = []

        # Find wild-type (parent) sequence
        parent = next((c for c in candidates if c.parent_id is None), None)
        variants = [c for c in candidates if c.parent_id is not None]

        if parent is None:
            warnings.append("No wild-type parent sequence found for ESM-1v scoring.")
            return StepResult(
                step_name=self.name, candidates=candidates,
                config_used=config, warnings=warnings,
            )

        # Build variant descriptors with mutation positions
        variant_descs: list[dict] = []
        cand_index: list[ProteinCandidate] = []
        for c in variants:
            muts = []
            for m in (c.mutations or []):
                muts.append({
                    "pos": m.position - 1,  # 0-indexed for the helper
                    "wt_aa": m.wt,
                    "mut_aa": m.mut,
                })
            if muts:
                variant_descs.append({"id": c.candidate_id, "mutations": muts})
                cand_index.append(c)

        if not variant_descs:
            warnings.append("No variants with mutations found.")
            return StepResult(
                step_name=self.name, candidates=candidates,
                config_used=config, warnings=warnings,
            )

        helper_input = {
            "wt_sequence": parent.sequence,
            "variants": variant_descs,
        }

        scores = _run_esm1v_subprocess(helper_input, conda_env, n_models)

        if scores is None:
            warnings.append(
                "ESM-1v scoring failed — check that the 'plm' conda env "
                "has fair-esm and CUDA torch installed."
            )
            return StepResult(
                step_name=self.name, candidates=candidates,
                config_used=config, warnings=warnings,
            )

        # Map scores back to candidates
        score_map = {s["id"]: s for s in scores}
        wt_pll = scores[0].get("wt_pll") if scores else None

        if wt_pll is not None:
            parent.scores["esm1v_pll"] = wt_pll

        for c in cand_index:
            entry = score_map.get(c.candidate_id)
            if entry and entry.get("esm1v_score") is not None:
                c.scores["esm1v_delta"] = entry["esm1v_score"]
                if wt_pll is not None:
                    c.scores["esm1v_pll"] = wt_pll + entry["esm1v_score"]

        n_scored = sum(1 for c in candidates if "esm1v_delta" in c.scores)
        n_improved = sum(1 for c in candidates if c.scores.get("esm1v_delta", -1) > 0)
        logger.info(
            f"ESM-1v scored {n_scored}/{len(variants)} variants "
            f"({n_improved} improved over WT)"
        )

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _run_esm1v_subprocess(
    helper_input: dict, conda_env: str, n_models: int
) -> list[dict] | None:
    """Dispatch ESM-1v scoring to a conda env with CUDA torch + fair-esm."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "esm1v_input.json"
            out_path = Path(tmpdir) / "esm1v_output.json"

            with open(in_path, "w") as f:
                json.dump(helper_input, f)

            helper = str(_HELPER_SCRIPT)
            if not Path(helper).exists():
                logger.error(f"ESM-1v helper script not found: {helper}")
                return None

            n_variants = len(helper_input.get("variants", helper_input.get("sequences", [])))
            mode = "variant" if "variants" in helper_input else "full-PLL"

            cmd = [
                "conda", "run", "--no-capture-output", "-n", conda_env,
                "python", helper,
                str(in_path), str(out_path),
                "--n-models", str(n_models),
            ]

            logger.info(
                f"Dispatching ESM-1v {mode} scoring to conda env '{conda_env}' "
                f"({n_variants} entries, {n_models} models)..."
            )

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600,
            )

            if result.returncode != 0:
                logger.error(f"ESM-1v helper failed:\n{result.stderr[:2000]}")
                return None

            if not out_path.exists():
                logger.error("ESM-1v helper produced no output file")
                return None

            with open(out_path) as f:
                output = json.load(f)

            return output.get("scores", [])

    except subprocess.TimeoutExpired:
        logger.error("ESM-1v scoring timed out after 3600s")
        return None
    except Exception as e:
        logger.error(f"ESM-1v subprocess dispatch failed: {e}")
        return None

