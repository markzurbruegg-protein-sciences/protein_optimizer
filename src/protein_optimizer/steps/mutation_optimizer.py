"""Tier 5 — Multi-Mutation Optimizer.

Pools beneficial mutations from all SSM scans and engineering steps,
builds an evidence matrix, generates a full combinatorial library
up to a configurable order, and scores ALL combinations with
Profluent E1 to capture epistatic effects.

Outputs:
  - Single Mutation Evidence Table (ranked by multi-source evidence)
  - Top Multi-Mutant Table (ranked by E1 fitness)
  - Epistasis analysis (E1_combo vs. additive prediction)
"""

from __future__ import annotations

import itertools
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

_E1_HELPER = Path(__file__).resolve().parents[3] / "scripts" / "e1_helper.py"

# Steps that contribute mutations to the pool
_MUTATION_SOURCES = {
    "stability_ddg", "e1_score", "solubility_ssm",
    "consensus_design", "pssm_analysis",
    "motif_scan", "cysteine_scan",
    "cavity_fill", "surface_patch", "disulfide_design",
}

# Score keys where lower is better (inverted for ranking)
_INVERT_SCORES = {"ddg", "surface_sap"}


class MutationOptimizerStep(BaseStep):
    name = "mutation_optimizer"
    tier = 5
    title = "Multi-Mutation Optimizer"
    description = "Combinatorial mutation optimization with E1 fitness scoring."
    requires = []  # soft deps — uses whatever prior results are available

    def run(self, step_input: StepResult, config: dict[str, Any]) -> StepResult:
        max_pool = config.get("max_pool_size", 20)
        max_order = config.get("max_combination_order", 5)
        max_library = config.get("max_library_size", 25000)
        max_charge_shift = config.get("max_charge_shift", 5)
        conda_env = config.get("e1_conda_env", "e1")
        e1_model = config.get("e1_model", "Profluent-Bio/E1-600m")
        use_retrieval = config.get("retrieval_augmented", True)
        max_homologs = config.get("max_homologs_context", 8)

        protected = parse_protected_residues(
            config.get("_global", {}).get("protected_residues")
        )

        prior_results: dict[str, StepResult] = config.get("_prior_results", {})

        candidates: list[ProteinCandidate] = list(step_input.candidates)
        warnings: list[str] = []

        # Get homolog sequences for E1 retrieval-augmented mode
        homolog_seqs: list[str] = []
        if use_retrieval and "find_homologs" in prior_results:
            fh_result = prior_results["find_homologs"]
            for c in getattr(fh_result, "candidates", []):
                meta = getattr(c, "metadata", {})
                if "homolog_sequences" in meta:
                    homolog_seqs = meta["homolog_sequences"][:max_homologs]
                    break

        for parent in step_input.candidates:
            if parent.parent_id is not None:
                continue

            wt_seq = parent.sequence

            # ── Step 1: Collect mutation pool ──
            pool = _collect_mutation_pool(
                parent, prior_results, step_input, protected
            )

            if len(pool) < 2:
                warnings.append(
                    f"{parent.name}: Only {len(pool)} mutations in pool — "
                    "minimum 2 needed for combinations."
                )
                continue

            logger.info(
                f"{parent.name}: Collected {len(pool)} unique mutations "
                f"from {len(_MUTATION_SOURCES)} source types"
            )

            # ── Step 2: Evidence scoring & ranking ──
            ranked = _rank_by_evidence(pool)

            # Select top-N, ensuring no position conflicts
            selected: list[dict] = []
            used_positions: set[int] = set()
            for entry in ranked:
                if entry["position"] in used_positions:
                    continue
                selected.append(entry)
                used_positions.add(entry["position"])
                if len(selected) >= max_pool:
                    break

            logger.info(
                f"{parent.name}: Selected top {len(selected)} mutations "
                f"for combinatorial optimization"
            )

            # Store the evidence table in parent metadata
            parent.metadata["mutation_evidence_table"] = [
                {
                    "position": e["position"],
                    "wt": e["wt"],
                    "mut": e["mut"],
                    "label": f"{e['wt']}{e['position']}{e['mut']}",
                    "e1_gain": e["scores"].get("e1_gain", None),
                    "ddg": e["scores"].get("ddg", None),
                    "delta_solubility": e["scores"].get("delta_solubility", None),
                    "evidence_sources": e["sources"],
                    "evidence_breadth": e["evidence_breadth"],
                    "evidence_score": round(e["evidence_score"], 4),
                }
                for e in selected
            ]

            # ── Step 3: Combinatorial library ──
            mutations_for_combo = [
                Mutation(
                    position=e["position"], wt=e["wt"], mut=e["mut"],
                    source_step="mutation_optimizer",
                    score=e["evidence_score"],
                    metadata={
                        "evidence_breadth": e["evidence_breadth"],
                        "sources": e["sources"],
                    },
                )
                for e in selected
            ]

            # Charge lookup for biological filter
            CHARGE = {"K": 1, "R": 1, "H": 0.5, "D": -1, "E": -1}

            combo_sequences: list[tuple[str, list[Mutation]]] = []
            combo_count = 0

            for order in range(2, max_order + 1):
                if combo_count >= max_library:
                    break
                for combo in itertools.combinations(mutations_for_combo, order):
                    if combo_count >= max_library:
                        break

                    # Position conflict check (redundant with selection, but safe)
                    positions = [m.position for m in combo]
                    if len(positions) != len(set(positions)):
                        continue

                    # Charge shift filter
                    charge_shift = 0.0
                    for m in combo:
                        charge_shift += CHARGE.get(m.mut, 0) - CHARGE.get(m.wt, 0)
                    if abs(charge_shift) > max_charge_shift:
                        continue

                    # Build the mutant sequence
                    seq_list = list(wt_seq)
                    valid = True
                    for m in combo:
                        idx = m.position - 1
                        if idx < 0 or idx >= len(seq_list) or seq_list[idx] != m.wt:
                            valid = False
                            break
                        seq_list[idx] = m.mut
                    if not valid:
                        continue

                    combo_sequences.append(("".join(seq_list), list(combo)))
                    combo_count += 1

            logger.info(
                f"{parent.name}: Generated {len(combo_sequences)} "
                f"combinatorial variants (order 2-{max_order})"
            )

            if not combo_sequences:
                warnings.append(
                    f"{parent.name}: No valid combinations generated."
                )
                continue

            # ── Step 4: Score ALL combinations with E1 ──
            # Deduplicate sequences
            unique_seqs: dict[str, list[int]] = {}
            for idx, (seq, _muts) in enumerate(combo_sequences):
                unique_seqs.setdefault(seq, []).append(idx)

            # Include WT for reference
            all_seqs_to_score = [wt_seq] + list(unique_seqs.keys())

            logger.info(
                f"{parent.name}: Scoring {len(all_seqs_to_score)} unique "
                f"sequences with Profluent E1..."
            )

            e1_scores = _score_with_e1(
                sequences=all_seqs_to_score,
                homolog_seqs=homolog_seqs,
                conda_env=conda_env,
                model_name=e1_model,
            )

            if not e1_scores:
                warnings.append(
                    f"{parent.name}: E1 scoring failed for combinatorial library."
                )
                continue

            wt_e1 = e1_scores[0] if e1_scores else 0.0

            # ── Step 5: Build scored candidates ──
            scored_combos: list[dict] = []
            seq_keys = list(unique_seqs.keys())

            for seq_idx, seq in enumerate(seq_keys):
                e1_fitness = e1_scores[seq_idx + 1] if (seq_idx + 1) < len(e1_scores) else None
                if e1_fitness is None:
                    continue

                # Get mutations for all combos with this sequence
                for combo_idx in unique_seqs[seq]:
                    _, muts = combo_sequences[combo_idx]

                    # Additive predictions
                    additive_ddg = 0.0
                    additive_sol = 0.0
                    n_ddg = 0
                    n_sol = 0
                    individual_e1 = 0.0

                    for m in muts:
                        # Look up individual scores from the pool
                        key = (m.position, m.wt, m.mut)
                        pool_entry = pool.get(key)
                        if pool_entry:
                            ddg_val = pool_entry["scores"].get("ddg")
                            if ddg_val is not None:
                                additive_ddg += ddg_val
                                n_ddg += 1
                            sol_val = pool_entry["scores"].get("delta_solubility")
                            if sol_val is not None:
                                additive_sol += sol_val
                                n_sol += 1
                            e1_val = pool_entry["scores"].get("e1_gain")
                            if e1_val is not None:
                                individual_e1 += e1_val

                    # Epistasis = E1_combo - (WT + sum of individual E1 gains)
                    epistasis = e1_fitness - (wt_e1 + individual_e1)

                    entry = {
                        "mutations": [m.label for m in muts],
                        "n_mutations": len(muts),
                        "e1_fitness": e1_fitness,
                        "e1_delta": e1_fitness - wt_e1,
                        "additive_ddg": round(additive_ddg, 3) if n_ddg > 0 else None,
                        "additive_sol": round(additive_sol / max(n_sol, 1), 4) if n_sol > 0 else None,
                        "epistasis_signal": round(epistasis, 4),
                        "sequence": seq,
                    }
                    scored_combos.append(entry)

                    # Create ProteinCandidate for pipeline
                    try:
                        variant = parent.apply_mutations(muts)
                        variant.scores["e1_fitness"] = e1_fitness
                        variant.scores["e1_delta"] = e1_fitness - wt_e1
                        if n_ddg > 0:
                            variant.scores["additive_ddg"] = additive_ddg
                        if n_sol > 0:
                            variant.scores["additive_sol"] = additive_sol / n_sol
                        variant.scores["epistasis_signal"] = epistasis
                        variant.metadata["combination_order"] = len(muts)
                        variant.metadata["component_mutations"] = [m.label for m in muts]
                        variant.metadata["optimizer_source"] = True
                        candidates.append(variant)
                    except ValueError as e:
                        logger.debug(f"Skipping combination: {e}")

            # Sort and store top combos in metadata
            scored_combos.sort(key=lambda x: x["e1_fitness"], reverse=True)
            parent.metadata["multi_mutant_table"] = scored_combos[:100]
            parent.metadata["optimizer_summary"] = {
                "n_pool": len(selected),
                "n_combinations": len(combo_sequences),
                "n_scored": len(scored_combos),
                "best_e1": scored_combos[0]["e1_fitness"] if scored_combos else None,
                "best_mutations": scored_combos[0]["mutations"] if scored_combos else [],
                "wt_e1": wt_e1,
                "max_order": max_order,
            }

            logger.info(
                f"{parent.name}: Optimization complete — "
                f"{len(scored_combos)} combos scored, "
                f"best E1 = {scored_combos[0]['e1_fitness']:.4f}" if scored_combos else ""
            )

        return StepResult(
            step_name=self.name,
            candidates=candidates,
            config_used=config,
            warnings=warnings,
        )


def _collect_mutation_pool(
    parent: ProteinCandidate,
    prior_results: dict[str, StepResult],
    step_input: StepResult,
    protected: set[int],
) -> dict[tuple[int, str, str], dict]:
    """Collect all beneficial single mutations from all prior steps.

    Returns dict: (position, wt, mut) → {position, wt, mut, scores, sources}.
    """
    pool: dict[tuple[int, str, str], dict] = {}

    # Collect from all prior step results
    for step_name, result in prior_results.items():
        if not isinstance(result, StepResult):
            continue
        for c in result.candidates:
            if c.parent_id is None:
                continue
            if c.parent_id != parent.candidate_id:
                continue
            if len(c.mutations) != 1:
                continue  # only single-point mutations

            m = c.mutations[0]
            if m.position in protected:
                continue

            source = m.source_step or step_name
            if source not in _MUTATION_SOURCES:
                continue

            key = (m.position, m.wt, m.mut)
            if key not in pool:
                pool[key] = {
                    "position": m.position,
                    "wt": m.wt,
                    "mut": m.mut,
                    "scores": {},
                    "sources": [],
                }

            # Merge evidence
            if source not in pool[key]["sources"]:
                pool[key]["sources"].append(source)

            # Store step-specific scores
            if source == "stability_ddg" and "ddg" in c.scores:
                pool[key]["scores"]["ddg"] = c.scores["ddg"]
            elif source == "e1_score":
                gain = (m.metadata or {}).get("e1_gain")
                if gain is not None:
                    pool[key]["scores"]["e1_gain"] = gain
            elif source == "solubility_ssm" and "delta_solubility" in c.scores:
                pool[key]["scores"]["delta_solubility"] = c.scores["delta_solubility"]
            elif source == "consensus_design":
                pool[key]["scores"]["consensus_conservation"] = m.score or 0.0
            elif source == "pssm_analysis":
                pool[key]["scores"]["pssm_log_odds"] = m.score or 0.0
            elif source in ("motif_scan", "cysteine_scan", "cavity_fill",
                            "surface_patch", "disulfide_design"):
                pool[key]["scores"][f"{source}_score"] = m.score or 0.0

    # Also check current step input
    for c in step_input.candidates:
        if c.parent_id is None or c.parent_id != parent.candidate_id:
            continue
        if len(c.mutations) != 1:
            continue
        m = c.mutations[0]
        if m.position in protected:
            continue
        source = m.source_step or "unknown"
        key = (m.position, m.wt, m.mut)
        if key not in pool:
            pool[key] = {
                "position": m.position,
                "wt": m.wt,
                "mut": m.mut,
                "scores": {},
                "sources": [],
            }
        if source not in pool[key]["sources"]:
            pool[key]["sources"].append(source)

    return pool


def _rank_by_evidence(
    pool: dict[tuple[int, str, str], dict],
) -> list[dict]:
    """Rank mutations by combined evidence from multiple sources.

    Computes z-scored normalized scores and a breadth bonus for
    mutations supported by multiple independent tools.
    """
    entries = list(pool.values())
    if not entries:
        return []

    # Compute z-scores for available metrics
    score_keys = ["e1_gain", "ddg", "delta_solubility",
                  "consensus_conservation", "pssm_log_odds"]
    stats: dict[str, tuple[float, float]] = {}
    for key in score_keys:
        vals = [e["scores"][key] for e in entries if key in e["scores"]]
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = var ** 0.5
            stats[key] = (mean, std)

    # Weights for each score type
    weights = {
        "e1_gain": 0.30,
        "ddg": 0.25,
        "delta_solubility": 0.20,
        "consensus_conservation": 0.10,
        "pssm_log_odds": 0.10,
    }
    breadth_bonus = 0.15  # per additional source beyond 1

    for entry in entries:
        score = 0.0
        total_weight = 0.0

        for key, weight in weights.items():
            if key not in entry["scores"]:
                continue
            raw = entry["scores"][key]
            if key in stats:
                mean, std = stats[key]
                if std > 1e-8:
                    z = (raw - mean) / std
                else:
                    z = 0.0
            else:
                z = 0.0

            # Invert ddg (more negative = better)
            if key == "ddg":
                z = -z

            score += weight * z
            total_weight += weight

        # Normalize by available weight
        if total_weight > 0:
            score /= total_weight

        # Breadth bonus
        n_sources = len(entry["sources"])
        entry["evidence_breadth"] = n_sources
        entry["evidence_score"] = score + breadth_bonus * max(0, n_sources - 1)

    entries.sort(key=lambda e: e["evidence_score"], reverse=True)
    return entries


def _score_with_e1(
    sequences: list[str],
    homolog_seqs: list[str],
    conda_env: str,
    model_name: str,
) -> list[float] | None:
    """Score sequences with Profluent E1 via subprocess."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_p = Path(tmpdir) / "optimizer_e1_in.json"
            out_p = Path(tmpdir) / "optimizer_e1_out.json"

            helper_input = {
                "model_name": model_name,
                "homolog_seqs": homolog_seqs,
                "sequences": [
                    {
                        "id": f"seq_{i}",
                        "sequence": seq,
                        "is_parent": (i == 0),
                    }
                    for i, seq in enumerate(sequences)
                ],
            }

            in_p.write_text(json.dumps(helper_input))

            helper = str(_E1_HELPER)
            if not Path(helper).exists():
                logger.error(f"E1 helper not found: {helper}")
                return None

            logger.info(
                f"Dispatching E1 to '{conda_env}' "
                f"({len(sequences)} sequences)..."
            )

            r = subprocess.run(
                [
                    "conda", "run", "--no-capture-output", "-n", conda_env,
                    "python", helper, str(in_p), str(out_p),
                ],
                capture_output=True, text=True, timeout=7200,
            )
            if r.returncode != 0:
                logger.error(f"E1 helper failed:\n{r.stderr[:2000]}")
                return None
            if not out_p.exists():
                logger.error("E1 helper produced no output")
                return None

            output = json.loads(out_p.read_text())
            scores_list = output.get("scores", [])

            # Build ordered score list
            result_scores = [0.0] * len(sequences)
            for entry in scores_list:
                idx = int(entry["id"].split("_")[1])
                val = entry.get("e1_fitness")
                if val is not None and idx < len(result_scores):
                    result_scores[idx] = val

            return result_scores

    except subprocess.TimeoutExpired:
        logger.error("E1 optimizer scoring timed out (7200s)")
        return None
    except Exception as e:
        logger.error(f"E1 optimizer dispatch failed: {e}")
        return None
