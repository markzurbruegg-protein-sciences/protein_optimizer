"""v2 HTML report — redesigned with clear, hierarchical layout.

Generates a self-contained HTML report with:
- Top bar: protein name, length, date
- Hero: Interactive 3Dmol.js viewer + biophysical metrics sidebar
- Executive Summary: key stats cards (candidates, E1 range, top pick, stages)
- Top Recommendations: single unified table, top 20 by composite score
- Pipeline Analysis: concise stage cards, top 5 per stage
- Sequence Liabilities: combined cysteine + motif table
- Full Variant Library: searchable, expandable table (single source of truth)
"""

from __future__ import annotations

import html as _html
import json
import logging
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from protein_optimizer.models import Mutation, ProteinCandidate, StepResult

logger = logging.getLogger(__name__)

# ── Amino-acid property tables ──────────────────────────────────────

_MW: dict[str, float] = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "E": 147.13, "Q": 146.15, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

_HYDROPATHY: dict[str, float] = {  # Kyte-Doolittle
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_CHARGE: dict[str, float] = {
    "D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5,
}

# ── Step display name lookup ────────────────────────────────────────
_STEP_DISPLAY_NAMES: dict[str, str] = {
    "protein_characterization": "Protein Characterization",
    "cysteine_scan":        "Cysteine Scan",
    "motif_scan":           "Motif Scan",
    "sequence_complexity":  "Sequence Complexity",
    "find_homologs":        "Find Homologs",
    "consensus_design":     "Consensus Design",
    "pssm_analysis":        "PSSM Analysis",
    "predict_structure":    "Predict Structure",
    "stability_ddg":        "Stability ΔΔG",
    "disulfide_design":     "Disulfide Design",
    "cavity_fill":          "Cavity Fill",
    "surface_patch":        "Surface Patch",
    "rfdiffusion_diversify": "RFdiffusion Diversify",
    "proteinmpnn_design":   "ProteinMPNN Design",
    "design_validate":      "Design Validate",
    "combine_variants":     "Combine Variants",
    "e1_score":             "E1 Score",
    "esm1v_score":          "ESM-1v Score",
    "esmif1_score":         "ESM-IF1 Score",
}


def _step_display_name(step_name: str) -> str:
    """Return a human-readable display name for a pipeline step."""
    return _STEP_DISPLAY_NAMES.get(step_name, step_name.replace("_", " ").title())


# ── Public API ──────────────────────────────────────────────────────


def generate_report_v2(
    results: dict[str, StepResult],
    output_path: str | Path,
    pdb_path: str | Path | None = None,
    title: str = "Protein Optimization Report",
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate a v2 HTML report.

    Args:
        results: step_name → StepResult mapping (ordered).
        output_path: Where to write the HTML file.
        pdb_path: Path to the PDB file for the 3D viewer.
        title: Report title.
        config: Pipeline config dict (optional).

    Returns:
        Path to the written HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Determine the canonical variant set ──
    # Prefer combine_variants (assembled library), then e1_score, then the step
    # with the most variants, then whatever is last.
    canonical_key = None
    for preferred in ("combine_variants", "e1_score"):
        if preferred in results:
            canonical_key = preferred
            break
    if canonical_key is None:
        canonical_key = max(
            results,
            key=lambda k: len([c for c in results[k].candidates if c.parent_id is not None]),
            default=list(results.keys())[-1] if results else None,
        )

    canonical_result = results.get(canonical_key) if canonical_key else None
    all_candidates = canonical_result.candidates if canonical_result else []
    parent = next((c for c in all_candidates if c.parent_id is None), None)

    # If parent not found in canonical, try other steps
    if parent is None:
        for sr in results.values():
            parent = next((c for c in sr.candidates if c.parent_id is None), None)
            if parent:
                break

    variants = [c for c in all_candidates if c.parent_id is not None]

    # ── Merge E1 scores into variants if available from a separate step ──
    e1_result = results.get("e1_score")
    if e1_result and canonical_key != "e1_score":
        e1_map: dict[str, float] = {}
        for c in e1_result.candidates:
            if "e1_fitness" in c.scores:
                e1_map[c.name] = c.scores["e1_fitness"]
        for v in variants:
            if v.name in e1_map and "e1_fitness" not in v.scores:
                v.scores["e1_fitness"] = e1_map[v.name]

    # ── Merge ESM-1v scores if available from a separate step ──
    esm1v_result = results.get("esm1v_score")
    if esm1v_result and canonical_key != "esm1v_score":
        esm1v_map: dict[str, dict[str, float]] = {}
        for c in esm1v_result.candidates:
            sc: dict[str, float] = {}
            if "esm1v_pll" in c.scores:
                sc["esm1v_pll"] = c.scores["esm1v_pll"]
            if "esm1v_delta" in c.scores:
                sc["esm1v_delta"] = c.scores["esm1v_delta"]
            if sc:
                esm1v_map[c.name] = sc
        for v in variants:
            if v.name in esm1v_map:
                for k, val in esm1v_map[v.name].items():
                    if k not in v.scores:
                        v.scores[k] = val

    # ── Merge ESM-IF1 scores if available from a separate step ──
    esmif1_result = results.get("esmif1_score")
    if esmif1_result and canonical_key != "esmif1_score":
        esmif1_map: dict[str, dict[str, float]] = {}
        for c in esmif1_result.candidates:
            sc2: dict[str, float] = {}
            if "esmif1_score" in c.scores:
                sc2["esmif1_score"] = c.scores["esmif1_score"]
            if "esmif1_delta" in c.scores:
                sc2["esmif1_delta"] = c.scores["esmif1_delta"]
            if sc2:
                esmif1_map[c.name] = sc2
        for v in variants:
            if v.name in esmif1_map:
                for k, val in esmif1_map[v.name].items():
                    if k not in v.scores:
                        v.scores[k] = val

    # Rank variants by E1 fitness (primary) then composite_score (fallback)
    def _rank_key(c: ProteinCandidate) -> float:
        return c.scores.get("e1_fitness", c.scores.get("composite_score", 0))

    variants_ranked = sorted(variants, key=_rank_key, reverse=True)

    # Top-10 unique E1 mutations for 3D viewer (deduped, sorted by e1_fitness)
    e1_top10 = _e1_top_mutations(variants_ranked, n=10)

    # PDB text (for inline embedding)
    pdb_text = ""
    if pdb_path and Path(pdb_path).exists():
        pdb_text = Path(pdb_path).read_text(encoding="utf-8", errors="replace")

    # Protein metrics
    seq = parent.sequence if parent else ""
    metrics = _compute_metrics(seq, pdb_text, all_candidates, results)

    # Per-step mutations
    step_mutations = _collect_step_mutations(results)

    # Build HTML with new architecture:
    # 1. Header (CSS/meta)
    # 2. Hero (top bar + 3D viewer + metrics sidebar)
    # 3. Executive Summary (key stats cards)
    # 4. Pipeline Status bar
    # 5. Step-by-step Analysis (per-step collapsible detail cards)
    # 6. Issues & Diagnostics
    # 7. Full Variant Library (expandable at bottom)
    # 8. Footer
    parts = [
        _header_html(title),
        _hero_section(parent, metrics, pdb_text, e1_top10, variants_ranked),
        '<div class="content">',
        _executive_summary_section(results, variants_ranked, metrics),
        _pipeline_status_bar(results),
        _step_details_section(results),
        _diagnostics_section(results),
        _full_variant_library_section(variants_ranked),
        '</div><!-- /content -->',
        _footer_html(),
    ]

    html_text = "\n".join(parts)
    output_path.write_text(html_text, encoding="utf-8")
    logger.info(f"Report v2 saved to {output_path}")
    return output_path


# ── Metrics computation ─────────────────────────────────────────────


def _compute_metrics(
    seq: str,
    pdb_text: str,
    candidates: list[ProteinCandidate],
    results: dict[str, StepResult],
) -> dict[str, Any]:
    """Compute biophysical metrics for the parent sequence."""
    n = len(seq)
    if n == 0:
        return {}

    # Molecular weight
    mw = sum(_MW.get(aa, 128.0) for aa in seq) - 18.015 * (n - 1)

    # Isoelectric point (simplified Henderson-Hasselbalch)
    pi = _compute_pI(seq)

    # GRAVY (Grand Average of Hydropathy)
    gravy = sum(_HYDROPATHY.get(aa, 0) for aa in seq) / n

    # Net charge at pH 7.4
    charge = _net_charge(seq, 7.4)

    # Instability index (Guruprasad et al., 1990)
    ii = _instability_index(seq)
    ii_label = "Stable" if ii < 40 else "Unstable"

    # Aliphatic index (thermostability proxy)
    ai = _aliphatic_index(seq)

    # Aggregation propensity — crude heuristic: fraction of hydrophobic
    # windows ≥ 5 residues with mean hydropathy > 1.6
    agg_score, agg_patches = _aggregation_propensity(seq)

    # Cysteine / disulfide info
    cys_count = seq.count("C")

    # pLDDT from PDB (B-factor column)
    plddt_mean, plddt_min = _plddt_from_pdb(pdb_text)

    # Expression risk flags
    rare_codons = sum(seq.count(aa) for aa in "WCM")  # rare in E. coli
    proline_count = seq.count("P")

    # Surface patches from results
    surface_result = results.get("surface_patch")
    n_surface_patches = 0
    if surface_result:
        for c in surface_result.candidates:
            if c.parent_id is None and "hydrophobic_patches" in c.metadata:
                n_surface_patches = len(c.metadata["hydrophobic_patches"])

    # ── Pull characterization data from protein_characterization step ──
    char: dict[str, Any] = {}
    char_result = results.get("protein_characterization")
    if char_result:
        for c in char_result.candidates:
            if c.parent_id is None and "characterization" in c.metadata:
                char = c.metadata["characterization"]
                break

    return {
        "length": n,
        "mw": mw,
        "pI": pi,
        "gravy": gravy,
        "charge_7_4": charge,
        "instability_index": ii,
        "instability_label": ii_label,
        "aliphatic_index": ai,
        "agg_score": agg_score,
        "agg_patches": agg_patches,
        "cys_count": cys_count,
        "plddt_mean": plddt_mean,
        "plddt_min": plddt_min,
        "rare_codons": rare_codons,
        "proline_count": proline_count,
        "n_surface_patches": n_surface_patches,
        # Characterization step data
        "signal_peptide": char.get("signal_peptide", {}),
        "disorder_fraction": char.get("disorder_fraction", 0),
        "disorder_regions": char.get("disorder_regions", []),
        "disorder_scores": char.get("disorder_scores", []),
        "disorder_method": char.get("disorder_method", ""),
        "domains": char.get("domains", []),
        "domain_count": char.get("domain_count", 0),
        "domain_method": char.get("domain_method", ""),
        "oligomeric_state": char.get("oligomeric_state", ""),
        "oligomeric_evidence": char.get("oligomeric_evidence", []),
        "cofactor_motifs": char.get("cofactor_motifs", []),
        "cofactor_summary": char.get("cofactor_summary", ""),
        "estimated_tm": char.get("estimated_tm", 0),
        "tm_confidence": char.get("tm_confidence", ""),
        "ph_curve": char.get("ph_curve", []),
        "ph_stable_range": char.get("ph_stable_range", []),
        "aggregation_regions": char.get("aggregation_regions", []),
        "aggregation_region_count": char.get("aggregation_region_count", 0),
        "charge_symmetry": char.get("charge_symmetry", 0),
        "charged_fraction": char.get("charged_fraction", 0),
        "solubility_score": char.get("solubility_score", 0),
        "solubility_class": char.get("solubility_class", ""),
        "extinction_coefficient": char.get("extinction_coefficient", {}),
        "disulfide_potential": char.get("disulfide_potential", {}),
        "rare_codon_details": char.get("rare_codon_details", ""),
        "rare_codon_fraction": char.get("rare_codon_fraction", 0),
        "aromaticity": char.get("aromaticity", 0),
        "sequence": char.get("sequence", seq),
    }


def _compute_pI(seq: str) -> float:
    """Estimate isoelectric point by bisection."""
    def _charge_at_pH(pH: float) -> float:
        # pKa values
        pKa = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.46,
                "H": 6.00, "K": 10.53, "R": 12.48}
        nterm_pKa, cterm_pKa = 9.69, 2.34

        charge = 1.0 / (1.0 + 10 ** (pH - nterm_pKa))  # N-term
        charge -= 1.0 / (1.0 + 10 ** (cterm_pKa - pH))  # C-term

        for aa in seq:
            if aa in ("D", "E", "C", "Y"):
                charge -= 1.0 / (1.0 + 10 ** (pKa[aa] - pH))
            elif aa in ("H", "K", "R"):
                charge += 1.0 / (1.0 + 10 ** (pH - pKa[aa]))
        return charge

    lo, hi = 0.0, 14.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _charge_at_pH(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def _net_charge(seq: str, pH: float) -> float:
    pKa = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.46,
            "H": 6.00, "K": 10.53, "R": 12.48}
    charge = 1.0 / (1.0 + 10 ** (pH - 9.69))
    charge -= 1.0 / (1.0 + 10 ** (2.34 - pH))
    for aa in seq:
        if aa in ("D", "E", "C", "Y"):
            charge -= 1.0 / (1.0 + 10 ** (pKa[aa] - pH))
        elif aa in ("H", "K", "R"):
            charge += 1.0 / (1.0 + 10 ** (pH - pKa[aa]))
    return round(charge, 1)


def _instability_index(seq: str) -> float:
    """Guruprasad instability index (simplified)."""
    # Dipeptide instability weight table (subset)
    _DIWV = {
        "WW": 1.0, "WC": 1.0, "WM": 24.68, "WF": 1.0, "WG": -9.37,
        "GG": -7.49, "GA": -7.49, "GD": 1.0, "GE": -6.54, "GK": -7.49,
        "CW": 24.68, "CC": 1.0, "CH": 1.0, "CM": 33.6, "CF": 1.0,
        "DG": 1.0, "DD": 1.0, "DE": 1.0, "DW": 1.0, "DF": -6.54,
        "LL": 1.0, "LV": 1.0, "LG": 1.0, "LP": 20.26,
    }
    n = len(seq)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        dipep = seq[i] + seq[i + 1]
        total += _DIWV.get(dipep, 1.0)
    return round(10.0 / n * total, 2)


def _aliphatic_index(seq: str) -> float:
    """Aliphatic index — higher = more thermostable."""
    n = len(seq)
    if n == 0:
        return 0.0
    a = seq.count("A") / n * 100
    v = seq.count("V") / n * 100
    i = seq.count("I") / n * 100
    l = seq.count("L") / n * 100
    return round(a + 2.9 * v + 3.9 * (i + l), 1)


def _aggregation_propensity(seq: str, window: int = 7, threshold: float = 1.6) -> tuple[float, int]:
    """Estimate aggregation-prone patches."""
    n = len(seq)
    patches = 0
    in_patch = False
    for i in range(n - window + 1):
        mean_h = sum(_HYDROPATHY.get(seq[j], 0) for j in range(i, i + window)) / window
        if mean_h > threshold:
            if not in_patch:
                patches += 1
                in_patch = True
        else:
            in_patch = False
    # Score 0-1: fraction of sequence in aggregation-prone windows
    prone_count = sum(
        1 for i in range(n - window + 1)
        if sum(_HYDROPATHY.get(seq[j], 0) for j in range(i, i + window)) / window > threshold
    )
    score = round(prone_count / max(n - window + 1, 1), 3)
    return score, patches


def _plddt_from_pdb(pdb_text: str) -> tuple[float, float]:
    """Extract mean and min pLDDT from PDB B-factor column (CA atoms)."""
    values = []
    for line in pdb_text.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                values.append(float(line[60:66].strip()))
            except (ValueError, IndexError):
                pass
    if not values:
        return 0.0, 0.0
    return round(sum(values) / len(values), 1), round(min(values), 1)


# ── Helpers ─────────────────────────────────────────────────────────


def _top_positions(variants: list[ProteinCandidate], n: int = 5) -> list[int]:
    """Get top-n unique mutation positions from ranked variants."""
    seen: set[int] = set()
    positions: list[int] = []
    for v in variants:
        for m in v.mutations:
            if m.position not in seen:
                seen.add(m.position)
                positions.append(m.position)
            if len(positions) >= n:
                return positions
    return positions


def _e1_top_mutations(
    variants_ranked: list[ProteinCandidate],
    n: int = 10,
) -> list[dict[str, Any]]:
    """Top-N unique E1 mutations from ranked variants (deduplicated by label)."""
    seen_labels: set[str] = set()
    result: list[dict[str, Any]] = []
    for v in variants_ranked:
        score = v.scores.get("e1_fitness", v.scores.get("composite_score", 0))
        for mut in v.mutations:
            lbl = f"{mut.wt}{mut.position}{mut.mut}"
            if lbl not in seen_labels:
                seen_labels.add(lbl)
                result.append({"pos": mut.position, "label": lbl, "score": score})
            if len(result) >= n:
                return result
    return result


def _collect_step_mutations(
    results: dict[str, StepResult],
) -> dict[str, list[ProteinCandidate]]:
    """Collect variant candidates per step, sorted by best score."""
    step_muts: dict[str, list[ProteinCandidate]] = {}
    for step_name, result in results.items():
        variants = [c for c in result.candidates if c.parent_id is not None]
        if not variants:
            continue
        # Sort by composite_score if available, else any score
        variants.sort(
            key=lambda c: c.scores.get(
                "composite_score",
                max(c.scores.values()) if c.scores else 0,
            ),
            reverse=True,
        )
        step_muts[step_name] = variants
    return step_muts


_TIER_MAP: dict[str, int] = {
    "protein_characterization": 1,
    "cysteine_scan": 1, "motif_scan": 1, "sequence_complexity": 1,
    "find_homologs": 2, "consensus_design": 2, "pssm_analysis": 2,
    "predict_structure": 3, "disulfide_design": 3, "cavity_fill": 3,
    "surface_patch": 3, "stability_ddg": 3,
    "rfdiffusion_diversify": 4, "proteinmpnn_design": 4,
    "design_validate": 4, "combine_variants": 4,
    "esm1v_score": 5, "e1_score": 5, "esmif1_score": 5,
    "motif_scaffold": 6,
}

_TIER_NAMES = {
    1: "Sequence Heuristics",
    2: "Evolutionary Analysis",
    3: "Structure-based Engineering",
    4: "Design & Combination",
    5: "PLM Scoring",
    6: "Generative Design",
}


def _fmt_score(v: float) -> str:
    if abs(v) > 100:
        return f"{v:.1f}"
    return f"{v:.4f}"


# ── HTML generators ─────────────────────────────────────────────────


def _header_html(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html.escape(title)}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
/* ── Reset & base ── */
:root {{
  --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff;
  --green: #3fb950; --red: #f85149; --yellow: #d29922; --purple: #bc8cff;
  --card-bg: #161b22; --border: #30363d; --hover: rgba(88,166,255,0.06);
}}
*, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.55;
}}

/* ── Top bar ── */
.topbar {{
  background: var(--card-bg); border-bottom: 1px solid var(--border);
  padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem;
}}
.topbar h1 {{ font-size: 1.35rem; color: var(--accent); }}
.topbar .meta {{ color: #8b949e; font-size: 0.82rem; margin-left: auto; }}

/* ── Hero grid (3D + metrics) ── */
.hero {{
  display: grid; grid-template-columns: 1fr 380px;
  gap: 0; border-bottom: 1px solid var(--border);
  min-height: 520px;
}}
@media (max-width: 960px) {{ .hero {{ grid-template-columns: 1fr; }} }}

.viewer-wrap {{
  position: relative; background: #000;
  min-height: 500px;
}}
.viewer-wrap .viewer-label {{
  position: absolute; top: 10px; left: 14px; z-index: 2;
  font-size: 0.75rem; color: #8b949e; background: rgba(0,0,0,.65);
  padding: 3px 8px; border-radius: 4px;
}}
#viewport {{ width: 100%; height: 100%; min-height: 500px; }}

/* ── Metrics sidebar ── */
.metrics-panel {{
  background: var(--card-bg); border-left: 1px solid var(--border);
  padding: 1.25rem 1.5rem; overflow-y: auto; max-height: 520px;
}}
.metrics-panel h2 {{ font-size: 1.05rem; color: var(--accent); margin-bottom: 0.9rem; }}
.metric-group {{ margin-bottom: 1rem; }}
.metric-group h3 {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: #8b949e; margin-bottom: 0.4rem; }}
.metric-row {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.3rem 0; border-bottom: 1px solid var(--border); font-size: 0.88rem;
}}
.metric-row .label {{ color: var(--fg); }}
.metric-row .value {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.metric-row .value.good {{ color: var(--green); }}
.metric-row .value.warn {{ color: var(--yellow); }}
.metric-row .value.bad {{ color: var(--red); }}

/* ── Content area ── */
.content {{ max-width: 1200px; margin: 0 auto; padding: 2rem 2rem 4rem; }}

/* ── Section ── */
.section {{ margin-bottom: 2.5rem; }}
.section h2 {{
  font-size: 1.15rem; color: var(--accent); margin-bottom: 0.6rem;
  padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
}}
.section .subtitle {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 1rem; }}

/* ── Table ── */
.tbl-wrap {{ overflow-x: auto; }}
table {{
  width: 100%; border-collapse: collapse; font-size: 0.84rem;
}}
th {{
  background: var(--card-bg); color: var(--accent); padding: 0.55rem 0.6rem;
  text-align: left; border-bottom: 2px solid var(--border); position: sticky; top: 0;
  font-weight: 600; white-space: nowrap;
}}
td {{
  padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border);
}}
tr:hover {{ background: var(--hover); }}
.mono {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 0.82rem; }}
.good {{ color: var(--green); }}
.bad {{ color: var(--red); }}
.warn {{ color: var(--yellow); }}
.muted {{ color: #8b949e; }}

/* ── Rank badge ── */
.rank {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; border-radius: 50%; font-size: 0.78rem; font-weight: 700;
  background: var(--card-bg); border: 1px solid var(--border); color: var(--accent);
}}
.rank.gold {{ background: #d4a017; color: #000; border-color: #d4a017; }}
.rank.silver {{ background: #8b949e; color: #000; border-color: #8b949e; }}
.rank.bronze {{ background: #a0522d; color: #fff; border-color: #a0522d; }}

/* ── Tier header ── */
.tier-header {{
  display: flex; align-items: center; gap: 0.6rem;
  margin: 1.5rem 0 0.5rem; padding: 0.5rem 0.7rem;
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px;
  cursor: default;
}}
.tier-badge {{
  font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px;
  background: var(--accent); color: #000; white-space: nowrap;
}}
.tier-header .tier-title {{ font-size: 0.95rem; font-weight: 600; }}

/* ── Step card ── */
.step-card {{
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
  margin: 0.6rem 0; overflow: hidden;
}}
.step-card summary {{
  padding: 0.65rem 1rem; cursor: pointer; font-weight: 600; font-size: 0.92rem;
  list-style: none; display: flex; align-items: center; gap: 0.6rem;
}}
.step-card summary::-webkit-details-marker {{ display: none; }}
.step-card summary::before {{
  content: '▸'; color: var(--accent); transition: transform 0.15s;
}}
.step-card[open] summary::before {{ transform: rotate(90deg); }}
.step-card .count-badge {{
  font-size: 0.72rem; padding: 1px 7px; border-radius: 10px;
  background: var(--border); color: var(--fg); margin-left: auto;
}}
.step-card .inner {{ padding: 0 1rem 1rem; }}

/* ── Show more button ── */
.show-more-btn {{
  display: inline-block; margin-top: 0.5rem; padding: 0.35rem 1rem;
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px;
  color: var(--accent); cursor: pointer; font-size: 0.82rem;
}}
.show-more-btn:hover {{ background: var(--border); }}
.hidden-rows {{ display: none; }}
.hidden-rows.show {{ display: table-row-group; }}

/* ── Legend highlight key ── */
.legend {{
  display: flex; gap: 1.2rem; flex-wrap: wrap; margin: 0.7rem 0 0;
  font-size: 0.78rem; color: #8b949e;
}}
.legend span {{ display: inline-flex; align-items: center; gap: 0.3rem; }}
.legend .dot {{
  width: 10px; height: 10px; border-radius: 50%; display: inline-block;
}}

/* ── Viewer toolbar & legend ── */
.viewer-toolbar {{
  position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
  z-index: 2; display: flex; align-items: center; gap: 3px;
  background: rgba(22,27,34,0.92); padding: 4px 6px;
  border-radius: 8px; border: 1px solid var(--border);
  backdrop-filter: blur(8px);
}}
.view-btn {{
  padding: 5px 12px; border-radius: 5px; border: 1px solid transparent;
  background: transparent; color: #8b949e; cursor: pointer;
  font-size: 0.73rem; font-weight: 600; transition: all 0.15s;
  white-space: nowrap; font-family: inherit;
}}
.view-btn:hover {{ background: rgba(88,166,255,0.12); color: var(--fg); }}
.view-btn.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}
.toolbar-sep {{ width: 1px; height: 20px; background: var(--border); margin: 0 4px; }}
.viewer-legend {{
  position: absolute; bottom: 50px; left: 50%; transform: translateX(-50%);
  z-index: 2; font-size: 0.73rem; color: #c9d1d9;
  background: rgba(22,27,34,0.85); padding: 5px 14px; border-radius: 6px;
  border: 1px solid var(--border); backdrop-filter: blur(8px);
  white-space: nowrap; display: flex; align-items: center; gap: 0.7rem;
}}

/* ── Tier selector bar (top of viewer) ── */
.tier-toolbar {{
  position: absolute; top: 36px; left: 50%; transform: translateX(-50%);
  z-index: 2; display: flex; align-items: center; gap: 3px;
  background: rgba(22,27,34,0.92); padding: 3px 6px;
  border-radius: 7px; border: 1px solid var(--border);
  backdrop-filter: blur(8px);
}}
.tier-btn {{
  padding: 4px 10px; border-radius: 4px; border: 1px solid transparent;
  background: transparent; color: #8b949e; cursor: pointer;
  font-size: 0.70rem; font-weight: 600; transition: all 0.15s;
  white-space: nowrap; font-family: inherit;
}}
.tier-btn:hover {{ background: rgba(88,166,255,0.12); color: var(--fg); }}
.tier-btn.active {{ background: var(--purple); color: #000; border-color: var(--purple); }}
.tier-btn .tier-pip {{
  display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  margin-right: 3px; vertical-align: middle;
}}

/* ── Step summary stats ── */
.summary-stats {{
  display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0 0.8rem;
}}
.stat-chip {{
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.3rem 0.75rem; border-radius: 6px;
  background: rgba(88,166,255,0.08); border: 1px solid var(--border);
  font-size: 0.8rem; color: var(--fg);
}}
.stat-chip strong {{ color: var(--accent); font-variant-numeric: tabular-nums; }}
.stat-chip.good strong {{ color: var(--green); }}
.stat-chip.warn strong {{ color: var(--yellow); }}
.stat-chip.bad strong {{ color: var(--red); }}

/* ── Pipeline status bar ── */
.status-bar {{
  display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0;
}}
.status-pip {{
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.3rem 0.7rem; border-radius: 6px; font-size: 0.78rem; font-weight: 600;
  background: var(--card-bg); border: 1px solid var(--border);
  cursor: default; transition: background 0.15s;
}}
.status-pip:hover {{ background: var(--border); }}
.status-pip.step-ok {{ border-color: var(--green); color: var(--green); }}
.status-pip.step-warn {{ border-color: var(--yellow); color: var(--yellow); }}
.status-pip.step-fail {{ border-color: var(--red); color: var(--red); }}

/* ── Diagnostics section ── */
.diag-card {{
  background: var(--card-bg); border-radius: 8px; padding: 1rem 1.25rem;
  margin: 0.75rem 0; border-left: 3px solid var(--border);
}}
.diag-card.diag-fail {{ border-left-color: var(--red); }}
.diag-card.diag-warn {{ border-left-color: var(--yellow); }}
.diag-header {{
  display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;
}}
.diag-warnings {{
  list-style: none; padding: 0; margin: 0.3rem 0;
}}
.diag-warnings li {{
  padding: 0.2rem 0; font-size: 0.85rem; color: #8b949e;
}}
.diag-warnings li::before {{ content: "\\2022  "; color: var(--fg); }}
.diag-fix {{
  margin-top: 0.5rem; padding: 0.6rem 0.8rem; border-radius: 6px;
  background: rgba(88,166,255,0.06); border: 1px solid var(--border);
  font-size: 0.82rem; line-height: 1.6;
}}
.diag-fix code {{
  background: rgba(88,166,255,0.12); padding: 1px 5px; border-radius: 3px;
  font-size: 0.8rem; font-family: 'SFMono-Regular', Consolas, monospace;
}}

/* ── Step status icon ── */
.step-status {{
  font-size: 0.85rem; margin-right: 0.15rem;
}}

/* ── Executive summary cards ── */
.exec-summary {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem; margin-bottom: 2rem;
}}
.exec-card {{
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
  padding: 1.1rem 1.25rem; display: flex; flex-direction: column; gap: 0.3rem;
  min-width: 0; overflow: hidden;
}}
.exec-card .ec-label {{
  font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.07em; color: #8b949e;
}}
.exec-card .ec-value {{
  font-size: 1.7rem; font-weight: 700; color: var(--accent);
  font-variant-numeric: tabular-nums; line-height: 1.1;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.exec-card .ec-sub {{
  font-size: 0.78rem; color: #8b949e;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.exec-card.highlight {{ border-color: var(--accent); }}
.exec-card.good-card {{ border-color: var(--green); }}
.exec-card.good-card .ec-value {{ color: var(--green); }}

/* ── Top recommendations section ── */
.recs-table td.rank-cell {{ width: 42px; text-align: center; }}
.score-badge {{
  display: inline-block; padding: 1px 8px; border-radius: 4px;
  font-size: 0.78rem; font-weight: 700;
}}
.score-badge.good {{ background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid var(--green); }}
.score-badge.bad {{ background: rgba(248,81,73,0.15); color: var(--red); border: 1px solid var(--red); }}
.score-badge.neutral {{ background: rgba(88,166,255,0.1); color: var(--accent); border: 1px solid var(--accent); }}

/* ── Pipeline analysis ── */
.pipeline-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 1rem; margin-top: 1rem;
}}
.pipeline-card {{
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden;
}}
.pipeline-card .pc-header {{
  padding: 0.75rem 1rem; display: flex; align-items: center; gap: 0.6rem;
  border-bottom: 1px solid var(--border);
}}
.pipeline-card .pc-header .stage-num {{
  font-size: 0.68rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;
  background: var(--accent); color: #000; white-space: nowrap;
}}
.pipeline-card .pc-header .pc-title {{
  font-size: 0.9rem; font-weight: 600; flex: 1;
}}
.pipeline-card .pc-header .pc-count {{
  font-size: 0.72rem; color: #8b949e;
}}
.pipeline-card .pc-desc {{
  padding: 0.5rem 1rem; font-size: 0.8rem; color: #8b949e; border-bottom: 1px solid var(--border);
  line-height: 1.5;
}}
.pipeline-card .pc-chips {{
  padding: 0.6rem 1rem; display: flex; flex-wrap: wrap; gap: 0.4rem;
  border-bottom: 1px solid var(--border);
}}
.pipeline-card .pc-top5 {{ padding: 0 1rem 0.75rem; }}
.pipeline-card .pc-top5 table {{ font-size: 0.8rem; }}
.pipeline-card .pc-top5 table td {{ padding: 0.3rem 0.4rem; }}

/* ── Sequence liabilities standalone ── */
.liabilities-section {{ margin-bottom: 2.5rem; }}

/* ── Full variant library ── */
.lib-controls {{
  display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem; flex-wrap: wrap;
}}
.lib-search {{
  flex: 1; min-width: 200px; padding: 0.4rem 0.75rem;
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px;
  color: var(--fg); font-size: 0.85rem; font-family: inherit;
}}
.lib-search:focus {{ outline: none; border-color: var(--accent); }}
.lib-toggle {{
  padding: 0.35rem 1rem; background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 6px; color: var(--accent); cursor: pointer; font-size: 0.82rem;
  font-family: inherit;
}}
.lib-toggle:hover {{ background: var(--border); }}

/* ── Step detail cards (per-step collapsible) ── */
.tier-divider {{
  display: flex; align-items: center; gap: 0.8rem;
  margin: 2rem 0 0.8rem; padding: 0.5rem 0;
  border-bottom: 1px solid var(--border);
}}
.tier-divider .td-badge {{
  font-size: 0.68rem; font-weight: 700; padding: 3px 9px; border-radius: 4px;
  background: var(--accent); color: #000; white-space: nowrap;
}}
.tier-divider .td-title {{
  font-size: 1rem; font-weight: 600; color: var(--fg);
}}
.tier-divider .td-desc {{
  color: #8b949e; font-size: 0.82rem; flex: 1;
}}

.sd-card {{
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
  margin: 0.5rem 0; overflow: hidden;
}}
.sd-card summary {{
  padding: 0.6rem 1rem; cursor: pointer; font-weight: 600; font-size: 0.9rem;
  list-style: none; display: flex; align-items: center; gap: 0.5rem;
  transition: background 0.15s;
}}
.sd-card summary:hover {{ background: var(--hover); }}
.sd-card summary::-webkit-details-marker {{ display: none; }}
.sd-card summary::before {{
  content: '\\25b8'; color: var(--accent); transition: transform 0.15s;
  font-size: 0.85rem;
}}
.sd-card[open] summary::before {{ transform: rotate(90deg); }}
.sd-card .sd-badge {{
  font-size: 0.7rem; padding: 2px 8px; border-radius: 10px;
  background: var(--border); color: var(--fg); margin-left: auto;
  font-weight: 500;
}}
.sd-card .sd-badge.sd-ok {{ background: rgba(63,185,80,0.15); color: var(--green); }}
.sd-card .sd-badge.sd-warn {{ background: rgba(210,153,34,0.15); color: var(--yellow); }}
.sd-card .sd-badge.sd-fail {{ background: rgba(248,81,73,0.15); color: var(--red); }}
.sd-card .sd-inner {{
  padding: 0.5rem 1rem 1rem; border-top: 1px solid var(--border);
}}
.sd-card .sd-summary {{
  font-size: 0.85rem; color: #8b949e; margin-bottom: 0.6rem; line-height: 1.5;
}}

/* Step detail tables */
.sd-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 0.5rem 0; }}
.sd-table th {{
  background: var(--card-bg); color: var(--accent); padding: 0.45rem 0.6rem;
  text-align: left; border-bottom: 2px solid var(--border); font-weight: 600;
  white-space: nowrap; position: sticky; top: 0;
}}
.sd-table td {{ padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--border); }}
.sd-table tr:hover {{ background: var(--hover); }}
.sd-table .extra-row {{ display: none; }}
.sd-table .extra-row.show {{ display: table-row; }}

/* Category badges */
.cat-badge {{
  display: inline-block; padding: 1px 8px; border-radius: 4px;
  font-size: 0.75rem; font-weight: 600; color: #000;
}}
.cat-deamidation {{ background: #58a6ff; }}
.cat-oxidation {{ background: #d29922; }}
.cat-proteolysis {{ background: #f85149; }}
.cat-aggregation {{ background: #bc8cff; }}
.cat-cysteine {{ background: #bc8cff; }}
.cat-homopolymer {{ background: #d29922; }}
.cat-proline {{ background: #f85149; }}
.cat-charge {{ background: #58a6ff; }}
.cat-low-complexity {{ background: #8b949e; }}
.cat-default {{ background: #8b949e; }}

/* Info chip row */
.sd-chips {{
  display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0 0.6rem;
}}
.sd-chip {{
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.25rem 0.65rem; border-radius: 5px;
  background: rgba(88,166,255,0.08); border: 1px solid var(--border);
  font-size: 0.78rem; color: var(--fg);
}}
.sd-chip strong {{ color: var(--accent); }}

/* Show more within step cards */
.sd-expand {{
  display: inline-block; margin-top: 0.4rem; padding: 0.3rem 0.8rem;
  background: var(--card-bg); border: 1px solid var(--border); border-radius: 5px;
  color: var(--accent); cursor: pointer; font-size: 0.78rem; font-family: inherit;
}}
.sd-expand:hover {{ background: var(--border); }}
</style>
</head>
<body>
"""


def _collect_tier_viewer_data(
    step_mutations: dict[str, list[ProteinCandidate]],
) -> dict[int, list[dict[str, Any]]]:
    """Collect unique mutations per tier for 3D viewer highlighting."""
    tier_pos: dict[int, dict[int, dict[str, Any]]] = {}
    for step_name, variants in step_mutations.items():
        tier = _TIER_MAP.get(step_name, 0)
        if tier == 0:
            continue
        if tier not in tier_pos:
            tier_pos[tier] = {}
        for v in variants:
            score = v.scores.get("composite_score", max(v.scores.values()) if v.scores else 0)
            for mut in v.mutations:
                prev = tier_pos[tier].get(mut.position)
                if prev is None or score > prev["score"]:
                    tier_pos[tier][mut.position] = {
                        "pos": mut.position,
                        "label": f"{mut.wt}{mut.position}{mut.mut}",
                        "score": score,
                    }
    result: dict[int, list[dict[str, Any]]] = {}
    for tier_num in sorted(tier_pos):
        items = sorted(tier_pos[tier_num].values(), key=lambda x: x["score"], reverse=True)
        result[tier_num] = items[:20]
    return result


# ── SVG sparkline generators ────────────────────────────────────


def _sparkline_svg(
    values: list[float],
    width: int = 140,
    height: int = 24,
    color: str = "#58a6ff",
    threshold: float | None = None,
    threshold_color: str = "#f85149",
    fill: bool = False,
) -> str:
    """Generate an inline SVG sparkline from a list of values."""
    if not values:
        return ""
    n = len(values)
    vmin = min(values)
    vmax = max(values)
    vrange = vmax - vmin if vmax != vmin else 1.0

    # Build polyline points
    points: list[str] = []
    for i, v in enumerate(values):
        x = (i / max(n - 1, 1)) * width
        y = height - ((v - vmin) / vrange) * (height - 2) - 1
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'style="display:inline-block;vertical-align:middle;margin-left:6px" '
             f'xmlns="http://www.w3.org/2000/svg">']

    if fill:
        # Area fill
        fill_points = f"0,{height} " + polyline + f" {width},{height}"
        parts.append(f'<polygon points="{fill_points}" fill="{color}" opacity="0.15"/>')

    parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" '
                 f'stroke-width="1.5" stroke-linecap="round"/>')

    # Threshold line
    if threshold is not None and vmin <= threshold <= vmax:
        ty = height - ((threshold - vmin) / vrange) * (height - 2) - 1
        parts.append(f'<line x1="0" y1="{ty:.1f}" x2="{width}" y2="{ty:.1f}" '
                     f'stroke="{threshold_color}" stroke-width="0.8" '
                     f'stroke-dasharray="3,2" opacity="0.7"/>')

    parts.append("</svg>")
    return "".join(parts)


def _ph_sparkline(curve: list[list[float]], stable_range: list[float],
                  width: int = 140, height: int = 28) -> str:
    """Generate a pH-vs-charge sparkline with stable window highlighted."""
    if not curve:
        return ""

    charges = [c for _, c in curve]
    pHs = [p for p, _ in curve]
    n = len(curve)
    cmin = min(charges)
    cmax = max(charges)
    crange = cmax - cmin if cmax != cmin else 1.0
    pH_min, pH_max = pHs[0], pHs[-1]
    pH_range = pH_max - pH_min if pH_max != pH_min else 1.0

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'style="display:inline-block;vertical-align:middle;margin-left:6px" '
             f'xmlns="http://www.w3.org/2000/svg">']

    # Stable window background
    if stable_range and len(stable_range) == 2:
        x1 = ((stable_range[0] - pH_min) / pH_range) * width
        x2 = ((stable_range[1] - pH_min) / pH_range) * width
        parts.append(f'<rect x="{x1:.1f}" y="0" width="{max(x2 - x1, 1):.1f}" '
                     f'height="{height}" fill="#3fb950" opacity="0.15" rx="2"/>')

    # Zero-charge line
    if cmin < 0 < cmax:
        zy = height - ((0 - cmin) / crange) * (height - 4) - 2
        parts.append(f'<line x1="0" y1="{zy:.1f}" x2="{width}" y2="{zy:.1f}" '
                     f'stroke="#c9d1d9" stroke-width="0.5" stroke-dasharray="2,2" opacity="0.4"/>')

    # Charge curve
    points: list[str] = []
    for i, (pH, charge) in enumerate(curve):
        x = ((pH - pH_min) / pH_range) * width
        y = height - ((charge - cmin) / crange) * (height - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    parts.append(f'<polyline points="{polyline}" fill="none" stroke="#58a6ff" '
                 f'stroke-width="1.5" stroke-linecap="round"/>')

    parts.append("</svg>")
    return "".join(parts)


def _disorder_sparkline(scores: list[float], threshold: float = 0.5,
                        width: int = 140, height: int = 20) -> str:
    """Generate a per-residue disorder sparkline (bar-style)."""
    if not scores:
        return ""

    n = len(scores)
    bar_w = max(width / n, 0.5)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'style="display:inline-block;vertical-align:middle;margin-left:6px" '
             f'xmlns="http://www.w3.org/2000/svg">']

    # Threshold line
    ty = height - threshold * (height - 2) - 1
    parts.append(f'<line x1="0" y1="{ty:.1f}" x2="{width}" y2="{ty:.1f}" '
                 f'stroke="#f85149" stroke-width="0.6" stroke-dasharray="2,2" opacity="0.5"/>')

    for i, s in enumerate(scores):
        x = (i / n) * width
        bh = s * (height - 2)
        y = height - bh - 1
        color = "#f85149" if s >= threshold else "#58a6ff"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.2f}" '
                     f'height="{bh:.1f}" fill="{color}" opacity="0.7"/>')

    parts.append("</svg>")
    return "".join(parts)


# ── Characterization metric helpers (conditional rendering) ─────


def _char_tm_metric(m: dict, _metric) -> str:
    """Render estimated Tm metric if characterization data available."""
    tm = m.get("estimated_tm", 0)
    if not tm:
        return ""
    conf = m.get("tm_confidence", "low")
    tm_css = "good" if tm >= 60 else "warn" if tm >= 45 else "bad"
    return f'      {_metric("Est. Tm", f"{tm:.0f}°C ({conf} conf.)", tm_css)}'


def _char_aggregation_metric(m: dict, _metric) -> str:
    """Render APR count if characterization data available."""
    apr_count = m.get("aggregation_region_count", 0)
    regions = m.get("aggregation_regions", [])
    if not apr_count and not regions:
        return ""
    apr_css = "good" if apr_count == 0 else "warn" if apr_count <= 2 else "bad"
    parts_text = ""
    if regions:
        locs = ", ".join(f"{r['start']}-{r['end']}" for r in regions[:3])
        parts_text = f" ({locs})"
    return f'      {_metric("APRs", f"{apr_count} region(s){parts_text}", apr_css)}'


def _char_solubility_metric(m: dict, _metric) -> str:
    """Render solubility prediction."""
    score = m.get("solubility_score", 0)
    label = m.get("solubility_class", "")
    if not label:
        return ""
    sol_css = "good" if label == "Soluble" else "warn" if label == "Borderline" else "bad"
    return f'      {_metric("Solubility (E. coli)", f"{score:.0f}% ({label})", sol_css)}'


def _char_disulfide_metric(m: dict, _metric) -> str:
    """Render disulfide bond potential."""
    dp = m.get("disulfide_potential", {})
    if not dp:
        return ""
    pairs = dp.get("possible_pairs", 0)
    unpaired = dp.get("unpaired_cys", 0)
    text = f"{pairs} possible pair(s)"
    if unpaired > 0:
        text += f", {unpaired} free"
    css = "warn" if unpaired > 0 else ""
    return f'      {_metric("Disulfide Bonds", text, css)}'


def _char_ph_metric(m: dict, _metric) -> str:
    """Render pH stability window with sparkline."""
    ph_range = m.get("ph_stable_range", [])
    ph_curve = m.get("ph_curve", [])
    if not ph_range:
        return ""
    sparkline = _ph_sparkline(ph_curve, ph_range)
    return (f'      {_metric("Stable pH Range", f"{ph_range[0]:.1f} – {ph_range[1]:.1f}", "")}'
            f'\n      <div class="metric-row"><span class="label">Charge vs pH</span>{sparkline}</div>')


def _char_colloidal_metric(m: dict, _metric) -> str:
    """Render colloidal stability metrics."""
    sigma = m.get("charge_symmetry", 0)
    cf = m.get("charged_fraction", 0)
    if not sigma and not cf:
        return ""
    sigma_css = "good" if sigma < 0.05 else "warn" if sigma < 0.15 else "bad"
    return (f'      {_metric("Charge Symmetry (σ)", f"{sigma:.3f}", sigma_css)}'
            f'\n      {_metric("Charged Fraction", f"{cf * 100:.1f}%", "")}')


def _char_rare_codon_metric(m: dict, _metric) -> str:
    """Render enhanced rare codon info."""
    details = m.get("rare_codon_details", "")
    frac = m.get("rare_codon_fraction", 0)
    if not details:
        return ""
    rc_css = "good" if frac < 0.05 else "warn" if frac < 0.15 else "bad"
    return f'      {_metric("Rare Codon Load", f"{frac * 100:.1f}%", rc_css)}'


def _char_sequence_features_group(m: dict, _metric) -> str:
    """Render Sequence Features metric group (signal peptides, IDRs)."""
    lines: list[str] = []

    # Signal peptide
    sp = m.get("signal_peptide", {})
    if sp:
        if sp.get("detected"):
            sp_text = f"Residues 1–{sp.get('cleavage_position', '?')}"
            sp_type = sp.get("type", "")
            if sp_type:
                sp_text += f" ({sp_type})"
            lines.append(_metric("Signal Peptide", sp_text, "warn"))
        else:
            lines.append(_metric("Signal Peptide", "None detected", "good"))

    # Intrinsically disordered regions
    disorder_frac = m.get("disorder_fraction", 0)
    disorder_regions = m.get("disorder_regions", [])
    disorder_scores = m.get("disorder_scores", [])
    if disorder_scores or disorder_frac:
        n_idr = len(disorder_regions)
        idr_residues = sum(r.get("length", 0) for r in disorder_regions)
        idr_css = "good" if disorder_frac < 0.1 else "warn" if disorder_frac < 0.3 else "bad"
        idr_text = f"{n_idr} IDR(s), {idr_residues} res ({disorder_frac * 100:.0f}%)"
        lines.append(_metric("Disordered Regions", idr_text, idr_css))
        # Disorder sparkline
        if disorder_scores:
            sparkline = _disorder_sparkline(disorder_scores)
            lines.append(f'<div class="metric-row"><span class="label">Disorder Profile</span>{sparkline}</div>')

    # Extinction coefficient
    ec = m.get("extinction_coefficient", {})
    if ec:
        ec_val = ec.get("reduced", 0)
        lines.append(_metric("ε₂₈₀ (reduced)", f"{ec_val:,} M⁻¹cm⁻¹", ""))

    # Aromaticity
    arom = m.get("aromaticity", 0)
    if arom:
        lines.append(_metric("Aromaticity", f"{arom:.3f}", ""))

    if not lines:
        return ""

    inner = "\n      ".join(lines)
    return f'''    <div class="metric-group">
      <h3>Sequence Features</h3>
      {inner}
    </div>'''


def _char_fold_architecture_group(m: dict, _metric) -> str:
    """Render Fold & Architecture metric group."""
    lines: list[str] = []

    # Domain architecture
    domains = m.get("domains", [])
    domain_count = m.get("domain_count", 0)
    if domains or domain_count:
        if domain_count <= 1:
            dom_text = "Single domain"
        else:
            boundaries = ", ".join(f"{d.get('start', '?')}–{d.get('end', '?')}" for d in domains[:4])
            dom_text = f"{domain_count} detected ({boundaries})"
        lines.append(_metric("Domains", dom_text, ""))

    # Oligomeric state
    oligo = m.get("oligomeric_state", "")
    if oligo:
        oligo_css = "" if "monomer" in oligo.lower() else "warn"
        lines.append(_metric("Oligomeric State", oligo, oligo_css))

    # Cofactor motifs
    cof_summary = m.get("cofactor_summary", "")
    if cof_summary:
        cof_css = "" if cof_summary == "None detected" else "warn"
        lines.append(_metric("Cofactor Motifs", cof_summary, cof_css))

    if not lines:
        return ""

    inner = "\n      ".join(lines)
    return f'''    <div class="metric-group">
      <h3>Fold &amp; Architecture</h3>
      {inner}
    </div>'''


def _hero_section(
    parent: ProteinCandidate | None,
    metrics: dict[str, Any],
    pdb_text: str,
    e1_mutations: list[dict[str, Any]],
    variants_ranked: list[ProteinCandidate],
) -> str:
    """Build the hero area: 3D viewer + metrics sidebar."""

    name = parent.name if parent else "Unknown"
    length = metrics.get("length", 0)

    # E1 top-10 mutation highlight data for JS
    _hl_colors = [
        "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#bc8cff",
        "#ff9ff3", "#54a0ff", "#f368e0", "#01a3a4", "#fd79a8",
    ]
    hl_positions_js = json.dumps([d["pos"] for d in e1_mutations])
    hl_labels_js = json.dumps([d["label"] for d in e1_mutations])
    hl_colors_js = json.dumps(_hl_colors[:max(len(e1_mutations), 1)])

    # Escape PDB for JS embedding
    pdb_escaped = pdb_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    # Metrics sidebar
    m = metrics

    def _metric(label: str, value: str, css: str = "") -> str:
        cls = f' class="value {css}"' if css else ' class="value"'
        return f'<div class="metric-row"><span class="label">{label}</span><span{cls}>{value}</span></div>'

    # pLDDT assessment
    plddt_mean = m.get("plddt_mean", 0)
    plddt_css = "good" if plddt_mean >= 80 else "warn" if plddt_mean >= 60 else "bad"

    # Instability assessment
    ii = m.get("instability_index", 0)
    ii_css = "good" if ii < 40 else "bad"

    # Aliphatic index assessment
    ai = m.get("aliphatic_index", 0)
    ai_css = "good" if ai >= 65 else "warn" if ai >= 45 else "bad"

    # Aggregation
    agg = m.get("agg_score", 0)
    agg_css = "good" if agg < 0.1 else "warn" if agg < 0.25 else "bad"

    # GRAVY
    gravy = m.get("gravy", 0)
    gravy_css = "good" if -0.5 < gravy < 0.0 else "warn" if -1.0 < gravy < 0.5 else "bad"

    # Charge
    charge = m.get("charge_7_4", 0)
    charge_css = "good" if -10 < charge < 10 else "warn"

    legend_dots = []
    for i, d in enumerate(e1_mutations):
        legend_dots.append(
            f'<span><span class="dot" style="background:{_hl_colors[i % len(_hl_colors)]}"></span>'
            f'{d["label"]}</span>'
        )

    return f"""
<div class="topbar">
  <h1>{_html.escape(name)}</h1>
  <span class="meta">{length} residues &middot; {m.get("mw", 0):.1f} Da &middot; Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
</div>

<div class="hero">
  <!-- 3D Viewer -->
  <div class="viewer-wrap">
    <span class="viewer-label">3Dmol.js &middot; Boltz-2 predicted structure &middot; pLDDT coloring</span>
    <div id="viewport"></div>
    <div class="viewer-legend" id="view-legend"></div>
    <div class="viewer-toolbar">
      <button class="view-btn mode active" data-mode="plddt" onclick="setView('plddt')">Confidence</button>
      <button class="view-btn mode" data-mode="hydrophobicity" onclick="setView('hydrophobicity')">Hydrophobicity</button>
      <button class="view-btn mode" data-mode="charge" onclick="setView('charge')">Charge</button>
      <button class="view-btn mode" data-mode="ss" onclick="setView('ss')">2&deg; Structure</button>
      <div class="toolbar-sep"></div>
      <button class="view-btn" id="btn-surface" onclick="toggleSurface()">Surface</button>
      <button class="view-btn" id="btn-mutations" onclick="toggleMutations()">Mutations</button>
    </div>
  </div>

  <!-- Metrics -->
  <div class="metrics-panel">
    <h2>Protein Metrics</h2>

    <div class="metric-group">
      <h3>Structure Quality</h3>
      {_metric("Mean pLDDT", f"{plddt_mean:.1f}", plddt_css)}
      {_metric("Min pLDDT", f"{m.get('plddt_min', 0):.1f}", plddt_css)}
    </div>

    <div class="metric-group">
      <h3>Thermal Stability</h3>
      {_metric("Instability Index", f"{ii:.1f} ({m.get('instability_label', '?')})", ii_css)}
      {_metric("Aliphatic Index", f"{ai:.1f}", ai_css)}
{_char_tm_metric(m, _metric)}
    </div>

    <div class="metric-group">
      <h3>Aggregation &amp; Solubility</h3>
      {_metric("Agg. Score", f"{agg:.3f}", agg_css)}
      {_metric("Hydrophobic Patches", f"{m.get('agg_patches', 0)}", agg_css)}
      {_metric("Surface Patches (SAP)", f"{m.get('n_surface_patches', 0)}", "")}
{_char_aggregation_metric(m, _metric)}
{_char_solubility_metric(m, _metric)}
    </div>

    <div class="metric-group">
      <h3>Physicochemical</h3>
      {_metric("GRAVY", f"{gravy:.3f}", gravy_css)}
      {_metric("Net Charge (pH 7.4)", f"{charge:+.1f}", charge_css)}
      {_metric("Isoelectric Point", f"{m.get('pI', 0):.2f}", "")}
      {_metric("Cysteines", f"{m.get('cys_count', 0)}", "")}
{_char_disulfide_metric(m, _metric)}
{_char_ph_metric(m, _metric)}
{_char_colloidal_metric(m, _metric)}
    </div>

    <div class="metric-group">
      <h3>Expression (E. coli)</h3>
      {_metric("Rare-codon AAs (W/C/M)", f"{m.get('rare_codons', 0)}", "")}
      {_metric("Prolines", f"{m.get('proline_count', 0)}", "")}
{_char_rare_codon_metric(m, _metric)}
    </div>

{_char_sequence_features_group(m, _metric)}
{_char_fold_architecture_group(m, _metric)}

    <div class="legend">
      <strong style="color:var(--fg)">Highlighted:</strong>
      {''.join(legend_dots)}
    </div>
  </div>
</div>

<script>
(function() {{
  var pdbData = `{pdb_escaped}`;
  var viewer = $3Dmol.createViewer("viewport", {{backgroundColor:"#000"}});
  viewer.addModel(pdbData, "pdb");

  var hlPos = {hl_positions_js};
  var hlLbl = {hl_labels_js};
  var hlClr = {hl_colors_js};
  var surfOn = false, mutOn = false, curView = "plddt";

  var aa3 = {{ALA:"A",ARG:"R",ASN:"N",ASP:"D",CYS:"C",GLU:"E",GLN:"Q",
    GLY:"G",HIS:"H",ILE:"I",LEU:"L",LYS:"K",MET:"M",PHE:"F",
    PRO:"P",SER:"S",THR:"T",TRP:"W",TYR:"Y",VAL:"V"}};
  var hyd = {{A:1.8,R:-4.5,N:-3.5,D:-3.5,C:2.5,E:-3.5,Q:-3.5,G:-0.4,
    H:-3.2,I:4.5,L:3.8,K:-3.9,M:1.9,F:2.8,P:-1.6,S:-0.8,
    T:-0.7,W:-0.9,Y:-1.3,V:4.2}};
  var chg = {{D:-1,E:-1,K:1,R:1,H:0.5}};

  function hex(r,g,b) {{
    return "#"+[r,g,b].map(function(v){{
      return Math.round(v*255).toString(16).padStart(2,"0");
    }}).join("");
  }}

  function plddtC(a) {{
    var b=a.b;
    if(b>=90) return "#3fb950"; if(b>=70) return "#58a6ff";
    if(b>=50) return "#d29922"; return "#f85149";
  }}

  function hydroC(a) {{
    var c=aa3[a.resn]||"G", h=hyd[c]||0, t=(h+4.5)/9;
    if(t<0.5) {{ var s=t*2; return hex(0.2+0.8*s, 0.4+0.6*s, 1); }}
    else {{ var s=(t-0.5)*2; return hex(1, 1-0.6*s, 1-0.9*s); }}
  }}

  function chargeC(a) {{
    var c=aa3[a.resn]||"G", q=chg[c]||0;
    if(q>0) return hex(0.3, 0.5, 1);
    if(q<0) return hex(1, 0.3, 0.3);
    return "#aaaaaa";
  }}

  function ssC(a) {{
    if(a.ss==="h") return "#ff6b9d";
    if(a.ss==="s") return "#ffd93d";
    return "#8b949e";
  }}

  var cfMap = {{plddt:plddtC, hydrophobicity:hydroC, charge:chargeC, ss:ssC}};

  function applyHL() {{
    viewer.removeAllLabels();
    if(!mutOn) return;
    for(var i=0; i<hlPos.length; i++) {{
      var p=hlPos[i], cl=hlClr[i % hlClr.length];
      viewer.addStyle({{resi:p}}, {{stick:{{radius:0.18, color:cl}}}});
      var atoms = viewer.selectedAtoms({{resi:p, atom:"CA"}});
      if(atoms.length)
        viewer.addLabel(hlLbl[i]||String(p), {{
          position:atoms[0], backgroundColor:cl, backgroundOpacity:0.85,
          fontColor:"#000", fontSize:12, showBackground:true
        }});
    }}
  }}

  function updSurf() {{
    viewer.removeAllSurfaces();
    if(!surfOn) return;
    viewer.addSurface($3Dmol.SurfaceType.VDW,
      {{opacity:0.82, colorfunc:cfMap[curView]}});
  }}

  function updLegend(m) {{
    var el = document.getElementById("view-legend");
    if(m==="plddt") el.innerHTML =
      '<span style="color:#3fb950">&#9679;</span>&thinsp;&#8805;90 '+
      '<span style="color:#58a6ff">&#9679;</span>&thinsp;70&#8211;90 '+
      '<span style="color:#d29922">&#9679;</span>&thinsp;50&#8211;70 '+
      '<span style="color:#f85149">&#9679;</span>&thinsp;&lt;50';
    else if(m==="hydrophobicity") el.innerHTML =
      '<span style="display:inline-block;width:90px;height:10px;border-radius:3px;'+
      'background:linear-gradient(to right,#3366ff,#ffffff,#ff6600);'+
      'vertical-align:middle"></span> Hydrophilic &#8594; Hydrophobic';
    else if(m==="charge") el.innerHTML =
      '<span style="color:#4d80ff">&#9679;</span> Positive (K,R,H) '+
      '<span style="color:#aaaaaa">&#9679;</span> Neutral '+
      '<span style="color:#ff4d4d">&#9679;</span> Negative (D,E)';
    else if(m==="ss") el.innerHTML =
      '<span style="color:#ff6b9d">&#9679;</span> &#945;-Helix '+
      '<span style="color:#ffd93d">&#9679;</span> &#946;-Sheet '+
      '<span style="color:#8b949e">&#9679;</span> Coil/Loop';
  }}

  window.setView = function(m) {{
    curView = m;
    viewer.setStyle({{}}, {{cartoon:{{colorfunc:cfMap[m]}}}});
    applyHL();
    if(surfOn) updSurf();
    viewer.render();
    document.querySelectorAll(".view-btn.mode").forEach(function(b){{
      b.classList.toggle("active", b.dataset.mode===m);
    }});
    var lbl = {{plddt:"pLDDT coloring", hydrophobicity:"Hydrophobicity (Kyte-Doolittle)",
      charge:"Charge distribution", ss:"Secondary structure"}};
    document.querySelector(".viewer-label").textContent =
      "3Dmol.js \\u00b7 Boltz-2 \\u00b7 " + lbl[m];
    updLegend(m);
  }};

  window.toggleSurface = function() {{
    surfOn = !surfOn; updSurf(); viewer.render();
    document.getElementById("btn-surface").classList.toggle("active", surfOn);
  }};

  window.toggleMutations = function() {{
    mutOn = !mutOn; window.setView(curView);
    document.getElementById("btn-mutations").classList.toggle("active", mutOn);
  }};

  // Initial render
  viewer.setStyle({{}}, {{cartoon:{{colorfunc:plddtC}}}});
  applyHL();
  viewer.zoomTo();
  viewer.render();
  viewer.zoom(1.1);
  updLegend("plddt");
}})();
</script>
"""


def _executive_summary_section(
    results: dict[str, StepResult],
    variants_ranked: list[ProteinCandidate],
    metrics: dict[str, Any],
) -> str:
    """Build the executive summary bar with key stat cards."""
    total_candidates = len(variants_ranked)
    n_stages = len([s for s in results if s != "input"])

    # E1 fitness range — pull from e1_score step directly, fall back to variants_ranked
    e1_result = results.get("e1_score")
    if e1_result:
        e1_scores = [c.scores["e1_fitness"] for c in e1_result.candidates if "e1_fitness" in c.scores]
    else:
        e1_scores = [c.scores["e1_fitness"] for c in variants_ranked if "e1_fitness" in c.scores]

    if e1_scores:
        e1_min, e1_max = min(e1_scores), max(e1_scores)
        e1_range = f"{e1_min:+.3f} → {e1_max:+.3f}"
        e1_sub = f"{sum(1 for s in e1_scores if s > 0)} predicted beneficial"
    else:
        e1_range = "N/A"
        e1_sub = "No E1 scores available"

    # Top pick — best E1 fitness variant
    e1_variants = [c for c in variants_ranked if "e1_fitness" in c.scores]
    if e1_variants:
        top = max(e1_variants, key=lambda c: c.scores["e1_fitness"])
    elif variants_ranked:
        top = variants_ranked[0]
    else:
        top = None

    if top:
        top_name = top.name
        top_score = top.scores.get("e1_fitness", top.scores.get("composite_score", 0))
        top_muts = ", ".join(m.label for m in top.mutations) if top.mutations else "—"
        top_sub = _html.escape(top_muts[:80] + ("…" if len(top_muts) > 80 else ""))
        score_label = "E1" if "e1_fitness" in top.scores else "score"
        top_score_str = f"{top_score:+.4f}"
        top_css = "good-card" if top_score > 0 else ""
    else:
        top_name = "—"
        top_score_str = "N/A"
        top_sub = "No candidates"
        score_label = "score"
        top_css = ""

    protein_len = metrics.get("length", 0)

    return f"""
<div class="section">
  <h2>Executive Summary</h2>
  <div class="exec-summary">
    <div class="exec-card">
      <div class="ec-label">Total Candidates</div>
      <div class="ec-value">{total_candidates}</div>
      <div class="ec-sub">variants generated across all stages</div>
    </div>
    <div class="exec-card">
      <div class="ec-label">E1 Fitness Range</div>
      <div class="ec-value" style="font-size:1.1rem;padding-top:0.4rem">{_html.escape(e1_range)}</div>
      <div class="ec-sub">{_html.escape(e1_sub)}</div>
    </div>
    <div class="exec-card {_html.escape(top_css)}">
      <div class="ec-label">Top Pick</div>
      <div class="ec-value" title="{_html.escape(top_name)}" style="font-size:1.1rem;padding-top:0.3rem">{_html.escape(top_name)}</div>
      <div class="ec-sub" title="{_html.escape(top_muts)}">{score_label} {top_score_str} &middot; {top_sub}</div>
    </div>
    <div class="exec-card">
      <div class="ec-label">Pipeline Stages</div>
      <div class="ec-value">{n_stages}</div>
      <div class="ec-sub">steps completed &middot; {protein_len} residue protein</div>
    </div>
  </div>
</div>
"""


# ── Step-by-step Analysis ───────────────────────────────────────────

_TIER_SHORT_DESC: dict[int, str] = {
    1: "Quick sequence-level scans flagging chemical liabilities.",
    2: "Evolutionary conservation from homologue alignments.",
    3: "Structure-aware engineering using the predicted 3D model.",
    4: "Generative protein design, validation, and combinatorial assembly.",
    5: "Zero-shot fitness evaluation with protein language models.",
}


# PLM scoring steps are pure annotators — omit their stage cards from the
# Step-by-Step Analysis (scores still appear in all candidate tables).
_PLM_STEPS = {"esm1v_score", "esmif1_score", "e1_score"}


def _step_details_section(results: dict[str, StepResult]) -> str:
    """Build the full step-by-step analysis with per-step collapsible cards."""
    if not results:
        return ""

    # Group steps by tier, skipping PLM-only scoring steps
    tier_steps: dict[int, list[str]] = {}
    for step_name in results:
        if step_name in _PLM_STEPS:
            continue
        tier = _TIER_MAP.get(step_name, 0)
        tier_steps.setdefault(tier, []).append(step_name)

    parts: list[str] = [
        '<div class="section">',
        '<h2>Step-by-Step Analysis</h2>',
        '<p class="subtitle">Detailed results from each optimization step. '
        'Click to expand individual steps.</p>',
    ]

    for tier_num in sorted(tier_steps):
        tier_label = _TIER_NAMES.get(tier_num, f"Tier {tier_num}")
        tier_desc = _TIER_SHORT_DESC.get(tier_num, "")
        steps = tier_steps[tier_num]

        parts.append(f"""
<div class="tier-divider">
  <span class="td-badge">STAGE {tier_num}</span>
  <span class="td-title">{_html.escape(tier_label)}</span>
  <span class="td-desc">{_html.escape(tier_desc)}</span>
</div>""")

        for step_name in steps:
            result = results[step_name]
            parts.append(_render_step_card(step_name, result))

    parts.append('</div><!-- /section -->')
    return "\n".join(parts)


def _render_step_card(step_name: str, result: StepResult) -> str:
    """Render a single step as a collapsible <details> card."""
    n_variants = len([c for c in result.candidates if c.parent_id is not None])
    has_warnings = len(result.warnings) > 0

    step_label = _step_display_name(step_name)

    # Badge
    if has_warnings and n_variants == 0:
        badge_css = "sd-fail"
        badge_text = "\u2718 failed"
    elif has_warnings:
        badge_css = "sd-warn"
        badge_text = f"{n_variants} variants \u00b7 \u26a0 warnings"
    elif n_variants > 0:
        badge_css = "sd-ok"
        badge_text = f"{n_variants} variants"
    else:
        badge_css = "sd-ok"
        badge_text = "\u2714 done"

    # Custom badge text overrides
    if step_name == "find_homologs":
        parent = next((c for c in result.candidates if c.parent_id is None), None)
        n_hom = parent.metadata.get("n_homologs", 0) if parent else 0
        badge_text = f"{n_hom} homologs"
    elif step_name == "predict_structure":
        parent = next((c for c in result.candidates if c.parent_id is None), None)
        plddt = parent.scores.get("plddt", 0) if parent else 0
        badge_text = f"pLDDT {plddt:.1f}"
    elif step_name == "sequence_complexity":
        parent = next((c for c in result.candidates if c.parent_id is None), None)
        n_issues = int(parent.scores.get("complexity_issues", 0)) if parent else 0
        badge_text = f"{n_issues} issues"
        badge_css = "sd-ok" if n_issues == 0 else "sd-warn"

    # Default open for producing steps with moderate data
    open_attr = ""
    if step_name in ("cysteine_scan", "consensus_design", "pssm_analysis",
                      "stability_ddg", "disulfide_design", "cavity_fill",
                      "surface_patch", "combine_variants", "e1_score"):
        open_attr = " open" if n_variants > 0 else ""

    # Render inner content
    inner = _render_step_inner(step_name, result)

    return f"""
<details class="sd-card"{open_attr}>
  <summary>
    {_html.escape(step_label)}
    <span class="sd-badge {badge_css}">{badge_text}</span>
  </summary>
  <div class="sd-inner">
    {inner}
  </div>
</details>"""


def _render_step_inner(step_name: str, result: StepResult) -> str:
    """Dispatch to per-step renderer."""
    renderer = _STEP_RENDERERS.get(step_name, _render_generic_step)
    return renderer(result)


# ── Per-step renderers ──────────────────────────────────────────────

def _render_cysteine_scan(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    if not variants:
        return '<p class="sd-summary">No cysteines requiring attention.</p>'

    single = [v for v in variants if len(v.mutations) == 1]
    combo = [v for v in variants if len(v.mutations) > 1]

    rows: list[str] = []
    for v in single:
        m = v.mutations[0]
        ctx = m.metadata.get("context", "") if m.metadata else ""
        risk = v.scores.get("cys_risk", m.score or 0)
        risk_css = "bad" if risk >= 0.7 else "warn" if risk >= 0.4 else "good"
        rows.append(
            f'<tr><td>{m.position}</td>'
            f'<td class="mono">{_html.escape(ctx or f"...{m.wt}...")}</td>'
            f'<td class="{risk_css}" style="font-weight:600">{risk:.2f}</td>'
            f'<td class="mono">{m.label}</td></tr>'
        )

    combo_note = ""
    if combo:
        c = combo[0]
        muts = ", ".join(m.label for m in c.mutations)
        risk = c.scores.get("cys_risk", 0)
        combo_note = (
            f'<div class="sd-chips"><div class="sd-chip">'
            f'<strong>Remove-all variant:</strong> {_html.escape(muts)} '
            f'(combined risk {risk:.2f})</div></div>'
        )

    return f"""
<p class="sd-summary">{len(single)} unpaired cysteine{"s" if len(single) != 1 else ""} detected.
Free cysteines can cause unwanted disulfide bonds or oxidation.</p>
{combo_note}
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>Pos</th><th>Context</th><th>Risk</th><th>Fix</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_motif_scan(result: StepResult) -> str:
    parent = next((c for c in result.candidates if c.parent_id is None), None)
    variants = [c for c in result.candidates if c.parent_id is not None]

    if not parent:
        return '<p class="sd-summary">No motif data available.</p>'

    hits = parent.metadata.get("motif_hits", [])
    if not hits:
        return '<p class="sd-summary good">No sequence liabilities detected.</p>'

    # Build fix map from variants
    fix_map: dict[int, str] = {}
    for v in variants:
        for m in v.mutations:
            fix_map[m.position] = m.label

    cat_css = {
        "deamidation": "cat-deamidation", "oxidation": "cat-oxidation",
        "proteolysis": "cat-proteolysis", "aggregation": "cat-aggregation",
    }

    rows: list[str] = []
    for hit in sorted(hits, key=lambda h: -h.get("risk", 0)):
        pos = hit.get("position", 0)
        cat = hit.get("category", "unknown")
        css = cat_css.get(cat, "cat-default")
        risk = hit.get("risk", 0)
        risk_css = "bad" if risk >= 0.65 else "warn" if risk >= 0.4 else "good"
        fix = fix_map.get(pos, fix_map.get(pos + 1, "\u2014"))

        rows.append(
            f'<tr><td>{pos}</td>'
            f'<td>{_html.escape(hit.get("residues", "?"))}</td>'
            f'<td><span class="cat-badge {css}">{_html.escape(cat.title())}</span></td>'
            f'<td>{_html.escape(hit.get("pattern", ""))}</td>'
            f'<td class="{risk_css}" style="font-weight:600">{risk:.2f}</td>'
            f'<td style="font-size:0.82rem">{_html.escape(hit.get("rationale", ""))}</td>'
            f'<td class="mono">{_html.escape(fix)}</td></tr>'
        )

    n_by_cat: dict[str, int] = {}
    for h in hits:
        cat = h.get("category", "unknown")
        n_by_cat[cat] = n_by_cat.get(cat, 0) + 1
    chips = "".join(
        f'<div class="sd-chip"><strong>{n}</strong> {_html.escape(cat)}</div>'
        for cat, n in sorted(n_by_cat.items(), key=lambda x: -x[1])
    )

    return f"""
<p class="sd-summary">{len(hits)} sequence liabilities detected across {len(n_by_cat)} categories.</p>
<div class="sd-chips">{chips}</div>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>Pos</th><th>Residues</th><th>Category</th><th>Pattern</th>
<th>Risk</th><th>Rationale</th><th>Fix</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_sequence_complexity(result: StepResult) -> str:
    parent = next((c for c in result.candidates if c.parent_id is None), None)
    if not parent:
        return '<p class="sd-summary">No complexity data.</p>'

    flags = parent.metadata.get("complexity_flags", [])
    n_issues = int(parent.scores.get("complexity_issues", 0))
    risk_total = parent.scores.get("complexity_risk_total", 0)

    if not flags:
        return (
            '<p class="sd-summary good">'
            '\u2714 No sequence complexity issues detected.</p>'
        )

    cat_css_map = {
        "homopolymer": "cat-homopolymer", "proline_run": "cat-proline",
        "proline_stall": "cat-proline", "charge_cluster": "cat-charge",
        "low_complexity": "cat-low-complexity",
    }

    rows: list[str] = []
    for f in sorted(flags, key=lambda x: -x.get("risk", 0)):
        cat = f.get("category", "")
        css = cat_css_map.get(cat, "cat-default")
        risk = f.get("risk", 0)
        risk_css = "bad" if risk >= 0.5 else "warn" if risk >= 0.2 else ""
        rows.append(
            f'<tr><td>{f.get("position", "?")}</td>'
            f'<td><span class="cat-badge {css}">{_html.escape(cat.replace("_", " ").title())}</span></td>'
            f'<td>{f.get("length", "")}</td>'
            f'<td class="mono">{_html.escape(f.get("residues", ""))}</td>'
            f'<td class="{risk_css}" style="font-weight:600">{risk:.2f}</td>'
            f'<td style="font-size:0.82rem">{_html.escape(f.get("description", ""))}</td></tr>'
        )

    return f"""
<p class="sd-summary">{n_issues} complexity flag{"s" if n_issues != 1 else ""} (total risk {risk_total:.2f}).</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>Pos</th><th>Category</th><th>Length</th><th>Residues</th>
<th>Risk</th><th>Description</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_find_homologs(result: StepResult) -> str:
    parent = next((c for c in result.candidates if c.parent_id is None), None)
    if not parent:
        return '<p class="sd-summary">No homolog data.</p>'

    meta = parent.metadata
    n_hom = meta.get("n_homologs", meta.get("num_homologs", 0))
    method = meta.get("search_method", "unknown")
    homolog_seqs = meta.get("homolog_sequences", [])
    avg_len = sum(len(s) for s in homolog_seqs) / len(homolog_seqs) if homolog_seqs else 0
    msa_path = meta.get("msa_fasta", "")

    chips = [
        f'<div class="sd-chip"><strong>{n_hom}</strong> homologs found</div>',
        f'<div class="sd-chip">Method: <strong>{_html.escape(str(method))}</strong></div>',
    ]
    if avg_len > 0:
        chips.append(f'<div class="sd-chip">Avg length: <strong>{avg_len:.0f}</strong> aa</div>')
    if msa_path:
        chips.append(f'<div class="sd-chip">MSA generated \u2714</div>')

    return f"""
<p class="sd-summary">Homolog search provides evolutionary context for conservation-based design.
This step does not produce mutations directly.</p>
<div class="sd-chips">{''.join(chips)}</div>"""


def _render_consensus_design(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    if not variants:
        return '<p class="sd-summary">No consensus mutations found.</p>'

    single = [v for v in variants if len(v.mutations) == 1]
    combo = [v for v in variants if len(v.mutations) > 1]

    single.sort(
        key=lambda v: v.scores.get("consensus_conservation", 0), reverse=True
    )

    rows: list[str] = []
    for v in single:
        m = v.mutations[0]
        cons = v.scores.get("consensus_conservation", 0)
        wt_freq = m.metadata.get("wt_frequency", 0) if m.metadata else 0
        depth = m.metadata.get("column_depth", 0) if m.metadata else 0
        cons_css = "good" if cons >= 0.7 else "" if cons >= 0.4 else "muted"
        rows.append(
            f'<tr><td>{m.position}</td>'
            f'<td class="mono">{m.wt} \u2192 {m.mut}</td>'
            f'<td class="{cons_css}" style="font-weight:600">{cons:.3f}</td>'
            f'<td>{wt_freq:.3f}</td>'
            f'<td>{depth}</td></tr>'
        )

    combo_note = ""
    if combo:
        c = combo[0]
        muts = ", ".join(m.label for m in c.mutations)
        score = c.scores.get("consensus_conservation", 0)
        combo_note = (
            f'<div class="sd-chips"><div class="sd-chip">'
            f'<strong>Top-{len(c.mutations)} combo:</strong> '
            f'{_html.escape(muts)} (sum conservation {score:.3f})</div></div>'
        )

    scores = [v.scores.get("consensus_conservation", 0) for v in single]
    range_str = f"{min(scores):.3f} \u2013 {max(scores):.3f}" if scores else "N/A"

    return f"""
<p class="sd-summary">{len(single)} consensus mutations (conservation {range_str}).
Positions where the consensus sequence differs from the wild-type, sorted by conservation frequency.</p>
{combo_note}
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>Pos</th><th>Mutation</th><th>Conservation</th>
<th>WT Frequency</th><th>Depth</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_pssm_analysis(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    if not variants:
        return '<p class="sd-summary">No PSSM-beneficial mutations found.</p>'

    variants.sort(
        key=lambda v: v.scores.get("pssm_log_odds", 0), reverse=True
    )

    rows: list[str] = []
    for v in variants:
        m = v.mutations[0] if v.mutations else None
        if not m:
            continue
        lo = v.scores.get("pssm_log_odds", 0)
        wt_score = m.metadata.get("wt_pssm_score", 0) if m.metadata else 0
        mut_score = m.metadata.get("mut_pssm_score", 0) if m.metadata else 0
        lo_css = "good" if lo >= 3 else "" if lo >= 1.5 else "muted"
        rows.append(
            f'<tr><td>{m.position}</td>'
            f'<td class="mono">{m.wt} \u2192 {m.mut}</td>'
            f'<td class="{lo_css}" style="font-weight:600">{lo:.3f}</td>'
            f'<td>{wt_score:.3f}</td>'
            f'<td>{mut_score:.3f}</td></tr>'
        )

    scores = [v.scores.get("pssm_log_odds", 0) for v in variants]
    range_str = f"{min(scores):.2f} \u2013 {max(scores):.2f}" if scores else "N/A"

    return f"""
<p class="sd-summary">{len(variants)} PSSM-positive mutations
(log-odds {range_str}). Higher values indicate stronger evolutionary support.</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>Pos</th><th>Mutation</th><th>Log-odds</th>
<th>WT Score</th><th>Mut Score</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_predict_structure(result: StepResult) -> str:
    parent = next((c for c in result.candidates if c.parent_id is None), None)
    if not parent:
        return '<p class="sd-summary">No structure prediction data.</p>'

    plddt = parent.scores.get("plddt", 0)
    method = parent.metadata.get("method", "boltz2")
    pdb_path = parent.metadata.get("structure_path", "")
    plddt_css = "good" if plddt >= 80 else "warn" if plddt >= 60 else "bad"

    chips = [
        f'<div class="sd-chip">Method: <strong>{_html.escape(str(method))}</strong></div>',
        f'<div class="sd-chip">Mean pLDDT: <strong class="{plddt_css}">{plddt:.1f}</strong></div>',
    ]
    if pdb_path:
        chips.append('<div class="sd-chip">Structure generated \u2714</div>')

    return f"""
<p class="sd-summary">3D structure predicted using {_html.escape(str(method))}. This structure
is used by downstream structure-based steps (stability, disulfide, cavity, surface).
No mutations are generated at this step.</p>
<div class="sd-chips">{''.join(chips)}</div>"""


def _render_stability_ddg(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    if not variants:
        if result.warnings:
            return f'<p class="sd-summary warn">\u26a0 {_html.escape(result.warnings[0])}</p>'
        return '<p class="sd-summary">No stabilizing mutations found.</p>'

    # Sort by ddg (most negative = most stabilizing first)
    variants.sort(key=lambda v: v.scores.get("ddg", 0))

    show_n = 15
    rows: list[str] = []
    for i, v in enumerate(variants):
        m = v.mutations[0] if v.mutations else None
        if not m:
            continue
        ddg = v.scores.get("ddg", 0)
        ddg_css = "good" if ddg < -0.7 else "" if ddg < -0.3 else "warn"
        extra = f' class="extra-row sd-ddg-extra"' if i >= show_n else ""
        rows.append(
            f'<tr{extra}><td>{i + 1}</td>'
            f'<td>{m.position}</td>'
            f'<td class="mono">{m.wt} \u2192 {m.mut}</td>'
            f'<td class="{ddg_css}" style="font-weight:600">{ddg:+.4f}</td></tr>'
        )

    ddgs = [v.scores.get("ddg", 0) for v in variants if "ddg" in v.scores]
    expand_btn = ""
    if len(variants) > show_n:
        expand_btn = (
            f'<button class="sd-expand" onclick="'
            f"document.querySelectorAll('.sd-ddg-extra').forEach(r=>r.classList.toggle('show'));"
            f"this.textContent=this.textContent.includes('Show')?'Hide':'Show {len(variants) - show_n} more'"
            f'">Show {len(variants) - show_n} more</button>'
        )

    return f"""
<p class="sd-summary">{len(variants)} stabilizing mutations (\u0394\u0394G &lt; threshold).
Ranked by predicted \u0394\u0394G (most stabilizing first). Range: {min(ddgs):+.4f} to {max(ddgs):+.4f} kcal/mol.</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Pos</th><th>Mutation</th><th>\u0394\u0394G (kcal/mol)</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
{expand_btn}"""


def _render_disulfide_design(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    if not variants:
        return '<p class="sd-summary">No disulfide candidates found.</p>'

    variants.sort(key=lambda v: v.scores.get("disulfide_cb_distance", 99))

    rows: list[str] = []
    for i, v in enumerate(variants, 1):
        pair = v.metadata.get("disulfide_pair", [])
        pair_str = f"{pair[0]} \u2194 {pair[1]}" if len(pair) == 2 else "\u2014"
        cb_dist = v.scores.get("disulfide_cb_distance", 0)
        energy = v.scores.get("disulfide_energy_estimate", 0)
        muts = ", ".join(m.label for m in v.mutations) if v.mutations else "\u2014"
        dist_css = "good" if 3.5 <= cb_dist <= 4.5 else "warn"
        energy_css = "good" if energy < -1.5 else "" if energy < -1.0 else "warn"

        rows.append(
            f'<tr><td>{i}</td>'
            f'<td class="mono">{_html.escape(pair_str)}</td>'
            f'<td class="{dist_css}" style="font-weight:600">{cb_dist:.3f}</td>'
            f'<td class="{energy_css}">{energy:.3f}</td>'
            f'<td class="mono">{_html.escape(muts)}</td></tr>'
        )

    return f"""
<p class="sd-summary">{len(variants)} potential disulfide bonds identified.
Ranked by C\u03b2\u2013C\u03b2 distance (ideal 3.5\u20134.5 \u00c5).</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Pair</th><th>C\u03b2 Dist (\u00c5)</th>
<th>Energy (kcal/mol)</th><th>Mutations</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_cavity_fill(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    if not variants:
        return '<p class="sd-summary">No cavity-filling mutations found.</p>'

    variants.sort(
        key=lambda v: v.scores.get("cavity_burial", 0), reverse=True
    )

    rows: list[str] = []
    for i, v in enumerate(variants, 1):
        m = v.mutations[0] if v.mutations else None
        if not m:
            continue
        burial = v.scores.get("cavity_burial", 0)
        burial_css = "good" if burial >= 1.5 else "" if burial >= 1.2 else "muted"
        rows.append(
            f'<tr><td>{i}</td>'
            f'<td>{m.position}</td>'
            f'<td class="mono">{m.wt} \u2192 {m.mut}</td>'
            f'<td class="{burial_css}" style="font-weight:600">{burial:.3f}</td></tr>'
        )

    return f"""
<p class="sd-summary">{len(variants)} cavity-filling mutations.
Small-to-large hydrophobic substitutions that fill interior cavities, ranked by burial score.</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Pos</th><th>Mutation</th><th>Burial Score</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_surface_patch(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    parent = next((c for c in result.candidates if c.parent_id is None), None)

    if not variants:
        return '<p class="sd-summary">No surface redesign mutations proposed.</p>'

    n_patches = 0
    if parent and "hydrophobic_patches" in parent.metadata:
        n_patches = len(parent.metadata["hydrophobic_patches"])

    variants.sort(
        key=lambda v: v.scores.get("surface_sap", 0), reverse=True
    )

    rows: list[str] = []
    for i, v in enumerate(variants, 1):
        m = v.mutations[0] if v.mutations else None
        if not m:
            continue
        sap = v.scores.get("surface_sap", 0)
        patch_size = m.metadata.get("patch_size", "") if m.metadata else ""
        sap_css = "bad" if sap > 2 else "warn" if sap > 1 else "good"
        rows.append(
            f'<tr><td>{i}</td>'
            f'<td>{m.position}</td>'
            f'<td class="mono">{m.wt} \u2192 {m.mut}</td>'
            f'<td class="{sap_css}" style="font-weight:600">{sap:.3f}</td>'
            f'<td>{patch_size}</td></tr>'
        )

    return f"""
<p class="sd-summary">{n_patches} hydrophobic surface patch{"es" if n_patches != 1 else ""} found,
{len(variants)} mutations proposed to reduce aggregation propensity. Ranked by SAP score (higher = more aggregation-prone).</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Pos</th><th>Mutation</th><th>SAP Score</th>
<th>Patch Size</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_rfdiffusion(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]

    if result.warnings and not variants:
        warn_html = "".join(
            f'<p class="warn" style="font-size:0.85rem">\u26a0 {_html.escape(w)}</p>'
            for w in result.warnings
        )
        return f"""
<p class="sd-summary">RFdiffusion backbone diversification was attempted but produced no outputs.</p>
{warn_html}"""

    if not variants:
        return '<p class="sd-summary">No RFdiffusion outputs.</p>'

    rows: list[str] = []
    for i, v in enumerate(variants, 1):
        output_pdb = v.metadata.get("rfdiffusion_output", "\u2014")
        partial_t = v.metadata.get("partial_T", "")
        rows.append(
            f'<tr><td>{i}</td>'
            f'<td>{_html.escape(v.name)}</td>'
            f'<td>{partial_t}</td>'
            f'<td class="mono" style="font-size:0.78rem">{_html.escape(str(output_pdb))}</td></tr>'
        )

    return f"""
<p class="sd-summary">{len(variants)} backbone-diversified structures generated
via partial diffusion. These are passed to ProteinMPNN for sequence design.</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Name</th><th>Partial T</th><th>Output PDB</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_proteinmpnn(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]

    if result.warnings and not variants:
        warn_html = "".join(
            f'<p class="warn" style="font-size:0.85rem">\u26a0 {_html.escape(w)}</p>'
            for w in result.warnings
        )
        return f"""
<p class="sd-summary">ProteinMPNN design was attempted but failed.</p>
{warn_html}"""

    if not variants:
        return '<p class="sd-summary">No MPNN designs produced.</p>'

    variants.sort(key=lambda v: v.scores.get("mpnn_score", 999))

    rows: list[str] = []
    for i, v in enumerate(variants, 1):
        mpnn_score = v.scores.get("mpnn_score", 0)
        recovery = v.scores.get("mpnn_recovery", 0)
        n_muts = len(v.mutations)
        score_css = "good" if mpnn_score < 1.0 else "" if mpnn_score < 1.5 else "warn"
        rows.append(
            f'<tr><td>{i}</td>'
            f'<td>{_html.escape(v.name)}</td>'
            f'<td class="{score_css}" style="font-weight:600">{mpnn_score:.4f}</td>'
            f'<td>{recovery:.1%}</td>'
            f'<td>{n_muts}</td></tr>'
        )

    return f"""
<p class="sd-summary">{len(variants)} sequence designs from ProteinMPNN (SolubleMPNN).
Lower MPNN score = better fit to backbone. These are full-sequence redesigns.</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Design</th><th>MPNN Score</th>
<th>Recovery</th><th>Mutations</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_design_validate(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]

    if not variants:
        return (
            '<p class="sd-summary">No designs to validate '
            '(requires RFdiffusion backbone outputs).</p>'
        )

    has_val = any("val_plddt" in c.scores for c in variants)
    if not has_val:
        return (
            f'<p class="sd-summary">{len(variants)} designs passed through '
            f'(validation not triggered — no RFdiffusion inputs available).</p>'
        )

    variants.sort(key=lambda v: v.scores.get("val_plddt", 0), reverse=True)

    rows: list[str] = []
    for i, v in enumerate(variants, 1):
        plddt = v.scores.get("val_plddt", 0)
        rmsd = v.scores.get("val_rmsd", 0)
        passed = v.metadata.get("validation_passed", None)
        plddt_css = "good" if plddt >= 70 else "warn" if plddt >= 50 else "bad"
        rmsd_css = "good" if rmsd < 2 else "warn" if rmsd < 4 else "bad"
        pass_str = "\u2714" if passed else ("\u2718" if passed is False else "\u2014")
        rows.append(
            f'<tr><td>{i}</td><td>{_html.escape(v.name)}</td>'
            f'<td class="{plddt_css}">{plddt:.1f}</td>'
            f'<td class="{rmsd_css}">{rmsd:.2f}</td>'
            f'<td>{pass_str}</td></tr>'
        )

    return f"""
<p class="sd-summary">{len(variants)} designs validated with Boltz-2 structure prediction.</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Design</th><th>Val pLDDT</th>
<th>RMSD (\u00c5)</th><th>Pass</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_combine_variants(result: StepResult) -> str:
    variants = [c for c in result.candidates if c.parent_id is not None]
    if not variants:
        return '<p class="sd-summary">No combinatorial variants generated.</p>'

    combos = [c for c in variants if "combo_mean" in c.scores]
    singles = [c for c in variants if "combo_mean" not in c.scores]

    combos.sort(
        key=lambda v: v.scores.get("combo_mean", 0), reverse=True
    )

    show_n = 20
    rows: list[str] = []
    for i, v in enumerate(combos):
        muts = ", ".join(m.label for m in v.mutations) if v.mutations else "\u2014"
        combo_mean = v.scores.get("combo_mean", 0)
        combo_sum = v.scores.get("combo_sum", 0)
        n_muts = int(v.scores.get("n_mutations", len(v.mutations)))
        mean_css = "good" if combo_mean >= 4 else "" if combo_mean >= 2 else "muted"
        extra = f' class="extra-row sd-combo-extra"' if i >= show_n else ""
        rows.append(
            f'<tr{extra}><td>{i + 1}</td>'
            f'<td class="mono" style="font-size:0.79rem">{_html.escape(muts)}</td>'
            f'<td class="{mean_css}" style="font-weight:600">{combo_mean:.3f}</td>'
            f'<td>{combo_sum:.3f}</td>'
            f'<td>{n_muts}</td></tr>'
        )

    expand_btn = ""
    if len(combos) > show_n:
        expand_btn = (
            f'<button class="sd-expand" onclick="'
            f"document.querySelectorAll('.sd-combo-extra').forEach(r=>r.classList.toggle('show'));"
            f"this.textContent=this.textContent.includes('Show')?'Hide':'Show {len(combos) - show_n} more'"
            f'">Show {len(combos) - show_n} more</button>'
        )

    chips: list[str] = [
        f'<div class="sd-chip"><strong>{len(combos)}</strong> combinatorial</div>',
        f'<div class="sd-chip"><strong>{len(singles)}</strong> single-source</div>',
        f'<div class="sd-chip"><strong>{len(variants)}</strong> total</div>',
    ]

    return f"""
<p class="sd-summary">Top single-point mutations combined into multi-mutant variants.
Ranked by mean component score. Order = number of mutations combined.</p>
<div class="sd-chips">{''.join(chips)}</div>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Mutations</th><th>Mean Score</th>
<th>Sum Score</th><th>Order</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
{expand_btn}"""


def _render_e1_score(result: StepResult) -> str:
    all_cands = result.candidates
    variants = [c for c in all_cands if c.parent_id is not None]
    scored = [c for c in variants if "e1_fitness" in c.scores]

    if result.warnings and not scored:
        warn_html = "".join(
            f'<p class="warn" style="font-size:0.85rem">\u26a0 {_html.escape(w)}</p>'
            for w in result.warnings
        )
        return f"""
<p class="sd-summary">E1 scoring was attempted but failed.</p>
{warn_html}"""

    if not scored:
        return '<p class="sd-summary">No E1 scores available.</p>'

    e1_scores = [c.scores["e1_fitness"] for c in scored]
    n_beneficial = sum(1 for s in e1_scores if s > 0)
    n_detrimental = sum(1 for s in e1_scores if s < 0)

    # Show top 15 by E1 fitness
    scored.sort(key=lambda v: v.scores.get("e1_fitness", -999), reverse=True)

    show_n = 15
    rows: list[str] = []
    for i, v in enumerate(scored[:show_n]):
        muts = ", ".join(m.label for m in v.mutations) if v.mutations else "\u2014"
        e1 = v.scores.get("e1_fitness", 0)
        e1_css = "good" if e1 > 0 else "bad" if e1 < -5 else "warn"
        rows.append(
            f'<tr><td>{i + 1}</td>'
            f'<td>{_html.escape(v.name)}</td>'
            f'<td class="mono" style="font-size:0.79rem">{_html.escape(muts[:60])}{"…" if len(muts) > 60 else ""}</td>'
            f'<td class="{e1_css}" style="font-weight:600">{e1:+.4f}</td></tr>'
        )

    chips = [
        f'<div class="sd-chip"><strong>{len(scored)}</strong> / {len(variants)} scored</div>',
        f'<div class="sd-chip"><strong class="good">{n_beneficial}</strong> beneficial (&gt;0)</div>',
        f'<div class="sd-chip"><strong class="bad">{n_detrimental}</strong> detrimental (&lt;0)</div>',
        f'<div class="sd-chip">Range: <strong>{min(e1_scores):+.3f}</strong> to <strong>{max(e1_scores):+.3f}</strong></div>',
    ]

    return f"""
<p class="sd-summary">Profluent E1 zero-shot fitness evaluation. Positive scores
predict beneficial variants. Top 15 shown below.</p>
<div class="sd-chips">{''.join(chips)}</div>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Variant</th><th>Mutations</th><th>E1 Fitness</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_esm_score(result: StepResult) -> str:
    """Shared renderer for esm1v_score and esmif1_score."""
    variants = [c for c in result.candidates if c.parent_id is not None]
    is_esm1v = result.step_name == "esm1v_score"
    model_label = "ESM-1v" if is_esm1v else "ESM-IF1"
    pll_key = "esm1v_pll" if is_esm1v else "esmif1_score"
    delta_key = "esm1v_delta" if is_esm1v else "esmif1_delta"

    if result.warnings:
        warn_html = "".join(
            f'<p class="warn" style="font-size:0.85rem">\u26a0 {_html.escape(w)}</p>'
            for w in result.warnings
        )
        return f"""
<p class="sd-summary">{model_label} scoring encountered issues.</p>
{warn_html}"""

    scored = [c for c in variants if pll_key in c.scores]
    if not scored:
        return f'<p class="sd-summary">No {model_label} scores available for this step.</p>'

    has_delta = any(delta_key in c.scores for c in scored)
    if has_delta:
        scored.sort(key=lambda v: v.scores.get(delta_key, -999), reverse=True)
    else:
        scored.sort(key=lambda v: v.scores.get(pll_key, -999), reverse=True)

    # Summary stats
    pll_values = [c.scores[pll_key] for c in scored]
    pll_min, pll_max = min(pll_values), max(pll_values)
    n_improved = sum(1 for c in scored if c.scores.get(delta_key, 0) > 0) if has_delta else 0

    summary_parts = [f"{len(scored)} variants scored"]
    if has_delta:
        summary_parts.append(f"{n_improved} improved over wild-type")
    summary_parts.append(f"PLL range: {pll_min:.3f} → {pll_max:.3f}")
    summary = " · ".join(summary_parts)

    rows: list[str] = []
    for i, v in enumerate(scored[:20]):
        muts = ", ".join(m.label for m in v.mutations) if v.mutations else "\u2014"
        pll = v.scores.get(pll_key, 0)
        if has_delta:
            delta = v.scores.get(delta_key, 0)
            css = "good" if delta > 0 else "bad" if delta < -0.5 else ""
            rows.append(
                f'<tr><td>{i + 1}</td>'
                f'<td>{_html.escape(v.name)}</td>'
                f'<td class="mono">{_html.escape(muts[:60])}</td>'
                f'<td>{pll:.4f}</td>'
                f'<td class="{css}" style="font-weight:600">{delta:+.4f}</td></tr>'
            )
        else:
            rows.append(
                f'<tr><td>{i + 1}</td>'
                f'<td>{_html.escape(v.name)}</td>'
                f'<td class="mono">{_html.escape(muts[:60])}</td>'
                f'<td style="font-weight:600">{pll:.4f}</td></tr>'
            )

    delta_col = f"<th>\u0394 {model_label}</th>" if has_delta else ""
    return f"""
<p class="sd-summary">{summary}</p>
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Variant</th><th>Mutations</th>
<th>PLL</th>{delta_col}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_generic_step(result: StepResult) -> str:
    """Fallback renderer for unrecognized steps."""
    variants = [c for c in result.candidates if c.parent_id is not None]

    if result.warnings:
        warn_html = "".join(
            f'<p class="warn" style="font-size:0.85rem">\u26a0 {_html.escape(w)}</p>'
            for w in result.warnings
        )
    else:
        warn_html = ""

    if not variants:
        return f"""
<p class="sd-summary">This step completed but produced no variant candidates.</p>
{warn_html}"""

    # Show first few variants with all their scores
    rows: list[str] = []
    all_keys: set[str] = set()
    for v in variants[:10]:
        all_keys.update(v.scores.keys())
    score_keys_sorted = sorted(all_keys - {"motif_risk_total", "motif_count",
                                            "complexity_issues", "complexity_risk_total"})[:4]

    for i, v in enumerate(variants[:10], 1):
        muts = ", ".join(m.label for m in v.mutations) if v.mutations else "\u2014"
        cells = "".join(
            f"<td>{_fmt_score(v.scores[k])}</td>" if k in v.scores else '<td class="muted">\u2014</td>'
            for k in score_keys_sorted
        )
        rows.append(
            f'<tr><td>{i}</td>'
            f'<td class="mono">{_html.escape(muts[:50])}</td>'
            f'{cells}</tr>'
        )

    hdrs = "".join(f"<th>{_html.escape(k)}</th>" for k in score_keys_sorted)

    return f"""
<p class="sd-summary">{len(variants)} variants produced.</p>
{warn_html}
<div class="tbl-wrap">
<table class="sd-table">
<thead><tr><th>#</th><th>Mutations</th>{hdrs}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>"""


def _render_protein_characterization(result: StepResult) -> str:
    """Render the protein characterization step card."""
    parent = None
    for c in result.candidates:
        if c.parent_id is None:
            parent = c
            break
    if not parent or "characterization" not in parent.metadata:
        return '<p class="sd-summary">No characterization data available.</p>'

    char = parent.metadata["characterization"]
    seq = char.get("sequence", "")
    n = len(seq)

    # ── Sequence display with numbered ruler ──
    ruler_lines: list[str] = []
    for i in range(0, n, 60):
        chunk = seq[i:i + 60]
        # Add spacing every 10 residues
        spaced = " ".join(chunk[j:j + 10] for j in range(0, len(chunk), 10))
        ruler_lines.append(f"{i + 1:>5}  {spaced}")
    seq_display = "\n".join(ruler_lines)

    # ── Summary cards ──
    mw = char.get("molecular_weight", 0)
    pI = char.get("isoelectric_point", 0)
    ii = char.get("instability_index", 0)
    ii_label = char.get("instability_label", "")
    cys = char.get("cysteine_count", 0)
    sp = char.get("signal_peptide", {})
    ec = char.get("extinction_coefficient", {})
    tm = char.get("estimated_tm", 0)
    sol = char.get("solubility_score", 0)
    sol_label = char.get("solubility_class", "")
    ph_range = char.get("ph_stable_range", [])

    # Disorder summary
    disorder_frac = char.get("disorder_fraction", 0)
    disorder_regions = char.get("disorder_regions", [])
    n_idr = len(disorder_regions)

    # Domains
    domains = char.get("domains", [])
    domain_count = char.get("domain_count", 0)

    # Cofactors
    cof_summary = char.get("cofactor_summary", "None detected")
    cof_motifs = char.get("cofactor_motifs", [])

    # Signal peptide text
    if sp.get("detected"):
        sp_text = f"Detected (cleavage at pos {sp.get('cleavage_position', '?')}, type: {sp.get('type', '?')})"
        sp_method = sp.get("method", "")
    else:
        sp_text = "Not detected"
        sp_method = sp.get("method", "")

    # pH range text
    ph_text = f"{ph_range[0]:.1f} – {ph_range[1]:.1f}" if ph_range else "N/A"

    # Build cofactor table
    cof_rows = ""
    if cof_motifs:
        cof_rows_list = []
        for cm in cof_motifs[:10]:
            cof_rows_list.append(
                f'<tr><td>{_html.escape(cm.get("name", ""))}</td>'
                f'<td>{cm.get("start", "")}-{cm.get("end", "")}</td>'
                f'<td><code>{_html.escape(cm.get("match", ""))}</code></td>'
                f'<td>{_html.escape(cm.get("description", ""))}</td></tr>'
            )
        cof_rows = "".join(cof_rows_list)

    # Build disorder region table
    idr_rows = ""
    if disorder_regions:
        idr_rows_list = []
        for r in disorder_regions[:10]:
            idr_rows_list.append(
                f'<tr><td>{r.get("start", "")}-{r.get("end", "")}</td>'
                f'<td>{r.get("length", "")}</td>'
                f'<td>{r.get("mean_score", 0):.3f}</td></tr>'
            )
        idr_rows = "".join(idr_rows_list)

    # Domain table
    dom_rows = ""
    if domains and domain_count > 1:
        dom_rows_list = []
        for d in domains:
            dom_rows_list.append(
                f'<tr><td>{_html.escape(str(d.get("name", "")))}</td>'
                f'<td>{d.get("start", "")}-{d.get("end", "")}</td>'
                f'<td>{d.get("length", d.get("end", 0) - d.get("start", 0) + 1)}</td></tr>'
            )
        dom_rows = "".join(dom_rows_list)

    return f"""
<p class="sd-summary">Comprehensive biophysical characterization of the wild-type protein.
Method notes: signal peptide ({_html.escape(sp_method)}), disorder ({_html.escape(char.get('disorder_method', 'heuristic'))}),
domains ({_html.escape(char.get('domain_method', 'heuristic'))}).</p>

<div class="sd-chips">
  <span class="chip">{n} residues</span>
  <span class="chip">{mw:,.0f} Da</span>
  <span class="chip">pI {pI:.2f}</span>
  <span class="chip">{cys} Cys</span>
  <span class="chip">{ii_label} (II={ii:.1f})</span>
  <span class="chip">Tm ≈ {tm:.0f}°C</span>
  <span class="chip">pH {ph_text}</span>
  <span class="chip">{sol_label} ({sol:.0f}%)</span>
  <span class="chip">{n_idr} IDR(s)</span>
  <span class="chip">{domain_count} domain(s)</span>
</div>

<details style="margin-top:8px">
  <summary style="cursor:pointer;color:var(--accent);font-size:0.85rem">
    ▸ Full Amino Acid Sequence ({n} residues)
  </summary>
  <pre style="font-family:Consolas,monospace;font-size:0.72rem;line-height:1.4;
  background:var(--bg);padding:10px;border-radius:6px;overflow-x:auto;
  border:1px solid var(--border);margin-top:6px;color:var(--fg)">{_html.escape(seq_display)}</pre>
</details>

{'<details style="margin-top:8px"><summary style="cursor:pointer;color:var(--accent);font-size:0.85rem">▸ Cofactor / Metal-Binding Motifs (' + str(len(cof_motifs)) + ')</summary><div class="tbl-wrap"><table class="sd-table"><thead><tr><th>Motif</th><th>Position</th><th>Match</th><th>Description</th></tr></thead><tbody>' + cof_rows + '</tbody></table></div></details>' if cof_motifs else ''}

{'<details style="margin-top:8px"><summary style="cursor:pointer;color:var(--accent);font-size:0.85rem">▸ Disordered Regions (' + str(n_idr) + ')</summary><div class="tbl-wrap"><table class="sd-table"><thead><tr><th>Region</th><th>Length</th><th>Mean Score</th></tr></thead><tbody>' + idr_rows + '</tbody></table></div></details>' if idr_rows else ''}

{'<details style="margin-top:8px"><summary style="cursor:pointer;color:var(--accent);font-size:0.85rem">▸ Domain Architecture (' + str(domain_count) + ')</summary><div class="tbl-wrap"><table class="sd-table"><thead><tr><th>Domain</th><th>Region</th><th>Length</th></tr></thead><tbody>' + dom_rows + '</tbody></table></div></details>' if dom_rows else ''}

<div style="margin-top:10px;font-size:0.78rem;color:var(--fg);opacity:0.7">
  ε₂₈₀ = {ec.get('reduced', 0):,} M⁻¹cm⁻¹ (reduced) / {ec.get('oxidized', 0):,} M⁻¹cm⁻¹ (oxidized)
  &middot; GRAVY = {char.get('gravy', 0):.3f}
  &middot; Aromaticity = {char.get('aromaticity', 0):.3f}
</div>"""


# Step renderer dispatch table
_STEP_RENDERERS: dict[str, Any] = {
    "protein_characterization": _render_protein_characterization,
    "cysteine_scan": _render_cysteine_scan,
    "motif_scan": _render_motif_scan,
    "sequence_complexity": _render_sequence_complexity,
    "find_homologs": _render_find_homologs,
    "consensus_design": _render_consensus_design,
    "pssm_analysis": _render_pssm_analysis,
    "predict_structure": _render_predict_structure,
    "stability_ddg": _render_stability_ddg,
    "disulfide_design": _render_disulfide_design,
    "cavity_fill": _render_cavity_fill,
    "surface_patch": _render_surface_patch,
    "rfdiffusion_diversify": _render_rfdiffusion,
    "proteinmpnn_design": _render_proteinmpnn,
    "design_validate": _render_design_validate,
    "combine_variants": _render_combine_variants,
    "e1_score": _render_e1_score,
    "esm1v_score": _render_esm_score,
    "esmif1_score": _render_esm_score,
}
def _pipeline_status_bar(results: dict[str, StepResult]) -> str:
    """Render a compact status bar showing pass/warn/fail for every step."""
    if not results:
        return ""

    items: list[str] = []
    for step_name, result in results.items():
        n_variants = len([c for c in result.candidates if c.parent_id is not None])
        has_warnings = len(result.warnings) > 0

        # Determine status: failed (warnings + no variants), warning, or success
        if has_warnings and n_variants == 0:
            css = "step-fail"
            icon = "\u2718"  # ✘
        elif has_warnings:
            css = "step-warn"
            icon = "\u26a0"  # ⚠
        else:
            css = "step-ok"
            icon = "\u2714"  # ✔

        label = _step_display_name(step_name)
        tooltip = f"{n_variants} variants"
        if has_warnings:
            tooltip += f" — {result.warnings[0][:60]}"

        items.append(
            f'<span class="status-pip {css}" title="{_html.escape(tooltip)}">'
            f'{icon} {_html.escape(label)}</span>'
        )

    n_ok = sum(1 for r in results.values()
               if not r.warnings)
    n_warn = sum(1 for r in results.values()
                 if r.warnings and len([c for c in r.candidates if c.parent_id is not None]) > 0)
    n_fail = sum(1 for r in results.values()
                 if r.warnings and len([c for c in r.candidates if c.parent_id is not None]) == 0)

    summary_parts = []
    if n_ok:
        summary_parts.append(f'<span class="good">{n_ok} passed</span>')
    if n_warn:
        summary_parts.append(f'<span class="warn">{n_warn} with warnings</span>')
    if n_fail:
        summary_parts.append(f'<span class="bad">{n_fail} failed</span>')

    return f"""
<div class="section">
  <h2>Pipeline Status</h2>
  <p class="subtitle">Step-level overview: {' &middot; '.join(summary_parts)}</p>
  <div class="status-bar">
    {''.join(items)}
  </div>
</div>
"""


# Known fix suggestions for common step failures
_FIX_SUGGESTIONS: dict[str, str] = {
    "e1_score": (
        "Create/activate the <code>e1</code> conda environment with "
        "<code>transformers</code>, <code>torch</code>, and the "
        "<code>Profluent-Bio/E1-600m</code> model downloaded."
    ),
    "esmif1_score": (
        "Install ESM-IF1 in the <code>protopt</code> conda env: "
        "<code>pip install fair-esm</code>. Ensure a PDB structure "
        "is available (run <code>predict_structure</code> first)."
    ),
    "esm1v_score": (
        "Install ESM-1v in the <code>plm</code> conda env: "
        "<code>pip install fair-esm</code>."
    ),
    "motif_scaffold": (
        "Set <code>motif_residues</code> in the pipeline YAML config "
        "(e.g. <code>motif_scaffold: {motif_residues: [10,20,30]}</code>) "
        "or populate <code>protected_residues</code> in the global config."
    ),
    "proteinmpnn_design": (
        "Set <code>PROTEINMPNN_DIR</code> environment variable or "
        "<code>proteinmpnn_dir</code> in the YAML config to point "
        "to a valid ProteinMPNN installation directory."
    ),
    "rfdiffusion_diversify": (
        "Ensure the <code>rfd3</code> conda env has <code>rf_diffusion</code> "
        "installed (check <code>PYTHONPATH</code> includes the RFdiffusion2 "
        "source directory). Also verify <code>rfdiffusion_dir</code> in config "
        "points to a valid installation."
    ),
    "stability_ddg": (
        "Ensure ThermoMPNN is installed: set <code>thermompnn_dir</code> "
        "in config and install <code>torch</code>, <code>omegaconf</code>, "
        "<code>pytorch-lightning</code> in the protopt conda env."
    ),
}


def _diagnostics_section(results: dict[str, StepResult]) -> str:
    """Render actionable diagnostics for failed or warning steps."""
    issues: list[str] = []

    for step_name, result in results.items():
        if not result.warnings:
            continue

        n_variants = len([c for c in result.candidates if c.parent_id is not None])
        is_failure = n_variants == 0
        severity_css = "diag-fail" if is_failure else "diag-warn"
        severity_label = "FAILED" if is_failure else "WARNING"
        severity_badge_css = "bad" if is_failure else "warn"

        step_label = _step_display_name(step_name)
        fix_html = _FIX_SUGGESTIONS.get(step_name, "")
        fix_block = f'<div class="diag-fix"><strong>Fix:</strong> {fix_html}</div>' if fix_html else ""

        warn_items = "".join(
            f'<li>{_html.escape(w)}</li>'
            for w in result.warnings
        )

        issues.append(f"""
<div class="diag-card {severity_css}">
  <div class="diag-header">
    <span class="score-badge {severity_badge_css}">{severity_label}</span>
    <strong>{_html.escape(step_label)}</strong>
  </div>
  <ul class="diag-warnings">{warn_items}</ul>
  {fix_block}
</div>""")

    if not issues:
        return ""

    return f"""
<div class="section">
  <h2>Issues &amp; Diagnostics</h2>
  <p class="subtitle">Steps that failed or produced warnings, with suggested fixes.</p>
  {''.join(issues)}
</div>
"""
def _full_variant_library_section(variants_ranked: list[ProteinCandidate]) -> str:
    """Full expandable variant library — single source of truth."""
    if not variants_ranked:
        return ""

    # Determine score columns
    skip = {"n_mutations", "combo_sum", "motif_risk_total", "motif_count",
            "complexity_issues", "complexity_risk_total"}
    counts: Counter[str] = Counter()
    for v in variants_ranked[:200]:
        for k in v.scores:
            if k not in skip:
                counts[k] += 1
    score_keys = [k for k, _ in counts.most_common(6)]

    score_hdrs = "".join(f"<th>{_html.escape(k.replace('_',' ').title())}</th>" for k in score_keys)

    rows: list[str] = []
    for i, v in enumerate(variants_ranked, 1):
        muts = ", ".join(m.label for m in v.mutations) if v.mutations else "\u2014"
        source = _step_display_name(v.mutations[0].source_step) if v.mutations else "\u2014"
        score_cells = "".join(
            f"<td>{_fmt_score(v.scores[k])}</td>" if k in v.scores else "<td class='muted'>\u2014</td>"
            for k in score_keys
        )
        hidden = ' class="lib-row-extra" style="display:none"' if i > 25 else ""
        rows.append(
            f"<tr{hidden}><td>{i}</td>"
            f"<td>{_html.escape(v.name)}</td>"
            f'<td class="mono" style="font-size:0.79rem">{_html.escape(muts)}</td>'
            f'<td class="muted" style="font-size:0.79rem">{_html.escape(source)}</td>'
            f"{score_cells}</tr>"
        )

    n_hidden = max(0, len(variants_ranked) - 25)

    return f"""
<div class="section">
  <h2>Full Variant Library</h2>
  <p class="subtitle">All {len(variants_ranked)} generated variants ranked by composite score. This is the single source of truth for the complete candidate set.</p>
  <div class="lib-controls">
    <input class="lib-search" type="text" placeholder="Filter by name or mutation…"
      oninput="filterLib(this.value)">
    {"" if n_hidden == 0 else f'<button class="lib-toggle" id="lib-btn" onclick="toggleLib(this)">Show all {len(variants_ranked)} variants</button>'}
  </div>
  <div class="tbl-wrap">
  <table id="lib-table">
    <thead>
      <tr><th>#</th><th>Variant</th><th>Mutations</th><th>Source</th>{score_hdrs}</tr>
    </thead>
    <tbody id="lib-tbody">
      {''.join(rows)}
    </tbody>
  </table>
  </div>
</div>
<script>
(function() {{
  var libExpanded = false;
  window.toggleLib = function(btn) {{
    libExpanded = !libExpanded;
    document.querySelectorAll('.lib-row-extra').forEach(function(r) {{
      r.style.display = libExpanded ? '' : 'none';
    }});
    btn.textContent = libExpanded ? 'Show fewer' : 'Show all {len(variants_ranked)} variants';
  }};
  window.filterLib = function(q) {{
    q = q.toLowerCase();
    document.querySelectorAll('#lib-tbody tr').forEach(function(r) {{
      var txt = r.textContent.toLowerCase();
      r.style.display = txt.indexOf(q) >= 0 ? '' : 'none';
    }});
  }};
}})();
</script>
"""
def _footer_html() -> str:
    return f"""
<div style="text-align:center; padding: 2rem; color: #484f58; font-size: 0.78rem; border-top: 1px solid var(--border);">
  protein-optimizer &middot; Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
</body>
</html>
"""
