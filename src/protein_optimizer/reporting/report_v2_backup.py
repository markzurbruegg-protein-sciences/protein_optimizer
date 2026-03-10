"""v2 HTML report — 3D structure viewer, protein metrics, ranked mutations.

Generates a self-contained HTML report with:
- Interactive 3Dmol.js viewer (PDB embedded inline)
- Protein biophysical metrics panel
- Top-10 mutation summary
- Per-step pipeline breakdown (top 20, expandable to 100)
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

    # Determine the final result (last step)
    final_key = list(results.keys())[-1] if results else None
    final_result = results.get(final_key) if final_key else None
    all_candidates = final_result.candidates if final_result else []
    parent = next((c for c in all_candidates if c.parent_id is None), None)
    variants = [c for c in all_candidates if c.parent_id is not None]

    # Rank variants
    variants_ranked = sorted(
        variants,
        key=lambda c: c.scores.get("composite_score", 0),
        reverse=True,
    )

    # Top-5 unique residue positions for 3D highlighting
    highlight_positions = _top_positions(variants_ranked, n=5)

    # PDB text (for inline embedding)
    pdb_text = ""
    if pdb_path and Path(pdb_path).exists():
        pdb_text = Path(pdb_path).read_text(encoding="utf-8", errors="replace")

    # Protein metrics
    seq = parent.sequence if parent else ""
    metrics = _compute_metrics(seq, pdb_text, all_candidates, results)

    # Per-step mutations
    step_mutations = _collect_step_mutations(results)

    # Per-tier mutation data for 3D viewer
    tier_mut_data = _collect_tier_viewer_data(step_mutations)

    # Build HTML
    parts = [
        _header_html(title),
        _hero_section(parent, metrics, pdb_text, highlight_positions, variants_ranked, tier_mut_data),
        _stage_sections(step_mutations, results),
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

/* ── Step status icon ── */
.step-status {{
  font-size: 0.85rem; margin-right: 0.15rem;
}}
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


def _hero_section(
    parent: ProteinCandidate | None,
    metrics: dict[str, Any],
    pdb_text: str,
    highlight_positions: list[int],
    variants_ranked: list[ProteinCandidate],
    tier_mut_data: dict[int, list[dict[str, Any]]] | None = None,
) -> str:
    """Build the hero area: 3D viewer + metrics sidebar."""

    name = parent.name if parent else "Unknown"
    length = metrics.get("length", 0)
    tier_mut_data = tier_mut_data or {}

    # Top-5 mutation labels for legend
    top5_labels = []
    seen_pos: set[int] = set()
    for v in variants_ranked:
        for m in v.mutations:
            if m.position not in seen_pos:
                seen_pos.add(m.position)
                top5_labels.append(f"{m.wt}{m.position}{m.mut}")
            if len(top5_labels) >= 5:
                break
        if len(top5_labels) >= 5:
            break

    # Highlight data for JS (as JSON for inline embedding)
    _hl_colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#bc8cff"]
    hl_positions_js = json.dumps(highlight_positions)
    hl_labels_js = json.dumps(top5_labels[:5])
    hl_colors_js = json.dumps(_hl_colors[:max(len(highlight_positions), 1)])

    # Tier data for JS
    tier_js = json.dumps({str(k): v for k, v in tier_mut_data.items()})

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
    colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#bc8cff"]
    for i, lbl in enumerate(top5_labels):
        legend_dots.append(f'<span><span class="dot" style="background:{colors[i]}"></span>{lbl}</span>')

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
    <div class="tier-toolbar" id="tier-bar"></div>
    <div class="viewer-legend" id="view-legend"></div>
    <div class="viewer-toolbar">
      <button class="view-btn mode active" data-mode="plddt" onclick="setView('plddt')">Confidence</button>
      <button class="view-btn mode" data-mode="hydrophobicity" onclick="setView('hydrophobicity')">Hydrophobicity</button>
      <button class="view-btn mode" data-mode="charge" onclick="setView('charge')">Charge</button>
      <button class="view-btn mode" data-mode="ss" onclick="setView('ss')">2&deg; Structure</button>
      <div class="toolbar-sep"></div>
      <button class="view-btn" id="btn-surface" onclick="toggleSurface()">Surface</button>
      <button class="view-btn active" id="btn-mutations" onclick="toggleMutations()">Mutations</button>
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
    </div>

    <div class="metric-group">
      <h3>Aggregation Propensity</h3>
      {_metric("Agg. Score", f"{agg:.3f}", agg_css)}
      {_metric("Hydrophobic Patches", f"{m.get('agg_patches', 0)}", agg_css)}
      {_metric("Surface Patches (SAP)", f"{m.get('n_surface_patches', 0)}", "")}
    </div>

    <div class="metric-group">
      <h3>Physicochemical</h3>
      {_metric("GRAVY", f"{gravy:.3f}", gravy_css)}
      {_metric("Net Charge (pH 7.4)", f"{charge:+.1f}", charge_css)}
      {_metric("Isoelectric Point", f"{m.get('pI', 0):.2f}", "")}
      {_metric("Cysteines", f"{m.get('cys_count', 0)}", "")}
    </div>

    <div class="metric-group">
      <h3>Expression (E. coli)</h3>
      {_metric("Rare-codon AAs (W/C/M)", f"{m.get('rare_codons', 0)}", "")}
      {_metric("Prolines", f"{m.get('proline_count', 0)}", "")}
    </div>

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
  var surfOn = false, mutOn = true, curView = "plddt";

  // Per-tier mutation data
  var tierData = {tier_js};
  var curTier = "all";
  var tierNames = {{1:"Sequence Heuristics",2:"Evolutionary Analysis",
    3:"Structure-based",4:"AI / ML Scoring",5:"Generative Design"}};
  var tierColors = {{1:"#3fb950",2:"#58a6ff",3:"#d29922",4:"#bc8cff",5:"#f85149"}};
  var mutColors = ["#ff6b6b","#ffd93d","#6bcb77","#4d96ff","#bc8cff",
    "#ff9ff3","#54a0ff","#5f27cd","#01a3a4","#f368e0",
    "#ff6348","#7bed9f","#70a1ff","#dfe6e9","#ffeaa7",
    "#fd79a8","#00cec9","#e17055","#0984e3","#6c5ce7"];

  // Build tier toolbar buttons
  (function buildTierBar() {{
    var bar = document.getElementById("tier-bar");
    var keys = Object.keys(tierData).sort();
    if(keys.length === 0) {{ bar.style.display = "none"; return; }}
    var allBtn = '<button class="tier-btn active" data-tier="all">' +
      '<span class="tier-pip" style="background:#c9d1d9"></span>Top 5</button>';
    bar.innerHTML = allBtn;
    keys.forEach(function(k) {{
      var n = tierData[k].length;
      var cl = tierColors[k] || "#8b949e";
      bar.innerHTML += '<button class="tier-btn" data-tier="'+k+'">' +
        '<span class="tier-pip" style="background:'+cl+'"></span>Tier '+k+
        ' <span style="opacity:0.6;font-size:0.65rem">('+n+')</span></button>';
    }});
    bar.addEventListener("click", function(e) {{
      var btn = e.target.closest(".tier-btn");
      if(btn) window.setTier(btn.dataset.tier);
    }});
  }})();

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
    var positions, labels, colors;
    if(curTier === "all") {{
      positions = hlPos; labels = hlLbl; colors = hlClr;
    }} else {{
      var items = tierData[curTier] || [];
      positions = items.map(function(d){{ return d.pos; }});
      labels = items.map(function(d){{ return d.label; }});
      colors = mutColors;
    }}
    for(var i=0; i<positions.length; i++) {{
      var p=positions[i], cl=colors[i % colors.length];
      viewer.addStyle({{resi:p}}, {{stick:{{radius:0.18, color:cl}}}});
      var atoms = viewer.selectedAtoms({{resi:p, atom:"CA"}});
      if(atoms.length)
        viewer.addLabel(labels[i]||String(p), {{
          position:atoms[0], backgroundColor:cl, backgroundOpacity:0.85,
          fontColor:"#000", fontSize:12, showBackground:true
        }});
    }}
  }}

  window.setTier = function(t) {{
    curTier = t;
    mutOn = true;
    document.getElementById("btn-mutations").classList.add("active");
    window.setView(curView);
    document.querySelectorAll(".tier-btn").forEach(function(b){{
      b.classList.toggle("active",
        (t==="all" && b.textContent.indexOf("Top 5")>=0) ||
        (t!=="all" && b.textContent.indexOf("Tier "+t)>=0));
    }});
  }};

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


def _sequence_liabilities_section(results: dict[str, StepResult]) -> str:
    """Combined cysteine-risk + motif-liability table (for Stage 1)."""
    cys_result = results.get("cysteine_scan")
    motif_result = results.get("motif_scan")

    if not cys_result and not motif_result:
        return ""

    rows: list[dict] = []

    if cys_result:
        for c in cys_result.candidates:
            if c.parent_id is None:
                continue
            for m in c.mutations:
                rows.append({
                    "position": m.position, "residue": m.wt,
                    "category": "Unpaired Cys",
                    "risk": m.score or c.scores.get("cys_risk", 0),
                    "rationale": "Free cysteine \u2014 may cause unwanted disulfides or oxidation",
                    "fix": m.label,
                })

    if motif_result:
        parent = next((c for c in motif_result.candidates if c.parent_id is None), None)
        hits = parent.metadata.get("motif_hits", []) if parent else []
        fix_map: dict[int, str] = {}
        for c in motif_result.candidates:
            if c.parent_id is None:
                continue
            for m in c.mutations:
                if m.source_step == "motif_scan":
                    fix_map[m.position] = m.label
        for hit in hits:
            pos = hit.get("position", 0)
            fix = fix_map.get(pos, fix_map.get(pos + 1, "\u2014"))
            rows.append({
                "position": pos,
                "residue": hit.get("residues", "?")[0],
                "category": hit.get("category", "unknown").title(),
                "risk": hit.get("risk", 0),
                "rationale": hit.get("rationale", ""),
                "fix": fix,
            })

    seen: set[tuple[int, str]] = set()
    unique: list[dict] = []
    for r in rows:
        key = (r["position"], r["category"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    rows = sorted(unique, key=lambda r: -r["risk"])

    cat_colours = {
        "Unpaired Cys": "#bc8cff", "Deamidation": "#58a6ff",
        "Oxidation": "#d29922", "Proteolysis": "#f85149",
    }

    def _risk_css(risk: float) -> str:
        if risk >= 0.65:
            return "bad"
        if risk >= 0.4:
            return "warn"
        return "good"

    table_rows = []
    for r in rows:
        cat_col = cat_colours.get(r["category"], "#8b949e")
        css = _risk_css(r["risk"])
        table_rows.append(
            f'<tr><td>{r["position"]}</td><td class="mono">{r["residue"]}</td>'
            f'<td><span style="background:{cat_col};color:#000;padding:1px 7px;'
            f'border-radius:4px;font-size:0.78rem;font-weight:600">'
            f'{_html.escape(r["category"])}</span></td>'
            f'<td class="{css}" style="font-weight:600">{r["risk"]:.2f}</td>'
            f'<td style="font-size:0.84rem">{_html.escape(r["rationale"])}</td>'
            f'<td class="mono">{_html.escape(r["fix"])}</td></tr>'
        )

    n_cys = sum(1 for r in rows if r["category"] == "Unpaired Cys")
    n_motif = len(rows) - n_cys

    return f"""
  <h3 style="color:var(--accent);margin:1rem 0 0.3rem">Sequence Liabilities</h3>
  <p class="subtitle">
    <strong>Unpaired cysteines</strong> may form non-native disulfides,
    <strong>deamidation</strong> sites undergo backbone rearrangements,
    <strong>oxidation</strong>-prone Met residues lose activity, and
    <strong>proteolytic</strong> dibasic sites are cleaved by host proteases.
    {n_cys} cysteine + {n_motif} motif liabilities detected.
  </p>
  <div class="tbl-wrap">
  <table>
    <thead><tr>
      <th>Pos</th><th>Res</th><th>Category</th><th>Risk</th>
      <th>Rationale</th><th>Suggested Fix</th>
    </tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>
  </div>
"""


_STAGE_DESCRIPTIONS: dict[int, str] = {
    1: (
        "Quick sequence-level scans that flag chemical liabilities \u2014 unpaired cysteines, "
        "deamidation/oxidation motifs, proteolytic sites, and low-complexity regions. "
        "These are fast to compute and catch issues that affect shelf-life and manufacturability."
    ),
    2: (
        "Evolutionary conservation from homologue alignments and PSSM analysis. "
        "Positions conserved across orthologs are generally safer to keep, while variable "
        "positions offer room for beneficial mutations."
    ),
    3: (
        "Structure-aware engineering using the Boltz-2 predicted 3D model. "
        "Includes stability prediction (ThermoMPNN \u0394\u0394G), disulfide design, "
        "cavity filling, and surface hydrophobic patch analysis "
        "to improve thermostability and reduce aggregation."
    ),
    4: (
        "Generative protein design: RFdiffusion2 generates backbone-diversified variants "
        "via partial diffusion, ProteinMPNN (SolubleMPNN) designs sequences on predicted "
        "structures, and Boltz-2 validates self-consistency. Combinatorial assembly of top "
        "point mutations rounds out the variant library."
    ),
    5: (
        "Zero-shot fitness evaluation with Profluent E1, a 600M-parameter protein language model. "
        "Wildtype-marginal scoring measures how well each variant fits the learned evolutionary "
        "landscape, providing an unbiased final evaluation of all candidates."
    ),
}


def _step_summary_card(step_name: str, result: StepResult) -> str:
    """Generate a summary info card for a step showing key stats."""
    parts: list[str] = []

    if step_name == "find_homologs":
        parent = next((c for c in result.candidates if c.parent_id is None), None)
        if parent:
            meta = parent.metadata
            n_homologs = meta.get("n_homologs", 0)
            method = meta.get("search_method", "unknown")
            homolog_seqs = meta.get("homolog_sequences", [])
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip"><strong>{n_homologs}</strong> homologs found</div>')
            parts.append(f'<div class="stat-chip">Method: <strong>{_html.escape(str(method))}</strong></div>')
            if homolog_seqs:
                avg_len = sum(len(s) for s in homolog_seqs) / len(homolog_seqs)
                parts.append(f'<div class="stat-chip">Avg length: <strong>{avg_len:.0f}</strong> aa</div>')
            parts.append('</div>')

    elif step_name == "consensus_design":
        variants = [c for c in result.candidates if c.parent_id is not None]
        if variants:
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> consensus mutations</div>')
            scores = [c.scores.get("consensus_conservation", 0) for c in variants if "consensus_conservation" in c.scores]
            if scores:
                parts.append(f'<div class="stat-chip">Conservation: <strong>{min(scores):.2f}</strong> \u2013 <strong>{max(scores):.2f}</strong></div>')
            parts.append('</div>')

    elif step_name == "predict_structure":
        parent = next((c for c in result.candidates if c.parent_id is None), None)
        if parent:
            plddt = parent.scores.get("plddt", 0)
            method = parent.metadata.get("method", "boltz2")
            pdb_path = parent.metadata.get("structure_path", "")
            css = "good" if plddt >= 80 else "warn" if plddt >= 60 else "bad"
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip">Method: <strong>{_html.escape(str(method))}</strong></div>')
            parts.append(f'<div class="stat-chip {css}">Mean pLDDT: <strong>{plddt:.1f}</strong></div>')
            if pdb_path:
                parts.append(f'<div class="stat-chip">Structure generated \u2714</div>')
            parts.append('</div>')

    elif step_name == "stability_ddg":
        variants = [c for c in result.candidates if c.parent_id is not None]
        if variants:
            ddgs = [c.scores.get("ddg", 0) for c in variants if "ddg" in c.scores]
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> stabilizing mutations</div>')
            if ddgs:
                parts.append(f'<div class="stat-chip">\u0394\u0394G range: <strong>{min(ddgs):.2f}</strong> to <strong>{max(ddgs):.2f}</strong> kcal/mol</div>')
            parts.append('</div>')

    elif step_name == "disulfide_design":
        variants = [c for c in result.candidates if c.parent_id is not None]
        if variants:
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> disulfide candidates</div>')
            dists = [c.scores.get("disulfide_cb_distance", 0) for c in variants if "disulfide_cb_distance" in c.scores]
            if dists:
                parts.append(f'<div class="stat-chip">C\u03b2 distance: <strong>{min(dists):.2f}</strong> \u2013 <strong>{max(dists):.2f}</strong> \u00c5</div>')
            parts.append('</div>')

    elif step_name == "pssm_analysis":
        variants = [c for c in result.candidates if c.parent_id is not None]
        if variants:
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> PSSM-positive mutations</div>')
            lo = [c.scores.get("pssm_log_odds", 0) for c in variants if "pssm_log_odds" in c.scores]
            if lo:
                parts.append(f'<div class="stat-chip">Log-odds range: <strong>{min(lo):.2f}</strong> to <strong>{max(lo):.2f}</strong></div>')
            parts.append('</div>')

    elif step_name == "sequence_complexity":
        parent = next((c for c in result.candidates if c.parent_id is None), None)
        if parent:
            issues = int(parent.scores.get("complexity_issues", 0))
            risk = parent.scores.get("complexity_risk_total", 0)
            css = "good" if issues == 0 else "warn" if issues < 3 else "bad"
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip {css}"><strong>{issues}</strong> complexity issues</div>')
            parts.append(f'<div class="stat-chip">Risk score: <strong>{risk:.2f}</strong></div>')
            parts.append('</div>')

    elif step_name == "cavity_fill":
        variants = [c for c in result.candidates if c.parent_id is not None]
        if variants:
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> cavity-filling mutations</div>')
            parts.append('</div>')

    elif step_name == "surface_patch":
        variants = [c for c in result.candidates if c.parent_id is not None]
        if variants:
            parts.append(f'<div class="summary-stats">')
            parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> surface redesign mutations</div>')
            sap = [c.scores.get("surface_sap", 0) for c in variants if "surface_sap" in c.scores]
            if sap:
                parts.append(f'<div class="stat-chip">SAP score: <strong>{min(sap):.2f}</strong> to <strong>{max(sap):.2f}</strong></div>')
            parts.append('</div>')

    elif step_name == "rfdiffusion_diversify":
        variants = [c for c in result.candidates if c.parent_id is not None]
        parts.append(f'<div class="summary-stats">')
        parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> RFdiffusion backbones</div>')
        parts.append('</div>')

    elif step_name == "proteinmpnn_design":
        variants = [c for c in result.candidates if c.parent_id is not None]
        if variants:
            parts.append(f'<div class="summary-stats">')
            n_designs = sum(1 for c in variants if "mpnn_score" in c.scores)
            parts.append(f'<div class="stat-chip"><strong>{n_designs}</strong> MPNN designs</div>')
            n_muts = sum(len(c.mutations) for c in variants if "mpnn_score" in c.scores)
            parts.append(f'<div class="stat-chip"><strong>{n_muts}</strong> total mutations</div>')
            scores = [c.scores.get("mpnn_score", 0) for c in variants if "mpnn_score" in c.scores]
            if scores:
                parts.append(f'<div class="stat-chip">MPNN score: <strong>{min(scores):.3f}</strong> \u2013 <strong>{max(scores):.3f}</strong></div>')
            parts.append('</div>')

    elif step_name == "combine_variants":
        variants = [c for c in result.candidates if c.parent_id is not None]
        n_combo = sum(1 for c in variants if "combo_mean" in c.scores)
        parts.append(f'<div class="summary-stats">')
        parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> total variants</div>')
        if n_combo:
            parts.append(f'<div class="stat-chip"><strong>{n_combo}</strong> combinatorial</div>')
        parts.append('</div>')

    elif step_name == "design_validate":
        variants = [c for c in result.candidates if c.parent_id is not None]
        parts.append(f'<div class="summary-stats">')
        parts.append(f'<div class="stat-chip"><strong>{len(variants)}</strong> validated designs</div>')
        parts.append('</div>')

    elif step_name == "e1_score":
        variants = [c for c in result.candidates if c.parent_id is not None]
        scored = [c for c in variants if "e1_fitness" in c.scores]
        parts.append(f'<div class="summary-stats">')
        parts.append(f'<div class="stat-chip"><strong>{len(scored)}</strong> / <strong>{len(variants)}</strong> scored</div>')
        e1_scores = [c.scores["e1_fitness"] for c in scored]
        if e1_scores:
            parts.append(f'<div class="stat-chip">E1 fitness: <strong>{min(e1_scores):.3f}</strong> to <strong>{max(e1_scores):.3f}</strong></div>')
            pos_count = sum(1 for s in e1_scores if s > 0)
            parts.append(f'<div class="stat-chip"><strong>{pos_count}</strong> predicted beneficial</div>')
        parts.append('</div>')

    if result.warnings:
        for w in result.warnings[:3]:
            parts.append(f'<p class="warn" style="font-size:0.82rem;margin:0.25rem 0">\u26a0 {_html.escape(w)}</p>')

    return "\n".join(parts)


def _stage_sections(
    step_mutations: dict[str, list[ProteinCandidate]],
    results: dict[str, StepResult],
) -> str:
    """Build per-stage sections with recommended mutations and step details."""

    # Group steps by tier
    tier_steps: dict[int, list[str]] = {}
    for step_name in results:
        tier = _TIER_MAP.get(step_name, 0)
        if tier == 0:
            continue
        tier_steps.setdefault(tier, []).append(step_name)

    parts: list[str] = ['<div class="content">']

    for tier_num in sorted(tier_steps):
        tier_label = _TIER_NAMES.get(tier_num, f"Tier {tier_num}")
        tier_desc = _STAGE_DESCRIPTIONS.get(tier_num, "")
        steps = tier_steps[tier_num]

        # Collect top mutations from steps in this tier
        _SCORING_TIERS = {5, 6}
        tier_step_set = set(steps)
        tier_variants: list[ProteinCandidate] = []
        for step in steps:
            for v in step_mutations.get(step, []):
                if tier_num in _SCORING_TIERS:
                    tier_variants.append(v)
                else:
                    if any(m.source_step in tier_step_set for m in v.mutations):
                        tier_variants.append(v)
        tier_variants.sort(
            key=lambda c: c.scores.get("composite_score", max(c.scores.values()) if c.scores else 0),
            reverse=True,
        )
        # Dedupe by mutation position set
        seen_pos: set[tuple[int, ...]] = set()
        deduped: list[ProteinCandidate] = []
        for v in tier_variants:
            key = tuple(sorted(m.position for m in v.mutations))
            if key not in seen_pos:
                seen_pos.add(key)
                deduped.append(v)
        tier_variants = deduped

        # Stage header
        parts.append(f"""
<div class="section">
  <div class="tier-header">
    <span class="tier-badge">STAGE {tier_num}</span>
    <span class="tier-title">{_html.escape(tier_label)}</span>
  </div>
  <p class="subtitle" style="margin-top:0.6rem">{_html.escape(tier_desc)}</p>
""")

        # Sequence liabilities table for stage 1
        if tier_num == 1:
            parts.append(_sequence_liabilities_section(results))

        # Recommended mutations table for stages that produce or rank variants
        if tier_num >= 2:
            recs = tier_variants[:10]
            if recs:
                parts.append(_stage_recs_table(recs, tier_num))

        # Per-step collapsible details — show ALL steps now
        for step_name in steps:
            result = results[step_name]
            variants = step_mutations.get(step_name, [])
            total = len(variants)
            n_warnings = len(result.warnings)
            warn_badge = f' &middot; <span class="warn">{n_warnings} warning{"s" if n_warnings != 1 else ""}</span>' if n_warnings else ""

            # Step title with descriptive subtitle
            step_title = step_name.replace('_', ' ').title()

            # Status indicator
            has_data = total > 0 or step_name in ("find_homologs", "predict_structure", "sequence_complexity")
            status_icon = '\u2714' if has_data and n_warnings == 0 else '\u26a0' if n_warnings > 0 else '\u2714'

            # Build count badge text
            if step_name == "find_homologs":
                parent = next((c for c in result.candidates if c.parent_id is None), None)
                n_hom = parent.metadata.get("n_homologs", 0) if parent else 0
                badge_text = f"{n_hom} homologs"
            elif step_name == "predict_structure":
                parent = next((c for c in result.candidates if c.parent_id is None), None)
                plddt_val = parent.scores.get("plddt", 0) if parent else 0
                badge_text = f"pLDDT {plddt_val:.1f}"
            elif step_name == "sequence_complexity":
                parent = next((c for c in result.candidates if c.parent_id is None), None)
                n_issues = int(parent.scores.get("complexity_issues", 0)) if parent else 0
                badge_text = f"{n_issues} issues"
            else:
                badge_text = f"{total} variant{'s' if total != 1 else ''}"

            uid = step_name.replace(" ", "_")

            # Default open for steps with data
            open_attr = " open" if total > 0 and total <= 30 else ""

            parts.append(f"""
<details class="step-card"{open_attr}>
  <summary>
    <span class="step-status">{status_icon}</span>
    {_html.escape(step_title)}
    <span class="count-badge">{badge_text}{warn_badge}</span>
  </summary>
  <div class="inner">
""")

            # Always show summary card
            parts.append(_step_summary_card(step_name, result))

            if not variants:
                parts.append("</div></details>")
                continue

            # Variant table
            score_keys = _best_score_keys(variants)
            score_hdrs = "".join(f"<th>{_html.escape(k)}</th>" for k in score_keys)

            parts.append(f'<div class="tbl-wrap"><table><thead><tr><th>#</th><th>Variant</th><th>Mutations</th>{score_hdrs}</tr></thead>')
            parts.append(f'<tbody id="tbody-{uid}">')

            for idx, v in enumerate(variants[:100], 1):
                muts = ", ".join(m.label for m in v.mutations) or "\u2014"
                score_cells = "".join(
                    f"<td>{_fmt_score(v.scores[k])}</td>" if k in v.scores else "<td class='muted'>\u2014</td>"
                    for k in score_keys
                )
                hidden_cls = ""
                if idx > 20:
                    hidden_cls = f' class="extra-row-{uid}" style="display:none"'
                parts.append(
                    f"<tr{hidden_cls}><td>{idx}</td><td>{_html.escape(v.name)}</td>"
                    f'<td class="mono">{_html.escape(muts)}</td>{score_cells}</tr>'
                )

            parts.append("</tbody></table></div>")

            if total > 20:
                showing_extra = min(total, 100) - 20
                parts.append(f"""
    <button class="show-more-btn" onclick="toggleRows('{uid}', this)" data-expanded="false">
      Show {showing_extra} more ({total} total)
    </button>""")

            parts.append("</div></details>")

        parts.append("</div><!-- /section -->")

    # Toggle-rows script
    parts.append("""
</div><!-- /content -->
<script>
function toggleRows(uid, btn) {
  var rows = document.querySelectorAll('.extra-row-' + uid);
  var expanded = btn.getAttribute('data-expanded') === 'true';
  rows.forEach(function(r) { r.style.display = expanded ? 'none' : ''; });
  btn.setAttribute('data-expanded', expanded ? 'false' : 'true');
  btn.textContent = expanded ? btn.textContent.replace('Show less', 'Show more') : 'Show less';
}
</script>
""")

    return "\n".join(parts)


def _stage_recs_table(recs: list[ProteinCandidate], tier_num: int) -> str:
    """Render a recommended-mutations table for a single stage."""
    rows: list[str] = []
    for i, v in enumerate(recs, 1):
        badge_cls = {1: "gold", 2: "silver", 3: "bronze"}.get(i, "")
        mut_strs = ", ".join(m.label for m in v.mutations) or "\u2014"
        source = v.mutations[0].source_step.replace("_", " ").title() if v.mutations else "\u2014"

        # Pick best available score
        comp = v.scores.get("composite_score", 0)
        best_key = "composite_score"
        if not comp:
            for k, val in v.scores.items():
                if k not in ("n_mutations",):
                    comp = val
                    best_key = k
                    break

        comp_css = "good" if comp > 0 else "bad" if comp < 0 else ""

        # Detail scores
        score_parts = []
        for key in ("esm1v_delta", "consensus_conservation", "pssm_log_odds",
                     "disulfide_cb_distance", "cavity_burial", "surface_sap",
                     "cys_risk", "motif_risk", "combo_mean"):
            if key in v.scores:
                short = key.replace("_", " ").split()[0]
                score_parts.append(f"{short}={_fmt_score(v.scores[key])}")

        _em_dash = '\u2014'
        score_detail = '; '.join(score_parts) if score_parts else _em_dash
        rows.append(f"""<tr>
  <td><span class="rank {badge_cls}">{i}</span></td>
  <td><strong>{_html.escape(v.name)}</strong></td>
  <td class="mono">{_html.escape(mut_strs)}</td>
  <td>{_html.escape(source)}</td>
  <td class="{comp_css}">{comp:+.4f}</td>
  <td class="muted" style="font-size:0.78rem">{score_detail}</td>
</tr>""")

    return f"""
  <h3 style="color:var(--accent);margin:0.8rem 0 0.3rem">Recommended Mutations</h3>
  <div class="tbl-wrap">
  <table>
    <thead><tr>
      <th>#</th><th>Variant</th><th>Mutations</th>
      <th>Source Step</th><th>Score</th><th>Details</th>
    </tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  </div>
"""


def _best_score_keys(variants: list[ProteinCandidate], max_keys: int = 5) -> list[str]:
    """Pick the most common score keys across variants (skip meta-scores)."""
    skip = {"composite_score", "n_mutations", "combo_sum", "combo_mean",
            "motif_risk_total", "motif_count", "complexity_issues",
            "complexity_risk_total"}
    counts: Counter[str] = Counter()
    for v in variants[:50]:
        for k in v.scores:
            if k not in skip:
                counts[k] += 1
    return [k for k, _ in counts.most_common(max_keys)]


def _footer_html() -> str:
    return f"""
<div style="text-align:center; padding: 2rem; color: #484f58; font-size: 0.78rem; border-top: 1px solid var(--border);">
  protein-optimizer &middot; Report generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
</body>
</html>
"""
