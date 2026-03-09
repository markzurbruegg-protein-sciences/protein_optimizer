"""HTML report generation for protein optimization results.

Produces a self-contained HTML report with:
- Summary statistics
- Ranked candidate table with all scores
- Mutation map (sequence logo-style)
- Per-step warnings
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult
from protein_optimizer.reporting.narratives import generate_step_narratives

logger = logging.getLogger(__name__)


def generate_html_report(
    results: dict[str, StepResult],
    output_path: str | Path,
    title: str = "Protein Optimization Report",
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate a comprehensive HTML report.

    Args:
        results: Dict mapping step_name → StepResult.
        output_path: Where to save the HTML file.
        title: Report title.
        config: Pipeline configuration used.

    Returns:
        Path to generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all candidates from the final step
    final_step = list(results.keys())[-1] if results else None
    final_result = results.get(final_step) if final_step else None

    all_candidates = final_result.candidates if final_result else []
    parents = [c for c in all_candidates if c.parent_id is None]
    variants = [c for c in all_candidates if c.parent_id is not None]

    # Collect all warnings
    all_warnings = []
    for step_name, result in results.items():
        for w in result.warnings:
            all_warnings.append((step_name, w))

    # Collect all score keys
    score_keys = set()
    for c in all_candidates:
        score_keys.update(c.scores.keys())
    score_keys = sorted(score_keys)

    # Generate narrative summaries for each step
    narratives = generate_step_narratives(results)

    # Build cross-tier recommendation summary
    from protein_optimizer.reporting.narratives import generate_recommendation_summary
    recommendation = generate_recommendation_summary(results)

    # Build HTML
    html_parts = [
        _html_header(title),
        _summary_section(results, parents, variants, score_keys),
        recommendation,
        _analysis_narrative_section(narratives),
        _candidate_table(variants, score_keys),
        _mutation_summary(variants),
        _warnings_section(all_warnings),
        _step_summary(results),
        _html_footer(),
    ]

    report_html = "\n".join(html_parts)
    output_path.write_text(report_html, encoding="utf-8")
    logger.info(f"Report saved to {output_path}")

    return output_path


def _html_header(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
  :root {{
    --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
    --card-bg: #161b22; --border: #30363d;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
         background: var(--bg); color: var(--fg); padding: 2rem; line-height: 1.5; }}
  h1 {{ color: var(--accent); margin-bottom: 0.5rem; }}
  h2 {{ color: var(--accent); margin: 1.5rem 0 0.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
  .timestamp {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }}
  .stat-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }}
  .stat-card .number {{ font-size: 2rem; font-weight: bold; color: var(--accent); }}
  .stat-card .label {{ font-size: 0.85rem; color: #8b949e; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.85rem; }}
  th {{ background: var(--card-bg); color: var(--accent); padding: 0.5rem; text-align: left;
       border-bottom: 2px solid var(--border); position: sticky; top: 0; }}
  td {{ padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--border); }}
  tr:hover {{ background: rgba(88, 166, 255, 0.05); }}
  .good {{ color: var(--green); }}
  .bad {{ color: var(--red); }}
  .warn {{ color: var(--yellow); }}
  .mutations {{ font-family: monospace; font-size: 0.8rem; }}
  .warning-list {{ list-style: none; }}
  .warning-list li {{ padding: 0.3rem 0; border-bottom: 1px solid var(--border); }}
  .warning-list li::before {{ content: "⚠ "; color: var(--yellow); }}
  .step-tag {{ display: inline-block; background: var(--card-bg); border: 1px solid var(--border);
              border-radius: 4px; padding: 0.1rem 0.4rem; margin: 0.1rem; font-size: 0.75rem; }}
  .overflow-x {{ overflow-x: auto; }}
  .narrative-block {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
                     padding: 1.25rem 1.5rem; margin: 1rem 0; }}
  .narrative-block h3 {{ color: var(--accent); font-size: 1.1rem; margin-bottom: 0.75rem; }}
  .narrative-block h4 {{ color: var(--fg); font-size: 0.95rem; margin: 1rem 0 0.4rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }}
  .narrative-content p {{ margin: 0.5rem 0; font-size: 0.9rem; line-height: 1.6; }}
  .narrative-content table {{ margin: 0.5rem 0; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
"""


def _summary_section(
    results: dict[str, StepResult],
    parents: list[ProteinCandidate],
    variants: list[ProteinCandidate],
    score_keys: list[str],
) -> str:
    n_steps = len(results)
    n_parents = len(parents)
    n_variants = len(variants)

    # Count unique mutation positions
    all_positions = set()
    for v in variants:
        for m in v.mutations:
            all_positions.add(m.position)

    return f"""
<h2>Summary</h2>
<div class="stats">
  <div class="stat-card"><div class="number">{n_steps}</div><div class="label">Steps Run</div></div>
  <div class="stat-card"><div class="number">{n_parents}</div><div class="label">Input Proteins</div></div>
  <div class="stat-card"><div class="number">{n_variants}</div><div class="label">Variants Generated</div></div>
  <div class="stat-card"><div class="number">{len(all_positions)}</div><div class="label">Positions Mutated</div></div>
  <div class="stat-card"><div class="number">{len(score_keys)}</div><div class="label">Score Types</div></div>
</div>
"""


def _candidate_table(
    variants: list[ProteinCandidate], score_keys: list[str]
) -> str:
    if not variants:
        return "<h2>Candidates</h2><p>No variant candidates generated.</p>"

    # Sort by composite_score if available, otherwise by first score
    sort_key = "composite_score" if "composite_score" in score_keys else (
        score_keys[0] if score_keys else None
    )
    if sort_key:
        variants = sorted(
            variants,
            key=lambda c: c.scores.get(sort_key, float("-inf")),
            reverse=True,
        )

    rows = []
    for rank, v in enumerate(variants[:100], 1):  # Top 100
        mut_str = ", ".join(m.label for m in v.mutations) if v.mutations else "—"
        score_cells = []
        for key in score_keys:
            val = v.scores.get(key)
            if val is not None:
                css = ""
                # Color-code key scores
                if key in ("e1_fitness", "esm1v_delta", "esmif1_delta", "composite_score"):
                    css = ' class="good"' if val > 0 else ' class="bad"' if val < -0.5 else ""
                elif key == "ddg":
                    css = ' class="good"' if val < 0 else ' class="bad"' if val > 2 else ""
                score_cells.append(f"<td{css}>{val:.4f}</td>")
            else:
                score_cells.append("<td>—</td>")

        rows.append(
            f"<tr><td>{rank}</td><td>{html.escape(v.name)}</td>"
            f'<td class="mutations">{html.escape(mut_str)}</td>'
            f"<td>{len(v.mutations)}</td>"
            f"{''.join(score_cells)}</tr>"
        )

    score_headers = "".join(f"<th>{html.escape(k)}</th>" for k in score_keys)

    return f"""
<h2>Top Candidates</h2>
<div class="overflow-x">
<table>
<thead>
<tr><th>#</th><th>Name</th><th>Mutations</th><th>Count</th>{score_headers}</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</div>
"""


def _mutation_summary(variants: list[ProteinCandidate]) -> str:
    """Summarize most frequently mutated positions."""
    if not variants:
        return ""

    position_counts: dict[int, dict[str, int]] = {}
    for v in variants:
        for m in v.mutations:
            if m.position not in position_counts:
                position_counts[m.position] = {}
            key = m.label
            position_counts[m.position][key] = (
                position_counts[m.position].get(key, 0) + 1
            )

    # Sort by total count
    sorted_positions = sorted(
        position_counts.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True,
    )[:30]

    rows = []
    for pos, mutations in sorted_positions:
        total = sum(mutations.values())
        mut_str = ", ".join(
            f"{k} ({v}x)" for k, v in sorted(
                mutations.items(), key=lambda x: -x[1]
            )
        )
        rows.append(f"<tr><td>{pos}</td><td>{total}</td><td>{mut_str}</td></tr>")

    return f"""
<h2>Most Mutated Positions</h2>
<table>
<thead><tr><th>Position</th><th>Total Variants</th><th>Mutations</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


def _analysis_narrative_section(narratives: list[str]) -> str:
    """Render the step-by-step analysis narrative."""
    if not narratives:
        return ""
    return (
        '<h2>Step-by-Step Analysis</h2>\n'
        '<p style="color:#8b949e;font-size:0.9rem;margin-bottom:1rem">'
        'Detailed explanation of what each optimization step found and proposed.</p>\n'
        + "\n".join(narratives)
    )


def _warnings_section(warnings: list[tuple[str, str]]) -> str:
    if not warnings:
        return ""

    items = []
    for step, msg in warnings:
        items.append(
            f'<li><span class="step-tag">{html.escape(step)}</span> '
            f"{html.escape(msg)}</li>"
        )

    return f"""
<h2>Warnings</h2>
<ul class="warning-list">{''.join(items)}</ul>
"""


def _step_summary(results: dict[str, StepResult]) -> str:
    rows = []
    for name, result in results.items():
        n_candidates = len(result.candidates)
        n_variants = len([c for c in result.candidates if c.parent_id is not None])
        n_warnings = len(result.warnings)
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{n_candidates}</td><td>{n_variants}</td>"
            f'<td class="{"warn" if n_warnings else ""}">{n_warnings}</td></tr>'
        )

    return f"""
<h2>Step Execution Summary</h2>
<table>
<thead><tr><th>Step</th><th>Candidates</th><th>Variants</th><th>Warnings</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
"""


def _html_footer() -> str:
    return """
</body>
</html>
"""
