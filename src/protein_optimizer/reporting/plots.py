"""Plotting utilities for protein optimization results.

Generates matplotlib/seaborn figures for:
- Score distributions
- Mutation heatmaps
- Score correlation matrices
- Position-wise analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from protein_optimizer.models import ProteinCandidate, StepResult

logger = logging.getLogger(__name__)


def plot_score_distribution(
    candidates: list[ProteinCandidate],
    score_key: str = "composite_score",
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Any:
    """Plot histogram of a score across candidates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return None

    values = [c.scores[score_key] for c in candidates if score_key in c.scores]
    if not values:
        logger.warning(f"No candidates have score '{score_key}'")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30, color="#58a6ff", edgecolor="#0d1117", alpha=0.8)
    ax.set_xlabel(score_key)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Distribution of {score_key}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")

    return fig


def plot_score_scatter(
    candidates: list[ProteinCandidate],
    x_score: str,
    y_score: str,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Any:
    """Scatter plot of two scores against each other."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return None

    x_vals, y_vals, names = [], [], []
    for c in candidates:
        if x_score in c.scores and y_score in c.scores:
            x_vals.append(c.scores[x_score])
            y_vals.append(c.scores[y_score])
            names.append(c.name)

    if not x_vals:
        logger.warning(f"No candidates have both '{x_score}' and '{y_score}'")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_vals, y_vals, c="#58a6ff", s=30, alpha=0.7, edgecolors="#0d1117")
    ax.set_xlabel(x_score)
    ax.set_ylabel(y_score)
    ax.set_title(title or f"{y_score} vs {x_score}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")

    return fig


def plot_mutation_heatmap(
    candidates: list[ProteinCandidate],
    score_key: str = "composite_score",
    output_path: str | Path | None = None,
    max_positions: int = 50,
) -> Any:
    """Heatmap of score by position and mutation type."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return None

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

    # Collect position → mutation → score
    data: dict[int, dict[str, float]] = {}
    for c in candidates:
        if score_key not in c.scores or not c.mutations:
            continue
        if len(c.mutations) != 1:
            continue  # only single mutants
        m = c.mutations[0]
        if m.position not in data:
            data[m.position] = {}
        data[m.position][m.mut] = c.scores[score_key]

    if not data:
        return None

    # Select top N positions by max score
    top_positions = sorted(
        data.keys(),
        key=lambda p: max(data[p].values()),
        reverse=True,
    )[:max_positions]

    matrix = np.full((len(AA_ORDER), len(top_positions)), np.nan)
    for j, pos in enumerate(top_positions):
        for i, aa in enumerate(AA_ORDER):
            if aa in data.get(pos, {}):
                matrix[i, j] = data[pos][aa]

    fig, ax = plt.subplots(figsize=(max(8, len(top_positions) * 0.4), 8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xticks(range(len(top_positions)))
    ax.set_xticklabels(top_positions, rotation=90, fontsize=7)
    ax.set_yticks(range(len(AA_ORDER)))
    ax.set_yticklabels(list(AA_ORDER), fontsize=8)
    ax.set_xlabel("Position")
    ax.set_ylabel("Mutation")
    ax.set_title(f"Mutation Heatmap ({score_key})")
    plt.colorbar(im, ax=ax, shrink=0.8, label=score_key)

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")

    return fig


def plot_score_correlation(
    candidates: list[ProteinCandidate],
    output_path: str | Path | None = None,
) -> Any:
    """Correlation matrix between all score types."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return None

    # Find all score keys
    score_keys = set()
    for c in candidates:
        score_keys.update(c.scores.keys())
    score_keys = sorted(score_keys)

    if len(score_keys) < 2:
        return None

    # Build matrix
    n = len(score_keys)
    corr = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = score_keys[i], score_keys[j]
            pairs = [
                (c.scores[ki], c.scores[kj])
                for c in candidates
                if ki in c.scores and kj in c.scores
            ]
            if len(pairs) >= 3:
                xs, ys = zip(*pairs)
                r = np.corrcoef(xs, ys)[0, 1]
                corr[i, j] = corr[j, i] = r if not np.isnan(r) else 0

    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(score_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(score_keys, fontsize=8)
    ax.set_title("Score Correlations")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate values
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr[i,j]) > 0.5 else "black")

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {output_path}")

    return fig
