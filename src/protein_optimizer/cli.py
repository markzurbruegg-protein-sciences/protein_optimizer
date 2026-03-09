"""Click CLI for the protein optimization pipeline.

Provides:
    protein-opt run           — run the full pipeline from a config YAML
    protein-opt <step-name>   — run a single step independently
    protein-opt list-steps    — show all available steps
    protein-opt rank          — aggregate and rank candidates
    protein-opt report        — generate HTML report
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from protein_optimizer.config import load_config
from protein_optimizer.io_utils import load_input
from protein_optimizer.models import StepResult
from protein_optimizer.pipeline import Pipeline, _import_all_steps, _get_step_config
from protein_optimizer.steps.base import get_step, list_steps as registry_list

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """protein-opt: Modular protein optimization pipeline."""
    _setup_logging(verbose)


# ── Full pipeline ────────────────────────────────────────────────────────


@main.command()
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to pipeline YAML config.")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True,
              help="Input FASTA or StepResult JSON.")
@click.option("-o", "--output", "output_dir", type=click.Path(), default="./results",
              help="Output directory.")
def run(config_path: str | None, input_path: str, output_dir: str) -> None:
    """Run the full optimization pipeline."""
    config = load_config(config_path)
    pipeline = Pipeline(config)
    result = pipeline.run(input_path, output_dir)
    console.print(f"\n[bold green]✓ Pipeline complete.[/] {len(result.candidates)} candidates.")
    console.print(f"  Results saved to: {output_dir}/")


# ── List steps ───────────────────────────────────────────────────────────


@main.command("list-steps")
def list_steps_cmd() -> None:
    """Show all available pipeline steps."""
    _import_all_steps()
    steps = registry_list()

    table = Table(title="Available Pipeline Steps")
    table.add_column("Name", style="cyan")
    table.add_column("Tier", justify="center")
    table.add_column("Title", style="green")
    table.add_column("Description")
    table.add_column("Requires")

    for name in sorted(steps, key=lambda n: (steps[n].tier, n)):
        cls = steps[name]
        table.add_row(
            name,
            str(cls.tier),
            cls.title,
            cls.description,
            ", ".join(cls.requires) if cls.requires else "—",
        )

    console.print(table)


# ── Generic single-step runner ────────────────────────────────────────────


@main.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.argument("step_name")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True,
              help="Input FASTA or StepResult JSON.")
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output StepResult JSON path.")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), default=None,
              help="Pipeline config YAML (step-specific section will be used).")
@click.option("--protected-residues", type=str, default=None,
              help="Comma-separated 1-based residue positions to protect from mutation.")
@click.pass_context
def step(ctx: click.Context, step_name: str, input_path: str,
         output_path: str | None, config_path: str | None,
         protected_residues: str | None) -> None:
    """Run a single pipeline step by name.

    Usage: protein-opt step <step-name> -i input.fasta -o output.json
    """
    _import_all_steps()

    config = load_config(config_path)
    step_config = _get_step_config(config, step_name)
    step_config["_global"] = config.get("global", {})
    step_config["_prior_results"] = {}

    # Override protected residues from CLI
    if protected_residues:
        step_config["_global"]["protected_residues"] = [
            int(x.strip()) for x in protected_residues.split(",") if x.strip()
        ]

    # Parse extra args as key=value config overrides
    for arg in ctx.args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
            # Try to parse as number/bool
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ("true", "yes"):
                        value = True
                    elif value.lower() in ("false", "no"):
                        value = False
            step_config[key] = value

    step_obj = get_step(step_name)
    step_input = load_input(input_path)
    result = step_obj.execute(step_input, step_config)

    # Determine output path
    if output_path is None:
        output_path = f"{step_name}_result.json"

    result.save(output_path)
    console.print(f"[bold green]✓ Step '{step_name}' complete.[/] "
                  f"{len(result.candidates)} candidates → {output_path}")


# ── Rank candidates ──────────────────────────────────────────────────────


@main.command()
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True,
              help="StepResult JSON to rank.")
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output ranked JSON.")
@click.option("-s", "--score", "score_key", default=None,
              help="Score key to rank by (default: auto-detect).")
@click.option("-n", "--top", "top_n", default=20, type=int,
              help="Number of top candidates to show.")
def rank(input_path: str, output_path: str | None, score_key: str | None, top_n: int) -> None:
    """Aggregate scores and rank candidates."""
    from protein_optimizer.scoring.aggregator import aggregate_scores, rank_candidates

    result = StepResult.load(input_path)
    aggregate_scores(result)

    key = score_key or "composite_score"
    ranked = rank_candidates(result, score_key=key)[:top_n]

    table = Table(title=f"Top {min(top_n, len(ranked))} Candidates by {key}")
    table.add_column("#", justify="right")
    table.add_column("Name", style="cyan")
    table.add_column("Mutations")
    table.add_column(key, justify="right", style="green")

    for i, c in enumerate(ranked, 1):
        mut_str = ", ".join(m.label for m in c.mutations) if c.mutations else "—"
        score_val = f"{c.scores.get(key, 0):.4f}"
        table.add_row(str(i), c.name, mut_str, score_val)

    console.print(table)

    if output_path:
        result.save(output_path)
        console.print(f"[bold green]✓ Saved ranked results to {output_path}[/]")


# ── Generate report ──────────────────────────────────────────────────────


@main.command()
@click.option("-d", "--results-dir", type=click.Path(exists=True), required=True,
              help="Directory containing step result JSON files.")
@click.option("-o", "--output", "output_path", type=click.Path(), default="report.html",
              help="Output HTML report path.")
def report(results_dir: str, output_path: str) -> None:
    """Generate an HTML report from pipeline results."""
    from protein_optimizer.reporting.html_report import generate_html_report

    results_path = Path(results_dir)
    results = {}
    for json_file in sorted(results_path.glob("*.json")):
        try:
            sr = StepResult.load(json_file)
            results[sr.step_name] = sr
        except Exception:
            pass

    if not results:
        console.print("[bold red]No valid result files found.[/]")
        return

    generate_html_report(results=results, output_path=output_path)
    console.print(f"[bold green]✓ Report generated: {output_path}[/]")
