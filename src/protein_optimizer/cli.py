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
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    # Force stdout and flush after every log line
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    # Make stdout unbuffered for nohup/redirect
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """protein-opt: Modular protein optimization pipeline."""
    _setup_logging(verbose)


# ── Full pipeline ────────────────────────────────────────────────────────


@main.command()
@click.argument("input_fasta", type=click.Path(exists=True))
@click.option("-o", "--output", "output_report", type=click.Path(), default=None,
              help="Output HTML report path (default: <name>_report.html next to FASTA).")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), default=None,
              help="Pipeline YAML config (default: configs/full_pipeline.yaml).")
def run(input_fasta: str, output_report: str | None, config_path: str | None) -> None:
    """Run the full optimization pipeline.

    \b
    Usage:  protopt run proteins/my_protein.fasta
            protopt run proteins/my_protein.fasta -o my_report.html
            protopt run proteins/my_protein.fasta -c configs/custom.yaml
    """
    fasta_path = Path(input_fasta).resolve()
    protein_name = fasta_path.stem

    # Resolve output dir: <fasta_dir>/<name>_results/
    output_dir = fasta_path.parent / f"{protein_name}_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve report path
    if output_report is None:
        output_report = str(fasta_path.parent / f"{protein_name}_report.html")

    # Default config
    if config_path is None:
        pkg_root = Path(__file__).resolve().parents[2]
        default_cfg = pkg_root / "configs" / "full_pipeline.yaml"
        if default_cfg.exists():
            config_path = str(default_cfg)

    config = load_config(config_path)

    # Override output_dir in config to match our resolved path
    config.setdefault("global", {})["output_dir"] = str(output_dir)

    pipeline = Pipeline(config)
    result = pipeline.run(str(fasta_path), str(output_dir))

    console.print(f"\n[bold green]✓ Pipeline complete.[/] {len(result.candidates)} candidates.")
    console.print(f"  Results: {output_dir}/")
    console.print(f"  Report:  {output_report}")


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
@click.option("--pdb", "pdb_path", type=click.Path(), default=None,
              help="Optional PDB/CIF file for 3D structure viewer in v2 report.")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), default=None,
              help="Pipeline YAML config used for the run.")
def report(results_dir: str, output_path: str, pdb_path: str | None,
           config_path: str | None) -> None:
    """Generate an HTML report from pipeline results.

    Uses the v2 enhanced report (3D viewer, metrics, exec summary) with
    automatic fallback to the v1 basic report.

    \b
    Usage:  protein-opt report -d run_proteins/jcDRM_results/ -o report.html
            protein-opt report -d results/ --pdb structures/protein.pdb
    """
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

    config = load_config(config_path)

    # Auto-detect PDB if not provided: look in structures/ sub-dir
    if pdb_path is None:
        struct_dir = results_path / "structures"
        for ext in ("*.pdb", "*.PDB", "*.cif", "*.CIF"):
            pdbs = list(struct_dir.glob(ext)) if struct_dir.is_dir() else []
            if pdbs:
                pdb_path = str(pdbs[0])
                break

    # Derive a report title from the parent candidate name
    final_result = list(results.values())[-1] if results else None
    parent_name = next(
        (c.name for c in (final_result.candidates if final_result else []) if c.parent_id is None),
        "",
    )
    report_title = f"{parent_name} — Protein Optimization Report" if parent_name else "Protein Optimization Report"

    # Try v2 first, fall back to v1
    try:
        from protein_optimizer.reporting.report_v2 import generate_report_v2
        generate_report_v2(
            results=results,
            output_path=output_path,
            pdb_path=pdb_path,
            title=report_title,
            config=config,
        )
        console.print(f"[bold green]✓ Report (v2) generated: {output_path}[/]")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning("v2 report failed, falling back to v1", exc_info=True)
        console.print(f"[yellow]v2 report failed ({e}), falling back to v1…[/]")
        from protein_optimizer.reporting.html_report import generate_html_report
        generate_html_report(results=results, output_path=output_path, config=config)
        console.print(f"[bold green]✓ Report (v1) generated: {output_path}[/]")
