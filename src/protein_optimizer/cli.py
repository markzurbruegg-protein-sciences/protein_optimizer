"""Click CLI for the protein optimization pipeline.

Provides:
    protopt -i my_protein.fasta         — run the full pipeline (shorthand)
    protopt run <fasta>                 — run the full pipeline (explicit)
    protopt report <fasta>             — (re)generate the HTML report
    protopt step <name> -i …           — run a single step independently
    protopt list-steps                 — show all available steps
    protopt rank -i …                  — aggregate and rank candidates

All outputs (results dir + report) are placed next to the input FASTA.
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
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)


# ── Shared helpers ───────────────────────────────────────────────────────


def _resolve_paths(fasta: str) -> tuple[Path, Path, Path]:
    """Return (fasta_path, results_dir, report_path) from a FASTA path."""
    fasta_path = Path(fasta).resolve()
    name = fasta_path.stem
    results_dir = fasta_path.parent / f"{name}_results"
    report_path = fasta_path.parent / f"{name}_report.html"
    return fasta_path, results_dir, report_path


def _auto_detect_pdb(results_dir: Path, results: dict[str, StepResult]) -> str | None:
    """Find the best PDB from structure prediction or structures/ dir."""
    # 1. From predict_structure metadata
    struct_result = results.get("predict_structure")
    if struct_result:
        for c in struct_result.candidates:
            sp = c.metadata.get("structure_path", "")
            if sp and Path(sp).exists():
                return sp

    # 2. From structures/ dir (prefer main protein PDB over rfdiff outputs)
    struct_dir = results_dir / "structures"
    if struct_dir.is_dir():
        pdbs = sorted(struct_dir.glob("*.pdb"))
        main_pdbs = [p for p in pdbs if "rfdiff" not in p.name.lower()]
        chosen = main_pdbs[0] if main_pdbs else (pdbs[0] if pdbs else None)
        if chosen:
            return str(chosen)

    return None


def _load_results(results_dir: Path) -> dict[str, StepResult]:
    """Load all StepResult JSONs from a directory."""
    results: dict[str, StepResult] = {}
    for json_file in sorted(results_dir.glob("*.json")):
        try:
            sr = StepResult.load(json_file)
            results[sr.step_name] = sr
        except Exception:
            pass
    return results


def _derive_title(results: dict[str, StepResult]) -> str:
    """Derive a report title from the protein name."""
    for key in reversed(list(results.keys())):
        for c in results[key].candidates:
            if c.parent_id is None:
                return f"{c.name} — Protein Optimization Report"
    return "Protein Optimization Report"


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-i", "--input", "input_fasta", type=click.Path(exists=True), default=None,
              metavar="FASTA", help="Input FASTA file — runs the full pipeline directly.")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), default=None,
              metavar="YAML", help="Pipeline YAML config (default: configs/full_pipeline.yaml).")
@click.pass_context
def main(ctx: click.Context, verbose: bool, input_fasta: str | None, config_path: str | None) -> None:
    """protopt — Modular protein optimization pipeline.

    \b
    Quick start (run full pipeline):
        protopt -i my_protein.fasta
        protopt -i my_protein.fasta -c configs/custom.yaml

    \b
    Subcommands:
        protopt run <fasta>
        protopt report <fasta>
        protopt step <name> -i <file>
        protopt list-steps
    """
    _setup_logging(verbose)
    # If -i was given and no subcommand, run the full pipeline immediately.
    if input_fasta is not None and ctx.invoked_subcommand is None:
        ctx.invoke(run, input_fasta=input_fasta, config_path=config_path)


# ── Full pipeline ────────────────────────────────────────────────────────


@main.command()
@click.argument("input_fasta", type=click.Path(exists=True))
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), default=None,
              help="Pipeline YAML config (default: configs/full_pipeline.yaml).")
def run(input_fasta: str, config_path: str | None) -> None:
    """Run the full optimization pipeline.

    \b
    Usage:  protopt run my_protein.fasta
            protopt run my_protein.fasta -c configs/custom.yaml

    Results are saved in <name>_results/ and the report as
    <name>_report.html, both next to the input FASTA.
    """
    fasta_path, output_dir, report_path = _resolve_paths(input_fasta)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default config
    if config_path is None:
        pkg_root = Path(__file__).resolve().parents[2]
        default_cfg = pkg_root / "configs" / "full_pipeline.yaml"
        if default_cfg.exists():
            config_path = str(default_cfg)

    config = load_config(config_path)
    config.setdefault("global", {})["output_dir"] = str(output_dir)

    pipeline = Pipeline(config)
    result = pipeline.run(str(fasta_path), str(output_dir))

    console.print(f"\n[bold green]✓ Pipeline complete.[/] {len(result.candidates)} candidates.")
    console.print(f"  Results: {output_dir}")
    console.print(f"  Report:  {report_path}")


# ── Generate / regenerate report ─────────────────────────────────────────


@main.command()
@click.argument("input_fasta", type=click.Path(exists=True))
@click.option("--pdb", "pdb_path", type=click.Path(exists=True), default=None,
              help="PDB file for 3D viewer (auto-detected if omitted).")
def report(input_fasta: str, pdb_path: str | None) -> None:
    """Generate (or regenerate) the HTML report.

    \b
    Usage:  protopt report my_protein.fasta

    Reads results from <name>_results/ next to the FASTA and writes
    <name>_report.html in the same directory.
    """
    from protein_optimizer.reporting.report_v2 import generate_report_v2

    fasta_path, results_dir, report_path = _resolve_paths(input_fasta)

    if not results_dir.is_dir():
        console.print(f"[bold red]Results directory not found: {results_dir}[/]")
        return

    results = _load_results(results_dir)
    if not results:
        console.print(f"[bold red]No valid result files in {results_dir}[/]")
        return

    if pdb_path is None:
        pdb_path = _auto_detect_pdb(results_dir, results)
        if pdb_path:
            console.print(f"  Auto-detected PDB: {pdb_path}")

    generate_report_v2(
        results=results,
        output_path=report_path,
        pdb_path=pdb_path,
        title=_derive_title(results),
    )
    console.print(f"[bold green]✓ Report generated: {report_path}[/]")


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
            name, str(cls.tier), cls.title, cls.description,
            ", ".join(cls.requires) if cls.requires else "—",
        )

    console.print(table)


# ── Single-step runner ───────────────────────────────────────────────────


@main.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.argument("step_name")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True,
              help="Input FASTA or StepResult JSON.")
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output StepResult JSON path.")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), default=None,
              help="Pipeline config YAML.")
@click.pass_context
def step(ctx: click.Context, step_name: str, input_path: str,
         output_path: str | None, config_path: str | None) -> None:
    """Run a single pipeline step by name.

    \b
    Usage: protopt step cysteine_scan -i my_protein.fasta
    """
    _import_all_steps()

    config = load_config(config_path)
    step_config = _get_step_config(config, step_name)
    step_config["_global"] = config.get("global", {})
    step_config["_prior_results"] = {}

    # Parse extra args as key=value config overrides
    for arg in ctx.args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lstrip("-")
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

    if output_path is None:
        output_path = f"{step_name}_result.json"

    result.save(output_path)
    console.print(f"[bold green]✓ Step '{step_name}' complete.[/] "
                  f"{len(result.candidates)} candidates → {output_path}")


# ── Rank candidates ─────────────────────────────────────────────────────


@main.command()
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True,
              help="StepResult JSON to rank.")
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Output ranked JSON.")
@click.option("-s", "--score", "score_key", default=None,
              help="Score key to rank by (default: composite_score).")
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
        table.add_row(str(i), c.name, mut_str, f"{c.scores.get(key, 0):.4f}")

    console.print(table)

    if output_path:
        result.save(output_path)
        console.print(f"[bold green]✓ Saved ranked results to {output_path}[/]")

