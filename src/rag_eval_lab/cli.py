"""Command-line interface: `rel run` and `rel report`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from rag_eval_lab.config import load_config
from rag_eval_lab.experiment import run_experiment
from rag_eval_lab.report import build_report

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="YAML config")],
    out: Annotated[Path | None, typer.Option(help="Output directory")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run every configuration of the grid (already finished ones are skipped)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        cfg = load_config(config)
    except ValidationError as exc:
        typer.echo(f"Invalid config {config}:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc
    out_dir = out or Path("results") / config.stem
    records = run_experiment(cfg, out_dir)
    failed = [r for r in records if r["status"] == "error"]
    typer.echo(f"{len(records)} configuration(s) run, {len(failed)} failed -> {out_dir}")


@app.command()
def report(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build report.md and charts from results.jsonl."""
    try:
        path = build_report(results_dir)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Report written to {path}")
