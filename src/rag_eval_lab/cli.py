"""Command-line interface: `rel run`, `rel report`, `rel generate`, `rel judge`,
`rel annotate` and `rel report-gen`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from rag_eval_lab.annotation import run_annotation_session
from rag_eval_lab.config import load_config
from rag_eval_lab.experiment import run_experiment
from rag_eval_lab.generation_config import load_generation_config
from rag_eval_lab.generation_experiment import run_generation, run_judging
from rag_eval_lab.ollama_client import OllamaClient, OllamaError
from rag_eval_lab.report import build_report
from rag_eval_lab.report_generation import build_generation_report

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
    except (ValidationError, yaml.YAMLError) as exc:
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


def build_client(timeout_s: float) -> OllamaClient:
    """Seam for tests: every generation command builds its client here."""
    return OllamaClient(timeout_s=timeout_s)


def _load_generation_config(path: Path):
    try:
        return load_generation_config(path)
    except (ValidationError, yaml.YAMLError) as exc:
        typer.echo(f"Invalid config {path}:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def generate(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="YAML config")],
    out: Annotated[Path | None, typer.Option(help="Output directory")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Generate one answer per (context, query); already generated ones are skipped."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = _load_generation_config(config)
    out_dir = out or Path("results") / config.stem
    try:
        records = run_generation(cfg, out_dir, client=build_client(cfg.generator.timeout_s))
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    failed = [r for r in records if r["status"] == "error"]
    succeeded = len(records) - len(failed)
    typer.echo(
        f"{len(records)} answer(s) attempted, {succeeded} ok, {len(failed)} failed -> {out_dir}"
    )
    if failed:
        typer.echo("Re-run the same command to retry only the failed items.")
    if len(failed) == len(records):
        raise typer.Exit(code=1)


@app.command()
def judge(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Judge every generated answer (already judged ones are skipped)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = _load_generation_config(config)
    try:
        records = run_judging(cfg, results_dir, client=build_client(cfg.judge.timeout_s))
    except OllamaError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    failed = [r for r in records if r["status"] == "error"]
    succeeded = len(records) - len(failed)
    typer.echo(
        f"{len(records)} answer(s) attempted, {succeeded} ok, {len(failed)} failed -> {results_dir}"
    )
    if failed:
        typer.echo("Re-run the same command to retry only the failed items.")
    if len(failed) == len(records):
        raise typer.Exit(code=1)


@app.command()
def annotate(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    n: Annotated[int, typer.Option(help="How many answers to label")] = 60,
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 7,
) -> None:
    """Label sampled answers yourself; the judge's verdict is never shown."""
    try:
        added = run_annotation_session(results_dir, n=n, seed=seed)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{added} annotation(s) added -> {results_dir / 'annotations.jsonl'}")


@app.command("report-gen")
def report_gen(
    results_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build the generation report (quality per context + judge calibration)."""
    try:
        path = build_generation_report(results_dir)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Report written to {path}")
