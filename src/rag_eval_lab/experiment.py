"""Run a grid of retrieval configurations and record results as JSON lines."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.chunking import parse_chunker
from rag_eval_lab.config import ExperimentConfig, RunSpec
from rag_eval_lab.data import load_benchmark
from rag_eval_lab.factory import build_reranker, build_retriever
from rag_eval_lab.metrics import evaluate
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.types import Dataset

logger = logging.getLogger(__name__)

_TRACKED_PACKAGES = ("rag-eval-lab", "sentence-transformers", "torch", "rank-bm25", "numpy")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
    }


def _completed_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "ok":
                done.add(record["config_id"])
    return done


def _append(results_path: Path, record: dict) -> None:
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_one(
    config: ExperimentConfig,
    spec: RunSpec,
    dataset: Dataset,
    cache: EmbeddingCache,
    out_dir: Path,
    config_id: str,
    env_info: dict,
) -> dict:
    pipeline = RetrievalPipeline(
        chunker=parse_chunker(spec.chunker),
        retriever=build_retriever(spec.retriever, cache),
        reranker=build_reranker(spec.reranker),
        candidates=config.candidates,
    )
    start = time.perf_counter()
    pipeline.index(dataset.documents)
    index_time_s = time.perf_counter() - start

    run: dict[str, list[str]] = {}
    latencies_ms: list[float] = []
    for i, query in enumerate(dataset.queries, start=1):
        start = time.perf_counter()
        run[query.id] = pipeline.search(query.text, config.k)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        if i % 100 == 0:
            logger.info("  %d/%d queries", i, len(dataset.queries))

    (out_dir / "runs" / f"{config_id}.json").write_text(json.dumps(run), encoding="utf-8")
    return {
        "metrics": evaluate(run, dataset.qrels, config.ks),
        "latency_ms": {
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
        },
        "index_time_s": index_time_s,
        "n_chunks": pipeline.n_chunks,
        "n_docs": len(dataset.documents),
        "n_queries": len(dataset.queries),
        "dataset": dataset.name,
        "env": env_info,
    }


def run_experiment(
    config: ExperimentConfig, out_dir: Path, dataset: Dataset | None = None
) -> list[dict]:
    out_dir = Path(out_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = _completed_ids(results_path)
    specs = config.expand()
    if dataset is None:
        dataset = load_benchmark(config.dataset)
    cache = EmbeddingCache(Path(config.cache_dir))
    env_info = environment_info()

    records = []
    for n, spec in enumerate(specs, start=1):
        config_id = config.run_id(spec)
        label = f"{spec.chunker} | {spec.retriever} | {spec.reranker}"
        if config_id in done:
            logger.info("[%d/%d] skip (already done): %s", n, len(specs), label)
            continue
        logger.info("[%d/%d] run: %s", n, len(specs), label)
        record = {"config_id": config_id, **spec.model_dump()}
        try:
            record.update(
                status="ok",
                **_run_one(config, spec, dataset, cache, out_dir, config_id, env_info),
            )
        except Exception as exc:  # one failing config must not stop the grid
            logger.exception("Configuration failed: %s", label)
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        record["timestamp"] = datetime.now(UTC).isoformat()
        _append(results_path, record)
        records.append(record)
    return records
