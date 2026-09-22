"""Turn results.jsonl into a Markdown report with charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_results(results_dir: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    path = Path(results_dir) / "results.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["config_id"]] = record
    return [r for r in latest.values() if r.get("status") == "ok"]


def pareto_front(points: list[tuple[float, float]]) -> set[int]:
    """Indices of (latency, quality) points not dominated by any other point."""
    front = set()
    for i, (lat_i, q_i) in enumerate(points):
        dominated = any(
            lat_j <= lat_i and q_j >= q_i and (lat_j < lat_i or q_j > q_i)
            for j, (lat_j, q_j) in enumerate(points)
            if j != i
        )
        if not dominated:
            front.add(i)
    return front


def _metric_sort_key(name: str) -> tuple[str, int]:
    base, _, k = name.partition("@")
    return base, int(k) if k.isdigit() else 0


def _label(record: dict) -> str:
    return f"{record['chunker']} | {record['retriever']} | {record['reranker']}"


def _plot_pareto(results: list[dict], primary: str, front: set[int], path: Path) -> None:
    latencies = [r["latency_ms"]["p50"] for r in results]
    scores = [r["metrics"][primary] for r in results]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(latencies, scores, color="#9aa5b1", label="configuration")
    front_sorted = sorted(front, key=lambda i: latencies[i])
    ax.plot(
        [latencies[i] for i in front_sorted],
        [scores[i] for i in front_sorted],
        "o-",
        color="#2563eb",
        label="Pareto front",
    )
    for i in front_sorted:
        ax.annotate(
            _label(results[i]),
            (latencies[i], scores[i]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    if min(latencies) > 0 and max(latencies) / min(latencies) > 20:
        ax.set_xscale("log")
    ax.set_xlabel("Query latency p50 (ms)")
    ax.set_ylabel(primary)
    ax.set_title(f"Quality vs latency ({primary})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ranking(results: list[dict], primary: str, path: Path) -> None:
    ordered = sorted(results, key=lambda r: r["metrics"][primary])
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(ordered))))
    ax.barh([_label(r) for r in ordered], [r["metrics"][primary] for r in ordered], color="#2563eb")
    ax.set_xlabel(primary)
    ax.set_title(f"Configurations ranked by {primary}")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    results = load_results(results_dir)
    if not results:
        raise ValueError(f"No successful results in {results_dir / 'results.jsonl'}")

    metric_names = sorted(results[0]["metrics"], key=_metric_sort_key)
    ndcg_names = [m for m in metric_names if m.startswith("ndcg@")]
    primary = max(ndcg_names, key=_metric_sort_key)
    results.sort(key=lambda r: r["metrics"][primary], reverse=True)
    front = pareto_front([(r["latency_ms"]["p50"], r["metrics"][primary]) for r in results])

    header = [
        "Rank",
        "Chunker",
        "Retriever",
        "Reranker",
        *metric_names,
        "p50 (ms)",
        "p95 (ms)",
        "Index (s)",
        "Chunks",
        "Pareto",
    ]
    lines = [
        "# Retrieval benchmark report",
        "",
        f"- Configurations: {len(results)}",
        f"- Queries: {results[0].get('n_queries', '?')}",
        f"- Primary metric: `{primary}`",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for i, r in enumerate(results):
        cells = [
            str(i + 1),
            r["chunker"],
            r["retriever"],
            r["reranker"],
            *(f"{r['metrics'][m]:.3f}" for m in metric_names),
            f"{r['latency_ms']['p50']:.1f}",
            f"{r['latency_ms']['p95']:.1f}",
            f"{r['index_time_s']:.1f}",
            str(r["n_chunks"]),
            "★" if i in front else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "![Quality vs latency](pareto.png)",
        "",
        "![Ranking](ranking.png)",
        "",
        "## Notes",
        "",
        "- Latency is measured per query on CPU, including query encoding and reranking.",
        "- Index time includes embedding computation only when the embedding cache was cold.",
        "- ★ marks configurations on the quality/latency Pareto front.",
        "",
    ]

    _plot_pareto(results, primary, front, results_dir / "pareto.png")
    _plot_ranking(results, primary, results_dir / "ranking.png")
    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
