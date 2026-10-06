"""Markdown report for the generation benchmark, including judge calibration."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rag_eval_lab.annotation import load_annotations  # noqa: E402
from rag_eval_lab.calibration import agreement  # noqa: E402
from rag_eval_lab.generation_experiment import load_answers  # noqa: E402

CRITERIA = ("faithfulness", "relevance", "correctness")


def load_judgments(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "judgments.jsonl"
    latest: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                latest[record["answer_id"]] = record
    return [r for r in latest.values() if r.get("status") == "ok"]


def _rate(values: list[bool | None]) -> float | None:
    kept = [v for v in values if isinstance(v, bool)]
    return sum(kept) / len(kept) if kept else None


def summarise(answers: list[dict], judgments: list[dict]) -> list[dict]:
    by_id = {j["answer_id"]: j for j in judgments}
    grouped: dict[str, list[dict]] = {}
    for answer in answers:
        if answer["answer_id"] in by_id:
            grouped.setdefault(answer["context"], []).append(answer)
    rows = []
    for context, records in sorted(grouped.items()):
        verdicts = [by_id[a["answer_id"]] for a in records]
        rows.append(
            {
                "context": context,
                "n": len(records),
                **{
                    f"{criterion}_rate": _rate([v.get(criterion) for v in verdicts])
                    for criterion in CRITERIA
                },
                "latency_ms_p50": statistics.median(a["latency_ms"] for a in records),
                "mean_passage_chars": statistics.mean(a["n_passage_chars"] for a in records),
            }
        )
    return rows


def _plot(rows: list[dict], path: Path) -> None:
    contexts = [row["context"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    positions = range(len(rows))
    faithful = [(row["faithfulness_rate"] or 0.0) for row in rows]
    correct = [(row["correctness_rate"] or 0.0) for row in rows]
    ax.bar(
        [p - width / 2 for p in positions], faithful, width, label="faithfulness", color="#9aa5b1"
    )
    ax.bar([p + width / 2 for p in positions], correct, width, label="correctness", color="#2563eb")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(contexts, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Answer quality by context")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_kappa(stats: dict) -> str:
    """Kappa is mathematically undefined when either rater never varies."""
    if stats["judge_positive_rate"] in (0.0, 1.0) or stats["human_positive_rate"] in (0.0, 1.0):
        return "undefined"
    return f"{stats['kappa']:.3f}"


def build_generation_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    judgments = load_judgments(out_dir)
    if not judgments:
        raise ValueError(f"No successful judgments in {out_dir / 'judgments.jsonl'}")
    answers = load_answers(out_dir)
    rows = summarise(answers, judgments)
    annotations = load_annotations(out_dir)
    calibration = agreement(judgments, annotations)

    lines = [
        "# Generation benchmark report",
        "",
        f"- Contexts: {len(rows)}",
        f"- Judged answers: {len(judgments)}",
        f"- Human annotations: {len(annotations)}",
        "",
        "| Context | n | faithfulness | relevance | correctness | latency p50 (ms) "
        "| mean passage chars |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['context']} | {row['n']} | {_format(row['faithfulness_rate'])} | "
            f"{_format(row['relevance_rate'])} | {_format(row['correctness_rate'])} | "
            f"{row['latency_ms_p50']:.0f} | {row['mean_passage_chars']:.0f} |"
        )
    lines += [
        "",
        "![Answer quality by context](generation.png)",
        "",
        "## Calibration — judge against human labels",
        "",
        "| Criterion | n | agreement | 95% CI | kappa | judge positive | human positive "
        "| tp/fp/fn/tn |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for criterion, stats in calibration.items():
        cells = stats["confusion"]
        lines.append(
            f"| {criterion} | {stats['n']} | {stats['agreement']:.3f} | "
            f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}] | {_format_kappa(stats)} | "
            f"{stats['judge_positive_rate']:.3f} | {stats['human_positive_rate']:.3f} | "
            f"{cells['tp']}/{cells['fp']}/{cells['fn']}/{cells['tn']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `faithfulness` and `relevance` are not judged for the `none` context: without passages,",
        "  grounding is undefined. Those cells read `n/a`.",
        "- Agreement counts only answers that carry both a human label and a judge verdict;",
        "  `?` answers during annotation are stored as null and excluded.",
        "- Confusion is reported judge-versus-human: `fp` means the judge said true where the",
        "  human said false (an indulgent judge), `fn` the opposite (a severe one).",
        "- Kappa reads `undefined` when the judge or the human gave the same label to every",
        "  answer (positive rate of exactly 0 or 1): chance agreement is then total and kappa",
        "  has no meaningful value. This is not the same as a kappa of 0, which would mean the",
        "  judge is no better than chance.",
        "",
    ]
    _plot(rows, out_dir / "generation.png")
    report_path = out_dir / "report-generation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
