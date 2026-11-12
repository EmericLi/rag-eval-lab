"""Markdown report for the generation benchmark, including judge calibration."""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rag_eval_lab.annotation import load_annotations  # noqa: E402
from rag_eval_lab.calibration import agreement, fisher_exact, wilson_interval  # noqa: E402
from rag_eval_lab.generation_experiment import load_answers  # noqa: E402

logger = logging.getLogger(__name__)

CRITERIA = ("faithfulness", "relevance", "correctness")


def load_judgments(out_dir: Path) -> list[dict]:
    """Last record per answer_id, successful ones only; unreadable lines are skipped."""
    path = Path(out_dir) / "judgments.jsonl"
    latest: dict[str, dict] = {}
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if "answer_id" in record:
                    latest[record["answer_id"]] = record
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning("skipping unreadable line %d of %s", number, path)
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


def _draw(rows: list[dict], ax) -> None:
    """Draw the bars; a criterion that does not apply (None) gets an `n/a` mark, not a bar."""
    contexts = [row["context"] for row in rows]
    width = 0.35
    positions = list(range(len(rows)))
    faith_x = [p - width / 2 for p, row in zip(positions, rows, strict=True)]
    corr_x = [p + width / 2 for p, row in zip(positions, rows, strict=True)]
    faith_bars = [(x, row["faithfulness_rate"]) for x, row in zip(faith_x, rows, strict=True)]
    corr_bars = [(x, row["correctness_rate"]) for x, row in zip(corr_x, rows, strict=True)]
    for series, label, color in (
        (faith_bars, "faithfulness", "#9aa5b1"),
        (corr_bars, "correctness", "#2563eb"),
    ):
        drawn = [(x, rate) for x, rate in series if rate is not None]
        ax.bar([x for x, _ in drawn], [rate for _, rate in drawn], width, label=label, color=color)
        for x, rate in series:
            if rate is None:
                ax.text(x, 0.02, "n/a", ha="center", va="bottom", color="#52606d", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(contexts, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.set_title("Answer quality by context")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()


def _plot(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _draw(rows, ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_kappa(stats: dict) -> str:
    """Kappa is mathematically undefined when either rater never varies."""
    if stats["n"] == 0:
        return "n/a"
    if stats["judge_positive_rate"] in (0.0, 1.0) or stats["human_positive_rate"] in (0.0, 1.0):
        return "undefined"
    return f"{stats['kappa']:.3f}"


def _human_rate_cell(values: list[bool]) -> str:
    if not values:
        return "n/a"
    low, high = wilson_interval(sum(values), len(values))
    return f"{sum(values) / len(values):.3f} [{low:.3f}, {high:.3f}]"


def _human_by_context(answers: list[dict], annotations: list[dict]) -> list[str]:
    """Markdown lines for the per-context human rates and the headline Fisher test."""
    context_of = {a["answer_id"]: a["context"] for a in answers}
    labels: dict[str, dict[str, list[bool]]] = {}
    counts: dict[str, int] = {}
    for annotation in annotations:
        context = context_of.get(annotation["answer_id"])
        if context is None:
            continue
        counts[context] = counts.get(context, 0) + 1
        bucket = labels.setdefault(context, {"faithfulness": [], "correctness": []})
        for criterion in bucket:
            if isinstance(annotation.get(criterion), bool):
                bucket[criterion].append(annotation[criterion])
    lines = ["", "## Human labels by context", ""]
    if not labels:
        return lines + ["Not calibrated yet: no human annotations to break down by context."]
    lines += [
        "| Context | annotated | human correctness (95% CI) | human faithfulness (95% CI) |",
        "|---|---|---|---|",
    ]
    for context in sorted(labels):
        corr, faith = labels[context]["correctness"], labels[context]["faithfulness"]
        lines.append(
            f"| {context} | {counts[context]} | {_human_rate_cell(corr)} | "
            f"{_human_rate_cell(faith)} |"
        )
    rated = {c: v["correctness"] for c, v in labels.items() if v["correctness"]}
    if len(rated) >= 2:
        best = max(rated, key=lambda c: sum(rated[c]) / len(rated[c]))
        worst = min(rated, key=lambda c: sum(rated[c]) / len(rated[c]))
        top, bottom = rated[best], rated[worst]
        p = fisher_exact(sum(top), len(top) - sum(top), sum(bottom), len(bottom) - sum(bottom))
        lines += [
            "",
            f"Fisher exact test, human correctness, `{best}` (highest) vs `{worst}` (lowest): "
            f"p = {p:.3f}, two-sided. Unadjusted for multiple comparisons, and the pair was "
            "selected after seeing the data, so read it as descriptive.",
        ]
    return lines


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
        if stats["n"] == 0:
            # Nothing measured: the zeros returned by agreement() must not read as a bad judge.
            agreement_cell = judge_cell = human_cell = ci_cell = "n/a"
        else:
            agreement_cell = f"{stats['agreement']:.3f}"
            judge_cell = f"{stats['judge_positive_rate']:.3f}"
            human_cell = f"{stats['human_positive_rate']:.3f}"
            ci_cell = f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]"
        lines.append(
            f"| {criterion} | {stats['n']} | {agreement_cell} | {ci_cell} | "
            f"{_format_kappa(stats)} | {judge_cell} | {human_cell} | "
            f"{cells['tp']}/{cells['fp']}/{cells['fn']}/{cells['tn']} |"
        )
    if not annotations:
        lines += [
            "",
            "The judge has not been calibrated yet: no human annotations. "
            "Run `rel annotate` to label answers, then regenerate this report.",
        ]
    lines += _human_by_context(answers, annotations)
    lines += [
        "",
        "## Notes",
        "",
        "- `faithfulness` and `relevance` are not judged for the `none` context: without passages,",
        "  grounding is undefined. Those cells read `n/a`.",
        "- Agreement counts only answers that carry both a human label and a judge verdict;",
        "  `?` answers during annotation are stored as null and excluded.",
        "- `annotated` counts the labelled answers of a context; a rate excludes the `?` ones,",
        "  so its own denominator can be smaller than that count.",
        "- Confusion is reported judge-versus-human: `fp` means the judge said true where the",
        "  human said false (an indulgent judge), `fn` the opposite (a severe one).",
        "- Kappa reads `undefined` when the judge or the human gave the same label to every",
        "  answer (positive rate of exactly 0 or 1): chance agreement is then total and kappa",
        "  has no meaningful value. This is not the same as a kappa of 0, which would mean the",
        "  judge is no better than chance.",
        "- A calibration row with n = 0 reads `n/a` throughout: nothing was measured, which is",
        "  not the same as a judge that never agrees.",
        "",
    ]
    _plot(rows, out_dir / "generation.png")
    report_path = out_dir / "report-generation.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
