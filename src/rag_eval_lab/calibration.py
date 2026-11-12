"""Judge calibration: how well the LLM judge agrees with human labels."""

from __future__ import annotations

import math

CRITERIA = ("faithfulness", "correctness")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0.0, 1.0) when there is no data."""
    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    spread = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
    return (max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator))


def confusion(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    """Counts with the judge as the predictor and the human as the reference."""
    cells = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for judge, human in pairs:
        if judge and human:
            cells["tp"] += 1
        elif judge and not human:
            cells["fp"] += 1
        elif not judge and human:
            cells["fn"] += 1
        else:
            cells["tn"] += 1
    return cells


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Agreement corrected for chance; 0.0 when one rater never varies."""
    n = len(pairs)
    if n == 0:
        return 0.0
    observed = sum(1 for judge, human in pairs if judge == human) / n
    judge_positive = sum(1 for judge, _ in pairs if judge) / n
    human_positive = sum(1 for _, human in pairs if human) / n
    expected = judge_positive * human_positive + (1 - judge_positive) * (1 - human_positive)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def agreement(
    judgments: list[dict], annotations: list[dict], criteria: tuple[str, ...] = CRITERIA
) -> dict[str, dict]:
    by_id = {j["answer_id"]: j for j in judgments if j.get("status") == "ok"}
    result: dict[str, dict] = {}
    for criterion in criteria:
        pairs: list[tuple[bool, bool]] = []
        for annotation in annotations:
            judgment = by_id.get(annotation["answer_id"])
            if judgment is None:
                continue
            judge_value, human_value = judgment.get(criterion), annotation.get(criterion)
            if isinstance(judge_value, bool) and isinstance(human_value, bool):
                pairs.append((judge_value, human_value))
        agreed = sum(1 for judge, human in pairs if judge == human)
        low, high = wilson_interval(agreed, len(pairs))
        result[criterion] = {
            "n": len(pairs),
            "agreement": agreed / len(pairs) if pairs else 0.0,
            "ci_low": low,
            "ci_high": high,
            "kappa": cohen_kappa(pairs),
            "confusion": confusion(pairs),
            "judge_positive_rate": (
                sum(1 for judge, _ in pairs if judge) / len(pairs) if pairs else 0.0
            ),
            "human_positive_rate": (
                sum(1 for _, human in pairs if human) / len(pairs) if pairs else 0.0
            ),
        }
    return result


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value of the 2x2 table [[a, b], [c, d]] (stdlib only)."""
    row1, col1, n = a + b, a + c, a + b + c + d

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(n - row1, col1 - x) / math.comb(n, col1)

    observed = prob(a)
    low, high = max(0, col1 - (n - row1)), min(row1, col1)
    # the 1e-7 slack keeps equally likely tables in the sum despite float rounding
    return min(1.0, sum(p for x in range(low, high + 1) if (p := prob(x)) <= observed * (1 + 1e-7)))
