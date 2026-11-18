"""Reproduce the refusal split of docs/results-v2.md (Finding 1) from results/v2.

    uv run python scripts/refusals.py [results/v2]

A refusal is an answer that says the extracts (or the model) cannot answer. Two patterns
match that wording but are not refusals, and are excluded by hand:
- "Non, je n'ai pas de jeu de tableau(x) de multiplication": a literal answer to
  "avez-vous un jeu de tableau de multiplication ?";
- "Il n'y a pas d'extraits qui traitent spécifiquement de ... Cependant, on peut en
  déduire ...": the answer goes on to answer.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from rag_eval_lab.annotation import load_annotations
from rag_eval_lab.calibration import agreement
from rag_eval_lab.generation_experiment import load_answers
from rag_eval_lab.report_generation import load_judgments

REFUSAL = re.compile(
    r"je ne (peux|suis) pas|je n'ai (pas|trouvé aucun)|n'ai trouvé aucun"
    r"|ne contien\w+ (pas|aucun)|aucune information|il n'y a pas d'extrait"
    r"|ne permett\w* pas|n'offrent aucun|pas (basée|fondée) sur les extraits|désolé",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    if "jeu de tableau" in answer or answer.startswith("Il n'y a pas d'extraits"):
        return False
    return bool(REFUSAL.search(answer))


def main(out_dir: Path) -> None:
    answers = load_answers(out_dir)
    judgments = load_judgments(out_dir)
    annotations = load_annotations(out_dir)
    refusals = {a["answer_id"] for a in answers if is_refusal(a["answer"])}
    by_context = Counter(a["context"] for a in answers if a["answer_id"] in refusals)
    print(f"{len(refusals)} refusals out of {len(answers)} answers: {dict(by_context)}")

    rows = {
        "all": annotations,
        "refusals only": [x for x in annotations if x["answer_id"] in refusals],
        "substantive answers only": [x for x in annotations if x["answer_id"] not in refusals],
    }
    split = {}
    for name, subset in rows.items():
        stats = agreement(judgments, subset, ("faithfulness",))["faithfulness"]
        c = stats["confusion"]
        split[name] = (stats["n"], round(stats["kappa"], 3), c["tp"], c["fp"], c["fn"], c["tn"])
        print(
            f"{name}: n={stats['n']} kappa={stats['kappa']:.3f} "
            f"tp/fp/fn/tn={c['tp']}/{c['fp']}/{c['fn']}/{c['tn']}"
        )

    if Path(out_dir).resolve() == Path("results/v2").resolve():
        # self-check: the published numbers of docs/results-v2.md
        assert len(refusals) == 29, len(refusals)
        assert by_context == {"oracle": 9, "bge-m3": 7, "bm25": 6, "e5-small": 5, "none": 2}
        assert split["refusals only"] == (9, -0.235, 2, 1, 6, 0), split
        assert split["substantive answers only"] == (39, -0.147, 15, 17, 5, 2), split
        print("matches docs/results-v2.md")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/v2"))
