"""Information-retrieval metrics computed from ranked document ids."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rag_eval_lab.types import Qrels

Run = dict[str, list[str]]
"""Retrieval output: query_id -> ranked doc ids (best first)."""


def _relevant_ids(relevant: dict[str, int]) -> set[str]:
    return {doc_id for doc_id, gain in relevant.items() if gain > 0}


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    rel = _relevant_ids(relevant)
    if not rel:
        return 0.0
    return len(rel.intersection(ranked[:k])) / len(rel)


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if relevant.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(run: Run, qrels: Qrels, ks: Sequence[int]) -> dict[str, float]:
    """Average recall/MRR/nDCG at each k over queries with at least one relevant doc."""
    query_ids = [qid for qid, rel in qrels.items() if _relevant_ids(rel)]
    totals: dict[str, float] = {}
    for k in ks:
        for name in ("recall", "mrr", "ndcg"):
            totals[f"{name}@{k}"] = 0.0
    for qid in query_ids:
        ranked = run.get(qid, [])
        relevant = qrels[qid]
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
            totals[f"mrr@{k}"] += mrr_at_k(ranked, relevant, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(ranked, relevant, k)
    n = len(query_ids)
    return {key: (value / n if n else 0.0) for key, value in totals.items()}
