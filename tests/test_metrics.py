import math

import pytest

from rag_eval_lab.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k

RANKED = ["a", "b", "c"]
RELEVANT = {"b": 1, "d": 1}


def test_recall_at_k():
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert recall_at_k(RANKED, RELEVANT, 2) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 3) == 0.5


def test_recall_ignores_zero_gain_and_empty_relevant():
    assert recall_at_k(RANKED, {"a": 0}, 3) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(RANKED, RELEVANT, 3) == 0.5
    assert mrr_at_k(RANKED, RELEVANT, 1) == 0.0
    assert mrr_at_k(RANKED, {"a": 1}, 3) == 1.0


def test_ndcg_at_k_hand_computed():
    # DCG = 1/log2(3); IDCG = 1 + 1/log2(3)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(expected)
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(0.386853, abs=1e-6)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["b", "d", "a"], RELEVANT, 3) == pytest.approx(1.0)


def test_evaluate_averages_over_queries():
    run = {"q1": ["a", "b"], "q2": ["x", "y"]}
    qrels = {"q1": {"a": 1}, "q2": {"y": 1}, "q3": {"z": 0}}
    scores = evaluate(run, qrels, ks=[1, 2])
    # q3 has no relevant doc -> ignored; q1 perfect, q2 hit at rank 2
    assert scores["recall@1"] == pytest.approx(0.5)
    assert scores["recall@2"] == pytest.approx(1.0)
    assert scores["mrr@2"] == pytest.approx((1.0 + 0.5) / 2)
    assert set(scores) == {"recall@1", "recall@2", "mrr@1", "mrr@2", "ndcg@1", "ndcg@2"}


def test_evaluate_missing_query_counts_as_empty():
    scores = evaluate({}, {"q1": {"a": 1}}, ks=[10])
    assert scores["recall@10"] == 0.0
