import pytest

from rag_eval_lab.calibration import (
    agreement,
    cohen_kappa,
    confusion,
    fisher_exact,
    wilson_interval,
)


def test_wilson_interval_brackets_the_rate():
    low, high = wilson_interval(45, 60)
    assert low < 0.75 < high
    assert 0.0 <= low < high <= 1.0


def test_wilson_interval_of_an_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_confusion_counts_each_cell():
    pairs = [(True, True), (True, False), (False, True), (False, False), (True, True)]
    assert confusion(pairs) == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([(True, True), (False, False)]) == pytest.approx(1.0)
    # judge always says True, human is split: raw agreement 0.5, kappa 0
    pairs = [(True, True), (True, False)]
    assert cohen_kappa(pairs) == pytest.approx(0.0)


def test_cohen_kappa_hand_computed():
    # 10 pairs: tp=6, fp=2, fn=1, tn=1 -> po=0.7
    # judge True 8/10, human True 7/10 -> pe = .8*.7 + .2*.3 = .62 -> kappa = .08/.38
    pairs = [(True, True)] * 6 + [(True, False)] * 2 + [(False, True)] + [(False, False)]
    assert cohen_kappa(pairs) == pytest.approx(0.08 / 0.38, abs=1e-6)


def test_cohen_kappa_of_a_single_class_is_zero():
    assert cohen_kappa([(True, True), (True, True)]) == 0.0


def test_agreement_skips_nulls_and_reports_both_criteria():
    judgments = [
        {"answer_id": "a", "status": "ok", "faithfulness": True, "correctness": True},
        {"answer_id": "b", "status": "ok", "faithfulness": None, "correctness": False},
        {"answer_id": "c", "status": "ok", "faithfulness": False, "correctness": True},
        {"answer_id": "d", "status": "error"},
    ]
    annotations = [
        {"answer_id": "a", "faithfulness": True, "correctness": False},
        {"answer_id": "b", "faithfulness": True, "correctness": False},
        {"answer_id": "c", "faithfulness": False, "correctness": None},
        {"answer_id": "d", "faithfulness": True, "correctness": True},
    ]
    result = agreement(judgments, annotations)
    assert result["faithfulness"]["n"] == 2
    assert result["faithfulness"]["agreement"] == pytest.approx(1.0)
    assert result["correctness"]["n"] == 2
    assert result["correctness"]["agreement"] == pytest.approx(0.5)
    assert result["correctness"]["confusion"] == {"tp": 0, "fp": 1, "fn": 0, "tn": 1}
    assert 0.0 <= result["faithfulness"]["ci_low"] <= result["faithfulness"]["ci_high"] <= 1.0


def test_agreement_without_overlap_returns_zero_sized_entries():
    result = agreement([], [])
    assert result["faithfulness"]["n"] == 0
    assert result["correctness"]["kappa"] == 0.0


def test_fisher_exact_known_values():
    assert fisher_exact(7, 5, 1, 9) == pytest.approx(0.031, abs=5e-4)
    assert fisher_exact(7, 5, 1, 8) == pytest.approx(0.067, abs=5e-4)
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
