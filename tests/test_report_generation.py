import json

import matplotlib.pyplot as plt
import pytest

from rag_eval_lab.report_generation import (
    _draw,
    build_generation_report,
    load_judgments,
    summarise,
)


def _answer(answer_id, context, chars=100, latency=1000.0):
    return {
        "answer_id": answer_id,
        "context": context,
        "query_id": answer_id,
        "question": "q",
        "answer": "a",
        "status": "ok",
        "passages": [] if context == "none" else [{"id": "d#0", "text": "x" * chars}],
        "n_passage_chars": 0 if context == "none" else chars,
        "latency_ms": latency,
    }


def _judgment(answer_id, faithful, correct, relevant=True):
    return {
        "answer_id": answer_id,
        "status": "ok",
        "faithfulness": faithful,
        "relevance": relevant,
        "correctness": correct,
        "latency_ms": 500.0,
        "rationales": {"correctness": "parce que"},
    }


def _write(tmp_path, name, records):
    with (tmp_path / name).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_load_judgments_keeps_last_ok_record(tmp_path):
    _write(
        tmp_path,
        "judgments.jsonl",
        [
            {"answer_id": "a", "status": "error", "error": "boom"},
            _judgment("a", True, True),
            {"answer_id": "b", "status": "error", "error": "boom"},
        ],
    )
    judgments = load_judgments(tmp_path)
    assert [j["answer_id"] for j in judgments] == ["a"]


def test_summarise_computes_rates_per_context():
    answers = [_answer("a1", "bm25"), _answer("a2", "bm25"), _answer("n1", "none")]
    judgments = [
        _judgment("a1", True, True),
        _judgment("a2", False, True),
        _judgment("n1", None, False, relevant=None),
    ]
    rows = {row["context"]: row for row in summarise(answers, judgments)}
    assert rows["bm25"]["n"] == 2
    assert rows["bm25"]["faithfulness_rate"] == pytest.approx(0.5)
    assert rows["bm25"]["correctness_rate"] == pytest.approx(1.0)
    assert rows["none"]["faithfulness_rate"] is None
    assert rows["none"]["correctness_rate"] == pytest.approx(0.0)


def test_build_report_writes_markdown_and_chart(tmp_path):
    _write(
        tmp_path,
        "answers.jsonl",
        [_answer("a1", "bm25"), _answer("o1", "oracle"), _answer("n1", "none")],
    )
    _write(
        tmp_path,
        "judgments.jsonl",
        [
            _judgment("a1", True, True),
            _judgment("o1", True, True),
            _judgment("n1", None, False, relevant=None),
        ],
    )
    _write(
        tmp_path,
        "annotations.jsonl",
        [{"answer_id": "a1", "faithfulness": True, "correctness": False}],
    )
    path = build_generation_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "report-generation.md"
    assert "oracle" in text and "bm25" in text
    assert "Calibration" in text
    assert "kappa" in text.lower()
    assert (tmp_path / "generation.png").exists()


def test_build_report_without_judgments_raises(tmp_path):
    _write(tmp_path, "answers.jsonl", [_answer("a1", "bm25")])
    _write(tmp_path, "judgments.jsonl", [{"answer_id": "a1", "status": "error", "error": "x"}])
    with pytest.raises(ValueError, match="No successful judgments"):
        build_generation_report(tmp_path)


def _calibration_rows(text):
    rows = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and cells[0] in ("faithfulness", "relevance", "correctness") and len(cells) == 8:
            rows[cells[0]] = cells
    return rows


def test_report_prints_undefined_kappa_when_a_rater_never_varies(tmp_path):
    ids = ["a1", "a2", "a3", "a4"]
    _write(tmp_path, "answers.jsonl", [_answer(i, "bm25") for i in ids])
    # Judge varies on faithfulness, human never does (always True): kappa is undefined.
    # Both rate correctness with variation: kappa is a real number.
    judge_faithful = [True, False, True, False]
    judge_correct = [True, True, False, False]
    _write(
        tmp_path,
        "judgments.jsonl",
        [_judgment(i, f, c) for i, f, c in zip(ids, judge_faithful, judge_correct, strict=True)],
    )
    human_correct = [True, False, False, False]
    _write(
        tmp_path,
        "annotations.jsonl",
        [
            {"answer_id": i, "faithfulness": True, "correctness": c}
            for i, c in zip(ids, human_correct, strict=True)
        ],
    )
    text = build_generation_report(tmp_path).read_text(encoding="utf-8")
    rows = _calibration_rows(text)
    assert rows["faithfulness"][4] == "undefined"
    assert rows["correctness"][4] != "undefined"
    assert float(rows["correctness"][4]) == pytest.approx(0.5)
    notes = text.split("## Notes")[1]
    assert "no better than chance" in notes


def test_summarise_rate_ignores_null_verdicts():
    answers = [_answer(f"a{i}", "bm25") for i in range(3)]
    judgments = [
        _judgment("a0", True, True),
        _judgment("a1", False, True),
        _judgment("a2", None, True),
    ]
    row = summarise(answers, judgments)[0]
    # Three answers, one null, two booleans of which one is true: 1/2, not 1/3.
    assert row["n"] == 3
    assert row["faithfulness_rate"] == pytest.approx(0.5)


def test_summarise_excludes_answers_without_judgment():
    answers = [_answer("a1", "bm25"), _answer("a2", "bm25"), _answer("a3", "bm25")]
    judgments = [_judgment("a1", True, True)]
    rows = summarise(answers, judgments)
    assert len(rows) == 1
    assert rows[0]["n"] == 1
    for criterion in ("faithfulness", "relevance", "correctness"):
        assert rows[0][f"{criterion}_rate"] == pytest.approx(1.0)
    assert rows[0]["latency_ms_p50"] == pytest.approx(1000.0)

    # A context whose answers have no judgment at all does not appear.
    assert summarise([_answer("z", "oracle")], judgments) == []


def test_report_without_annotations_does_not_read_as_a_failing_judge(tmp_path):
    _write(tmp_path, "answers.jsonl", [_answer("a1", "bm25")])
    _write(tmp_path, "judgments.jsonl", [_judgment("a1", True, True)])
    text = build_generation_report(tmp_path).read_text(encoding="utf-8")
    rows = _calibration_rows(text)
    for criterion in ("faithfulness", "correctness"):
        _, n, agree, ci, kappa, judge_pos, human_pos, confusion = rows[criterion]
        assert (n, agree, ci, kappa, judge_pos, human_pos) == ("0", *["n/a"] * 5)
        assert confusion == "0/0/0/0"
    calibration = text.split("## Calibration")[1].split("## Notes")[0]
    assert "0.000" not in calibration
    assert "not been calibrated yet" in calibration
    assert "rel annotate" in calibration


def test_report_with_annotations_has_no_not_calibrated_line(tmp_path):
    _write(tmp_path, "answers.jsonl", [_answer("a1", "bm25")])
    _write(tmp_path, "judgments.jsonl", [_judgment("a1", True, True)])
    _write(
        tmp_path,
        "annotations.jsonl",
        [{"answer_id": "a1", "faithfulness": True, "correctness": True}],
    )
    text = build_generation_report(tmp_path).read_text(encoding="utf-8")
    assert "not been calibrated yet" not in text
    assert _calibration_rows(text)["correctness"][2] == "1.000"


def test_chart_marks_a_non_applicable_rate_instead_of_drawing_a_zero_bar():
    rows = summarise(
        [_answer("a1", "bm25"), _answer("n1", "none")],
        [_judgment("a1", True, True), _judgment("n1", None, False, relevant=None)],
    )
    fig, ax = plt.subplots()
    try:
        _draw(rows, ax)
        # 2 faithfulness bars would be drawn if None were coerced to 0; only 1 + 2 correctness.
        assert len(ax.patches) == 3
        assert [t.get_text() for t in ax.texts] == ["n/a"]
        faithfulness_bars = ax.containers[0]
        assert len(faithfulness_bars) == 1
    finally:
        plt.close(fig)


def test_build_report_chart_survives_a_none_rate(tmp_path):
    _write(tmp_path, "answers.jsonl", [_answer("a1", "bm25"), _answer("n1", "none")])
    _write(
        tmp_path,
        "judgments.jsonl",
        [_judgment("a1", True, True), _judgment("n1", None, False, relevant=None)],
    )
    build_generation_report(tmp_path)
    assert (tmp_path / "generation.png").stat().st_size > 0
