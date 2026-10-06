import json

import pytest

from rag_eval_lab.report_generation import build_generation_report, load_judgments, summarise


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
    assert "undefined" in text.split("## Notes")[1]
