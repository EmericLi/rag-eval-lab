import json

import pytest

from rag_eval_lab.report import build_report, load_results, pareto_front


def _record(cid, retriever, ndcg, p50, status="ok"):
    record = {
        "config_id": cid,
        "status": status,
        "chunker": "none",
        "retriever": retriever,
        "reranker": "none",
    }
    if status == "ok":
        record.update(
            metrics={"ndcg@10": ndcg, "mrr@10": ndcg, "recall@5": ndcg, "recall@10": ndcg},
            latency_ms={"p50": p50, "p95": p50 * 2},
            index_time_s=1.5,
            n_chunks=10,
            n_queries=5,
        )
    else:
        record["error"] = "RuntimeError: boom"
    return record


def _write(results_dir, records):
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_pareto_front():
    points = [(10.0, 0.5), (20.0, 0.6), (30.0, 0.55), (5.0, 0.3)]
    assert pareto_front(points) == {0, 1, 3}


def test_load_results_keeps_last_ok_record(tmp_path):
    _write(
        tmp_path,
        [
            _record("a", "bm25", 0.0, 1.0, status="error"),
            _record("a", "bm25", 0.4, 1.0),
            _record("b", "dense:x", 0.6, 30.0),
            _record("c", "dense:y", 0.0, 0.0, status="error"),
        ],
    )
    results = load_results(tmp_path)
    assert sorted(r["config_id"] for r in results) == ["a", "b"]


def test_build_report(tmp_path):
    _write(
        tmp_path,
        [
            _record("a", "bm25", 0.4, 2.0),
            _record("b", "dense:x", 0.6, 30.0),
            _record("c", "hybrid:bm25+x", 0.55, 60.0),
        ],
    )
    path = build_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert path.name == "report.md"
    assert text.index("dense:x") < text.index("hybrid:bm25+x") < text.index("| bm25")
    assert "ndcg@10" in text
    assert (tmp_path / "pareto.png").exists()
    assert (tmp_path / "ranking.png").exists()


def test_build_report_without_results_raises(tmp_path):
    _write(tmp_path, [_record("a", "bm25", 0.0, 1.0, status="error")])
    with pytest.raises(ValueError, match="No successful"):
        build_report(tmp_path)
