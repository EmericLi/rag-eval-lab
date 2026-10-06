import json

import pytest

from rag_eval_lab.annotation import (
    append_annotation,
    load_annotations,
    run_annotation_session,
    sample_for_annotation,
)


def _answers(n_per_context=4, contexts=("none", "bm25", "oracle")):
    return [
        {
            "answer_id": f"{context}-{i}",
            "context": context,
            "query_id": f"q{i}",
            "question": f"question {i}",
            "answer": f"réponse {i}",
            "passages": [] if context == "none" else [{"id": "d1#0", "text": "extrait"}],
            "status": "ok",
        }
        for context in contexts
        for i in range(n_per_context)
    ]


def test_sample_is_balanced_across_contexts():
    sample = sample_for_annotation(_answers(), n=6, seed=7)
    assert len(sample) == 6
    counts = {}
    for record in sample:
        counts[record["context"]] = counts.get(record["context"], 0) + 1
    assert counts == {"none": 2, "bm25": 2, "oracle": 2}


def test_sample_is_deterministic():
    first = [a["answer_id"] for a in sample_for_annotation(_answers(), n=5, seed=7)]
    second = [a["answer_id"] for a in sample_for_annotation(_answers(), n=5, seed=7)]
    assert first == second
    assert first != [a["answer_id"] for a in sample_for_annotation(_answers(), n=5, seed=8)]


def test_sample_larger_than_available_returns_everything():
    assert len(sample_for_annotation(_answers(n_per_context=2), n=99, seed=1)) == 6


def test_append_and_load_annotations(tmp_path):
    append_annotation(tmp_path, {"answer_id": "a", "faithfulness": True, "correctness": False})
    append_annotation(tmp_path, {"answer_id": "b", "faithfulness": None, "correctness": True})
    loaded = load_annotations(tmp_path)
    assert [r["answer_id"] for r in loaded] == ["a", "b"]
    assert loaded[1]["faithfulness"] is None


def test_load_annotations_keeps_only_the_last_record_per_answer(tmp_path):
    append_annotation(tmp_path, {"answer_id": "a", "faithfulness": True, "correctness": False})
    append_annotation(tmp_path, {"answer_id": "b", "faithfulness": None, "correctness": True})
    append_annotation(tmp_path, {"answer_id": "a", "faithfulness": False, "correctness": True})
    loaded = {r["answer_id"]: r for r in load_annotations(tmp_path)}
    assert len(load_annotations(tmp_path)) == 2
    assert loaded["a"]["faithfulness"] is False and loaded["a"]["correctness"] is True
    assert loaded["b"]["correctness"] is True


def _write_answers(tmp_path, answers):
    with (tmp_path / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for answer in answers:
            handle.write(json.dumps(answer, ensure_ascii=False) + "\n")


def test_session_records_answers_and_skips_faithfulness_without_passages(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("none", "bm25")))
    # sorted contexts: bm25 first (faithful=o, correct=n), then none (correct=o)
    replies = iter(["o", "n", "o"])
    added = run_annotation_session(
        tmp_path, n=2, seed=1, read=lambda _: next(replies), write=lambda _: None
    )
    assert added == 2
    records = {r["answer_id"]: r for r in load_annotations(tmp_path)}
    none_record = next(r for r in records.values() if r["answer_id"].startswith("none"))
    bm25_record = next(r for r in records.values() if r["answer_id"].startswith("bm25"))
    assert none_record["faithfulness"] is None and none_record["correctness"] is True
    assert bm25_record["faithfulness"] is True and bm25_record["correctness"] is False


def test_session_question_mark_records_null(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("bm25",)))
    replies = iter(["?", "?"])
    run_annotation_session(
        tmp_path, n=1, seed=1, read=lambda _: next(replies), write=lambda _: None
    )
    record = load_annotations(tmp_path)[0]
    assert record["faithfulness"] is None and record["correctness"] is None


def test_session_quit_stops_without_recording(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=2, contexts=("bm25",)))
    replies = iter(["o", "o", "q"])
    added = run_annotation_session(
        tmp_path, n=2, seed=1, read=lambda _: next(replies), write=lambda _: None
    )
    assert added == 1
    assert len(load_annotations(tmp_path)) == 1


def test_session_resumes_and_never_shows_the_judge(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("bm25",)))
    (tmp_path / "judgments.jsonl").write_text(
        json.dumps({"answer_id": "bm25-0", "status": "ok", "faithfulness": True}) + "\n",
        encoding="utf-8",
    )
    shown: list[str] = []
    run_annotation_session(
        tmp_path, n=1, seed=1, read=lambda _: "o", write=lambda text: shown.append(text)
    )
    assert "juge" not in " ".join(shown).lower()
    assert (
        run_annotation_session(tmp_path, n=1, seed=1, read=lambda _: "o", write=lambda _: None) == 0
    )


def test_session_rejects_an_unreadable_reply(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("bm25",)))
    replies = iter(["x", "o", "n"])
    run_annotation_session(
        tmp_path, n=1, seed=1, read=lambda _: next(replies), write=lambda _: None
    )
    record = load_annotations(tmp_path)[0]
    assert record["faithfulness"] is True and record["correctness"] is False


def test_session_without_answers_raises(tmp_path):
    with pytest.raises(ValueError, match="no answers"):
        run_annotation_session(tmp_path, n=1, seed=1, read=lambda _: "o", write=lambda _: None)


NL = chr(10)


def _long_answer(total_length=3421):
    answer = _answers(n_per_context=1, contexts=("bm25",))[0]
    head = "a" * 1190
    tail = "FIN-DU-TEXTE".ljust(total_length - len(head), "z")
    answer["passages"] = [{"id": "d1#0", "text": head + tail}]
    return answer


def _run(tmp_path, replies, n=1):
    shown: list[str] = []
    feed = iter(replies)
    run_annotation_session(tmp_path, n=n, seed=1, read=lambda _: next(feed), write=shown.append)
    return NL.join(shown)


def test_long_passage_is_truncated_with_a_visible_marker(tmp_path):
    _write_answers(tmp_path, [_long_answer()])
    output = _run(tmp_path, ["o", "o"])
    assert "a" * 1190 in output
    assert "FIN-DU-TEXTE" not in output
    assert "tronqué, 3421 caractères au total" in output
    assert "tape v pour tout afficher" in output


def test_view_reply_reprints_the_full_passage_then_records_the_next_label(tmp_path):
    _write_answers(tmp_path, [_long_answer()])
    output = _run(tmp_path, ["v", "n", "o"])
    assert "FIN-DU-TEXTE" in output
    assert output.count("Question : question 0") == 2  # once truncated, once in full
    record = load_annotations(tmp_path)[0]
    assert record["faithfulness"] is False and record["correctness"] is True


def test_view_reply_is_never_recorded_as_a_label(tmp_path):
    _write_answers(tmp_path, [_long_answer()])
    _run(tmp_path, ["v", "v", "o", "v", "n"])
    record = load_annotations(tmp_path)[0]
    assert record["faithfulness"] is True and record["correctness"] is False
    assert all(record[k] in (True, False, None) for k in ("faithfulness", "correctness"))


def test_full_display_applies_to_the_current_item_only(tmp_path):
    second = _long_answer()
    second["answer_id"] = "bm25-1"
    _write_answers(tmp_path, [_long_answer(), second])
    output = _run(tmp_path, ["v", "o", "o", "o", "o"], n=2)
    first_item, second_item = output.split("--- 2/2")
    assert "FIN-DU-TEXTE" in first_item
    assert "FIN-DU-TEXTE" not in second_item


def test_short_passage_is_shown_without_marker(tmp_path):
    _write_answers(tmp_path, _answers(n_per_context=1, contexts=("bm25",)))
    output = _run(tmp_path, ["o", "o"])
    assert "extrait" in output
    assert "tronqué" not in output


def test_load_annotations_skips_a_torn_trailing_line(tmp_path):
    append_annotation(tmp_path, {"answer_id": "a", "faithfulness": True, "correctness": False})
    append_annotation(tmp_path, {"answer_id": "a", "faithfulness": False, "correctness": True})
    append_annotation(tmp_path, {"answer_id": "b", "faithfulness": None, "correctness": True})
    with (tmp_path / "annotations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"answer_id": "c", "faithf')
    loaded = {r["answer_id"]: r for r in load_annotations(tmp_path)}
    assert set(loaded) == {"a", "b"}
    assert loaded["a"]["faithfulness"] is False


def test_load_annotations_skips_lines_without_answer_id(tmp_path):
    append_annotation(tmp_path, {"answer_id": "a", "faithfulness": True, "correctness": True})
    with (tmp_path / "annotations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"faithfulness": true}' + NL + "[1, 2]" + NL)
    assert [r["answer_id"] for r in load_annotations(tmp_path)] == ["a"]
