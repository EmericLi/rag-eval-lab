import http.client
import json
import logging

from rag_eval_lab import generation_experiment
from rag_eval_lab.generation_config import ContextConfig, GenerationConfig
from rag_eval_lab.generation_experiment import (
    _append,
    _completed_ids,
    build_dataset,
    load_answers,
    run_generation,
    run_judging,
)
from rag_eval_lab.ollama_client import OllamaClient
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.prompts import ANSWER_PROMPT_VERSION, JUDGE_PROMPT_VERSION


class FakeTransport:
    def __init__(self, fail_on=(), raise_exc=None):
        self.fail_on = fail_on
        self.raise_exc = raise_exc
        self.prompts = []

    def __call__(self, method, path, payload):
        if path == "/api/tags":
            return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
        self.prompts.append(payload["prompt"])
        if any(marker in payload["prompt"] for marker in self.fail_on):
            raise self.raise_exc or TimeoutError("too slow")
        return {"response": "réponse", "eval_count": 5, "prompt_eval_count": 50}


def _config(mini_jsonl, tmp_path, **overrides) -> GenerationConfig:
    data = {
        "dataset": {"name": "jsonl", "path": str(mini_jsonl), "n_queries": 2},
        "n_passages": 2,
        "oracle_chunker": "fixed_200_50",
        "generator": {"model": "fake-model"},
        "judge": {"model": "fake-model"},
        "cache_dir": str(tmp_path / "cache"),
        "contexts": [
            {"name": "none", "kind": "empty"},
            {"name": "bm25", "kind": "retrieved", "chunker": "fixed_200_50", "retriever": "bm25"},
            {"name": "oracle", "kind": "oracle"},
        ],
    }
    data.update(overrides)
    return GenerationConfig.model_validate(data)


def _client(transport):
    return OllamaClient(transport=transport, retries=0)


def test_build_dataset_truncates_to_n_queries(mini_jsonl, tmp_path):
    dataset = build_dataset(_config(mini_jsonl, tmp_path))
    assert len(dataset.queries) == 2
    assert set(dataset.qrels) == {q.id for q in dataset.queries}


def test_run_generation_writes_one_record_per_context_and_query(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    records = run_generation(_config(mini_jsonl, tmp_path), out, client=_client(FakeTransport()))
    assert len(records) == 6
    assert {r["status"] for r in records} == {"ok"}
    assert {r["context"] for r in records} == {"none", "bm25", "oracle"}
    lines = (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    record = json.loads(lines[0])
    assert record["generator"]["digest"] == "0123456789ab"
    assert record["prompt_version"]
    assert record["answer"] == "réponse"
    assert record["reference"] == "Le volcan est un sujet important."


def test_run_generation_stores_passage_texts(mini_jsonl, tmp_path):
    records = run_generation(
        _config(mini_jsonl, tmp_path), tmp_path / "out", client=_client(FakeTransport())
    )
    empty = next(r for r in records if r["context"] == "none")
    retrieved = next(r for r in records if r["context"] == "bm25")
    assert empty["passages"] == []
    assert len(retrieved["passages"]) == 2
    assert all(p["text"] for p in retrieved["passages"])
    assert retrieved["n_passage_chars"] == sum(len(p["text"]) for p in retrieved["passages"])


def test_run_generation_resumes(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    assert run_generation(cfg, out, client=_client(FakeTransport())) == []
    assert len((out / "answers.jsonl").read_text(encoding="utf-8").splitlines()) == 6


def test_run_generation_isolates_errors_and_retries_them(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    records = run_generation(cfg, out, client=_client(FakeTransport(fail_on=("connaissances",))))
    statuses = {r["context"]: r["status"] for r in records}
    assert statuses["none"] == "error"
    assert statuses["bm25"] == "ok"
    failed = next(r for r in records if r["status"] == "error")
    assert "OllamaError" in failed["error"]
    retried = run_generation(cfg, out, client=_client(FakeTransport()))
    assert {r["context"] for r in retried} == {"none"}
    assert {r["status"] for r in retried} == {"ok"}


def _ok(answer_id):
    return {"answer_id": answer_id, "status": "ok"}


def test_completed_ids_skips_a_truncated_trailing_line(tmp_path, caplog):
    path = tmp_path / "answers.jsonl"
    path.write_text(
        json.dumps(_ok("a")) + "\n" + json.dumps(_ok("b")) + '\n{"answer_id": "c", "sta',
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        assert _completed_ids(path) == {"a", "b"}
    assert "answers.jsonl" in caplog.text
    assert "line 3" in caplog.text


def test_append_after_a_truncated_line_keeps_records_separate(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps(_ok("a")) + '\n{"answer_id": "b", "sta', encoding="utf-8")
    _append(path, _ok("c"))
    _append(path, _ok("d"))
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    parsed, broken = [], 0
    for line in text.splitlines():
        try:
            parsed.append(json.loads(line)["answer_id"])
        except json.JSONDecodeError:
            broken += 1
    assert parsed == ["a", "c", "d"]
    assert broken == 1
    assert _completed_ids(path) == {"a", "c", "d"}


def test_append_does_not_add_a_blank_line_to_a_clean_file(tmp_path):
    path = tmp_path / "answers.jsonl"
    _append(path, _ok("a"))
    _append(path, _ok("b"))
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 2
    assert "\n\n" not in text


def test_run_generation_resumes_after_a_truncated_line(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "answers.jsonl").write_text('{"answer_id": "x", "sta', encoding="utf-8")
    records = run_generation(_config(mini_jsonl, tmp_path), out, client=_client(FakeTransport()))
    assert len(records) == 6
    lines = (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7
    assert [json.loads(line)["status"] for line in lines[1:]] == ["ok"] * 6


def test_run_generation_survives_unexpected_transport_errors(mini_jsonl, tmp_path):
    transport = FakeTransport(
        fail_on=("connaissances",), raise_exc=http.client.IncompleteRead(b"par")
    )
    records = run_generation(
        _config(mini_jsonl, tmp_path), tmp_path / "out", client=_client(transport)
    )
    errors = [r for r in records if r["status"] == "error"]
    assert len(records) == 6
    assert {r["context"] for r in errors} == {"none"}
    assert all(r["error"].startswith("IncompleteRead") for r in errors)
    assert {r["status"] for r in records if r["context"] != "none"} == {"ok"}


class ExplodingContext:
    name = "boom"

    def passages(self, query):
        raise RuntimeError("retrieval blew up")


def test_run_generation_records_unexpected_exceptions_and_continues(
    mini_jsonl, tmp_path, monkeypatch, caplog
):
    real = generation_experiment.build_context_sources

    def with_exploding(cfg, dataset, only=None):
        return [ExplodingContext() if s.name == "boom" else s for s in real(cfg, dataset, only)]

    monkeypatch.setattr(generation_experiment, "build_context_sources", with_exploding)
    cfg = _config(mini_jsonl, tmp_path)
    cfg = cfg.model_copy(
        update={"contexts": [ContextConfig(name="boom", kind="empty"), *cfg.contexts]}
    )
    with caplog.at_level(logging.ERROR):
        records = run_generation(cfg, tmp_path / "out", client=_client(FakeTransport()))
    boom = [r for r in records if r["context"] == "boom"]
    assert len(boom) == 2
    assert {r["status"] for r in boom} == {"error"}
    assert all(r["error"] == "RuntimeError: retrieval blew up" for r in boom)
    assert {r["status"] for r in records if r["context"] != "boom"} == {"ok"}
    assert "Traceback" in caplog.text


def test_expected_ollama_errors_are_logged_without_a_traceback(mini_jsonl, tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        run_generation(
            _config(mini_jsonl, tmp_path),
            tmp_path / "out",
            client=_client(FakeTransport(fail_on=("connaissances",))),
        )
    assert "generation failed for none" in caplog.text
    assert "Traceback" not in caplog.text


def _count_indexing(monkeypatch):
    calls = []
    real_index = RetrievalPipeline.index

    def counting(self, documents):
        calls.append(self)
        return real_index(self, documents)

    monkeypatch.setattr(RetrievalPipeline, "index", counting)
    return calls


def test_a_complete_run_indexes_nothing_on_resume(mini_jsonl, tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    calls = _count_indexing(monkeypatch)
    transport = FakeTransport()
    assert run_generation(cfg, out, client=_client(transport)) == []
    assert calls == []
    assert transport.prompts == []


def test_only_contexts_with_pending_answers_are_indexed(mini_jsonl, tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = _config(
        mini_jsonl,
        tmp_path,
        contexts=[
            {"name": "bm25", "kind": "retrieved", "chunker": "fixed_200_50", "retriever": "bm25"},
            {
                "name": "hashing",
                "kind": "retrieved",
                "chunker": "fixed_200_50",
                "retriever": "dense:hashing",
            },
        ],
    )
    run_generation(cfg, out, client=_client(FakeTransport()))
    path = out / "answers.jsonl"
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if '"bm25"' in line]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    calls = _count_indexing(monkeypatch)
    records = run_generation(cfg, out, client=_client(FakeTransport()))
    assert len(calls) == 1
    assert {r["context"] for r in records} == {"hashing"}
    assert len(records) == 2


class JudgeTransport:
    def __init__(self, bad_json_for=(), raise_exc=None):
        self.bad_json_for = bad_json_for
        self.raise_exc = raise_exc
        self.prompts = []

    def __call__(self, method, path, payload):
        if path == "/api/tags":
            return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
        prompt = payload["prompt"]
        self.prompts.append(prompt)
        if self.raise_exc is not None:
            raise self.raise_exc
        if any(marker in prompt for marker in self.bad_json_for):
            return {"response": "je ne sais pas"}
        if "faithful" in prompt:
            return {"response": '{"faithful": true, "relevant": true, "reason": "ok"}'}
        return {"response": '{"correct": false, "reason": "non"}'}


def test_run_judging_judges_every_answer(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    records = run_judging(cfg, out, client=_client(JudgeTransport()))
    assert len(records) == 6
    assert {r["status"] for r in records} == {"ok"}
    by_context = {
        json.loads(line)["answer_id"]: json.loads(line)
        for line in (out / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
    }
    answers = {a["answer_id"]: a for a in load_answers(out)}
    for answer_id, judgment in by_context.items():
        if answers[answer_id]["context"] == "none":
            assert judgment["faithfulness"] is None
            assert judgment["relevance"] is None
        else:
            assert judgment["faithfulness"] is True
        assert judgment["correctness"] is False
        assert judgment["judge"]["digest"] == "0123456789ab"


def test_judgments_carry_the_judge_prompt_version_not_the_answers(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    answers = run_generation(cfg, out, client=_client(FakeTransport()))
    judgments = run_judging(cfg, out, client=_client(JudgeTransport()))
    assert {a["prompt_version"] for a in answers} == {ANSWER_PROMPT_VERSION}
    assert {j["prompt_version"] for j in judgments} == {JUDGE_PROMPT_VERSION}
    assert ANSWER_PROMPT_VERSION != JUDGE_PROMPT_VERSION


def test_run_judging_resumes(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    run_judging(cfg, out, client=_client(JudgeTransport()))
    assert run_judging(cfg, out, client=_client(JudgeTransport())) == []


def test_run_judging_records_parse_errors(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    records = run_judging(cfg, out, client=_client(JudgeTransport(bad_json_for=("faithful",))))
    failed = [r for r in records if r["status"] == "error"]
    assert failed, "grounding calls should have failed"
    assert failed[0]["raw_output"] == "je ne sais pas"
    assert all(
        r["status"] == "ok"
        for r in records
        if r["answer_id"] not in {f["answer_id"] for f in failed}
    )


def test_load_answers_keeps_only_successful_records(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport(fail_on=("connaissances",))))
    answers = load_answers(out)
    assert {a["context"] for a in answers} == {"bm25", "oracle"}


def test_load_answers_skips_unreadable_lines_and_keeps_the_last_record(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    lines = [
        json.dumps({"answer_id": "a", "status": "error"}),
        json.dumps({"answer_id": "a", "status": "ok", "answer": "x"}),
        '{"answer_id": "b", "sta',
        json.dumps({"answer_id": "c", "status": "ok"}),
        json.dumps({"answer_id": "c", "status": "error"}),
    ]
    (out / "answers.jsonl").write_text("\n".join(lines), encoding="utf-8")
    assert [a["answer_id"] for a in load_answers(out)] == ["a"]


def test_run_judging_retries_failed_judgments_on_resume(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    first = run_judging(cfg, out, client=_client(JudgeTransport(bad_json_for=("faithful",))))
    failed = {r["answer_id"] for r in first if r["status"] == "error"}
    retried = run_judging(cfg, out, client=_client(JudgeTransport()))
    assert {r["answer_id"] for r in retried} == failed
    assert {r["status"] for r in retried} == {"ok"}


def test_run_judging_survives_unexpected_errors(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    transport = JudgeTransport(raise_exc=http.client.IncompleteRead(b"par"))
    records = run_judging(cfg, out, client=_client(transport))
    assert len(records) == 6
    assert {r["status"] for r in records} == {"error"}
    assert all(r["error"].startswith("IncompleteRead") for r in records)
    assert all(r["raw_output"] == "" for r in records)


def test_run_judging_uses_the_reference_stored_in_the_answer(mini_jsonl, tmp_path):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_generation(cfg, out, client=_client(FakeTransport()))
    transport = JudgeTransport()
    run_judging(cfg, out, client=_client(transport))
    assert any("Le volcan est un sujet important." in p for p in transport.prompts)
