import http.client
import json
import logging

from rag_eval_lab import generation_experiment
from rag_eval_lab.generation_config import ContextConfig, GenerationConfig
from rag_eval_lab.generation_experiment import (
    _append,
    _completed_ids,
    build_dataset,
    run_generation,
)
from rag_eval_lab.ollama_client import OllamaClient


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

    def with_exploding(cfg, dataset):
        return [ExplodingContext() if s.name == "boom" else s for s in real(cfg, dataset)]

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
