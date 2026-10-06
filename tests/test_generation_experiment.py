import json

from rag_eval_lab.generation_config import GenerationConfig
from rag_eval_lab.generation_experiment import build_dataset, run_generation
from rag_eval_lab.ollama_client import OllamaClient


class FakeTransport:
    def __init__(self, fail_on=()):
        self.fail_on = fail_on
        self.prompts = []

    def __call__(self, method, path, payload):
        if path == "/api/tags":
            return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
        self.prompts.append(payload["prompt"])
        if any(marker in payload["prompt"] for marker in self.fail_on):
            raise TimeoutError("too slow")
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
