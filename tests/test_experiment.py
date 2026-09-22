import json

from rag_eval_lab import experiment
from rag_eval_lab.config import ExperimentConfig
from rag_eval_lab.experiment import environment_info, run_experiment


def _config(mini_jsonl, tmp_path, retrievers=("bm25", "dense:hashing")) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "dataset": {"name": "jsonl", "path": str(mini_jsonl)},
            "k": 5,
            "ks": [1, 5],
            "candidates": 20,
            "cache_dir": str(tmp_path / "cache"),
            "grid": {"chunker": ["none"], "retriever": list(retrievers)},
        }
    )


def _lines(out_dir):
    return (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()


def test_run_experiment_writes_results(tmp_path, mini_jsonl):
    out = tmp_path / "out"
    records = run_experiment(_config(mini_jsonl, tmp_path), out)
    assert [r["status"] for r in records] == ["ok", "ok"]
    assert len(_lines(out)) == 2
    record = records[0]
    assert record["metrics"]["recall@5"] >= 0.8
    assert set(record["latency_ms"]) == {"p50", "p95"}
    assert record["n_queries"] == 5
    assert record["n_chunks"] == 20
    run = json.loads((out / "runs" / f"{record['config_id']}.json").read_text(encoding="utf-8"))
    assert set(run) == {"q0", "q1", "q2", "q3", "q4"}


def test_run_experiment_resumes(tmp_path, mini_jsonl):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    run_experiment(cfg, out)
    assert run_experiment(cfg, out) == []
    assert len(_lines(out)) == 2


def test_run_experiment_records_errors_and_retries(tmp_path, mini_jsonl, monkeypatch):
    out = tmp_path / "out"
    cfg = _config(mini_jsonl, tmp_path)
    real_build = experiment.build_retriever

    def flaky(spec, cache=None):
        if spec == "bm25":
            raise RuntimeError("boom")
        return real_build(spec, cache)

    monkeypatch.setattr(experiment, "build_retriever", flaky)
    records = run_experiment(cfg, out)
    statuses = {r["retriever"]: r["status"] for r in records}
    assert statuses == {"bm25": "error", "dense:hashing": "ok"}
    error = next(r for r in records if r["status"] == "error")
    assert error["error"] == "RuntimeError: boom"

    monkeypatch.undo()
    retried = run_experiment(cfg, out)
    assert [(r["retriever"], r["status"]) for r in retried] == [("bm25", "ok")]


def test_environment_info_keys():
    info = environment_info()
    assert {"python", "platform", "processor", "git_commit", "packages"} <= set(info)
