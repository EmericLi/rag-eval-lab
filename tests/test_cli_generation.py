import yaml
from typer.testing import CliRunner

from rag_eval_lab.cli import app

runner = CliRunner()


def _write_config(tmp_path, mini_jsonl):
    config = {
        "dataset": {"name": "jsonl", "path": str(mini_jsonl), "n_queries": 2},
        "n_passages": 2,
        "oracle_chunker": "fixed_200_50",
        "generator": {"model": "fake-model"},
        "judge": {"model": "fake-model"},
        "cache_dir": str(tmp_path / "cache"),
        "contexts": [
            {"name": "none", "kind": "empty"},
            {"name": "bm25", "kind": "retrieved", "chunker": "fixed_200_50", "retriever": "bm25"},
        ],
    }
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_generate_judge_annotate_report(tmp_path, mini_jsonl, monkeypatch):
    from rag_eval_lab import cli

    class FakeTransport:
        def __call__(self, method, path, payload):
            if path == "/api/tags":
                return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
            prompt = payload["prompt"]
            if "faithful" in prompt:
                return {"response": '{"faithful": true, "relevant": true, "reason": "ok"}'}
            if "correct" in prompt:
                return {"response": '{"correct": true, "reason": "ok"}'}
            return {"response": "réponse", "eval_count": 3, "prompt_eval_count": 10}

    monkeypatch.setattr(
        cli,
        "build_client",
        lambda timeout_s: cli.OllamaClient(transport=FakeTransport(), retries=0),
    )

    config = _write_config(tmp_path, mini_jsonl)
    out = tmp_path / "results"
    assert runner.invoke(app, ["generate", str(config), "--out", str(out)]).exit_code == 0
    assert len((out / "answers.jsonl").read_text(encoding="utf-8").splitlines()) == 4

    result = runner.invoke(app, ["judge", str(out), str(config)])
    assert result.exit_code == 0, result.output
    assert len((out / "judgments.jsonl").read_text(encoding="utf-8").splitlines()) == 4

    result = runner.invoke(app, ["annotate", str(out), "--n", "2"], input="o\nn\no\n")
    assert result.exit_code == 0, result.output
    assert len((out / "annotations.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    result = runner.invoke(app, ["report-gen", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "report-generation.md").exists()


def test_generate_reports_a_missing_ollama(tmp_path, mini_jsonl, monkeypatch):
    from rag_eval_lab import cli

    def explode(timeout_s):
        raise cli.OllamaError(
            "cannot reach Ollama at http://127.0.0.1:11434. Start it with `ollama serve`."
        )

    monkeypatch.setattr(cli, "build_client", explode)
    result = runner.invoke(app, ["generate", str(_write_config(tmp_path, mini_jsonl))])
    assert result.exit_code == 1
    assert "ollama serve" in result.output
    assert "Traceback" not in result.output


def test_report_gen_without_judgments(tmp_path):
    (tmp_path / "judgments.jsonl").write_text("", encoding="utf-8")
    result = runner.invoke(app, ["report-gen", str(tmp_path)])
    assert result.exit_code == 1


def test_generate_exits_1_when_all_items_fail(tmp_path, mini_jsonl, monkeypatch):
    from rag_eval_lab import cli

    class AlwaysFailTransport:
        def __call__(self, method, path, payload):
            if path == "/api/tags":
                return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
            raise Exception("boom")

    monkeypatch.setattr(
        cli,
        "build_client",
        lambda timeout_s: cli.OllamaClient(transport=AlwaysFailTransport(), retries=0),
    )

    config = _write_config(tmp_path, mini_jsonl)
    out = tmp_path / "results"
    result = runner.invoke(app, ["generate", str(config), "--out", str(out)])
    assert result.exit_code == 1
    assert "4 answer(s) attempted, 0 ok, 4 failed" in result.output
    assert "retry" in result.output.lower()


def test_generate_exits_0_when_some_items_succeed(tmp_path, mini_jsonl, monkeypatch):
    from rag_eval_lab import cli

    attempt_count = 0

    class SometimesFailTransport:
        def __call__(self, method, path, payload):
            nonlocal attempt_count
            if path == "/api/tags":
                return {"models": [{"name": "fake-model", "digest": "0123456789abcdef"}]}
            attempt_count += 1
            if attempt_count == 1:
                raise Exception("boom")
            return {"response": "réponse", "eval_count": 3, "prompt_eval_count": 10}

    monkeypatch.setattr(
        cli,
        "build_client",
        lambda timeout_s: cli.OllamaClient(transport=SometimesFailTransport(), retries=0),
    )

    config = _write_config(tmp_path, mini_jsonl)
    out = tmp_path / "results"
    result = runner.invoke(app, ["generate", str(config), "--out", str(out)])
    assert result.exit_code == 0
    assert "attempted" in result.output
    assert "ok" in result.output
    assert "failed" in result.output
    assert "retry" in result.output.lower()
