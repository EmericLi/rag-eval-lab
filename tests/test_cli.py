import yaml
from typer.testing import CliRunner

from rag_eval_lab.cli import app

runner = CliRunner()


def _write_config(tmp_path, mini_jsonl, retrievers):
    config = {
        "dataset": {"name": "jsonl", "path": str(mini_jsonl)},
        "k": 5,
        "ks": [1, 5],
        "candidates": 20,
        "cache_dir": str(tmp_path / "cache"),
        "grid": {"chunker": ["none", "fixed_200_50"], "retriever": retrievers},
    }
    path = tmp_path / "mini.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_cli_run_and_report_end_to_end(tmp_path, mini_jsonl):
    config = _write_config(tmp_path, mini_jsonl, ["bm25", "dense:hashing", "hybrid:bm25+hashing"])
    out = tmp_path / "results"
    result = runner.invoke(app, ["run", str(config), "--out", str(out)])
    assert result.exit_code == 0, result.output
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert '"status": "error"' not in "".join(lines)

    result = runner.invoke(app, ["report", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "report.md").exists()


def test_cli_rejects_invalid_config(tmp_path, mini_jsonl):
    config = _write_config(tmp_path, mini_jsonl, ["dense:nope"])
    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code == 1
    assert "Unknown encoder" in result.output


def test_cli_report_without_results(tmp_path):
    (tmp_path / "results.jsonl").write_text("", encoding="utf-8")
    result = runner.invoke(app, ["report", str(tmp_path)])
    assert result.exit_code == 1
