import pytest
import yaml
from pydantic import ValidationError

from rag_eval_lab import generation_config, prompts
from rag_eval_lab.generation_config import GenerationConfig, load_generation_config

BASE = {
    "dataset": {"name": "alloprof", "n_queries": 80, "seed": 42},
    "n_passages": 5,
    "generator": {"model": "llama3.2:latest"},
    "judge": {"model": "llama3.1:8B"},
    "contexts": [
        {"name": "none", "kind": "empty"},
        {"name": "bm25", "kind": "retrieved", "chunker": "none", "retriever": "bm25"},
        {"name": "oracle", "kind": "oracle"},
    ],
    "annotation": {"n": 60, "seed": 7},
}


def _config(**overrides) -> GenerationConfig:
    return GenerationConfig.model_validate({**BASE, **overrides})


def test_defaults_and_accessors():
    cfg = _config()
    assert cfg.n_passages == 5
    assert cfg.generator.max_tokens == 300
    assert cfg.generator.num_ctx == 8192
    assert cfg.judge.num_ctx == 8192
    assert cfg.judge.timeout_s == 300.0
    assert cfg.oracle_chunker == "fixed_1000_200"
    assert [c.name for c in cfg.contexts] == ["none", "bm25", "oracle"]


def test_answer_id_is_stable_and_discriminates():
    cfg = _config()
    first = cfg.answer_id("bm25", "q1")
    assert first == _config().answer_id("bm25", "q1")
    assert len(first) == 12
    assert first != cfg.answer_id("oracle", "q1")
    assert first != cfg.answer_id("bm25", "q2")
    assert first != _config(generator={"model": "llama3.1:8B"}).answer_id("bm25", "q1")


def _with_context(**changes) -> GenerationConfig:
    contexts = [dict(c) for c in BASE["contexts"]]
    contexts[1].update(changes)
    return _config(contexts=contexts)


def test_answer_id_changes_when_a_context_definition_changes_without_renaming():
    base = _config().answer_id("bm25", "q1")
    assert _with_context(retriever="bm25").answer_id("bm25", "q1") == base
    assert _with_context(retriever="dense:hashing").answer_id("bm25", "q1") != base
    assert _with_context(chunker="fixed_200_50").answer_id("bm25", "q1") != base


def test_answer_id_changes_with_n_passages_and_generator_settings():
    base = _config().answer_id("bm25", "q1")
    assert _config(n_passages=3).answer_id("bm25", "q1") != base
    assert _config(oracle_chunker="fixed_200_50").answer_id("oracle", "q1") != _config().answer_id(
        "oracle", "q1"
    )
    generator = BASE["generator"]
    assert _config(generator={**generator, "temperature": 0.7}).answer_id("bm25", "q1") != base
    assert _config(generator={**generator, "max_tokens": 100}).answer_id("bm25", "q1") != base


def test_answer_id_changes_with_the_context_window():
    base = _config().answer_id("bm25", "q1")
    generator = BASE["generator"]
    assert _config(generator={**generator, "num_ctx": 8192}).answer_id("bm25", "q1") == base
    assert _config(generator={**generator, "num_ctx": 4096}).answer_id("bm25", "q1") != base
    assert len(base) == 12


def test_the_judge_window_does_not_change_answer_ids():
    base = _config().answer_id("bm25", "q1")
    assert _config(judge={**BASE["judge"], "num_ctx": 4096}).answer_id("bm25", "q1") == base


def test_answer_id_rejects_an_unknown_context():
    with pytest.raises(ValueError, match="nope"):
        _config().answer_id("nope", "q1")


def test_retrieved_context_requires_chunker_and_retriever():
    with pytest.raises(ValidationError, match="retrieved"):
        _config(contexts=[{"name": "x", "kind": "retrieved", "chunker": "none"}])


def test_empty_context_rejects_retrieval_fields():
    with pytest.raises(ValidationError, match="empty"):
        _config(contexts=[{"name": "x", "kind": "empty", "retriever": "bm25"}])


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValidationError, match="Unknown encoder"):
        _config(
            contexts=[
                {"name": "x", "kind": "retrieved", "chunker": "none", "retriever": "dense:nope"}
            ]
        )


def test_unknown_chunker_is_rejected():
    with pytest.raises(ValidationError, match="Unknown chunker"):
        _config(
            contexts=[{"name": "x", "kind": "retrieved", "chunker": "banana", "retriever": "bm25"}]
        )


def test_duplicate_context_names_are_rejected():
    with pytest.raises(ValidationError, match="unique"):
        _config(contexts=[{"name": "a", "kind": "empty"}, {"name": "a", "kind": "oracle"}])


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        _config(typo=1)


def test_load_generation_config(tmp_path):
    path = tmp_path / "v2.yaml"
    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    assert load_generation_config(path).generator.model == "llama3.2:latest"


def test_answer_id_depends_on_the_answer_prompt_version_only(monkeypatch):
    base = _config().answer_id("bm25", "q1")
    # Editing a judge prompt must not invalidate any answer.
    monkeypatch.setattr(prompts, "JUDGE_PROMPT_VERSION", "changed!")
    assert _config().answer_id("bm25", "q1") == base
    assert not hasattr(generation_config, "JUDGE_PROMPT_VERSION")
    # Editing an answer prompt must.
    monkeypatch.setattr(generation_config, "ANSWER_PROMPT_VERSION", "changed!")
    assert _config().answer_id("bm25", "q1") != base
