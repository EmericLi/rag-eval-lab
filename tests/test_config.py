import pytest
import yaml
from pydantic import ValidationError

from rag_eval_lab.config import ExperimentConfig, RunSpec, load_config
from rag_eval_lab.encoders import SentenceTransformerEncoder
from rag_eval_lab.factory import build_reranker, build_retriever
from rag_eval_lab.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

BASE = {
    "dataset": {"name": "jsonl", "path": "data/mini"},
    "grid": {
        "chunker": ["none", "fixed_100_20"],
        "retriever": ["bm25", "dense:hashing"],
        "reranker": ["none", "minilm"],
    },
    "exclude": [{"chunker": "none", "reranker": "minilm"}],
}


def _config(**overrides) -> ExperimentConfig:
    data = {**BASE, **overrides}
    return ExperimentConfig.model_validate(data)


def test_expand_grid_applies_exclude():
    specs = _config().expand()
    assert len(specs) == 6
    assert RunSpec(chunker="none", retriever="bm25", reranker="minilm") not in specs
    assert RunSpec(chunker="fixed_100_20", retriever="bm25", reranker="minilm") in specs


def test_unknown_encoder_rejected():
    grid = {**BASE["grid"], "retriever": ["dense:unknown"]}
    with pytest.raises(ValidationError, match="Unknown encoder"):
        _config(grid=grid)


def test_unknown_chunker_rejected():
    grid = {**BASE["grid"], "chunker": ["banana"]}
    with pytest.raises(ValidationError, match="Unknown chunker"):
        _config(grid=grid)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        _config(typo_field=1)


def test_jsonl_dataset_requires_path():
    with pytest.raises(ValidationError, match="path"):
        _config(dataset={"name": "jsonl"})


def test_ks_must_not_exceed_k():
    with pytest.raises(ValidationError, match="ks"):
        _config(k=5, ks=[5, 10])


def test_run_id_is_stable_and_depends_on_dataset():
    spec = RunSpec(chunker="none", retriever="bm25")
    first = _config().run_id(spec)
    assert first == _config().run_id(spec)
    assert len(first) == 12
    other = _config(dataset={"name": "jsonl", "path": "data/mini", "seed": 7}).run_id(spec)
    assert other != first
    other_ks = _config(ks=[5]).run_id(spec)
    assert other_ks != first
    assert other_ks == _config(ks=[5]).run_id(spec)


def test_load_config_from_yaml(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    assert load_config(path).grid.retriever == ["bm25", "dense:hashing"]


def test_build_retriever_variants():
    assert isinstance(build_retriever("bm25"), BM25Retriever)
    dense = build_retriever("dense:e5-small")
    assert isinstance(dense, DenseRetriever)
    assert isinstance(dense.encoder, SentenceTransformerEncoder)
    assert dense.encoder.model_id == "intfloat/multilingual-e5-small"
    assert dense.encoder._model is None
    hybrid = build_retriever("hybrid:bm25+hashing")
    assert isinstance(hybrid, HybridRetriever)
    assert len(hybrid.retrievers) == 2
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever("sparse:splade")
    with pytest.raises(ValueError):
        build_retriever("hybrid:bm25")


def test_build_reranker():
    assert build_reranker("none") is None
    reranker = build_reranker("minilm")
    assert reranker.model_id == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    with pytest.raises(ValueError, match="Unknown reranker"):
        build_reranker("nope")
