import pytest

from rag_eval_lab.config import DatasetConfig
from rag_eval_lab.data import alloprof_from_rows, load_benchmark, load_jsonl, sample_dataset
from rag_eval_lab.types import Dataset, Document, Query


def test_load_jsonl(mini_jsonl):
    dataset = load_jsonl(mini_jsonl)
    assert dataset.name == "mini"
    assert len(dataset.documents) == 20
    assert len(dataset.queries) == 5
    assert dataset.qrels["q0"] == {"d0": 1}


def test_alloprof_from_rows():
    docs = [{"uuid": "u1", "title": "Le volcan", "text": "Un volcan est...", "topic": "x"}]
    queries = [
        {"id": 7, "text": "c'est quoi un volcan", "relevant": ["u1"], "subject": "s"},
        {"id": 7, "text": "doublon", "relevant": ["u1"]},
    ]
    dataset = alloprof_from_rows(docs, queries)
    assert dataset.documents == [Document("u1", "Le volcan\n\nUn volcan est...")]
    assert dataset.queries == [Query("7", "c'est quoi un volcan")]
    assert dataset.qrels == {"7": {"u1": 1}}


def test_sample_queries_is_deterministic(mini_dataset):
    a = sample_dataset(mini_dataset, n_queries=3, n_docs=None, seed=1)
    b = sample_dataset(mini_dataset, n_queries=3, n_docs=None, seed=1)
    assert [q.id for q in a.queries] == [q.id for q in b.queries]
    assert len(a.queries) == 3
    assert set(a.qrels) == {q.id for q in a.queries}
    assert len(a.documents) == 20


def test_sample_docs_keeps_relevant_documents(mini_dataset):
    sampled = sample_dataset(mini_dataset, n_queries=2, n_docs=6, seed=0)
    doc_ids = {d.id for d in sampled.documents}
    assert len(doc_ids) == 6
    for rel in sampled.qrels.values():
        assert set(rel) <= doc_ids


def test_sample_drops_queries_without_relevant_documents():
    dataset = Dataset(
        name="t",
        documents=[Document("d1", "texte")],
        queries=[Query("q1", "a"), Query("q2", "b")],
        qrels={"q1": {"d1": 1}, "q2": {"missing": 1}},
    )
    sampled = sample_dataset(dataset, n_queries=None, n_docs=None, seed=0)
    assert [q.id for q in sampled.queries] == ["q1"]
    assert sampled.qrels == {"q1": {"d1": 1}}


def test_sample_rejects_too_small_n_docs(mini_dataset):
    with pytest.raises(ValueError, match="n_docs"):
        sample_dataset(mini_dataset, n_queries=5, n_docs=2, seed=0)


def test_load_benchmark_jsonl(mini_jsonl):
    cfg = DatasetConfig(name="jsonl", path=str(mini_jsonl), n_queries=2, seed=3)
    dataset = load_benchmark(cfg)
    assert len(dataset.queries) == 2
