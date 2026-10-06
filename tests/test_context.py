from rag_eval_lab.chunking import FixedChunker
from rag_eval_lab.context import EmptyContext, OracleContext, RetrievedContext
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.retrieval import BM25Retriever
from rag_eval_lab.types import Document, Query


def test_empty_context_returns_nothing():
    source = EmptyContext()
    assert source.name == "none"
    assert source.passages(Query("q1", "une question")) == []


def test_retrieved_context_returns_n_passages(mini_dataset):
    pipeline = RetrievalPipeline(FixedChunker(size=200, overlap=50), BM25Retriever())
    pipeline.index(mini_dataset.documents)
    source = RetrievedContext("bm25", pipeline, n_passages=3)
    passages = source.passages(mini_dataset.queries[0])
    assert len(passages) == 3
    assert passages[0].doc_id == "d0"


def test_oracle_context_uses_gold_documents_only(mini_dataset):
    source = OracleContext(
        "oracle",
        mini_dataset.qrels,
        mini_dataset.documents,
        FixedChunker(size=100, overlap=0),
        n_passages=2,
    )
    passages = source.passages(mini_dataset.queries[1])
    assert len(passages) == 2
    assert {p.doc_id for p in passages} == {"d1"}


def test_oracle_context_alternates_between_gold_documents():
    documents = [Document("d1", "a" * 300), Document("d2", "b" * 300)]
    qrels = {"q1": {"d1": 1, "d2": 1}}
    source = OracleContext("oracle", qrels, documents, FixedChunker(size=100, overlap=0), 3)
    passages = source.passages(Query("q1", "peu importe"))
    assert [p.doc_id for p in passages] == ["d1", "d2", "d1"]


def test_oracle_context_ignores_the_question_text():
    documents = [Document("d1", "x" * 250)]
    source = OracleContext(
        "oracle", {"q1": {"d1": 1}}, documents, FixedChunker(size=100, overlap=0), 2
    )
    first = source.passages(Query("q1", "question A"))
    second = source.passages(Query("q1", "question B totalement différente"))
    assert [p.id for p in first] == [p.id for p in second]


def test_oracle_context_without_gold_documents_returns_nothing():
    source = OracleContext(
        "oracle", {}, [Document("d1", "texte")], FixedChunker(size=100, overlap=0), 2
    )
    assert source.passages(Query("q1", "q")) == []
