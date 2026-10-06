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
    # Create a document with 3 distinct 100-char chunks so similarity-based selection
    # would rank them differently, but oracle ignores question and returns in doc order
    chunk1 = "alpha " + "a" * 94  # 100 chars total, starts with "alpha"
    chunk2 = "beta " + "b" * 95  # 100 chars total, starts with "beta"
    chunk3 = "gamma " + "g" * 94  # 100 chars total, starts with "gamma"
    text = chunk1 + chunk2 + chunk3

    documents = [Document("d1", text)]
    source = OracleContext(
        "oracle", {"q1": {"d1": 1}}, documents, FixedChunker(size=100, overlap=0), 3
    )

    # Query mentioning "gamma" — similarity-based selector would rank chunk3 first
    with_gamma = source.passages(Query("q1", "What about gamma?"))
    # Query with unrelated text — no special ranking
    unrelated = source.passages(Query("q1", "xyz xyz xyz"))

    # Oracle returns chunks in document order regardless of question
    chunk_ids = [p.id for p in with_gamma]
    assert chunk_ids == [p.id for p in unrelated]
    assert chunk_ids == ["d1#0", "d1#1", "d1#2"]


def test_oracle_context_n_passages_larger_than_available():
    # Requesting more passages than available should return all without error or padding
    documents = [Document("d1", "x" * 150)]  # 2 chunks with size=100, overlap=0
    source = OracleContext(
        "oracle", {"q1": {"d1": 1}}, documents, FixedChunker(size=100, overlap=0), 10
    )
    passages = source.passages(Query("q1", "test"))
    assert len(passages) == 2
    assert [p.doc_id for p in passages] == ["d1", "d1"]


def test_oracle_context_unequal_chunk_counts():
    # Round-robin with different document sizes: alternate while available, then drain
    doc1_text = "a" * 250  # 3 chunks: a#0, a#1, a#2
    doc2_text = "b" * 100  # 1 chunk: b#0
    documents = [Document("d1", doc1_text), Document("d2", doc2_text)]
    source = OracleContext(
        "oracle",
        {"q1": {"d1": 1, "d2": 1}},
        documents,
        FixedChunker(size=100, overlap=0),
        5,
    )
    passages = source.passages(Query("q1", "test"))
    # Round-robin: d1#0, d2#0, d1#1, (d2 exhausted), d1#2
    assert [p.id for p in passages] == ["d1#0", "d2#0", "d1#1", "d1#2"]


def test_oracle_context_absent_gold_id():
    # A qrel referencing a document not in corpus should be skipped
    documents = [Document("d1", "x" * 100), Document("d2", "y" * 100)]
    qrels = {"q1": {"d1": 1, "d3": 1}}  # d3 does not exist
    source = OracleContext("oracle", qrels, documents, FixedChunker(size=100, overlap=0), 2)
    passages = source.passages(Query("q1", "test"))
    # Only d1 is available, so return only its chunk(s)
    assert len(passages) == 1
    assert passages[0].doc_id == "d1"


def test_retrieved_context_returns_chunk_instances():
    # RetrievedContext.passages() should return Chunk instances, not ids
    from rag_eval_lab.types import Chunk

    pipeline = RetrievalPipeline(FixedChunker(size=200, overlap=50), BM25Retriever())
    documents = [Document("d1", "sample document text for retrieval testing")]
    pipeline.index(documents)
    source = RetrievedContext("bm25", pipeline, n_passages=1)
    passages = source.passages(Query("q1", "sample"))

    assert len(passages) >= 1
    assert isinstance(passages[0], Chunk)


def test_oracle_context_without_gold_documents_returns_nothing():
    source = OracleContext(
        "oracle", {}, [Document("d1", "texte")], FixedChunker(size=100, overlap=0), 2
    )
    assert source.passages(Query("q1", "q")) == []
