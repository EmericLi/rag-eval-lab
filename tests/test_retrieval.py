import numpy as np
import pytest

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.encoders import HashingEncoder
from rag_eval_lab.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from rag_eval_lab.types import Chunk, Hit

CHUNKS = [
    Chunk("a#0", "a", "le volcan est actif"),
    Chunk("b#0", "b", "la photosynthese des plantes vertes"),
    Chunk("c#0", "c", "une fraction et un denominateur"),
]


def test_bm25_ranks_matching_chunk_first():
    retriever = BM25Retriever()
    retriever.index(CHUNKS)
    hits = retriever.search("volcan", k=2)
    assert len(hits) == 2
    assert hits[0].chunk.id == "a#0"
    assert hits[0].score >= hits[1].score


def test_search_k_larger_than_corpus():
    retriever = BM25Retriever()
    retriever.index(CHUNKS)
    assert len(retriever.search("volcan", k=50)) == 3


def test_dense_retriever_ranks_and_caches(tmp_path):
    cache = EmbeddingCache(tmp_path)
    for _ in range(2):
        retriever = DenseRetriever(HashingEncoder(dim=256), cache=cache)
        retriever.index(CHUNKS)
        hits = retriever.search("una fraction", k=3)
        assert hits[0].chunk.id == "c#0"
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(list(tmp_path.glob("*.npy"))) == 1


class FixedRetriever:
    def __init__(self, ranked_ids: list[str]) -> None:
        self.ranked_ids = ranked_ids

    def index(self, chunks):
        self.by_id = {c.id: c for c in chunks}

    def search(self, query, k):
        return [Hit(self.by_id[i], 1.0) for i in self.ranked_ids[:k]]


def test_hybrid_rrf_fusion():
    hybrid = HybridRetriever(
        [FixedRetriever(["a#0", "b#0", "c#0"]), FixedRetriever(["b#0", "c#0", "a#0"])], rrf_k=60
    )
    hybrid.index(CHUNKS)
    hits = hybrid.search("anything", k=3)
    assert [h.chunk.id for h in hits] == ["b#0", "a#0", "c#0"]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_requires_two_retrievers():
    with pytest.raises(ValueError):
        HybridRetriever([BM25Retriever()])


def test_hybrid_with_real_retrievers():
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(HashingEncoder(dim=256))])
    hybrid.index(CHUNKS)
    assert hybrid.search("photosynthese", k=1)[0].chunk.id == "b#0"
    assert isinstance(hybrid.search("photosynthese", k=1)[0].score, float)
    assert np.isfinite(hybrid.search("photosynthese", k=1)[0].score)
