import numpy as np

from rag_eval_lab.chunking import FixedChunker
from rag_eval_lab.pipeline import RetrievalPipeline, aggregate_to_docs
from rag_eval_lab.reranking import CrossEncoderReranker
from rag_eval_lab.retrieval import BM25Retriever
from rag_eval_lab.types import Chunk, Hit


class LengthModel:
    def predict(self, pairs, **kwargs):
        return np.array([len(text) for _, text in pairs], dtype=float)


class KeywordModel:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def predict(self, pairs, **kwargs):
        return np.array([1.0 if self.keyword in text else 0.0 for _, text in pairs])


def test_cross_encoder_reranker_reorders():
    hits = [
        Hit(Chunk("a#0", "a", "court"), 0.9),
        Hit(Chunk("b#0", "b", "beaucoup plus long"), 0.1),
    ]
    reranker = CrossEncoderReranker("fake", model=LengthModel())
    out = reranker.rerank("q", hits, k=2)
    assert [h.chunk.id for h in out] == ["b#0", "a#0"]
    assert out[0].score == len("beaucoup plus long")


def test_cross_encoder_reranker_empty_and_truncation():
    reranker = CrossEncoderReranker("fake", model=LengthModel())
    assert reranker.rerank("q", [], k=5) == []
    hits = [Hit(Chunk(f"x#{i}", "x", "t" * i), 0.0) for i in range(1, 5)]
    assert len(reranker.rerank("q", hits, k=2)) == 2


def test_cross_encoder_reranker_is_lazy():
    assert CrossEncoderReranker("some/model")._model is None


def test_aggregate_to_docs_uses_max_score_and_dedups():
    hits = [
        Hit(Chunk("a#1", "a", "x"), 0.9),
        Hit(Chunk("b#0", "b", "y"), 0.8),
        Hit(Chunk("a#0", "a", "z"), 0.7),
        Hit(Chunk("c#0", "c", "w"), 0.95),
    ]
    assert aggregate_to_docs(hits, k=2) == ["c", "a"]
    assert aggregate_to_docs(hits, k=10) == ["c", "a", "b"]


def test_pipeline_bm25_on_mini_dataset(mini_dataset):
    pipeline = RetrievalPipeline(FixedChunker(size=200, overlap=50), BM25Retriever())
    pipeline.index(mini_dataset.documents)
    assert pipeline.n_chunks > len(mini_dataset.documents)
    assert pipeline.search("Comment expliquer le volcan ?", k=3)[0] == "d0"


def test_pipeline_with_reranker_overrides_first_stage(mini_dataset):
    pipeline = RetrievalPipeline(
        FixedChunker(size=200, overlap=50),
        BM25Retriever(),
        reranker=CrossEncoderReranker("fake", model=KeywordModel("atome")),
        candidates=1000,
    )
    pipeline.index(mini_dataset.documents)
    assert pipeline.search("Comment expliquer le volcan ?", k=3)[0] == "d5"
