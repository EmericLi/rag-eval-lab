"""Assemble chunker, retriever and optional reranker into a document ranker."""

from __future__ import annotations

from rag_eval_lab.chunking import Chunker
from rag_eval_lab.reranking import Reranker
from rag_eval_lab.retrieval import Retriever
from rag_eval_lab.types import Document, Hit


def aggregate_to_docs(hits: list[Hit], k: int) -> list[str]:
    """Map chunk hits to doc ids: a document scores as its best chunk."""
    best: dict[str, float] = {}
    for hit in hits:
        doc_id = hit.chunk.doc_id
        if doc_id not in best or hit.score > best[doc_id]:
            best[doc_id] = hit.score
    return sorted(best, key=lambda d: best[d], reverse=True)[:k]


class RetrievalPipeline:
    def __init__(
        self,
        chunker: Chunker,
        retriever: Retriever,
        reranker: Reranker | None = None,
        candidates: int = 100,
    ) -> None:
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.candidates = candidates
        self.n_chunks = 0

    def index(self, documents: list[Document]) -> None:
        chunks = [chunk for doc in documents for chunk in self.chunker.chunk(doc)]
        self.retriever.index(chunks)
        self.n_chunks = len(chunks)
        if self.reranker is not None:
            self.reranker.load()

    def search(self, query: str, k: int) -> list[str]:
        hits = self.retriever.search(query, max(self.candidates, k))
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, len(hits))
        return aggregate_to_docs(hits, k)
