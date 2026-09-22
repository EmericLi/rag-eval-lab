"""First-stage retrievers over chunks: lexical, dense and hybrid (RRF)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from rag_eval_lab.cache import EmbeddingCache, make_key
from rag_eval_lab.encoders import Encoder
from rag_eval_lab.text import tokenize
from rag_eval_lab.types import Chunk, Hit


class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[Hit]: ...


def _top_k(chunks: list[Chunk], scores: np.ndarray, k: int) -> list[Hit]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [Hit(chunks[i], float(scores[i])) for i in order]


class BM25Retriever:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self._chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.search called before index")
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        return _top_k(self._chunks, scores, k)


class DenseRetriever:
    def __init__(self, encoder: Encoder, cache: EmbeddingCache | None = None) -> None:
        self.encoder = encoder
        self.cache = cache
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]

        def compute() -> np.ndarray:
            return self.encoder.encode_passages(texts)

        if self.cache is None:
            self._embeddings = compute()
        else:
            key = make_key(self.encoder.name, *(c.id for c in self._chunks), *texts)
            self._embeddings = self.cache.get_or_compute(key, compute)

    def search(self, query: str, k: int) -> list[Hit]:
        if self._embeddings is None:
            raise RuntimeError("DenseRetriever.search called before index")
        query_vector = self.encoder.encode_queries([query])[0]
        scores = self._embeddings @ query_vector
        return _top_k(self._chunks, scores, k)


class HybridRetriever:
    """Reciprocal Rank Fusion of several retrievers."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 100) -> None:
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever needs at least two retrievers")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def index(self, chunks: list[Chunk]) -> None:
        for retriever in self.retrievers:
            retriever.index(chunks)

    def search(self, query: str, k: int) -> list[Hit]:
        depth = max(k, self.depth)
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            for rank, hit in enumerate(retriever.search(query, depth), start=1):
                chunks[hit.chunk.id] = hit.chunk
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
        return [Hit(chunks[cid], scores[cid]) for cid in ranked]
