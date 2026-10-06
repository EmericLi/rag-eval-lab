"""Context sources: what passages a generator is given for a query."""

from __future__ import annotations

from typing import Protocol

from rag_eval_lab.chunking import Chunker
from rag_eval_lab.pipeline import RetrievalPipeline
from rag_eval_lab.types import Chunk, Document, Qrels, Query


class ContextSource(Protocol):
    name: str

    def passages(self, query: Query) -> list[Chunk]: ...


class EmptyContext:
    """No passage at all: measures what the generator answers from memory."""

    def __init__(self, name: str = "none") -> None:
        self.name = name

    def passages(self, query: Query) -> list[Chunk]:
        return []


class RetrievedContext:
    """Top passages from a v1 retrieval pipeline."""

    def __init__(self, name: str, pipeline: RetrievalPipeline, n_passages: int) -> None:
        self.name = name
        self.pipeline = pipeline
        self.n_passages = n_passages

    def passages(self, query: Query) -> list[Chunk]:
        return [hit.chunk for hit in self.pipeline.search_passages(query.text, self.n_passages)]


class OracleContext:
    """Upper bound: the first chunks of the labelled documents, round-robin.

    Passages are never selected by similarity to the question — that would build a
    perfect context out of the question itself and inflate the bound.
    """

    def __init__(
        self,
        name: str,
        qrels: Qrels,
        documents: list[Document],
        chunker: Chunker,
        n_passages: int,
    ) -> None:
        self.name = name
        self.qrels = qrels
        self.chunker = chunker
        self.n_passages = n_passages
        self._chunks = {doc.id: chunker.chunk(doc) for doc in documents}

    def passages(self, query: Query) -> list[Chunk]:
        gold = sorted(d for d, gain in self.qrels.get(query.id, {}).items() if gain > 0)
        selected: list[Chunk] = []
        position = 0
        while len(selected) < self.n_passages:
            available = [d for d in gold if len(self._chunks.get(d, [])) > position]
            if not available:
                break
            for doc_id in available:
                selected.append(self._chunks[doc_id][position])
                if len(selected) == self.n_passages:
                    break
            position += 1
        return selected
