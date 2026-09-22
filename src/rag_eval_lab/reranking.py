"""Second-stage rerankers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from rag_eval_lab.types import Hit


class Reranker(Protocol):
    def load(self) -> None: ...

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class CrossEncoderReranker:
    """Cross-encoder scoring (query, chunk) pairs; model loaded lazily."""

    def __init__(
        self,
        model_id: str,
        batch_size: int = 32,
        max_length: int = 512,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = model

    def load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, max_length=self.max_length, device="cpu")

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        self.load()
        pairs = [(query, hit.chunk.text) for hit in hits]
        scores = np.asarray(
            self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
            dtype=np.float64,
        )
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(hits[i].chunk, float(scores[i])) for i in order]
