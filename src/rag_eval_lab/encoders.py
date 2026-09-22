"""Text encoders producing L2-normalized float32 embeddings."""

from __future__ import annotations

import zlib
from typing import Any, Protocol

import numpy as np

from rag_eval_lab.text import tokenize


class Encoder(Protocol):
    name: str

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...


class HashingEncoder:
    """Bag-of-words hashing trick. Model-free baseline, also used in tests."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


class SentenceTransformerEncoder:
    """Wrapper around a sentence-transformers model, loaded lazily."""

    def __init__(
        self,
        name: str,
        model_id: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        max_seq_length: int = 512,
        model: Any | None = None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device="cpu")
            self._model.max_seq_length = self.max_seq_length
        return self._model

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        vectors = self._get_model().encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)
