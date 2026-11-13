"""Build pipeline components from compact spec strings used in configs."""

from __future__ import annotations

from collections.abc import Callable

from rag_eval_lab.cache import EmbeddingCache
from rag_eval_lab.encoders import Encoder, HashingEncoder, SentenceTransformerEncoder
from rag_eval_lab.reranking import CrossEncoderReranker, Reranker
from rag_eval_lab.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from rag_eval_lab.text import analyze_fr, tokenize

ENCODERS: dict[str, dict[str, str]] = {
    "e5-small": {
        "model_id": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "e5-base": {
        "model_id": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "bge-m3": {"model_id": "BAAI/bge-m3"},
}

# `bm25` keeps the raw lowercase tokenizer of the first v1 run, as a reference point;
# `bm25-fr` is the tuned baseline.
LEXICAL: dict[str, Callable[[str], list[str]]] = {"bm25": tokenize, "bm25-fr": analyze_fr}

RERANKERS: dict[str, str] = {
    "minilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
}


def build_encoder(name: str) -> Encoder:
    if name == "hashing":
        return HashingEncoder()
    if name in ENCODERS:
        return SentenceTransformerEncoder(name=name, **ENCODERS[name])
    known = ", ".join(["hashing", *ENCODERS])
    raise ValueError(f"Unknown encoder '{name}'. Known encoders: {known}")


def build_retriever(spec: str, cache: EmbeddingCache | None = None) -> Retriever:
    kind, _, arg = spec.partition(":")
    if spec in LEXICAL:
        return BM25Retriever(LEXICAL[spec])
    if kind == "dense" and arg:
        return DenseRetriever(build_encoder(arg), cache=cache)
    if kind == "hybrid" and arg:
        parts = arg.split("+")
        if len(parts) < 2:
            raise ValueError(f"Hybrid retriever '{spec}' needs at least two parts")
        return HybridRetriever(
            [build_retriever(p if p in LEXICAL else f"dense:{p}", cache) for p in parts]
        )
    raise ValueError(
        f"Unknown retriever spec '{spec}'. Expected 'bm25', 'bm25-fr', 'dense:<encoder>' "
        "or 'hybrid:<a>+<b>'."
    )


def build_reranker(spec: str) -> Reranker | None:
    if spec == "none":
        return None
    if spec in RERANKERS:
        return CrossEncoderReranker(RERANKERS[spec])
    known = ", ".join(["none", *RERANKERS])
    raise ValueError(f"Unknown reranker '{spec}'. Known rerankers: {known}")
