"""Core data types shared by every module."""

from __future__ import annotations

from dataclasses import dataclass

Qrels = dict[str, dict[str, int]]
"""Relevance judgments: query_id -> {doc_id: gain}."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: list[Document]
    queries: list[Query]
    qrels: Qrels


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
